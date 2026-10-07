// Fused multi-layer b1 Tensor-Core ANF megakernel — full-machine cycle-accurate sim.
//
// One warp owns 8 stimuli; the net-state V[8][n_nets] stays resident in shared memory.
// A single kernel loops over ALL combinational layers and ALL clock cycles: each layer is
// evaluated by the two-stage b1 `and.popc` MMA (stage1 monomial features via X&A, stage2
// GF(2) combine via phi&C), signals route between layers through V, and clock edges commit
// next-state — so there are NO per-layer kernel launches and NO host-side routing.
// Per-layer ANF data (A, C, deg, input/output net indices) is passed as flat blobs with
// per-layer offsets in `meta`. Bit-exact vs iverilog golden.

#include <mma.h>
#include <cstdint>
#include <cuda_runtime.h>
using namespace nvcuda;

#define WM 8
#define WN 8
#define WK 128
#define KW (WK / 32)
#ifndef MAXSTATE
#define MAXSTATE 512
#endif
// meta layout per layer (12 ints):
#define M_F 0
#define M_NIN 1
#define M_G 2
#define M_NMT 3
#define M_NKT 4
#define M_NOT 5
#define M_NFT 6
#define M_AOFF 7
#define M_COFF 8
#define M_DOFF 9
#define M_IOFF 10
#define M_OOFF 11

extern "C" __global__ void b1_mega(
    int batch, int n_nets, int n_layers, int cycles,
    const int* __restrict__ meta,               // [n_layers][12]
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_nets_all, const int* __restrict__ out_nets_all,
    int n_state, const int* __restrict__ state_nets, const int* __restrict__ dff_d_nets,
    int n_input, const int* __restrict__ input_nets, int one_net,
    int n_po, const int* __restrict__ po_nets,
    const signed char* __restrict__ u,          // [cycles, n_input, batch]
    signed char* __restrict__ po_out,           // [cycles, n_po, batch]
    int maxNkt, int maxPhiPadW)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;                          // fragment row (stimulus)
    int fc = (lane & 3) * 2;                     // fragment col base

    extern __shared__ signed char smem[];
    // per-warp: V[8*n_nets] int8 (padded to 16B so the wmma scratch stays 16B-aligned),
    // then XkAll[maxNkt*8*4] u32, phi[8*maxPhiPadW] u32
    int Vpad = (8 * n_nets + 15) & ~15;
    int perWarpBytes = Vpad + (maxNkt * WM * KW + WM * maxPhiPadW) * 4;
    signed char* wbase = smem + warp_in_block * perWarpBytes;
    signed char* V = wbase;                      // V[s*n_nets + net]
    unsigned* XkAll = (unsigned*)(wbase + Vpad);
    unsigned* phi   = XkAll + maxNkt * WM * KW;

    int nMtiles = (batch + WM - 1) / WM;
    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int mtile = gwarp; mtile < nMtiles; mtile += total_warps) {
        int stim0 = mtile * WM;
        for (int i = lane; i < 8 * n_nets; i += 32) V[i] = 0;       // x0 = 0
        if (one_net >= 0) for (int s = lane; s < WM; s += 32) V[s * n_nets + one_net] = 1;
        __syncwarp();

        for (int t = 0; t < cycles; t++) {
            // set primary inputs
            for (int i = 0; i < n_input; i++) {
                int net = input_nets[i];
                for (int s = lane; s < WM; s += 32) {
                    int stim = stim0 + s;
                    V[s * n_nets + net] = (stim < batch) ? u[(size_t)(t * n_input + i) * batch + stim] : 0;
                }
            }
            __syncwarp();

            for (int l = 0; l < n_layers; l++) {
                const int* md = meta + l * 12;
                int F = md[M_F], NIN = md[M_NIN], G = md[M_G];
                int n_mt = md[M_NMT], n_kt = md[M_NKT], n_ot = md[M_NOT], n_ft = md[M_NFT];
                const unsigned* A = A_all + md[M_AOFF];
                const unsigned* C = C_all + md[M_COFF];
                const int* deg = deg_all + md[M_DOFF];
                const int* in_nets = in_nets_all + md[M_IOFF];
                const int* out_nets = out_nets_all + md[M_OOFF];
                int phiW = (F + 31) / 32;
                int phiPadW = (phiW + KW - 1) / KW * KW;

                // pack X (layer inputs) for all K-tiles from V
                for (int idx = lane; idx < n_kt * WM * KW; idx += 32) {
                    int kt = idx / (WM * KW), r = idx % (WM * KW), s = r / KW, w = r % KW;
                    unsigned v = 0u;
                    #pragma unroll
                    for (int b = 0; b < 32; b++) {
                        int k = kt * WK + w * 32 + b;
                        if (k < NIN && V[s * n_nets + in_nets[k]]) v |= (1u << b);
                    }
                    XkAll[idx] = v;
                }
                for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
                __syncwarp();

                // stage 1: phi = (popc(X & A) == deg)
                for (int mt = 0; mt < n_mt; mt++) {
                    wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
                    wmma::fill_fragment(acc, 0);
                    for (int kt = 0; kt < n_kt; kt++) {
                        wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::experimental::precision::b1, wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::experimental::precision::b1, wmma::col_major> b;
                        wmma::load_matrix_sync(a, XkAll + kt * WM * KW, WK);
                        wmma::load_matrix_sync(b, A + ((size_t)mt * n_kt + kt) * WN * KW, WK);
                        wmma::bmma_sync(acc, a, b, acc, wmma::experimental::bmmaBitOpAND,
                                        wmma::experimental::bmmaAccumulateOpPOPC);
                    }
                    int m0 = mt * WN + fc, m1 = m0 + 1;
                    if (m0 < F && acc.x[0] == deg[m0]) atomicOr(&phi[fr * phiPadW + (m0 >> 5)], 1u << (m0 & 31));
                    if (m1 < F && acc.x[1] == deg[m1]) atomicOr(&phi[fr * phiPadW + (m1 >> 5)], 1u << (m1 & 31));
                }
                __syncwarp();

                // stage 2: y = popc(phi & C) & 1, write to V[out_nets]
                for (int ot = 0; ot < n_ot; ot++) {
                    wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
                    wmma::fill_fragment(acc, 0);
                    for (int ft = 0; ft < n_ft; ft++) {
                        wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::experimental::precision::b1, wmma::row_major> a;
                        wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::experimental::precision::b1, wmma::col_major> b;
                        wmma::load_matrix_sync(a, phi + ft * KW, phiPadW * 32);
                        wmma::load_matrix_sync(b, C + ((size_t)ot * n_ft + ft) * WN * KW, WK);
                        wmma::bmma_sync(acc, a, b, acc, wmma::experimental::bmmaBitOpAND,
                                        wmma::experimental::bmmaAccumulateOpPOPC);
                    }
                    int o0 = ot * WN + fc, o1 = o0 + 1;
                    if (o0 < G) V[fr * n_nets + out_nets[o0]] = (signed char)(acc.x[0] & 1);
                    if (o1 < G) V[fr * n_nets + out_nets[o1]] = (signed char)(acc.x[1] & 1);
                }
                __syncwarp();
            }

            // sample POs
            for (int k = 0; k < n_po; k++) {
                int net = po_nets[k];
                for (int s = lane; s < WM; s += 32) {
                    int stim = stim0 + s;
                    if (stim < batch) po_out[(size_t)(t * n_po + k) * batch + stim] = V[s * n_nets + net] & 1;
                }
            }
            // commit next state (read all D before writing Q — D may alias a state net)
            for (int s = lane; s < WM; s += 32) {
                signed char nxt[MAXSTATE];
                for (int j = 0; j < n_state; j++) nxt[j] = V[s * n_nets + dff_d_nets[j]] & 1;
                for (int j = 0; j < n_state; j++) V[s * n_nets + state_nets[j]] = nxt[j];
            }
            __syncwarp();
        }
    }
}

extern "C" void launch_b1_mega(
    int batch, int n_nets, int n_layers, int cycles, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_nets_all, const int* out_nets_all,
    int n_state, const int* state_nets, const int* dff_d_nets,
    int n_input, const int* input_nets, int one_net,
    int n_po, const int* po_nets, const signed char* u, signed char* po_out,
    int maxNkt, int maxPhiPadW, int warps_per_block, int grid_blocks, int smem_bytes)
{
    cudaFuncSetAttribute(b1_mega, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    b1_mega<<<grid_blocks, warps_per_block * 32, smem_bytes>>>(
        batch, n_nets, n_layers, cycles, meta, A_all, C_all, deg_all,
        in_nets_all, out_nets_all, n_state, state_nets, dff_d_nets,
        n_input, input_nets, one_net, n_po, po_nets, u, po_out, maxNkt, maxPhiPadW);
}
