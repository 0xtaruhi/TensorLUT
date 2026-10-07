// b1 (single-bit) Tensor-Core ANF evaluation — the core contribution.
//
//   stage 1:  cnt[stim,mono] = popc_k( X[stim,k]  & A[k,mono] );  phi = (cnt == deg)
//   stage 2:  acc[stim,out]  = popc_f( phi[stim,f] & C[f,out] );  y   = acc & 1
//
// Both stages use wmma b1 `and.popc` (8x8x128). ncu showed the earlier version was
// OCCUPANCY-BOUND: caching A/C/deg in shared memory used ~52 KB/block, capping the SM at
// 1 resident block (16.7% occupancy) so the (synchronous) MMA + compare latency could not
// be hidden. Fix: keep A/C/deg in GLOBAL (they are small and stay hot in L2, reused by all
// blocks) and put ONLY the small per-warp scratch in shared -> many blocks/SM -> the extra
// warps hide the epilogue latency. X and phi are still pre-packed once per tile.
// Bit-exact vs numpy ANF ref.

#include <mma.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
using namespace nvcuda;
namespace cg = cooperative_groups;

#define WM 8
#define WN 8
#define WK 128
#define KW (WK / 32)
#define LV_F 0
#define LV_NIN 1
#define LV_G 2
#define LV_NMT 3
#define LV_NKT 4
#define LV_NOT 5
#define LV_NFT 6
#define LV_AOFF 7
#define LV_COFF 8
#define LV_DOFF 9
#define LV_IOFF 10
#define LV_OOFF 11
#define LV_FTC 12
#define LV_BPOFF 13
#define LV_NBYP 14
#define LV_LAYER_META_WORDS 15

extern "C" __global__ void b1_anf(
    const signed char* __restrict__ X, int batch, int NIN,
    const unsigned* __restrict__ A, int F, int n_mt, int n_kt,
    const int* __restrict__ deg,
    const unsigned* __restrict__ C, int G, int n_ot, int n_ft,
    signed char* __restrict__ Y)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int phiW = (F + 31) / 32;

    int phiPadW = (phiW + KW - 1) / KW * KW;               // pad rows to a KW multiple so
    extern __shared__ unsigned smem[];                     // stage-2 can load phi directly
    int perWarp = n_kt * WM * KW + WM * phiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll     = base;                            // [kt][s][w]
    unsigned* phi       = XkAll + n_kt * WM * KW;          // [s][phiPadW]  (row-major)
    // wmma 8x8 s32 accumulator fragment layout: lane L holds C[L/4][(L%4)*2 + i], i=0,1
    int fr = lane >> 2;              // fragment row  (stimulus within tile)
    int fc = (lane & 3) * 2;         // fragment col base (output within tile)

    int nMtiles = (batch + WM - 1) / WM;
    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int mtile = gwarp; mtile < nMtiles; mtile += total_warps) {
        int stim0 = mtile * WM;
        for (int idx = lane; idx < n_kt * WM * KW; idx += 32) {
            int kt = idx / (WM * KW), r = idx % (WM * KW), s = r / KW, w = r % KW;
            int stim = stim0 + s;
            unsigned v = 0u;
            #pragma unroll
            for (int b = 0; b < 32; b++) {
                int k = kt * WK + w * 32 + b;
                if (k < NIN && stim < batch && X[(size_t)stim * NIN + k]) v |= (1u << b);
            }
            XkAll[idx] = v;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

        // ---- stage 1: A read from global (hot in L2) ----
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
            // register-level epilogue: read cnt straight from the fragment (no store->shared)
            int m0 = mt * WN + fc, m1 = m0 + 1;
            if (m0 < F && acc.x[0] == deg[m0]) atomicOr(&phi[fr * phiPadW + (m0 >> 5)], 1u << (m0 & 31));
            if (m1 < F && acc.x[1] == deg[m1]) atomicOr(&phi[fr * phiPadW + (m1 >> 5)], 1u << (m1 & 31));
        }
        __syncwarp();

        // ---- stage 2: C read from global (hot in L2); phi read directly (no repack) ----
        for (int ot = 0; ot < n_ot; ot++) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
            wmma::fill_fragment(acc, 0);
            for (int ft = 0; ft < n_ft; ft++) {
                wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::experimental::precision::b1, wmma::row_major> a;
                wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::experimental::precision::b1, wmma::col_major> b;
                wmma::load_matrix_sync(a, phi + ft * KW, phiPadW * 32);   // row stride = phiPadW words
                wmma::load_matrix_sync(b, C + ((size_t)ot * n_ft + ft) * WN * KW, WK);
                wmma::bmma_sync(acc, a, b, acc, wmma::experimental::bmmaBitOpAND,
                                wmma::experimental::bmmaAccumulateOpPOPC);
            }
            // register-level epilogue: write outputs straight from the fragment
            int o0 = ot * WN + fc, o1 = o0 + 1, stim = stim0 + fr;
            if (stim < batch) {
                if (o0 < G) Y[(size_t)stim * G + o0] = (signed char)(acc.x[0] & 1);
                if (o1 < G) Y[(size_t)stim * G + o1] = (signed char)(acc.x[1] & 1);
            }
        }
    }
}

extern "C" __global__ void b1_anf_v(
    signed char* __restrict__ V, int batch, int n_cols,
    const int* __restrict__ in_cols, int NIN,
    const unsigned* __restrict__ A, int F, int n_mt, int n_kt,
    const int* __restrict__ deg,
    const unsigned* __restrict__ C, int G, int n_ot, int n_ft,
    const int* __restrict__ out_cols)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int phiW = (F + 31) / 32;

    int phiPadW = (phiW + KW - 1) / KW * KW;
    extern __shared__ unsigned smem[];
    int perWarp = n_kt * WM * KW + WM * phiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + n_kt * WM * KW;
    int fr = lane >> 2;
    int fc = (lane & 3) * 2;

    int nMtiles = (batch + WM - 1) / WM;
    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int mtile = gwarp; mtile < nMtiles; mtile += total_warps) {
        int stim0 = mtile * WM;
        for (int idx = lane; idx < n_kt * WM * KW; idx += 32) {
            int kt = idx / (WM * KW), r = idx % (WM * KW), s = r / KW, w = r % KW;
            int stim = stim0 + s;
            unsigned v = 0u;
            #pragma unroll
            for (int b = 0; b < 32; b++) {
                int k = kt * WK + w * 32 + b;
                if (k < NIN && stim < batch && V[(size_t)stim * n_cols + in_cols[k]]) v |= (1u << b);
            }
            XkAll[idx] = v;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

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
            int o0 = ot * WN + fc, o1 = o0 + 1, stim = stim0 + fr;
            if (stim < batch) {
                if (o0 < G) V[(size_t)stim * n_cols + out_cols[o0]] = (signed char)(acc.x[0] & 1);
                if (o1 < G) V[(size_t)stim * n_cols + out_cols[o1]] = (signed char)(acc.x[1] & 1);
            }
        }
    }
}

extern "C" __global__ void b1_anf_v8(
    unsigned char* __restrict__ V, int n_tiles, int n_cols,
    const int* __restrict__ in_cols, int NIN,
    const unsigned* __restrict__ A, int F, int n_mt, int n_kt,
    const int* __restrict__ deg,
    const unsigned* __restrict__ C, int G, int n_ot, int n_ft,
    const int* __restrict__ out_cols)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int phiW = (F + 31) / 32;

    int phiPadW = (phiW + KW - 1) / KW * KW;
    extern __shared__ unsigned smem[];
    int perWarp = n_kt * WM * KW + WM * phiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + n_kt * WM * KW;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int tile = gwarp; tile < n_tiles; tile += total_warps) {
        // Warp-ballot transpose: 32 lanes read 32 input bytes and directly form
        // the 8 row words consumed by BMMA.
        for (int idx = 0; idx < n_kt * KW; idx++) {
            int kt = idx / KW, w = idx % KW;
            int k = kt * WK + w * 32 + lane;
            unsigned bits = (k < NIN) ? (unsigned)V[(size_t)tile * n_cols + in_cols[k]] : 0u;
            unsigned xs0 = __ballot_sync(0xffffffffu, (bits & 0x01u) != 0);
            unsigned xs1 = __ballot_sync(0xffffffffu, (bits & 0x02u) != 0);
            unsigned xs2 = __ballot_sync(0xffffffffu, (bits & 0x04u) != 0);
            unsigned xs3 = __ballot_sync(0xffffffffu, (bits & 0x08u) != 0);
            unsigned xs4 = __ballot_sync(0xffffffffu, (bits & 0x10u) != 0);
            unsigned xs5 = __ballot_sync(0xffffffffu, (bits & 0x20u) != 0);
            unsigned xs6 = __ballot_sync(0xffffffffu, (bits & 0x40u) != 0);
            unsigned xs7 = __ballot_sync(0xffffffffu, (bits & 0x80u) != 0);
            size_t off = (size_t)kt * WM * KW + w;
            if (lane == 0) XkAll[off + 0 * KW] = xs0;
            if (lane == 1) XkAll[off + 1 * KW] = xs1;
            if (lane == 2) XkAll[off + 2 * KW] = xs2;
            if (lane == 3) XkAll[off + 3 * KW] = xs3;
            if (lane == 4) XkAll[off + 4 * KW] = xs4;
            if (lane == 5) XkAll[off + 5 * KW] = xs5;
            if (lane == 6) XkAll[off + 6 * KW] = xs6;
            if (lane == 7) XkAll[off + 7 * KW] = xs7;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

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
            unsigned mask0 = __ballot_sync(0xffffffffu, (acc.x[0] & 1) != 0);
            unsigned mask1 = __ballot_sync(0xffffffffu, (acc.x[1] & 1) != 0);
            if (fr == 0) {
                unsigned char y0 = 0, y1 = 0;
                #pragma unroll
                for (int r = 0; r < WM; r++) {
                    y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                    y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                }
                int o0 = ot * WN + fc, o1 = o0 + 1;
                if (o0 < G) V[(size_t)tile * n_cols + out_cols[o0]] = y0;
                if (o1 < G) V[(size_t)tile * n_cols + out_cols[o1]] = y1;
            }
        }
    }
}

extern "C" __global__ void b1_anf_xcols(
    const signed char* __restrict__ X, int batch, int XNIN,
    const int* __restrict__ x_cols, int NIN,
    const unsigned* __restrict__ A, int F, int n_mt, int n_kt,
    const int* __restrict__ deg,
    const unsigned* __restrict__ C, int G, int n_ot, int n_ft,
    signed char* __restrict__ Y)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int phiW = (F + 31) / 32;
    int phiPadW = (phiW + KW - 1) / KW * KW;
    extern __shared__ unsigned smem[];
    int perWarp = n_kt * WM * KW + WM * phiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + n_kt * WM * KW;
    int fr = lane >> 2;
    int fc = (lane & 3) * 2;

    int nMtiles = (batch + WM - 1) / WM;
    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int mtile = gwarp; mtile < nMtiles; mtile += total_warps) {
        int stim0 = mtile * WM;
        for (int idx = lane; idx < n_kt * WM * KW; idx += 32) {
            int kt = idx / (WM * KW), r = idx % (WM * KW), s = r / KW, w = r % KW;
            int stim = stim0 + s;
            unsigned v = 0u;
            #pragma unroll
            for (int b = 0; b < 32; b++) {
                int k = kt * WK + w * 32 + b;
                if (k < NIN && stim < batch && X[(size_t)stim * XNIN + x_cols[k]]) v |= (1u << b);
            }
            XkAll[idx] = v;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

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
            int o0 = ot * WN + fc, o1 = o0 + 1, stim = stim0 + fr;
            if (stim < batch) {
                if (o0 < G) Y[(size_t)stim * G + o0] = (signed char)(acc.x[0] & 1);
                if (o1 < G) Y[(size_t)stim * G + o1] = (signed char)(acc.x[1] & 1);
            }
        }
    }
}

extern "C" __global__ void b1_layer_v8(
    unsigned char* __restrict__ V, int n_tiles, int n_cols, int n_chunks,
    const int* __restrict__ meta,
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_cols_all, const int* __restrict__ out_cols_all,
    const int* __restrict__ bypass_all,
    int maxNkt, int maxPhiPadW)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    extern __shared__ unsigned smem[];
    int perWarp = maxNkt * WM * KW + WM * maxPhiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + maxNkt * WM * KW;

    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;
    int total_work = n_tiles * n_chunks;

    for (int work = gwarp; work < total_work; work += total_warps) {
        int tile = work / n_chunks;
        int ci = work - tile * n_chunks;
        const int* md = meta + ci * LV_LAYER_META_WORDS;
        int F = md[LV_F], NIN = md[LV_NIN], G = md[LV_G];
        int n_mt = md[LV_NMT], n_kt = md[LV_NKT], n_ot = md[LV_NOT], n_ft = md[LV_NFT];
        int Ftc = md[LV_FTC], n_bypass = md[LV_NBYP];
        const unsigned* A = A_all + md[LV_AOFF];
        const unsigned* C = C_all + md[LV_COFF];
        const int* deg = deg_all + md[LV_DOFF];
        const int* in_cols = in_cols_all + md[LV_IOFF];
        const int* out_cols = out_cols_all + md[LV_OOFF];
        const int* bypass = bypass_all + md[LV_BPOFF];
        int phiW = (F + 31) / 32;
        int phiPadW = (phiW + KW - 1) / KW * KW;

        for (int idx = 0; idx < n_kt * KW; idx++) {
            int kt = idx / KW, w = idx % KW;
            int k = kt * WK + w * 32 + lane;
            unsigned bits = (k < NIN) ? (unsigned)V[(size_t)tile * n_cols + in_cols[k]] : 0u;
            unsigned xs0 = __ballot_sync(0xffffffffu, (bits & 0x01u) != 0);
            unsigned xs1 = __ballot_sync(0xffffffffu, (bits & 0x02u) != 0);
            unsigned xs2 = __ballot_sync(0xffffffffu, (bits & 0x04u) != 0);
            unsigned xs3 = __ballot_sync(0xffffffffu, (bits & 0x08u) != 0);
            unsigned xs4 = __ballot_sync(0xffffffffu, (bits & 0x10u) != 0);
            unsigned xs5 = __ballot_sync(0xffffffffu, (bits & 0x20u) != 0);
            unsigned xs6 = __ballot_sync(0xffffffffu, (bits & 0x40u) != 0);
            unsigned xs7 = __ballot_sync(0xffffffffu, (bits & 0x80u) != 0);
            size_t off = (size_t)kt * WM * KW + w;
            if (lane == 0) XkAll[off + 0 * KW] = xs0;
            if (lane == 1) XkAll[off + 1 * KW] = xs1;
            if (lane == 2) XkAll[off + 2 * KW] = xs2;
            if (lane == 3) XkAll[off + 3 * KW] = xs3;
            if (lane == 4) XkAll[off + 4 * KW] = xs4;
            if (lane == 5) XkAll[off + 5 * KW] = xs5;
            if (lane == 6) XkAll[off + 6 * KW] = xs6;
            if (lane == 7) XkAll[off + 7 * KW] = xs7;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

        for (int j = lane; j < n_bypass; j += 32) {
            int pidx = bypass[2 * j + 0];
            int src_col = bypass[2 * j + 1];
            unsigned bits = (src_col < 0) ? 0xffu : (unsigned)V[(size_t)tile * n_cols + src_col];
            unsigned mask = 1u << (pidx & 31);
            int word = pidx >> 5;
            #pragma unroll
            for (int r = 0; r < WM; r++) {
                if ((bits >> r) & 1u) atomicOr(&phi[r * phiPadW + word], mask);
            }
        }
        __syncwarp();

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
            if (m0 < Ftc && acc.x[0] == deg[m0]) atomicOr(&phi[fr * phiPadW + (m0 >> 5)], 1u << (m0 & 31));
            if (m1 < Ftc && acc.x[1] == deg[m1]) atomicOr(&phi[fr * phiPadW + (m1 >> 5)], 1u << (m1 & 31));
        }
        __syncwarp();

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
            unsigned mask0 = __ballot_sync(0xffffffffu, (acc.x[0] & 1) != 0);
            unsigned mask1 = __ballot_sync(0xffffffffu, (acc.x[1] & 1) != 0);
            if (fr == 0) {
                unsigned char y0 = 0, y1 = 0;
                #pragma unroll
                for (int r = 0; r < WM; r++) {
                    y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                    y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                }
                int o0 = ot * WN + fc, o1 = o0 + 1;
                if (o0 < G) V[(size_t)tile * n_cols + out_cols[o0]] = y0;
                if (o1 < G) V[(size_t)tile * n_cols + out_cols[o1]] = y1;
            }
        }
    }
}

extern "C" __global__ void b1_layer_v8_k1(
    unsigned char* __restrict__ V, int n_tiles, int n_cols, int n_chunks,
    const int* __restrict__ meta,
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_cols_all, const int* __restrict__ out_cols_all,
    const int* __restrict__ bypass_all,
    int maxNkt, int maxPhiPadW)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    extern __shared__ unsigned smem[];
    int perWarp = maxNkt * WM * KW + WM * maxPhiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + maxNkt * WM * KW;

    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;
    int total_work = n_tiles * n_chunks;

    for (int work = gwarp; work < total_work; work += total_warps) {
        int tile = work / n_chunks;
        int ci = work - tile * n_chunks;
        const int* md = meta + ci * LV_LAYER_META_WORDS;
        int F = md[LV_F], NIN = md[LV_NIN], G = md[LV_G];
        int n_mt = md[LV_NMT], n_ot = md[LV_NOT], n_ft = md[LV_NFT];
        int Ftc = md[LV_FTC], n_bypass = md[LV_NBYP];
        const unsigned* A = A_all + md[LV_AOFF];
        const unsigned* C = C_all + md[LV_COFF];
        const int* deg = deg_all + md[LV_DOFF];
        const int* in_cols = in_cols_all + md[LV_IOFF];
        const int* out_cols = out_cols_all + md[LV_OOFF];
        const int* bypass = bypass_all + md[LV_BPOFF];
        int phiW = (F + 31) / 32;
        int phiPadW = (phiW + KW - 1) / KW * KW;

        for (int i = lane; i < WM * KW; i += 32) XkAll[i] = 0u;
        int n_words = (NIN + 31) >> 5;
        for (int w = 0; w < n_words; w++) {
            int k = w * 32 + lane;
            unsigned bits = (k < NIN) ? (unsigned)V[(size_t)tile * n_cols + in_cols[k]] : 0u;
            unsigned xs0 = __ballot_sync(0xffffffffu, (bits & 0x01u) != 0);
            unsigned xs1 = __ballot_sync(0xffffffffu, (bits & 0x02u) != 0);
            unsigned xs2 = __ballot_sync(0xffffffffu, (bits & 0x04u) != 0);
            unsigned xs3 = __ballot_sync(0xffffffffu, (bits & 0x08u) != 0);
            unsigned xs4 = __ballot_sync(0xffffffffu, (bits & 0x10u) != 0);
            unsigned xs5 = __ballot_sync(0xffffffffu, (bits & 0x20u) != 0);
            unsigned xs6 = __ballot_sync(0xffffffffu, (bits & 0x40u) != 0);
            unsigned xs7 = __ballot_sync(0xffffffffu, (bits & 0x80u) != 0);
            if (lane == 0) XkAll[w + 0 * KW] = xs0;
            if (lane == 1) XkAll[w + 1 * KW] = xs1;
            if (lane == 2) XkAll[w + 2 * KW] = xs2;
            if (lane == 3) XkAll[w + 3 * KW] = xs3;
            if (lane == 4) XkAll[w + 4 * KW] = xs4;
            if (lane == 5) XkAll[w + 5 * KW] = xs5;
            if (lane == 6) XkAll[w + 6 * KW] = xs6;
            if (lane == 7) XkAll[w + 7 * KW] = xs7;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

        for (int j = lane; j < n_bypass; j += 32) {
            int pidx = bypass[2 * j + 0];
            int src_col = bypass[2 * j + 1];
            unsigned bits = (src_col < 0) ? 0xffu : (unsigned)V[(size_t)tile * n_cols + src_col];
            unsigned mask = 1u << (pidx & 31);
            int word = pidx >> 5;
            #pragma unroll
            for (int r = 0; r < WM; r++) {
                if ((bits >> r) & 1u) atomicOr(&phi[r * phiPadW + word], mask);
            }
        }
        __syncwarp();

        for (int mt = 0; mt < n_mt; mt++) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
            wmma::fill_fragment(acc, 0);
            wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::experimental::precision::b1, wmma::row_major> a;
            wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::experimental::precision::b1, wmma::col_major> b;
            wmma::load_matrix_sync(a, XkAll, WK);
            wmma::load_matrix_sync(b, A + (size_t)mt * WN * KW, WK);
            wmma::bmma_sync(acc, a, b, acc, wmma::experimental::bmmaBitOpAND,
                            wmma::experimental::bmmaAccumulateOpPOPC);
            int m0 = mt * WN + fc, m1 = m0 + 1;
            if (m0 < Ftc && acc.x[0] == deg[m0]) atomicOr(&phi[fr * phiPadW + (m0 >> 5)], 1u << (m0 & 31));
            if (m1 < Ftc && acc.x[1] == deg[m1]) atomicOr(&phi[fr * phiPadW + (m1 >> 5)], 1u << (m1 & 31));
        }
        __syncwarp();

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
            unsigned mask0 = __ballot_sync(0xffffffffu, (acc.x[0] & 1) != 0);
            unsigned mask1 = __ballot_sync(0xffffffffu, (acc.x[1] & 1) != 0);
            if (fr == 0) {
                unsigned char y0 = 0, y1 = 0;
                #pragma unroll
                for (int r = 0; r < WM; r++) {
                    y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                    y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                }
                int o0 = ot * WN + fc, o1 = o0 + 1;
                if (o0 < G) V[(size_t)tile * n_cols + out_cols[o0]] = y0;
                if (o1 < G) V[(size_t)tile * n_cols + out_cols[o1]] = y1;
            }
        }
    }
}


// Multi-tile variant of b1_layer_v8_k1: each warp evaluates T consecutive 8-stimulus tiles of one
// chunk. Every A/C fragment is loaded from L2 once and reused for T bmma calls, so operand traffic
// per stimulus drops by T; shared scratch grows by T.
template <int T>
__global__ void b1_layer_v8_k1_t(
    unsigned char* __restrict__ V, int n_tiles, int n_cols, int n_chunks,
    const int* __restrict__ meta,
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_cols_all, const int* __restrict__ out_cols_all,
    const int* __restrict__ bypass_all,
    int maxPhiPadW)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    extern __shared__ unsigned smem[];
    int perTile = WM * KW + WM * maxPhiPadW;
    unsigned* base = smem + warp_in_block * perTile * T;

    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;
    int n_groups = (n_tiles + T - 1) / T;
    int total_work = n_groups * n_chunks;

    for (int work = gwarp; work < total_work; work += total_warps) {
        int group = work / n_chunks;
        int ci = work - group * n_chunks;
        const int* md = meta + ci * LV_LAYER_META_WORDS;
        int F = md[LV_F], NIN = md[LV_NIN], G = md[LV_G];
        int n_mt = md[LV_NMT], n_ot = md[LV_NOT], n_ft = md[LV_NFT];
        int Ftc = md[LV_FTC], n_bypass = md[LV_NBYP];
        const unsigned* A = A_all + md[LV_AOFF];
        const unsigned* C = C_all + md[LV_COFF];
        const int* deg = deg_all + md[LV_DOFF];
        const int* in_cols = in_cols_all + md[LV_IOFF];
        const int* out_cols = out_cols_all + md[LV_OOFF];
        const int* bypass_src = bypass_all + md[LV_BPOFF];  // front layout: literal j -> phi bit j
        int tc_off = (n_bypass + 31) & ~31;
        int phiW = (F + 31) / 32;
        int phiPadW = (phiW + KW - 1) / KW * KW;
        int n_words = (NIN + 31) >> 5;

        #pragma unroll
        for (int t = 0; t < T; t++) {
            int tile = group * T + t;
            unsigned* Xk = base + t * perTile;
            unsigned* phi = Xk + WM * KW;
            for (int i = lane; i < WM * KW; i += 32) Xk[i] = 0u;
            for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
            __syncwarp();
            if (tile >= n_tiles) continue;
            const unsigned char* Vt = V + (size_t)tile * n_cols;
            for (int w = 0; w < n_words; w++) {
                int k = w * 32 + lane;
                unsigned bits = (k < NIN) ? (unsigned)Vt[in_cols[k]] : 0u;
                #pragma unroll
                for (int r = 0; r < WM; r++) {
                    unsigned xs = __ballot_sync(0xffffffffu, (bits >> r) & 1u);
                    if (lane == r) Xk[w + r * KW] = xs;
                }
            }
            __syncwarp();
            for (int jb = 0; jb < tc_off; jb += 32) {
                int j = jb + lane;
                int src_col = (j < n_bypass) ? bypass_src[j] : -2;
                unsigned bits = (src_col == -2) ? 0u : (src_col < 0) ? 0xffu : (unsigned)Vt[src_col];
                #pragma unroll
                for (int r = 0; r < WM; r++) {
                    unsigned w = __ballot_sync(0xffffffffu, (bits >> r) & 1u);
                    if (lane == r) phi[r * phiPadW + (jb >> 5)] = w;
                }
            }
        }
        __syncwarp();

        for (int mt = 0; mt < n_mt; mt++) {
            wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::experimental::precision::b1, wmma::col_major> b;
            wmma::load_matrix_sync(b, A + (size_t)mt * WN * KW, WK);
            int m0 = mt * WN + fc, m1 = m0 + 1;
            int d0 = (m0 < Ftc) ? deg[m0] : -1;
            int d1 = (m1 < Ftc) ? deg[m1] : -1;
            int p0 = tc_off + m0, p1 = tc_off + m1;
            #pragma unroll
            for (int t = 0; t < T; t++) {
                unsigned* Xk = base + t * perTile;
                unsigned* phi = Xk + WM * KW;
                wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc;
                wmma::fill_fragment(acc, 0);
                wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::experimental::precision::b1, wmma::row_major> a;
                wmma::load_matrix_sync(a, Xk, WK);
                wmma::bmma_sync(acc, a, b, acc, wmma::experimental::bmmaBitOpAND,
                                wmma::experimental::bmmaAccumulateOpPOPC);
                if (acc.x[0] == d0) atomicOr(&phi[fr * phiPadW + (p0 >> 5)], 1u << (p0 & 31));
                if (acc.x[1] == d1) atomicOr(&phi[fr * phiPadW + (p1 >> 5)], 1u << (p1 & 31));
            }
        }
        __syncwarp();

        for (int ot = 0; ot < n_ot; ot++) {
            wmma::fragment<wmma::accumulator, WM, WN, WK, int> acc[T];
            #pragma unroll
            for (int t = 0; t < T; t++) wmma::fill_fragment(acc[t], 0);
            for (int ft = 0; ft < n_ft; ft++) {
                wmma::fragment<wmma::matrix_b, WM, WN, WK, wmma::experimental::precision::b1, wmma::col_major> b;
                wmma::load_matrix_sync(b, C + ((size_t)ot * n_ft + ft) * WN * KW, WK);
                #pragma unroll
                for (int t = 0; t < T; t++) {
                    unsigned* phi = base + t * perTile + WM * KW;
                    wmma::fragment<wmma::matrix_a, WM, WN, WK, wmma::experimental::precision::b1, wmma::row_major> a;
                    wmma::load_matrix_sync(a, phi + ft * KW, phiPadW * 32);
                    wmma::bmma_sync(acc[t], a, b, acc[t], wmma::experimental::bmmaBitOpAND,
                                    wmma::experimental::bmmaAccumulateOpPOPC);
                }
            }
            int o0 = ot * WN + fc, o1 = o0 + 1;
            #pragma unroll
            for (int t = 0; t < T; t++) {
                int tile = group * T + t;
                unsigned mask0 = __ballot_sync(0xffffffffu, (acc[t].x[0] & 1) != 0);
                unsigned mask1 = __ballot_sync(0xffffffffu, (acc[t].x[1] & 1) != 0);
                if (fr == 0 && tile < n_tiles) {
                    unsigned char y0 = 0, y1 = 0;
                    #pragma unroll
                    for (int r = 0; r < WM; r++) {
                        y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                        y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                    }
                    if (o0 < G) V[(size_t)tile * n_cols + out_cols[o0]] = y0;
                    if (o1 < G) V[(size_t)tile * n_cols + out_cols[o1]] = y1;
                }
            }
        }
    }
}

extern "C" __global__ void b1_program_v8(
    unsigned char* __restrict__ V, int n_tiles, int n_cols, int n_layers, int cycles,
    const int* __restrict__ layer_off, const int* __restrict__ layer_count,
    const int* __restrict__ meta,
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_cols_all, const int* __restrict__ out_cols_all,
    int n_state, const int* __restrict__ state_cols, const int* __restrict__ dff_cols,
    int n_input, const int* __restrict__ input_cols, int one_col,
    int n_po, const int* __restrict__ po_cols,
    const unsigned char* __restrict__ u,
    unsigned char* __restrict__ po_out,
    unsigned char* __restrict__ final_state,
    int maxNkt, int maxPhiPadW, int capture_po)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    int statePad = (n_state + 15) & ~15;
    int scratchWords = maxNkt * WM * KW + WM * maxPhiPadW;
    int perWarpBytes = statePad + scratchWords * 4;
    extern __shared__ unsigned char smem_raw[];
    unsigned char* wbase = smem_raw + warp_in_block * perWarpBytes;
    unsigned char* nxt = wbase;
    unsigned* XkAll = (unsigned*)(wbase + statePad);
    unsigned* phi = XkAll + maxNkt * WM * KW;

    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int tile = gwarp; tile < n_tiles; tile += total_warps) {
        unsigned char* Vt = V + (size_t)tile * n_cols;
        for (int i = lane; i < n_cols; i += 32) Vt[i] = 0;
        if (one_col >= 0 && lane == 0) Vt[one_col] = 0xff;
        __syncwarp();

        for (int t = 0; t < cycles; t++) {
            const unsigned char* ut = u + (size_t)(t * n_tiles + tile) * n_input;
            for (int i = lane; i < n_input; i += 32) Vt[input_cols[i]] = ut[i];
            __syncwarp();

            for (int l = 0; l < n_layers; l++) {
                int c_begin = layer_off[l];
                int c_end = c_begin + layer_count[l];
                for (int ci = c_begin; ci < c_end; ci++) {
                    const int* md = meta + ci * 12;
                    int F = md[LV_F], NIN = md[LV_NIN], G = md[LV_G];
                    int n_mt = md[LV_NMT], n_kt = md[LV_NKT], n_ot = md[LV_NOT], n_ft = md[LV_NFT];
                    const unsigned* A = A_all + md[LV_AOFF];
                    const unsigned* C = C_all + md[LV_COFF];
                    const int* deg = deg_all + md[LV_DOFF];
                    const int* in_cols = in_cols_all + md[LV_IOFF];
                    const int* out_cols = out_cols_all + md[LV_OOFF];
                    int phiW = (F + 31) / 32;
                    int phiPadW = (phiW + KW - 1) / KW * KW;

                    for (int i = lane; i < n_kt * WM * KW; i += 32) XkAll[i] = 0u;
                    int n_words = (NIN + 31) >> 5;
                    for (int idx = 0; idx < n_words; idx++) {
                        int kt = idx / KW, w = idx % KW;
                        int k = kt * WK + w * 32 + lane;
                        unsigned bits = (k < NIN) ? (unsigned)Vt[in_cols[k]] : 0u;
                        unsigned xs0 = __ballot_sync(0xffffffffu, (bits & 0x01u) != 0);
                        unsigned xs1 = __ballot_sync(0xffffffffu, (bits & 0x02u) != 0);
                        unsigned xs2 = __ballot_sync(0xffffffffu, (bits & 0x04u) != 0);
                        unsigned xs3 = __ballot_sync(0xffffffffu, (bits & 0x08u) != 0);
                        unsigned xs4 = __ballot_sync(0xffffffffu, (bits & 0x10u) != 0);
                        unsigned xs5 = __ballot_sync(0xffffffffu, (bits & 0x20u) != 0);
                        unsigned xs6 = __ballot_sync(0xffffffffu, (bits & 0x40u) != 0);
                        unsigned xs7 = __ballot_sync(0xffffffffu, (bits & 0x80u) != 0);
                        size_t off = (size_t)kt * WM * KW + w;
                        if (lane == 0) XkAll[off + 0 * KW] = xs0;
                        if (lane == 1) XkAll[off + 1 * KW] = xs1;
                        if (lane == 2) XkAll[off + 2 * KW] = xs2;
                        if (lane == 3) XkAll[off + 3 * KW] = xs3;
                        if (lane == 4) XkAll[off + 4 * KW] = xs4;
                        if (lane == 5) XkAll[off + 5 * KW] = xs5;
                        if (lane == 6) XkAll[off + 6 * KW] = xs6;
                        if (lane == 7) XkAll[off + 7 * KW] = xs7;
                    }
                    for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
                    __syncwarp();

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
                        unsigned mask0 = __ballot_sync(0xffffffffu, (acc.x[0] & 1) != 0);
                        unsigned mask1 = __ballot_sync(0xffffffffu, (acc.x[1] & 1) != 0);
                        if (fr == 0) {
                            unsigned char y0 = 0, y1 = 0;
                            #pragma unroll
                            for (int r = 0; r < WM; r++) {
                                y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                                y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                            }
                            int o0 = ot * WN + fc, o1 = o0 + 1;
                            if (o0 < G) Vt[out_cols[o0]] = y0;
                            if (o1 < G) Vt[out_cols[o1]] = y1;
                        }
                    }
                    __syncwarp();
                }
            }

            if (capture_po) {
                unsigned char* pot = po_out + (size_t)(t * n_tiles + tile) * n_po;
                for (int k = lane; k < n_po; k += 32) pot[k] = Vt[po_cols[k]];
            }
            for (int j = lane; j < n_state; j += 32) nxt[j] = Vt[dff_cols[j]];
            __syncwarp();
            for (int j = lane; j < n_state; j += 32) Vt[state_cols[j]] = nxt[j];
            __syncwarp();
        }

        unsigned char* fs = final_state + (size_t)tile * n_state;
        for (int j = lane; j < n_state; j += 32) fs[j] = Vt[state_cols[j]];
    }
}

extern "C" __global__ void b1_block_program_v8(
    int n_tiles, int n_cols, int n_layers, int cycles,
    const int* __restrict__ layer_off, const int* __restrict__ layer_count,
    const int* __restrict__ meta,
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_cols_all, const int* __restrict__ out_cols_all,
    int n_state, const int* __restrict__ state_cols, const int* __restrict__ dff_cols,
    int n_input, const int* __restrict__ input_cols, int one_col,
    int n_po, const int* __restrict__ po_cols,
    const unsigned char* __restrict__ u,
    unsigned char* __restrict__ po_out,
    unsigned char* __restrict__ final_state,
    int maxNkt, int maxPhiPadW, int capture_po)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    int Vpad = (n_cols + 15) & ~15;
    int statePad = (n_state + 15) & ~15;
    int scratchWords = maxNkt * WM * KW + WM * maxPhiPadW;
    int scratchBytes = scratchWords * 4;
    extern __shared__ unsigned char smem_raw[];
    unsigned char* Vsh = smem_raw;
    unsigned char* nxt = Vsh + Vpad;
    unsigned char* scratchBase = nxt + statePad;
    unsigned* XkAll = (unsigned*)(scratchBase + warp_in_block * scratchBytes);
    unsigned* phi = XkAll + maxNkt * WM * KW;

    for (int tile = blockIdx.x; tile < n_tiles; tile += gridDim.x) {
        for (int i = threadIdx.x; i < n_cols; i += blockDim.x) Vsh[i] = 0;
        if (one_col >= 0 && threadIdx.x == 0) Vsh[one_col] = 0xff;
        __syncthreads();

        for (int t = 0; t < cycles; t++) {
            const unsigned char* ut = u + (size_t)(t * n_tiles + tile) * n_input;
            for (int i = threadIdx.x; i < n_input; i += blockDim.x) Vsh[input_cols[i]] = ut[i];
            __syncthreads();

            for (int l = 0; l < n_layers; l++) {
                int c_begin = layer_off[l];
                int c_end = c_begin + layer_count[l];
                for (int ci = c_begin + warp_in_block; ci < c_end; ci += wpb) {
                    const int* md = meta + ci * 12;
                    int F = md[LV_F], NIN = md[LV_NIN], G = md[LV_G];
                    int n_mt = md[LV_NMT], n_kt = md[LV_NKT], n_ot = md[LV_NOT], n_ft = md[LV_NFT];
                    const unsigned* A = A_all + md[LV_AOFF];
                    const unsigned* C = C_all + md[LV_COFF];
                    const int* deg = deg_all + md[LV_DOFF];
                    const int* in_cols = in_cols_all + md[LV_IOFF];
                    const int* out_cols = out_cols_all + md[LV_OOFF];
                    int phiW = (F + 31) / 32;
                    int phiPadW = (phiW + KW - 1) / KW * KW;

                    for (int i = lane; i < n_kt * WM * KW; i += 32) XkAll[i] = 0u;
                    int n_words = (NIN + 31) >> 5;
                    for (int idx = 0; idx < n_words; idx++) {
                        int kt = idx / KW, w = idx % KW;
                        int k = kt * WK + w * 32 + lane;
                        unsigned bits = (k < NIN) ? (unsigned)Vsh[in_cols[k]] : 0u;
                        unsigned xs0 = __ballot_sync(0xffffffffu, (bits & 0x01u) != 0);
                        unsigned xs1 = __ballot_sync(0xffffffffu, (bits & 0x02u) != 0);
                        unsigned xs2 = __ballot_sync(0xffffffffu, (bits & 0x04u) != 0);
                        unsigned xs3 = __ballot_sync(0xffffffffu, (bits & 0x08u) != 0);
                        unsigned xs4 = __ballot_sync(0xffffffffu, (bits & 0x10u) != 0);
                        unsigned xs5 = __ballot_sync(0xffffffffu, (bits & 0x20u) != 0);
                        unsigned xs6 = __ballot_sync(0xffffffffu, (bits & 0x40u) != 0);
                        unsigned xs7 = __ballot_sync(0xffffffffu, (bits & 0x80u) != 0);
                        size_t off = (size_t)kt * WM * KW + w;
                        if (lane == 0) XkAll[off + 0 * KW] = xs0;
                        if (lane == 1) XkAll[off + 1 * KW] = xs1;
                        if (lane == 2) XkAll[off + 2 * KW] = xs2;
                        if (lane == 3) XkAll[off + 3 * KW] = xs3;
                        if (lane == 4) XkAll[off + 4 * KW] = xs4;
                        if (lane == 5) XkAll[off + 5 * KW] = xs5;
                        if (lane == 6) XkAll[off + 6 * KW] = xs6;
                        if (lane == 7) XkAll[off + 7 * KW] = xs7;
                    }
                    for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
                    __syncwarp();

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
                        unsigned mask0 = __ballot_sync(0xffffffffu, (acc.x[0] & 1) != 0);
                        unsigned mask1 = __ballot_sync(0xffffffffu, (acc.x[1] & 1) != 0);
                        if (fr == 0) {
                            unsigned char y0 = 0, y1 = 0;
                            #pragma unroll
                            for (int r = 0; r < WM; r++) {
                                y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                                y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                            }
                            int o0 = ot * WN + fc, o1 = o0 + 1;
                            if (o0 < G) Vsh[out_cols[o0]] = y0;
                            if (o1 < G) Vsh[out_cols[o1]] = y1;
                        }
                    }
                    __syncwarp();
                }
                __syncthreads();
            }

            if (capture_po) {
                unsigned char* pot = po_out + (size_t)(t * n_tiles + tile) * n_po;
                for (int k = threadIdx.x; k < n_po; k += blockDim.x) pot[k] = Vsh[po_cols[k]];
            }
            for (int j = threadIdx.x; j < n_state; j += blockDim.x) nxt[j] = Vsh[dff_cols[j]];
            __syncthreads();
            for (int j = threadIdx.x; j < n_state; j += blockDim.x) Vsh[state_cols[j]] = nxt[j];
            __syncthreads();
        }

        unsigned char* fs = final_state + (size_t)tile * n_state;
        for (int j = threadIdx.x; j < n_state; j += blockDim.x) fs[j] = Vsh[state_cols[j]];
        __syncthreads();
    }
}

extern "C" __global__ void b1_coop_program_v8(
    unsigned char* __restrict__ V, int n_tiles, int n_cols, int n_layers, int cycles,
    const int* __restrict__ layer_off, const int* __restrict__ layer_count,
    const int* __restrict__ meta,
    const unsigned* __restrict__ A_all, const unsigned* __restrict__ C_all,
    const int* __restrict__ deg_all,
    const int* __restrict__ in_cols_all, const int* __restrict__ out_cols_all,
    int n_state, const int* __restrict__ state_cols, const int* __restrict__ dff_cols,
    int n_input, const int* __restrict__ input_cols, int one_col,
    int n_po, const int* __restrict__ po_cols,
    const unsigned char* __restrict__ u,
    unsigned char* __restrict__ po_out,
    unsigned char* __restrict__ final_state,
    int maxNkt, int maxPhiPadW, int capture_po)
{
    cg::grid_group grid = cg::this_grid();
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int fr = lane >> 2;
    int cpair = lane & 3;
    int fc = cpair * 2;

    extern __shared__ unsigned smem[];
    int perWarp = maxNkt * WM * KW + WM * maxPhiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + maxNkt * WM * KW;

    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int nthreads = gridDim.x * blockDim.x;

    for (int i = tid; i < n_tiles * n_cols; i += nthreads) V[i] = 0;
    if (one_col >= 0) {
        for (int tile = tid; tile < n_tiles; tile += nthreads) {
            V[(size_t)tile * n_cols + one_col] = 0xff;
        }
    }
    grid.sync();

    for (int t = 0; t < cycles; t++) {
        int n_input_work = n_tiles * n_input;
        for (int i = tid; i < n_input_work; i += nthreads) {
            int tile = i / n_input;
            int inp = i - tile * n_input;
            V[(size_t)tile * n_cols + input_cols[inp]] = u[(size_t)(t * n_tiles + tile) * n_input + inp];
        }
        grid.sync();

        for (int l = 0; l < n_layers; l++) {
            int c_begin = layer_off[l];
            int n_chunks = layer_count[l];
            int total_work = n_tiles * n_chunks;
            for (int work = gwarp; work < total_work; work += total_warps) {
                int tile = work / n_chunks;
                int ci = c_begin + (work - tile * n_chunks);
                unsigned char* Vt = V + (size_t)tile * n_cols;
                const int* md = meta + ci * 12;
                int F = md[LV_F], NIN = md[LV_NIN], G = md[LV_G];
                int n_mt = md[LV_NMT], n_kt = md[LV_NKT], n_ot = md[LV_NOT], n_ft = md[LV_NFT];
                const unsigned* A = A_all + md[LV_AOFF];
                const unsigned* C = C_all + md[LV_COFF];
                const int* deg = deg_all + md[LV_DOFF];
                const int* in_cols = in_cols_all + md[LV_IOFF];
                const int* out_cols = out_cols_all + md[LV_OOFF];
                int phiW = (F + 31) / 32;
                int phiPadW = (phiW + KW - 1) / KW * KW;

                for (int idx = 0; idx < n_kt * KW; idx++) {
                    int kt = idx / KW, w = idx % KW;
                    int k = kt * WK + w * 32 + lane;
                    unsigned bits = (k < NIN) ? (unsigned)Vt[in_cols[k]] : 0u;
                    unsigned xs0 = __ballot_sync(0xffffffffu, (bits & 0x01u) != 0);
                    unsigned xs1 = __ballot_sync(0xffffffffu, (bits & 0x02u) != 0);
                    unsigned xs2 = __ballot_sync(0xffffffffu, (bits & 0x04u) != 0);
                    unsigned xs3 = __ballot_sync(0xffffffffu, (bits & 0x08u) != 0);
                    unsigned xs4 = __ballot_sync(0xffffffffu, (bits & 0x10u) != 0);
                    unsigned xs5 = __ballot_sync(0xffffffffu, (bits & 0x20u) != 0);
                    unsigned xs6 = __ballot_sync(0xffffffffu, (bits & 0x40u) != 0);
                    unsigned xs7 = __ballot_sync(0xffffffffu, (bits & 0x80u) != 0);
                    size_t off = (size_t)kt * WM * KW + w;
                    if (lane == 0) XkAll[off + 0 * KW] = xs0;
                    if (lane == 1) XkAll[off + 1 * KW] = xs1;
                    if (lane == 2) XkAll[off + 2 * KW] = xs2;
                    if (lane == 3) XkAll[off + 3 * KW] = xs3;
                    if (lane == 4) XkAll[off + 4 * KW] = xs4;
                    if (lane == 5) XkAll[off + 5 * KW] = xs5;
                    if (lane == 6) XkAll[off + 6 * KW] = xs6;
                    if (lane == 7) XkAll[off + 7 * KW] = xs7;
                }
                for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
                __syncwarp();

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
                    unsigned mask0 = __ballot_sync(0xffffffffu, (acc.x[0] & 1) != 0);
                    unsigned mask1 = __ballot_sync(0xffffffffu, (acc.x[1] & 1) != 0);
                    if (fr == 0) {
                        unsigned char y0 = 0, y1 = 0;
                        #pragma unroll
                        for (int r = 0; r < WM; r++) {
                            y0 |= (unsigned char)(((mask0 >> (r * 4 + cpair)) & 1u) << r);
                            y1 |= (unsigned char)(((mask1 >> (r * 4 + cpair)) & 1u) << r);
                        }
                        int o0 = ot * WN + fc, o1 = o0 + 1;
                        if (o0 < G) Vt[out_cols[o0]] = y0;
                        if (o1 < G) Vt[out_cols[o1]] = y1;
                    }
                }
            }
            grid.sync();
        }

        if (capture_po) {
            int n_po_work = n_tiles * n_po;
            for (int i = tid; i < n_po_work; i += nthreads) {
                int tile = i / n_po;
                int po = i - tile * n_po;
                po_out[(size_t)(t * n_tiles + tile) * n_po + po] =
                    V[(size_t)tile * n_cols + po_cols[po]];
            }
        }
        int n_state_work = n_tiles * n_state;
        for (int i = tid; i < n_state_work; i += nthreads) {
            int tile = i / n_state;
            int st = i - tile * n_state;
            final_state[(size_t)tile * n_state + st] =
                V[(size_t)tile * n_cols + dff_cols[st]];
        }
        grid.sync();
        for (int i = tid; i < n_state_work; i += nthreads) {
            int tile = i / n_state;
            int st = i - tile * n_state;
            V[(size_t)tile * n_cols + state_cols[st]] =
                final_state[(size_t)tile * n_state + st];
        }
        grid.sync();
    }
}

extern "C" __global__ void v8_scatter_cols(
    unsigned char* __restrict__ V, int n_tiles, int n_cols,
    const int* __restrict__ cols, int n_set,
    const unsigned char* __restrict__ X)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tiles * n_set;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int tile = i / n_set;
        int j = i - tile * n_set;
        V[(size_t)tile * n_cols + cols[j]] = X[(size_t)tile * n_set + j];
    }
}

extern "C" __global__ void v8_commit_direct(
    unsigned char* __restrict__ V, int n_tiles, int n_cols,
    const int* __restrict__ state_cols, const int* __restrict__ dff_cols,
    int n_state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_tiles * n_state;
    for (int i = tid; i < total; i += blockDim.x * gridDim.x) {
        int tile = i / n_state;
        int j = i - tile * n_state;
        V[(size_t)tile * n_cols + state_cols[j]] =
            V[(size_t)tile * n_cols + dff_cols[j]];
    }
}

extern "C" __global__ void b1_anf_x_to_v(
    const signed char* __restrict__ X, int batch, int NIN,
    const unsigned* __restrict__ A, int F, int n_mt, int n_kt,
    const int* __restrict__ deg,
    const unsigned* __restrict__ C, int G, int n_ot, int n_ft,
    signed char* __restrict__ V, int n_cols, const int* __restrict__ out_cols)
{
    int wpb = blockDim.x / 32;
    int warp_in_block = threadIdx.x / 32;
    int lane = threadIdx.x & 31;
    int phiW = (F + 31) / 32;
    int phiPadW = (phiW + KW - 1) / KW * KW;
    extern __shared__ unsigned smem[];
    int perWarp = n_kt * WM * KW + WM * phiPadW;
    unsigned* base = smem + warp_in_block * perWarp;
    unsigned* XkAll = base;
    unsigned* phi = XkAll + n_kt * WM * KW;
    int fr = lane >> 2;
    int fc = (lane & 3) * 2;

    int nMtiles = (batch + WM - 1) / WM;
    int gwarp = blockIdx.x * wpb + warp_in_block;
    int total_warps = gridDim.x * wpb;

    for (int mtile = gwarp; mtile < nMtiles; mtile += total_warps) {
        int stim0 = mtile * WM;
        for (int idx = lane; idx < n_kt * WM * KW; idx += 32) {
            int kt = idx / (WM * KW), r = idx % (WM * KW), s = r / KW, w = r % KW;
            int stim = stim0 + s;
            unsigned v = 0u;
            #pragma unroll
            for (int b = 0; b < 32; b++) {
                int k = kt * WK + w * 32 + b;
                if (k < NIN && stim < batch && X[(size_t)stim * NIN + k]) v |= (1u << b);
            }
            XkAll[idx] = v;
        }
        for (int i = lane; i < WM * phiPadW; i += 32) phi[i] = 0u;
        __syncwarp();

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
            int o0 = ot * WN + fc, o1 = o0 + 1, stim = stim0 + fr;
            if (stim < batch) {
                if (o0 < G) V[(size_t)stim * n_cols + out_cols[o0]] = (signed char)(acc.x[0] & 1);
                if (o1 < G) V[(size_t)stim * n_cols + out_cols[o1]] = (signed char)(acc.x[1] & 1);
            }
        }
    }
}

extern "C" void launch_b1_anf(
    const signed char* X, int batch, int NIN,
    const unsigned* A, int F, int n_mt, int n_kt, const int* deg,
    const unsigned* C, int G, int n_ot, int n_ft,
    signed char* Y, int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_anf, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_anf<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        X, batch, NIN, A, F, n_mt, n_kt, deg, C, G, n_ot, n_ft, Y);
}

extern "C" void launch_b1_anf_xcols(
    const signed char* X, int batch, int XNIN, const int* x_cols, int NIN,
    const unsigned* A, int F, int n_mt, int n_kt, const int* deg,
    const unsigned* C, int G, int n_ot, int n_ft,
    signed char* Y, int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_anf_xcols, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_anf_xcols<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        X, batch, XNIN, x_cols, NIN, A, F, n_mt, n_kt, deg, C, G, n_ot, n_ft, Y);
}

extern "C" void launch_b1_anf_x_to_v(
    const signed char* X, int batch, int NIN,
    const unsigned* A, int F, int n_mt, int n_kt, const int* deg,
    const unsigned* C, int G, int n_ot, int n_ft,
    signed char* V, int n_cols, const int* out_cols,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_anf_x_to_v, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_anf_x_to_v<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        X, batch, NIN, A, F, n_mt, n_kt, deg, C, G, n_ot, n_ft, V, n_cols, out_cols);
}

extern "C" void launch_b1_anf_v(
    signed char* V, int batch, int n_cols, const int* in_cols, int NIN,
    const unsigned* A, int F, int n_mt, int n_kt, const int* deg,
    const unsigned* C, int G, int n_ot, int n_ft, const int* out_cols,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_anf_v, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_anf_v<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        V, batch, n_cols, in_cols, NIN, A, F, n_mt, n_kt, deg, C, G, n_ot, n_ft,
        out_cols);
}

extern "C" void launch_b1_anf_v8(
    unsigned char* V, int n_tiles, int n_cols, const int* in_cols, int NIN,
    const unsigned* A, int F, int n_mt, int n_kt, const int* deg,
    const unsigned* C, int G, int n_ot, int n_ft, const int* out_cols,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_anf_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_anf_v8<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        V, n_tiles, n_cols, in_cols, NIN, A, F, n_mt, n_kt, deg, C, G, n_ot, n_ft,
        out_cols);
}

extern "C" void launch_b1_layer_v8(
    unsigned char* V, int n_tiles, int n_cols, int n_chunks, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_cols_all, const int* out_cols_all, const int* bypass_all,
    int maxNkt, int maxPhiPadW,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_layer_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_layer_v8<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        V, n_tiles, n_cols, n_chunks, meta, A_all, C_all, deg_all, in_cols_all,
        out_cols_all, bypass_all, maxNkt, maxPhiPadW);
}

extern "C" void launch_b1_layer_v8_k1(
    unsigned char* V, int n_tiles, int n_cols, int n_chunks, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_cols_all, const int* out_cols_all, const int* bypass_all,
    int maxNkt, int maxPhiPadW,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_layer_v8_k1, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_layer_v8_k1<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        V, n_tiles, n_cols, n_chunks, meta, A_all, C_all, deg_all, in_cols_all,
        out_cols_all, bypass_all, maxNkt, maxPhiPadW);
}


extern "C" void launch_b1_layer_v8_k1_t(
    unsigned char* V, int n_tiles, int n_cols, int n_chunks, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_cols_all, const int* out_cols_all, const int* bypass_all,
    int maxPhiPadW, int tiles_per_warp,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
#define LAUNCH_T(TT)                                                                              \
    cudaFuncSetAttribute(b1_layer_v8_k1_t<TT>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); \
    b1_layer_v8_k1_t<TT><<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(              \
        V, n_tiles, n_cols, n_chunks, meta, A_all, C_all, deg_all, in_cols_all, out_cols_all,       \
        bypass_all, maxPhiPadW)
    switch (tiles_per_warp) {
        case 2: LAUNCH_T(2); break;
        case 4: LAUNCH_T(4); break;
        case 8: LAUNCH_T(8); break;
        default: LAUNCH_T(1); break;
    }
#undef LAUNCH_T
}

extern "C" void launch_b1_program_v8(
    unsigned char* V, int n_tiles, int n_cols, int n_layers, int cycles,
    const int* layer_off, const int* layer_count, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_cols_all, const int* out_cols_all,
    int n_state, const int* state_cols, const int* dff_cols,
    int n_input, const int* input_cols, int one_col,
    int n_po, const int* po_cols, const unsigned char* u,
    unsigned char* po_out, unsigned char* final_state,
    int maxNkt, int maxPhiPadW, int capture_po,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_program_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_program_v8<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        V, n_tiles, n_cols, n_layers, cycles, layer_off, layer_count, meta, A_all, C_all,
        deg_all, in_cols_all, out_cols_all, n_state, state_cols, dff_cols, n_input,
        input_cols, one_col, n_po, po_cols, u, po_out, final_state, maxNkt,
        maxPhiPadW, capture_po);
}

extern "C" void launch_b1_block_program_v8(
    int n_tiles, int n_cols, int n_layers, int cycles,
    const int* layer_off, const int* layer_count, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_cols_all, const int* out_cols_all,
    int n_state, const int* state_cols, const int* dff_cols,
    int n_input, const int* input_cols, int one_col,
    int n_po, const int* po_cols, const unsigned char* u,
    unsigned char* po_out, unsigned char* final_state,
    int maxNkt, int maxPhiPadW, int capture_po,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_block_program_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    b1_block_program_v8<<<grid_blocks, warps_per_block * 32, smem_bytes, stream>>>(
        n_tiles, n_cols, n_layers, cycles, layer_off, layer_count, meta, A_all, C_all,
        deg_all, in_cols_all, out_cols_all, n_state, state_cols, dff_cols, n_input,
        input_cols, one_col, n_po, po_cols, u, po_out, final_state, maxNkt,
        maxPhiPadW, capture_po);
}

extern "C" void launch_b1_coop_program_v8(
    unsigned char* V, int n_tiles, int n_cols, int n_layers, int cycles,
    const int* layer_off, const int* layer_count, const int* meta,
    const unsigned* A_all, const unsigned* C_all, const int* deg_all,
    const int* in_cols_all, const int* out_cols_all,
    int n_state, const int* state_cols, const int* dff_cols,
    int n_input, const int* input_cols, int one_col,
    int n_po, const int* po_cols, const unsigned char* u,
    unsigned char* po_out, unsigned char* final_state,
    int maxNkt, int maxPhiPadW, int capture_po,
    int warps_per_block, int grid_blocks, int smem_bytes, void* stream_ptr)
{
    cudaFuncSetAttribute(b1_coop_program_v8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    void* args[] = {
        &V, &n_tiles, &n_cols, &n_layers, &cycles, &layer_off, &layer_count, &meta,
        &A_all, &C_all, &deg_all, &in_cols_all, &out_cols_all,
        &n_state, &state_cols, &dff_cols, &n_input, &input_cols, &one_col,
        &n_po, &po_cols, &u, &po_out, &final_state, &maxNkt, &maxPhiPadW, &capture_po
    };
    cudaLaunchCooperativeKernel((void*)b1_coop_program_v8, grid_blocks,
                                warps_per_block * 32, args, smem_bytes, stream);
}

extern "C" void launch_v8_scatter_cols(
    unsigned char* V, int n_tiles, int n_cols, const int* cols, int n_set,
    const unsigned char* X, int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    v8_scatter_cols<<<grid_blocks, 256, 0, stream>>>(V, n_tiles, n_cols, cols, n_set, X);
}

extern "C" void launch_v8_commit_direct(
    unsigned char* V, int n_tiles, int n_cols, const int* state_cols,
    const int* dff_cols, int n_state, int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    v8_commit_direct<<<grid_blocks, 256, 0, stream>>>(
        V, n_tiles, n_cols, state_cols, dff_cols, n_state);
}
