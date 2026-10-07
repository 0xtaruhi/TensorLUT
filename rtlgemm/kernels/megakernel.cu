// One-launch bit-parallel cycle-accurate simulation megakernel.
//
// Each THREAD simulates a group of 32 stimuli that are bit-packed into uint32 words
// (bit b of a word = that signal's value for stimulus b). Because stimuli are
// independent there is NO cross-thread dependency; the only dependencies (layer->layer,
// cycle->cycle) are within a thread. So the ENTIRE simulation — all combinational
// layers, all clock cycles — runs inside a single kernel launch:
//   * the constant netlist structure (LUT tables + wiring) is loaded once into shared
//     memory and reused every cycle,
//   * the net-state lives in per-thread local storage (registers/L1),
//   * global traffic is only the streamed inputs + recorded outputs.
// LUTs are evaluated bit-parallel (32 stimuli at once) via minterm OR of ANDed inputs.
//
// Compiled standalone by nvcc into a .so and driven from Python via ctypes.

#include <cuda_runtime.h>
#include <stdint.h>

#ifndef MAX_NETS
#define MAX_NETS 2048
#endif
#ifndef MAX_STATE
#define MAX_STATE 512
#endif

extern "C" __global__ void sim_mega(
    int n_luts, const int* __restrict__ lut_out, const int* __restrict__ lut_w,
    const int* __restrict__ lut_in,                 // n_luts * 6
    const unsigned long long* __restrict__ lut_tab,
    int n_nets, int one_idx,
    int n_state, const int* __restrict__ state_net, const int* __restrict__ dff_d,
    int n_input, const int* __restrict__ input_net,
    int n_po, const int* __restrict__ po_net,
    int cycles, int ngroups,
    const uint32_t* __restrict__ u,                 // [cycles, n_input, ngroups]
    uint32_t* __restrict__ po_out)                  // [cycles, n_po, ngroups]
{
    // ---- load constant structure into shared memory (once per block) ----
    // s_tab (8-byte) FIRST so it inherits the 16-byte-aligned shared base; the 4-byte
    // int arrays follow (putting s_tab last would misalign it unless the int region
    // happened to be a multiple of 8 bytes).
    extern __shared__ char smem_raw[];
    unsigned long long* s_tab = (unsigned long long*)smem_raw;   // n_luts (8B each)
    int* smem = (int*)(s_tab + n_luts);
    int* s_out = smem;                              // n_luts
    int* s_w   = s_out + n_luts;                    // n_luts
    int* s_in  = s_w + n_luts;                      // n_luts*6
    int* s_dd  = s_in + n_luts * 6;                 // n_state
    int* s_sn  = s_dd + n_state;                    // n_state
    int* s_pn  = s_sn + n_state;                    // n_po
    int* s_inp = s_pn + n_po;                       // n_input
    for (int i = threadIdx.x; i < n_luts; i += blockDim.x) {
        s_out[i] = lut_out[i]; s_w[i] = lut_w[i]; s_tab[i] = lut_tab[i];
        #pragma unroll
        for (int j = 0; j < 6; j++) s_in[i * 6 + j] = lut_in[i * 6 + j];
    }
    for (int i = threadIdx.x; i < n_state; i += blockDim.x) { s_dd[i] = dff_d[i]; s_sn[i] = state_net[i]; }
    for (int i = threadIdx.x; i < n_po; i += blockDim.x) s_pn[i] = po_net[i];
    for (int i = threadIdx.x; i < n_input; i += blockDim.x) s_inp[i] = input_net[i];
    __syncthreads();

    int g = blockIdx.x * blockDim.x + threadIdx.x;  // stimulus group (32 stimuli)
    if (g >= ngroups) return;

    uint32_t val[MAX_NETS];
    for (int i = 0; i < n_nets; i++) val[i] = 0u;   // x0 = 0 (reset driven via inputs)
    if (one_idx >= 0) val[one_idx] = 0xFFFFFFFFu;   // constant-1 sentinel net

    for (int t = 0; t < cycles; t++) {
        for (int i = 0; i < n_input; i++)
            val[s_inp[i]] = u[(size_t)(t * n_input + i) * ngroups + g];
        for (int l = 0; l < n_luts; l++) {
            int w = s_w[l];
            unsigned long long tab = s_tab[l];
            const int* in = s_in + l * 6;
            uint32_t res = 0u;
            int ne = 1 << w;
            for (int e = 0; e < ne; e++) {
                if ((tab >> e) & 1ull) {
                    uint32_t term = 0xFFFFFFFFu;
                    for (int j = 0; j < w; j++) {
                        uint32_t iv = val[in[j]];
                        term &= ((e >> j) & 1) ? iv : ~iv;
                    }
                    res |= term;
                }
            }
            val[s_out[l]] = res;
        }
        for (int k = 0; k < n_po; k++)
            po_out[(size_t)(t * n_po + k) * ngroups + g] = val[s_pn[k]];
        uint32_t nxt[MAX_STATE];
        for (int j = 0; j < n_state; j++) nxt[j] = val[s_dd[j]];
        for (int j = 0; j < n_state; j++) val[s_sn[j]] = nxt[j];
    }
}

extern "C" void launch_mega(
    int n_luts, const int* lut_out, const int* lut_w, const int* lut_in,
    const unsigned long long* lut_tab, int n_nets, int one_idx,
    int n_state, const int* state_net, const int* dff_d,
    int n_input, const int* input_net, int n_po, const int* po_net,
    int cycles, int ngroups, const uint32_t* u, uint32_t* po_out,
    int threads, int smem)
{
    if (smem > 48 * 1024)
        cudaFuncSetAttribute(sim_mega, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    int grid = (ngroups + threads - 1) / threads;
    sim_mega<<<grid, threads, smem>>>(
        n_luts, lut_out, lut_w, lut_in, lut_tab, n_nets, one_idx,
        n_state, state_net, dff_d, n_input, input_net, n_po, po_net,
        cycles, ngroups, u, po_out);
}
