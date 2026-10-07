// CUDA-core LUT/DFF simulator baseline.
//
// This intentionally avoids Tensor Cores: each layer is evaluated by ordinary CUDA
// threads, one (stimulus, LUT) pair per thread, using truth-table lookup. Kernel
// boundaries provide the global synchronization between logic levels.

#include <cuda_runtime.h>
#include <stdint.h>

extern "C" __global__ void lut_init_state(
    unsigned char* __restrict__ V, int batch, int n_cols,
    int zero_col, int one_col, int n_state,
    const int* __restrict__ state_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * (n_state + 2);
    if (idx >= total) return;
    int row = idx / (n_state + 2);
    int k = idx - row * (n_state + 2);
    if (k < n_state) {
        V[(size_t)row * n_cols + state_cols[k]] = 0u;
    } else if (k == n_state) {
        V[(size_t)row * n_cols + zero_col] = 0u;
    } else {
        V[(size_t)row * n_cols + one_col] = 1u;
    }
}

extern "C" __global__ void lut_scatter_inputs(
    unsigned char* __restrict__ V, int batch, int n_cols,
    int n_input, const int* __restrict__ input_cols,
    const unsigned char* __restrict__ u)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_input;
    if (idx >= total) return;
    int row = idx / n_input;
    int i = idx - row * n_input;
    V[(size_t)row * n_cols + input_cols[i]] = u[(size_t)row * n_input + i] & 1u;
}

extern "C" __global__ void lut_eval_layer(
    unsigned char* __restrict__ V, int batch, int n_cols,
    int n_lut,
    const int* __restrict__ in_cols,
    const int* __restrict__ out_cols,
    const unsigned char* __restrict__ widths,
    const unsigned long long* __restrict__ tables)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_lut;
    if (idx >= total) return;
    int row = idx / n_lut;
    int l = idx - row * n_lut;
    const int* in = in_cols + l * 6;
    int w = widths[l];
    unsigned code = 0u;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        if (j < w) {
            unsigned bit = (unsigned)V[(size_t)row * n_cols + in[j]] & 1u;
            code |= bit << j;
        }
    }
    V[(size_t)row * n_cols + out_cols[l]] =
        (unsigned char)((tables[l] >> code) & 1ull);
}

extern "C" __global__ void lut_commit_state(
    unsigned char* __restrict__ V, int batch, int n_cols,
    int n_state,
    const int* __restrict__ state_cols,
    const int* __restrict__ dff_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_state;
    if (idx >= total) return;
    int row = idx / n_state;
    int s = idx - row * n_state;
    V[(size_t)row * n_cols + state_cols[s]] =
        V[(size_t)row * n_cols + dff_cols[s]] & 1u;
}

extern "C" __global__ void lut_gather_cols(
    const unsigned char* __restrict__ V, int batch, int n_cols,
    int n_out, const int* __restrict__ cols,
    unsigned char* __restrict__ out)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch * n_out;
    if (idx >= total) return;
    int row = idx / n_out;
    int k = idx - row * n_out;
    out[(size_t)row * n_out + k] = V[(size_t)row * n_cols + cols[k]] & 1u;
}

extern "C" void launch_lut_init_state(
    unsigned char* V, int batch, int n_cols, int zero_col, int one_col,
    int n_state, const int* state_cols, int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    lut_init_state<<<grid_blocks, 256, 0, stream>>>(
        V, batch, n_cols, zero_col, one_col, n_state, state_cols);
}

extern "C" void launch_lut_scatter_inputs(
    unsigned char* V, int batch, int n_cols, int n_input,
    const int* input_cols, const unsigned char* u,
    int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    lut_scatter_inputs<<<grid_blocks, 256, 0, stream>>>(
        V, batch, n_cols, n_input, input_cols, u);
}

extern "C" void launch_lut_eval_layer(
    unsigned char* V, int batch, int n_cols, int n_lut,
    const int* in_cols, const int* out_cols,
    const unsigned char* widths, const unsigned long long* tables,
    int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    lut_eval_layer<<<grid_blocks, 256, 0, stream>>>(
        V, batch, n_cols, n_lut, in_cols, out_cols, widths, tables);
}

extern "C" void launch_lut_commit_state(
    unsigned char* V, int batch, int n_cols, int n_state,
    const int* state_cols, const int* dff_cols,
    int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    lut_commit_state<<<grid_blocks, 256, 0, stream>>>(
        V, batch, n_cols, n_state, state_cols, dff_cols);
}

extern "C" void launch_lut_gather_cols(
    const unsigned char* V, int batch, int n_cols, int n_out,
    const int* cols, unsigned char* out,
    int grid_blocks, void* stream_ptr)
{
    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    lut_gather_cols<<<grid_blocks, 256, 0, stream>>>(
        V, batch, n_cols, n_out, cols, out);
}
