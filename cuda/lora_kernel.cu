#include <torch/extension.h>
#include <vector>
#include <mma.h> // Header for wmma (Tensor Core) intrinsics

using namespace nvcuda;

__global__ void lora_a_gemm_kernel(
    const half* __restrict__ x, const half* __restrict__ lora_a,
    const int* __restrict__ adapter_indices, half* out,
    const int batch_size, const int in_features, const int r);

__global__ void lora_b_gemm_kernel(
    const half* __restrict__ intermediate, const half* __restrict__ lora_b,
    const int* __restrict__ adapter_indices, half* out,
    const int batch_size, const int r, const int out_features);

// non-sharded batched LoRA forward ONLY
torch::Tensor batched_lora_forward(
    torch::Tensor x,
    torch::Tensor lora_a,
    torch::Tensor lora_b,
    torch::Tensor adapter_indices)
{
    // TODO: support other precisions
    TORCH_CHECK(x.is_cuda() && x.scalar_type() == torch::kFloat16,
                "Input tensor x must be a FP16 CUDA tensor.");
    TORCH_CHECK(lora_a.is_cuda() && lora_a.scalar_type() == torch::kFloat16,
                "LoRA A weights must be a FP16 CUDA tensor.");
    TORCH_CHECK(lora_b.is_cuda() && lora_b.scalar_type() == torch::kFloat16,
                "LoRA B weights must be a FP16 CUDA tensor.");

    const int batch_size = x.size(0);
    const int in_features = x.size(1);
    const int r = lora_a.size(1);
    const int out_features = lora_b.size(1);

    auto intermediate = torch::empty({batch_size, r}, x.options());
    auto final_output = torch::empty({batch_size, out_features}, x.options());

    // TODO: optimize kernels based on grid/block dims
    // lora_a_gemm_kernel<<<grid_a, block_a>>>(...);
    // lora_b_gemm_kernel<<<grid_b, block_b>>>(...);

    // TODO: replace matmul with custom kernel (do entire loop in parallel at once)
    for (int i = 0; i < batch_size; ++i) {
        int adapter_idx = adapter_indices[i].item<int>();
        auto lora_a_i = lora_a[adapter_idx];
        auto lora_b_i = lora_b[adapter_idx];
        auto x_i = x[i].unsqueeze(0); // Shape: [1, in_features]

        auto intermediate_res = torch::matmul(x_i, lora_a_i.t());
        final_output[i] = torch::matmul(intermediate_res, lora_b_i.t()).squeeze(0);
    }

    return final_output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &batched_lora_forward, "Batched LoRA Forward (Non-Sharded, CUDA)");
}