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

class BatchedLoRAFunction : public torch::autograd::Function<BatchedLoRAFunction> {
public:
    // non-sharded batched LoRA forward ONLY
    static torch::Tensor forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor x,
        torch::Tensor lora_a,
        torch::Tensor lora_b,
        torch::Tensor adapter_indices)
    {
        // TODO: optimize kernels based on grid/block dims
        // lora_a_gemm_kernel<<<grid_a, block_a>>>(...);
        // lora_b_gemm_kernel<<<grid_b, block_b>>>(...);

        // TODO: replace matmul with custom kernel (do entire loop in parallel at once)
        const int batch_size = x.size(0);

        std::vector<torch::Tensor> intermediate_slices;
        std::vector<torch::Tensor> output_slices;
        intermediate_slices.reserve(batch_size);
        output_slices.reserve(batch_size);

        for (int i = 0; i < batch_size; ++i) {
            int adapter_idx = adapter_indices[i].item<int>();
            auto lora_a_i = lora_a[adapter_idx];
            auto lora_b_i = lora_b[adapter_idx];
            auto x_i = x.select(0, i).unsqueeze(0);

            auto intermediate_res = torch::matmul(x_i, lora_a_i.t());
            auto final_output_res = torch::matmul(intermediate_res, lora_b_i.t());

            intermediate_slices.push_back(intermediate_res);
            output_slices.push_back(final_output_res);
        }

        // torch::cat is differentiable for bwd pass
        auto intermediate = torch::cat(intermediate_slices, 0);
        auto final_output = torch::cat(output_slices, 0);

        final_output = final_output.squeeze(1);

        ctx->save_for_backward({x, lora_a, lora_b, adapter_indices, intermediate});
        return final_output;
    }

    static torch::autograd::variable_list backward(
        torch::autograd::AutogradContext* ctx,
        torch::autograd::variable_list grad_outputs)
    {
        auto grad_output = grad_outputs[0];
        auto saved_variables = ctx->get_saved_variables();
        auto x = saved_variables[0];
        auto lora_a = saved_variables[1];
        auto lora_b = saved_variables[2];
        auto adapter_indices = saved_variables[3];
        auto intermediate = saved_variables[4];

        const int batch_size = x.size(0);

        auto grad_x = torch::zeros_like(x);
        auto grad_lora_a = torch::zeros_like(lora_a);
        auto grad_lora_b = torch::zeros_like(lora_b);

        for (int i = 0; i < batch_size; ++i) {
            int adapter_idx = adapter_indices[i].item<int>();

            auto x_i = x.select(0, i).unsqueeze(0);                         // [1, in_features]
            auto lora_a_i = lora_a[adapter_idx];                           // [r, in_features]
            auto lora_b_i = lora_b[adapter_idx];                           // [out_features, r]
            auto intermediate_i = intermediate.select(0, i).unsqueeze(0);  // [1, r]
            auto grad_output_i = grad_output.select(0, i).unsqueeze(0);    // [1, out_features]

            // grad_intermediate = grad_output @ lora_b
            auto grad_intermediate = torch::matmul(grad_output_i, lora_b_i);

            // grad_x = grad_intermediate @ lora_a
            grad_x.select(0, i).copy_(torch::matmul(grad_intermediate, lora_a_i).squeeze(0));

            // grad_lora_b += grad_output.T @ intermediate
            grad_lora_b[adapter_idx] += torch::matmul(grad_output_i.t(), intermediate_i);

            // grad_lora_a += grad_intermediate.T @ x
            grad_lora_a[adapter_idx] += torch::matmul(grad_intermediate.t(), x_i);
        }

        return {grad_x, grad_lora_a, grad_lora_b, torch::Tensor()};
    }
};

torch::Tensor batched_lora_forward(
    torch::Tensor x,
    torch::Tensor lora_a,
    torch::Tensor lora_b,
    torch::Tensor adapter_indices)
{
    return BatchedLoRAFunction::apply(x, lora_a, lora_b, adapter_indices);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &batched_lora_forward, "Batched LoRA with backward pass (CUDA)");
}