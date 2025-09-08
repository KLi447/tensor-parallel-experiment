import torch
import triton
import triton.testing
from lora_kernel import batched_lora_heterogeneous, prepare_heterogeneous_lora_weights

def reference_lora_forward(x, lora_a_list, lora_b_list, adapter_indices):
    out_features = lora_b_list[0].shape[0]
    output = torch.empty(x.size(0), out_features, device=x.device, dtype=x.dtype)

    sorted_indices, sorted_positions = torch.sort(adapter_indices)

    unsorter = torch.empty_like(sorted_positions)
    unsorter[sorted_positions] = torch.arange(len(sorted_positions), device=x.device)

    x_sorted = x[sorted_positions]

    unique_adapters, group_sizes = torch.unique_consecutive(sorted_indices, return_counts=True)

    group_boundaries = torch.cumsum(group_sizes, dim=0)

    start_idx = 0
    for i, adapter_id in enumerate(unique_adapters):
        group_size = group_sizes[i].item()
        end_idx = start_idx + group_size

        x_group = x_sorted[start_idx:end_idx]

        a_matrix = lora_a_list[adapter_id]
        b_matrix = lora_b_list[adapter_id]

        intermediate = x_group @ a_matrix.T
        final_output = intermediate @ b_matrix.T

        output[start_idx:end_idx] = final_output

        start_idx = end_idx

    return output[unsorter]

if __name__ == '__main__':
    BATCH_SIZE = 32
    IN_FEATURES = 4096
    OUT_FEATURES = 4096
    device = 'cuda'
    torch.manual_seed(0)

    rank_list = [2, 16, 32, 4, 128]
    lora_a_adapters_cpu = [torch.randn(r, IN_FEATURES, dtype=torch.float32) for r in rank_list]
    lora_b_adapters_cpu = [torch.randn(OUT_FEATURES, r, dtype=torch.float32) for r in rank_list]
    NUM_ADAPTERS = len(lora_a_adapters_cpu)

    lora_a_flat, lora_b_flat, metadata = prepare_heterogeneous_lora_weights(
        lora_a_adapters_cpu, lora_b_adapters_cpu, device
    )

    x = torch.randn(BATCH_SIZE, IN_FEATURES, device=device, dtype=torch.float32, requires_grad=True)
    adapter_indices = torch.randint(0, NUM_ADAPTERS, (BATCH_SIZE,), device=device, dtype=torch.int32)

    x_ref = x.detach().clone().requires_grad_()
    lora_a_ref = [a.to(device).detach().clone().requires_grad_() for a in lora_a_adapters_cpu]
    lora_b_ref = [b.to(device).detach().clone().requires_grad_() for b in lora_b_adapters_cpu]

    lora_a_flat.requires_grad_()
    lora_b_flat.requires_grad_()

    print("Verifying Triton kernel against reference implementation...")

    output_triton = batched_lora_heterogeneous(x, lora_a_flat, lora_b_flat, metadata, adapter_indices, OUT_FEATURES)
    torch.cuda.synchronize()
    (output_triton.sum()).backward()
    torch.cuda.synchronize()

    output_ref = reference_lora_forward(x_ref, lora_a_ref, lora_b_ref, adapter_indices)
    torch.cuda.synchronize()
    (output_ref.sum()).backward()
    torch.cuda.synchronize()

    grad_lora_a_ref_flat = torch.cat([a.grad.flatten() for a in lora_a_ref])
    grad_lora_b_ref_flat = torch.cat([b.grad.flatten() for b in lora_b_ref])

    fwd_close = torch.allclose(output_ref, output_triton, atol=1e-2, rtol=1e-2)
    x_grad_close = torch.allclose(x_ref.grad, x.grad, atol=1e-2, rtol=1e-2)
    a_grad_close = torch.allclose(grad_lora_a_ref_flat, lora_a_flat.grad, atol=1e-2, rtol=1e-2)
    b_grad_close = torch.allclose(grad_lora_b_ref_flat, lora_b_flat.grad, atol=1e-2, rtol=1e-2)

    print(f"Forward pass is close: {fwd_close}")
    print(f"Gradient 'x' is close:   {x_grad_close}")
    print(f"Gradient 'a' is close:   {a_grad_close}")
    print(f"Gradient 'b' is close:   {b_grad_close}")
    # print(grad_lora_a_ref_flat)
    # print(lora_a_flat.grad)

    quantiles = [0.25, 0.5, 0.75]

    x_fwd = x.detach()
    lora_a_flat_fwd = lora_a_flat.detach()
    lora_b_flat_fwd = lora_b_flat.detach()

    lora_a_ref_fwd = [a.detach() for a in lora_a_ref]
    lora_b_ref_fwd = [b.detach() for b in lora_b_ref]

    ref_fwd_latency = triton.testing.do_bench(
        lambda: reference_lora_forward(x_fwd, lora_a_ref_fwd, lora_b_ref_fwd, adapter_indices),
        quantiles=quantiles
    )
    triton_fwd_latency = triton.testing.do_bench(
        lambda: batched_lora_heterogeneous(x_fwd, lora_a_flat_fwd, lora_b_flat_fwd, metadata, adapter_indices, OUT_FEATURES),
        quantiles=quantiles
    )

    median_ref_fwd = ref_fwd_latency[1]
    median_triton_fwd = triton_fwd_latency[1]
    speedup_fwd = median_ref_fwd / max(1e-12, median_triton_fwd)
    print(f"\nReference Forward (Median Latency): {median_ref_fwd:.6f} s")
    print(f"Triton Forward   (Median Latency): {median_triton_fwd:.6f} s")
    print(f"Forward Speedup: {speedup_fwd:.2f}x")

    def triton_backward_once():
        x_b = x.detach().requires_grad_()
        a_b = lora_a_flat.detach().requires_grad_()
        b_b = lora_b_flat.detach().requires_grad_()
        out = batched_lora_heterogeneous(x_b, a_b, b_b, metadata, adapter_indices, OUT_FEATURES)
        out.sum().backward(retain_graph=False)
        torch.cuda.synchronize()

    def ref_backward_once():
        x_b = x_ref.detach().requires_grad_()
        a_list = [a.detach().requires_grad_() for a in lora_a_ref]
        b_list = [b.detach().requires_grad_() for b in lora_b_ref]
        out_ref = reference_lora_forward(x_b, a_list, b_list, adapter_indices)
        out_ref.sum().backward(retain_graph=False)
        torch.cuda.synchronize()

    ref_bwd_latency = triton.testing.do_bench(ref_backward_once, quantiles=quantiles)
    triton_bwd_latency = triton.testing.do_bench(triton_backward_once, quantiles=quantiles)

    median_ref_bwd = ref_bwd_latency[1]
    median_triton_bwd = triton_bwd_latency[1]
    speedup_bwd = median_ref_bwd / max(1e-12, median_triton_bwd)

    print(f"Reference Backward (Median Latency): {median_ref_bwd:.6f} s")
    print(f"Triton Backward   (Median Latency): {median_triton_bwd:.6f} s")
    print(f"Backward Speedup: {speedup_bwd:.2f}x")

    print(f"Reference Total (Median Latency): {(median_ref_fwd + median_ref_bwd):.6f} s")
    print(f"Triton Total (Median Latency): {(median_triton_fwd + median_triton_bwd):.6f} s")
    total_speedup = (median_ref_fwd + median_ref_bwd) / max(1e-12, (median_triton_fwd + median_triton_bwd))
    print(f"Total Speedup: {total_speedup:.2f}x")
    print("\nDone.")