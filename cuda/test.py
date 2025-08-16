import torch
import triton
from lora_kernel import batched_lora_heterogeneous, prepare_heterogeneous_lora_weights

def reference_lora_forward(x, lora_a_list, lora_b_list, adapter_indices):
    output_slices = []
    for i in range(x.size(0)):
        adapter_idx = adapter_indices[i].item()
        x_i, lora_a_i, lora_b_i = x[i].unsqueeze(0), lora_a_list[adapter_idx], lora_b_list[adapter_idx]
        intermediate = torch.matmul(x_i, lora_a_i.t())
        final_output = torch.matmul(intermediate, lora_b_i.t())
        output_slices.append(final_output)
    return torch.cat(output_slices, 0)

if __name__ == '__main__':
    BATCH_SIZE, IN_FEATURES, OUT_FEATURES = 1024, 4096, 4096
    device = 'cuda'

    lora_a_adapters_cpu = [torch.randn(r, IN_FEATURES, dtype=torch.float32) for r in [8, 16, 32, 4, 128]]
    lora_b_adapters_cpu = [torch.randn(OUT_FEATURES, r, dtype=torch.float32) for r in [8, 16, 32, 4, 128]]
    NUM_ADAPTERS = len(lora_a_adapters_cpu)

    lora_a_flat, lora_b_flat, metadata = prepare_heterogeneous_lora_weights(lora_a_adapters_cpu, lora_b_adapters_cpu, device)

    x = torch.randn(BATCH_SIZE, IN_FEATURES, device=device, dtype=torch.float32, requires_grad=True)
    adapter_indices = torch.randint(0, NUM_ADAPTERS, (BATCH_SIZE,), device=device, dtype=torch.int32)

    x_ref = x.detach().clone().requires_grad_()
    lora_a_ref = [a.to(device).requires_grad_() for a in lora_a_adapters_cpu]
    lora_b_ref = [b.to(device).requires_grad_() for b in lora_b_adapters_cpu]

    print("Verifying Triton kernel against reference implementation...")

    lora_a_flat.requires_grad_()
    lora_b_flat.requires_grad_()
    output_triton = batched_lora_heterogeneous(x, lora_a_flat, lora_b_flat, metadata, adapter_indices)
    output_triton.sum().backward()

    output_ref = reference_lora_forward(x_ref, lora_a_ref, lora_b_ref, adapter_indices)
    output_ref.sum().backward()

    grad_lora_a_ref_flat = torch.cat([g.flatten() for g in [a.grad for a in lora_a_ref]])
    grad_lora_b_ref_flat = torch.cat([g.flatten() for g in [b.grad for b in lora_b_ref]])

    print(f"Forward pass is close: {torch.allclose(output_ref, output_triton, atol=1e-2, rtol=1e-2)}")
    print(f"Gradient 'x' is close:   {torch.allclose(x_ref.grad, x.grad, atol=1e-2, rtol=1e-2)}")
    print(f"Gradient 'a' is close:   {torch.allclose(grad_lora_a_ref_flat, lora_a_flat.grad, atol=1e-2, rtol=1e-2)}")
    print(f"Gradient 'b' is close:   {torch.allclose(grad_lora_b_ref_flat, lora_b_flat.grad, atol=1e-2, rtol=1e-2)}")

    print("\n--- Benchmarking Forward Pass ---")
    quantiles = [0.25, 0.5, 0.75]

    x_fwd = x.detach()
    lora_a_flat_fwd = lora_a_flat.detach()
    lora_b_flat_fwd = lora_b_flat.detach()
    lora_a_ref_fwd = [a.detach() for a in lora_a_ref]
    lora_b_ref_fwd = [b.detach() for b in lora_b_ref]

    ref_fwd_latency = triton.testing.do_bench(lambda: reference_lora_forward(x_fwd, lora_a_ref_fwd, lora_b_ref_fwd, adapter_indices), quantiles=quantiles)
    triton_fwd_latency = triton.testing.do_bench(lambda: batched_lora_heterogeneous(x_fwd, lora_a_flat_fwd, lora_b_flat_fwd, metadata, adapter_indices), quantiles=quantiles)
    
    speedup_fwd = ref_fwd_latency[1] / triton_fwd_latency[1]
    print(f"Reference Forward (Median Latency): {ref_fwd_latency[1]:.4f} ms")
    print(f"Triton Forward (Median Latency):    {triton_fwd_latency[1]:.4f} ms")
    print(f"Forward Speedup: {speedup_fwd:.2f}x")

    print("\n--- Benchmarking Backward Pass ---")
    output_triton_bench = batched_lora_heterogeneous(x, lora_a_flat, lora_b_flat, metadata, adapter_indices)
    loss_triton_bench = output_triton_bench.sum()

    output_ref_bench = reference_lora_forward(x_ref, lora_a_ref, lora_b_ref, adapter_indices)
    loss_ref_bench = output_ref_bench.sum()

    ref_bwd_latency = triton.testing.do_bench(lambda: loss_ref_bench.backward(retain_graph=True), quantiles=quantiles)
    triton_bwd_latency = triton.testing.do_bench(lambda: loss_triton_bench.backward(retain_graph=True), quantiles=quantiles)

    speedup_bwd = ref_bwd_latency[1] / triton_bwd_latency[1]
    print(f"Reference Backward (Median Latency): {ref_bwd_latency[1]:.4f} ms")
    print(f"Triton Backward (Median Latency):    {triton_bwd_latency[1]:.4f} ms")
    print(f"Backward Speedup: {speedup_bwd:.2f}x")
