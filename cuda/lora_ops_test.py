import torch
import time

try:
    import lora_cuda_ops_lib
except ImportError:
    print("Could not import 'lora_cuda_ops_lib'. Please build the extension with 'python setup.py install'")
    lora_cuda_ops_lib = None

def batched_lora_op(
    x: torch.Tensor,
    base_output: torch.Tensor,
    lora_weights: dict,
    adapter_indices: torch.Tensor
) -> torch.Tensor:
    if lora_cuda_ops_lib is None:
        raise ImportError("CUDA extension not available.")

    lora_output = lora_cuda_ops_lib.forward(
        x,
        lora_weights['lora_a'],
        lora_weights['lora_b'],
        adapter_indices
    )
    return base_output + lora_output.to(base_output.dtype)

def sequential_lora_op(
    x: torch.Tensor,
    base_output: torch.Tensor,
    lora_weights: dict,
    adapter_indices: torch.Tensor
) -> torch.Tensor:
    lora_output = torch.zeros_like(base_output, dtype=x.dtype)

    for i in range(x.size(0)):
        adapter_idx = adapter_indices[i].item()
        lora_a = lora_weights['lora_a'][adapter_idx]
        lora_b = lora_weights['lora_b'][adapter_idx]

        intermediate = x[i].unsqueeze(0) @ lora_a.t()
        lora_output[i] = (intermediate @ lora_b.t()).squeeze(0)

    return base_output + lora_output.to(base_output.dtype)

def benchmark_forward(func, *args, **kwargs):
    # warmup
    for _ in range(10):
        func(*args, **kwargs)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        func(*args, **kwargs)
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 100.0

def benchmark_backward(func, inputs, *args, **kwargs):
    # warmup
    for _ in range(10):
        for inp in inputs:
            inp.grad = None
        output = func(*args, **kwargs)
        loss = output.sum()
        loss.backward()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(100):
        for inp in inputs:
            inp.grad = None
        output = func(*args, **kwargs)
        loss = output.sum()
        loss.backward()
    end_event.record()
    
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / 100.0

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        exit()
    if lora_cuda_ops_lib is None:
        exit()

    device = 'cuda'
    dtype = torch.float32
    in_features = 4096
    out_features = 4096
    r = 32
    num_adapters = 4
    batch_sizes = [16, 8, 32, 24]
    total_batch_size = sum(batch_sizes)

    print("--- Test Setup ---")
    print(f"Device: {device}, DType: {dtype}")
    print(f"Dimensions: In={in_features}, Out={out_features}, r={r}")
    print(f"Adapters: {num_adapters}, Total Batch Size: {total_batch_size}")
    print("-" * 20)

    lora_weights = {
        'lora_a': torch.randn(num_adapters, r, in_features, device=device, dtype=dtype, requires_grad=True),
        'lora_b': torch.randn(num_adapters, out_features, r, device=device, dtype=dtype, requires_grad=True)
    }
    x_input = torch.randn(total_batch_size, in_features, device=device, dtype=dtype, requires_grad=True)
    base_output = torch.randn(total_batch_size, out_features, device=device, dtype=dtype)
    
    adapter_indices_list = []
    for i, bs in enumerate(batch_sizes):
        adapter_indices_list.append(torch.full((bs,), i, dtype=torch.int32))
    # Adapter indices must be on CPU for the custom kernel
    adapter_indices = torch.cat(adapter_indices_list).to('cpu')

    # --- Correctness Test (Forward) ---
    print("\n--- Running Forward Correctness Test ---")
    custom_op_output = batched_lora_op(x_input, base_output, lora_weights, adapter_indices)
    sequential_op_output = sequential_lora_op(x_input, base_output, lora_weights, adapter_indices)
    
    are_close_fwd = torch.allclose(custom_op_output, sequential_op_output, atol=1e-2, rtol=1e-2)
    print(f"Forward verification successful: {are_close_fwd}")
    if not are_close_fwd:
        print("Warning: Forward outputs do not match.")
    print("-" * 20)

    # --- Correctness Test (Backward) ---
    print("\n--- Running Backward Correctness Test ---")
    loss_custom = custom_op_output.sum()
    loss_custom.backward()
    grad_x_custom = x_input.grad.clone()
    grad_a_custom = lora_weights['lora_a'].grad.clone()
    grad_b_custom = lora_weights['lora_b'].grad.clone()

    # Reset gradients
    x_input.grad = None
    lora_weights['lora_a'].grad = None
    lora_weights['lora_b'].grad = None

    loss_seq = sequential_op_output.sum()
    loss_seq.backward()
    grad_x_seq = x_input.grad.clone()
    grad_a_seq = lora_weights['lora_a'].grad.clone()
    grad_b_seq = lora_weights['lora_b'].grad.clone()

    are_close_gx = torch.allclose(grad_x_custom, grad_x_seq, atol=1e-3, rtol=1e-3)
    are_close_ga = torch.allclose(grad_a_custom, grad_a_seq, atol=1e-3, rtol=1e-3)
    are_close_gb = torch.allclose(grad_b_custom, grad_b_seq, atol=1e-3, rtol=1e-3)

    print(f"Gradient 'x' verification successful: {are_close_gx}")
    print(f"Gradient 'lora_a' verification successful: {are_close_ga}")
    print(f"Gradient 'lora_b' verification successful: {are_close_gb}")
    if not all([are_close_gx, are_close_ga, are_close_gb]):
        print("Warning: Gradients do not match. There may be an issue in the backward implementation.")
    print("-" * 20)
    
    # --- Performance Benchmark ---
    print("\n--- Running Performance Benchmark (Forward Only) ---")
    custom_fwd_latency = benchmark_forward(batched_lora_op, x_input, base_output, lora_weights, adapter_indices)
    print(f"Custom Kernel Avg Latency:     {custom_fwd_latency:.6f} ms")
    seq_fwd_latency = benchmark_forward(sequential_lora_op, x_input, base_output, lora_weights, adapter_indices)
    print(f"Sequential Loop Avg Latency:   {seq_fwd_latency:.6f} ms")
    print(f"Speedup: {seq_fwd_latency / custom_fwd_latency:.2f}x")
    print("-" * 20)

    print("\n--- Running Performance Benchmark (Forward + Backward) ---")
    grad_inputs = [x_input, lora_weights['lora_a'], lora_weights['lora_b']]
    custom_bwd_latency = benchmark_backward(batched_lora_op, grad_inputs, x_input, base_output, lora_weights, adapter_indices)
    print(f"Custom Kernel Avg Latency:     {custom_bwd_latency:.6f} ms")
    seq_bwd_latency = benchmark_backward(sequential_lora_op, grad_inputs, x_input, base_output, lora_weights, adapter_indices)
    print(f"Sequential Loop Avg Latency:   {seq_bwd_latency:.6f} ms")
    print(f"Speedup: {seq_bwd_latency / custom_bwd_latency:.2f}x")
    print("-" * 20)