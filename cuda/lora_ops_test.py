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
        x.to(torch.float16),
        lora_weights['lora_a'].to(torch.float16),
        lora_weights['lora_b'].to(torch.float16),
        adapter_indices
    )

    return base_output + lora_output.to(base_output.dtype)

def sequential_lora_op(
    x: torch.Tensor,
    base_output: torch.Tensor,
    lora_weights: dict,
    adapter_indices: torch.Tensor
) -> torch.Tensor:
    lora_output = torch.zeros_like(base_output)

    for i in range(x.size(0)):
        adapter_idx = adapter_indices[i].item()
        lora_a = lora_weights['lora_a'][adapter_idx]
        lora_b = lora_weights['lora_b'][adapter_idx]
        lora_output[i] = (x[i] @ lora_a.t()) @ lora_b.t()

    return base_output + lora_output

def benchmark(func, *args, **kwargs):
    #warmup
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


if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        exit()
    if lora_cuda_ops_lib is None:
        exit()

    device = 'cuda'
    dtype = torch.float16
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
        'lora_a': torch.randn(num_adapters, r, in_features, device=device, dtype=dtype),
        'lora_b': torch.randn(num_adapters, out_features, r, device=device, dtype=dtype)
    }
    x_input = torch.randn(total_batch_size, in_features, device=device, dtype=dtype)
    base_output = torch.randn(total_batch_size, out_features, device=device, dtype=dtype)

    adapter_indices_list = []
    for i, bs in enumerate(batch_sizes):
        adapter_indices_list.append(torch.full((bs,), i, dtype=torch.int32))
    adapter_indices = torch.cat(adapter_indices_list).to(device)

    print("\n--- Running Correctness Test ---")
    custom_op_output = batched_lora_op(x_input, base_output, lora_weights, adapter_indices)
    sequential_op_output = sequential_lora_op(x_input, base_output, lora_weights, adapter_indices)

    are_close = torch.allclose(custom_op_output, sequential_op_output, atol=1e-2, rtol=1e-2)
    print(f"Verification successful: {are_close}")
    if not are_close:
        print("Warning: Outputs do not match. There may be an issue in the kernel implementation.")
    print("-" * 20)

    print("\n--- Running Performance Benchmark ---")
    custom_op_latency = benchmark(batched_lora_op, x_input, base_output, lora_weights, adapter_indices)
    print(f"Custom Kernel Avg Latency:  {custom_op_latency:.6f} ms")

    sequential_op_latency = benchmark(sequential_lora_op, x_input, base_output, lora_weights, adapter_indices)
    print(f"Sequential Loop Avg Latency: {sequential_op_latency:.6f} ms")

    speedup = sequential_op_latency / custom_op_latency
    print(f"\nSpeedup: {speedup:.2f}x")
    print("-" * 20)