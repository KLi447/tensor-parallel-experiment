import torch
import triton
import triton.testing
from transformers import AutoModelForCausalLM, AutoTokenizer
import weakref
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from lora_kernel import batched_lora_heterogeneous, prepare_heterogeneous_lora_weights

def reference_lora_layer_forward(x, lora_a_list, lora_b_list, adapter_indices):
    batch_size, seq_len, in_features = x.shape
    x_reshaped = x.view(batch_size * seq_len, in_features)

    out_features = lora_b_list[0].shape[0]
    output = torch.empty(x_reshaped.size(0), out_features, device=x.device, dtype=x.dtype)

    unique_indices = torch.unique(adapter_indices)

    for idx in unique_indices:
        mask = (adapter_indices == idx)
        expanded_mask = mask.unsqueeze(1).expand(-1, seq_len).reshape(-1)
        x_group = x_reshaped[expanded_mask]

        if x_group.shape[0] == 0:
            continue
            
        a_i = lora_a_list[idx.item()]
        b_i = lora_b_list[idx.item()]

        intermediate = x_group @ a_i.T
        final_output = intermediate @ b_i.T

        output[expanded_mask] = final_output

    return output.view(batch_size, seq_len, out_features)

class LoraLinear(nn.Module):
    def __init__(self, linear_layer, layer_name, model_ref, use_triton_kernel=True):
        super().__init__()
        self.linear = linear_layer
        self.layer_name = layer_name
        self.model_ref = weakref.ref(model_ref) 
        self.use_triton_kernel = use_triton_kernel

    def forward(self, x: torch.Tensor):
        base_output = self.linear(x)

        model = self.model_ref()
        if model is None:
            return base_output

        adapter_indices = model._current_adapter_indices
        lora_params_for_layer = model._lora_params[self.layer_name]

        batch_size, seq_len, in_features = x.shape
        x_reshaped = x.view(-1, in_features)

        adapter_indices_expanded = adapter_indices.unsqueeze(1).expand(-1, seq_len).reshape(-1).contiguous()

        if self.use_triton_kernel:
            lora_output = batched_lora_heterogeneous(
                x_reshaped,
                lora_params_for_layer['a_flat'],
                lora_params_for_layer['b_flat'],
                lora_params_for_layer['metadata'],
                adapter_indices_expanded,
                self.linear.out_features
            )
        else:
             lora_output = reference_lora_layer_forward(
                x,
                lora_params_for_layer['lora_a_list'],
                lora_params_for_layer['lora_b_list'],
                adapter_indices,
            ).view(-1, self.linear.out_features)

        lora_output = lora_output.view(batch_size, seq_len, self.linear.out_features)

        return base_output + lora_output

def replace_linear_with_lora(module, top_level_model, target_modules, use_triton_kernel, prefix=""):
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear) and any(target in name for target in target_modules):
            custom_layer = LoraLinear(child, full_name, top_level_model, use_triton_kernel)
            setattr(module, name, custom_layer)
        else:
            replace_linear_with_lora(child, top_level_model, target_modules, use_triton_kernel, prefix=full_name)

def prepare_all_lora_weights(model, rank_list, device):
    lora_params = {}
    
    for name, module in model.named_modules():
        if isinstance(module, LoraLinear):
            in_features = module.linear.in_features
            out_features = module.linear.out_features

            lora_a_cpu = [torch.randn(r, in_features, dtype=torch.float32) for r in rank_list]
            lora_b_cpu = [torch.randn(out_features, r, dtype=torch.float32) for r in rank_list]

            a_flat, b_flat, metadata = prepare_heterogeneous_lora_weights(lora_a_cpu, lora_b_cpu, device)
            
            lora_params[name] = {
                'a_flat': a_flat,
                'b_flat': b_flat,
                'metadata': metadata,
                'lora_a_list': [a.to(device) for a in lora_a_cpu],
                'lora_b_list': [b.to(device) for b in lora_b_cpu],
            }
            
    return lora_params

def benchmark_model_pass(model, input_ids, adapter_indices, lora_params):
    model._current_adapter_indices = adapter_indices
    model._lora_params = lora_params
    model.zero_grad(set_to_none=True)

    outputs = model(input_ids=input_ids).logits

    loss = outputs.sum()
    loss.backward()
    torch.cuda.synchronize()


if __name__ == '__main__':
    MODEL_ID = "meta-llama/Meta-Llama-3-8B"
    # MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DEVICE = 'cuda'
    DTYPE = torch.float32
    BATCH_SIZE = 4
    SEQ_LEN = 512
    
    rank_list = [2, 16, 32, 64, 128]
    # rank_list = [4, 4, 32, 32, 64]
    NUM_ADAPTERS = len(rank_list)
    TARGET_MODULES = ["q_proj", "v_proj"] 

    median_ref = 0.0
    median_triton = 0.0
    quantiles = [0.25, 0.5, 0.75]

    print("--- Benchmarking Reference Model (PyTorch loop) ---")
    model_ref = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE
    )
    replace_linear_with_lora(model_ref, model_ref, TARGET_MODULES, use_triton_kernel=False)
    lora_params_ref = prepare_all_lora_weights(model_ref, rank_list, DEVICE)
    input_ids = torch.randint(0, model_ref.config.vocab_size, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
    adapter_indices = torch.randint(0, NUM_ADAPTERS, (BATCH_SIZE,), device=DEVICE, dtype=torch.int32)

    model_ref.eval().zero_grad()
    ref_latency = triton.testing.do_bench(
        lambda: benchmark_model_pass(model_ref, input_ids, adapter_indices, lora_params_ref),
        warmup=10, rep=100 ,quantiles=quantiles
    )
    median_ref = ref_latency[1]
    print(f"Reference Median Latency: {median_ref:.6f} s")
    
    print("Clearing reference model from memory...")
    del model_ref, lora_params_ref
    gc.collect()
    torch.cuda.empty_cache()
    print("Memory cleared.\n")

    print("--- Benchmarking Triton-Optimized Model ---")
    model_triton = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=DTYPE, device_map=DEVICE
    )
    replace_linear_with_lora(model_triton, model_triton, TARGET_MODULES, use_triton_kernel=True)

    lora_params_triton = prepare_all_lora_weights(model_triton, rank_list, DEVICE)

    model_triton.eval().zero_grad()
    with torch.autograd.profiler.profile(use_device='cuda') as prof:
        triton_latency = triton.testing.do_bench(
            lambda: benchmark_model_pass(model_triton, input_ids, adapter_indices, lora_params_triton),
            warmup=10, rep=100, quantiles=quantiles
        )

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
    median_triton = triton_latency[1]
    print(f"Triton Median Latency: {median_triton:.6f} s")

    del model_triton, lora_params_triton
    gc.collect()
    torch.cuda.empty_cache()

    print("\n--- Final Results ---")
    speedup = median_ref / max(1e-12, median_triton)
    print(f"Reference Full Pass (Median Latency): {median_ref:.6f} s")
    print(f"Triton Full Pass    (Median Latency): {median_triton:.6f} s")
    print(f"End-to-End Speedup: {speedup:.2f}x")
    
    print("\nDone.")