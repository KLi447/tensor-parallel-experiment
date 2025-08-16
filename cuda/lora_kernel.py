import torch
import triton
import triton.language as tl

@triton.jit
def _batched_lora_forward_kernel_heterogeneous(
    X_ptr,
    LORA_A_FLAT_ptr, LORA_B_FLAT_ptr,
    METADATA_ptr,
    ADAPTER_INDICES_ptr,
    OUTPUT_ptr,
    in_features, out_features,
    stride_x_batch, stride_x_in,
    stride_out_batch, stride_out_d,
    stride_meta_adapter, stride_meta_field,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_RANK_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0)

    adapter_idx = tl.load(ADAPTER_INDICES_ptr + batch_idx)

    meta_base_ptr = METADATA_ptr + adapter_idx * stride_meta_adapter
    offset_a = tl.load(meta_base_ptr + 0 * stride_meta_field)
    offset_b = tl.load(meta_base_ptr + 1 * stride_meta_field)
    rank = tl.load(meta_base_ptr + 2 * stride_meta_field)

    lora_a_i_ptr = LORA_A_FLAT_ptr + offset_a
    lora_b_i_ptr = LORA_B_FLAT_ptr + offset_b
    x_i_ptr = X_ptr + batch_idx * stride_x_batch

    acc_h = tl.zeros((MAX_RANK_BLOCK,), dtype=tl.float32)

    r_offsets = tl.arange(0, MAX_RANK_BLOCK)
    for k_start in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offsets = (k_start * BLOCK_SIZE_K) + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features

        x_ptrs = x_i_ptr + k_offsets * stride_x_in
        x_chunk = tl.load(x_ptrs, mask=k_mask, other=0.0)

        a_ptrs = lora_a_i_ptr + \
                 (r_offsets[:, None] * in_features) + \
                 (k_offsets[None, :])
        a_mask = (r_offsets[:, None] < rank) & k_mask[None, :]
        a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0)

        acc_h += tl.sum(x_chunk[None, :] * a_chunk, axis=1)

    for n_start in range(0, tl.cdiv(out_features, BLOCK_SIZE_N)):
        n_offsets = (n_start * BLOCK_SIZE_N) + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < out_features

        acc_final = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

        b_ptrs = lora_b_i_ptr + \
                 (n_offsets[:, None] * rank) + \
                 (r_offsets[None, :])
        b_mask = n_mask[:, None] & (r_offsets[None, :] < rank)
        b_chunk = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc_final += tl.sum(b_chunk * acc_h[None, :], axis=1)

        output_ptrs = OUTPUT_ptr + batch_idx * stride_out_batch + n_offsets * stride_out_d
        tl.store(output_ptrs, acc_final, mask=n_mask)

@triton.jit
def _batched_lora_backward_kernel_heterogeneous(
    X_ptr, LORA_A_FLAT_ptr, LORA_B_FLAT_ptr, METADATA_ptr, ADAPTER_INDICES_ptr, GRAD_OUTPUT_ptr,
    GRAD_X_ptr, GRAD_LORA_A_FLAT_ptr, GRAD_LORA_B_FLAT_ptr,
    in_features, out_features,
    stride_x_batch, stride_x_in,
    stride_grad_out_batch, stride_grad_out_d,
    stride_grad_x_batch, stride_grad_x_in,
    stride_meta_adapter, stride_meta_field,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    MAX_RANK_BLOCK: tl.constexpr,
):
    batch_idx = tl.program_id(axis=0)

    adapter_idx = tl.load(ADAPTER_INDICES_ptr + batch_idx)
    meta_base_ptr = METADATA_ptr + adapter_idx * stride_meta_adapter
    offset_a = tl.load(meta_base_ptr + 0 * stride_meta_field)
    offset_b = tl.load(meta_base_ptr + 1 * stride_meta_field)
    rank = tl.load(meta_base_ptr + 2 * stride_meta_field)

    x_i_ptr = X_ptr + batch_idx * stride_x_batch
    lora_a_i_ptr = LORA_A_FLAT_ptr + offset_a
    lora_b_i_ptr = LORA_B_FLAT_ptr + offset_b
    grad_out_i_ptr = GRAD_OUTPUT_ptr + batch_idx * stride_grad_out_batch
    grad_x_i_ptr = GRAD_X_ptr + batch_idx * stride_grad_x_batch
    grad_lora_a_i_ptr = GRAD_LORA_A_FLAT_ptr + offset_a
    grad_lora_b_i_ptr = GRAD_LORA_B_FLAT_ptr + offset_b
    
    r_offsets = tl.arange(0, MAX_RANK_BLOCK)
    r_mask = r_offsets < rank

    acc_h = tl.zeros((MAX_RANK_BLOCK,), dtype=tl.float32)
    for k_start in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offsets = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        x_chunk = tl.load(x_i_ptr + k_offsets * stride_x_in, mask=k_mask, other=0.0)
        a_ptrs = lora_a_i_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
        a_mask = r_mask[:, None] & k_mask[None, :]
        a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0)
        acc_h += tl.sum(x_chunk[None, :] * a_chunk, axis=1)

    acc_grad_h = tl.zeros((MAX_RANK_BLOCK,), dtype=tl.float32)
    for n_start in range(0, tl.cdiv(out_features, BLOCK_SIZE_N)):
        n_offsets = n_start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < out_features
        grad_out_chunk = tl.load(grad_out_i_ptr + n_offsets * stride_grad_out_d, mask=n_mask, other=0.0)
        b_ptrs = lora_b_i_ptr + (n_offsets[:, None] * rank) + r_offsets[None, :]
        b_mask = n_mask[:, None] & r_mask[None, :]
        b_chunk = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc_grad_h += tl.sum(grad_out_chunk[:, None] * b_chunk, axis=0)

    for k_start in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offsets = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        a_ptrs = lora_a_i_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
        a_mask = r_mask[:, None] & k_mask[None, :]
        a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0)
        grad_x_chunk = tl.sum(acc_grad_h[:, None] * a_chunk, axis=0)
        tl.store(grad_x_i_ptr + k_offsets * stride_grad_x_in, grad_x_chunk, mask=k_mask)

    for n_start in range(0, tl.cdiv(out_features, BLOCK_SIZE_N)):
        n_offsets = n_start * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = n_offsets < out_features
        grad_out_chunk = tl.load(grad_out_i_ptr + n_offsets * stride_grad_out_d, mask=n_mask, other=0.0)
        grad_b_chunk = grad_out_chunk[:, None] * acc_h[None, :]
        grad_b_ptrs = grad_lora_b_i_ptr + (n_offsets[:, None] * rank) + r_offsets[None, :]
        grad_b_mask = n_mask[:, None] & r_mask[None, :]
        tl.atomic_add(grad_b_ptrs, grad_b_chunk, mask=grad_b_mask)

    for k_start in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
        k_offsets = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        k_mask = k_offsets < in_features
        x_chunk = tl.load(x_i_ptr + k_offsets * stride_x_in, mask=k_mask, other=0.0)
        grad_a_chunk = acc_grad_h[:, None] * x_chunk[None, :]
        grad_a_ptrs = grad_lora_a_i_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
        grad_a_mask = r_mask[:, None] & k_mask[None, :]
        tl.atomic_add(grad_a_ptrs, grad_a_chunk, mask=grad_a_mask)

class BatchedLoRAFunctionHeterogeneous(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lora_a_flat, lora_b_flat, metadata, adapter_indices):
        x = x.contiguous()
        batch_size, in_features = x.shape
        out_features = in_features #FIXME eventually?
        
        output = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
        grid = (batch_size,)
        
        max_rank = int(torch.max(metadata[:, 2]).item())
        MAX_RANK_BLOCK = triton.next_power_of_2(max_rank)

        _batched_lora_forward_kernel_heterogeneous[grid](
            X_ptr=x, LORA_A_FLAT_ptr=lora_a_flat, LORA_B_FLAT_ptr=lora_b_flat,
            METADATA_ptr=metadata, ADAPTER_INDICES_ptr=adapter_indices, OUTPUT_ptr=output,
            in_features=in_features, out_features=out_features,
            stride_x_batch=x.stride(0), stride_x_in=x.stride(1),
            stride_out_batch=output.stride(0), stride_out_d=output.stride(1),
            stride_meta_adapter=metadata.stride(0), stride_meta_field=metadata.stride(1),
            BLOCK_SIZE_K=32, BLOCK_SIZE_N=32, MAX_RANK_BLOCK=MAX_RANK_BLOCK,
        )
        
        ctx.save_for_backward(x, lora_a_flat, lora_b_flat, metadata, adapter_indices)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.max_rank_block = MAX_RANK_BLOCK
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        x, lora_a_flat, lora_b_flat, metadata, adapter_indices = ctx.saved_tensors
        
        grad_x = torch.empty_like(x)
        grad_lora_a_flat = torch.zeros_like(lora_a_flat)
        grad_lora_b_flat = torch.zeros_like(lora_b_flat)
        
        grid = (x.shape[0],)

        _batched_lora_backward_kernel_heterogeneous[grid](
            X_ptr=x, LORA_A_FLAT_ptr=lora_a_flat, LORA_B_FLAT_ptr=lora_b_flat,
            METADATA_ptr=metadata, ADAPTER_INDICES_ptr=adapter_indices, GRAD_OUTPUT_ptr=grad_output,
            GRAD_X_ptr=grad_x, GRAD_LORA_A_FLAT_ptr=grad_lora_a_flat, GRAD_LORA_B_FLAT_ptr=grad_lora_b_flat,
            in_features=ctx.in_features, out_features=ctx.out_features,
            stride_x_batch=x.stride(0), stride_x_in=x.stride(1),
            stride_grad_out_batch=grad_output.stride(0), stride_grad_out_d=grad_output.stride(1),
            stride_grad_x_batch=grad_x.stride(0), stride_grad_x_in=grad_x.stride(1),
            stride_meta_adapter=metadata.stride(0), stride_meta_field=metadata.stride(1),
            BLOCK_SIZE_K=32, BLOCK_SIZE_N=32, MAX_RANK_BLOCK=ctx.max_rank_block,
        )
        
        return grad_x, grad_lora_a_flat, grad_lora_b_flat, None, None

batched_lora_heterogeneous = BatchedLoRAFunctionHeterogeneous.apply

def prepare_heterogeneous_lora_weights(lora_a_list, lora_b_list, device):
    lora_a_flat = torch.cat([t.flatten() for t in lora_a_list]).to(device)
    lora_b_flat = torch.cat([t.flatten() for t in lora_b_list]).to(device)
    num_adapters = len(lora_a_list)
    metadata = torch.zeros((num_adapters, 3), dtype=torch.int32, device=device)
    offset_a, offset_b = 0, 0
    for i in range(num_adapters):
        rank_a, _ = lora_a_list[i].shape
        _, rank_b = lora_b_list[i].shape
        assert rank_a == rank_b, f"Mismatched ranks for adapter {i}"
        metadata[i, 0], metadata[i, 1], metadata[i, 2] = offset_a, offset_b, rank_a
        offset_a += lora_a_list[i].numel()
        offset_b += lora_b_list[i].numel()
    return lora_a_flat, lora_b_flat, metadata