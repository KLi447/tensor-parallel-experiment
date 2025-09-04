import torch
import triton
import triton.language as tl

@triton.jit
def _batched_lora_forward(
    X_ptr,
    LORA_A_FLAT_ptr, LORA_B_FLAT_ptr,
    METADATA_ptr,
    ADAPTER_INDICES_ptr,
    OUTPUT_ptr,
    in_features, out_features,
    stride_x_tok, stride_x_in,
    stride_out_tok, stride_out_d,
    stride_meta_adapter, stride_meta_field,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    R_TILE: tl.constexpr,
):
    pid_tok = tl.program_id(axis=0)
    pid_nt  = tl.program_id(axis=1)

    adapter_idx = tl.load(ADAPTER_INDICES_ptr + pid_tok)

    meta_base_ptr = METADATA_ptr + adapter_idx * stride_meta_adapter
    offset_a = tl.load(meta_base_ptr + 0 * stride_meta_field)
    offset_b = tl.load(meta_base_ptr + 1 * stride_meta_field)
    rank = tl.load(meta_base_ptr + 2 * stride_meta_field)

    x_tok_ptr = X_ptr + pid_tok * stride_x_tok
    out_tok_ptr = OUTPUT_ptr + pid_tok * stride_out_tok
    lora_a_i_ptr = LORA_A_FLAT_ptr + offset_a
    lora_b_i_ptr = LORA_B_FLAT_ptr + offset_b

    n_offsets = pid_nt * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < out_features

    acc_final = tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32)

    r_base = 0
    while r_base < rank:
        r_offsets = r_base + tl.arange(0, R_TILE)
        r_mask = r_offsets < rank

        h_tile = tl.zeros((R_TILE,), dtype=tl.float32)

        for k_start in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
            k_offsets = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < in_features

            x_chunk = tl.load(x_tok_ptr + k_offsets * stride_x_in, mask=k_mask, other=0.0)

            a_ptrs = lora_a_i_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
            a_mask = r_mask[:, None] & k_mask[None, :]
            a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0)

            h_tile += tl.sum(x_chunk[None, :] * a_chunk, axis=1)

        b_ptrs = lora_b_i_ptr + (n_offsets[:, None] * rank) + r_offsets[None, :]
        b_mask = n_mask[:, None] & r_mask[None, :]
        b_chunk = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc_final += tl.sum(b_chunk * h_tile[None, :], axis=1)

        r_base += R_TILE

    out_ptrs = out_tok_ptr + n_offsets * stride_out_d
    tl.store(out_ptrs, acc_final, mask=n_mask)

@triton.jit
def _batched_lora_backwardA(
    X_ptr, GRAD_OUT_ptr,
    LORA_A_FLAT_ptr, LORA_B_FLAT_ptr,
    METADATA_ptr, ADAPTER_INDICES_ptr,
    ACC_H_SCRATCH_ptr, GRAD_LORA_B_FLAT_ptr,
    in_features, out_features,
    stride_x_tok, stride_x_in,
    stride_grad_out_tok, stride_grad_out_d,
    stride_meta_adapter, stride_meta_field,
    stride_acc_h_tok, stride_acc_h_r,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    R_TILE: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_nt  = tl.program_id(1)

    adapter_idx = tl.load(ADAPTER_INDICES_ptr + pid_tok)
    meta_base = METADATA_ptr + adapter_idx * stride_meta_adapter
    offset_a = tl.load(meta_base + 0 * stride_meta_field)
    offset_b = tl.load(meta_base + 1 * stride_meta_field)
    rank     = tl.load(meta_base + 2 * stride_meta_field)

    x_tok_ptr      = X_ptr + pid_tok * stride_x_tok
    grad_out_tok   = GRAD_OUT_ptr + pid_tok * stride_grad_out_tok
    lora_a_ptr     = LORA_A_FLAT_ptr + offset_a
    lora_b_ptr     = LORA_B_FLAT_ptr + offset_b
    grad_b_ptr     = GRAD_LORA_B_FLAT_ptr + offset_b
    acc_h_base_ptr = ACC_H_SCRATCH_ptr + pid_tok * stride_acc_h_tok

    n_offsets = pid_nt * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    n_mask = n_offsets < out_features

    r_base = 0
    while r_base < rank:
        r_offsets = r_base + tl.arange(0, R_TILE)
        r_mask = r_offsets < rank

        h_tile = tl.zeros((R_TILE,), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(in_features, BLOCK_SIZE_K)):
            k_offsets = k_start * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
            k_mask = k_offsets < in_features

            x_chunk = tl.load(x_tok_ptr + k_offsets * stride_x_in, mask=k_mask, other=0.0)
            a_ptrs = lora_a_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
            a_mask = r_mask[:, None] & k_mask[None, :]
            a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0)

            h_tile += tl.sum(a_chunk * x_chunk[None, :], axis=1)

        grad_out_chunk = tl.load(grad_out_tok + n_offsets * stride_grad_out_d, mask=n_mask, other=0.0)

        b_ptrs = lora_b_ptr + (n_offsets[:, None] * rank) + r_offsets[None, :]
        b_mask = n_mask[:, None] & r_mask[None, :]
        b_chunk = tl.load(b_ptrs, mask=b_mask, other=0.0)

        partial_acc_h = tl.sum(grad_out_chunk[:, None] * b_chunk, axis=0)

        acc_h_ptrs = acc_h_base_ptr + r_offsets * stride_acc_h_r
        tl.atomic_add(acc_h_ptrs, partial_acc_h, mask=r_mask)

        grad_b_chunk = grad_out_chunk[:, None] * h_tile[None, :]
        grad_b_ptrs = grad_b_ptr + (n_offsets[:, None] * rank) + r_offsets[None, :]
        grad_b_mask = n_mask[:, None] & r_mask[None, :]
        tl.atomic_add(grad_b_ptrs, grad_b_chunk, mask=grad_b_mask)

        r_base += R_TILE

@triton.jit
def _batched_lora_backwardB(
    X_ptr, ACC_H_SCRATCH_ptr,
    LORA_A_FLAT_ptr, METADATA_ptr, ADAPTER_INDICES_ptr,
    GRAD_X_ptr, GRAD_LORA_A_FLAT_ptr,
    in_features, rank_max,
    stride_x_tok, stride_x_in,
    stride_acc_h_tok, stride_acc_h_r,
    stride_meta_adapter, stride_meta_field,
    stride_grad_x_tok, stride_grad_x_in,
    BLOCK_SIZE_K: tl.constexpr,
    R_TILE: tl.constexpr,
):
    pid_tok = tl.program_id(0)
    pid_kt  = tl.program_id(1)

    adapter_idx = tl.load(ADAPTER_INDICES_ptr + pid_tok)
    meta_base = METADATA_ptr + adapter_idx * stride_meta_adapter
    offset_a = tl.load(meta_base + 0 * stride_meta_field)
    rank = tl.load(meta_base + 2 * stride_meta_field)

    x_tok_ptr = X_ptr + pid_tok * stride_x_tok
    grad_x_tok_ptr = GRAD_X_ptr + pid_tok * stride_grad_x_tok
    acc_h_base = ACC_H_SCRATCH_ptr + pid_tok * stride_acc_h_tok
    lora_a_ptr = LORA_A_FLAT_ptr + offset_a
    grad_a_ptr = GRAD_LORA_A_FLAT_ptr + offset_a

    k_offsets = pid_kt * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    k_mask = k_offsets < in_features

    x_chunk = tl.load(x_tok_ptr + k_offsets * stride_x_in, mask=k_mask, other=0.0)

    grad_x_local = tl.zeros((BLOCK_SIZE_K,), dtype=tl.float32)

    r_base = 0
    while r_base < rank:
        r_offsets = r_base + tl.arange(0, R_TILE)
        r_mask = r_offsets < rank

        acc_grad_h_tile = tl.load(acc_h_base + r_offsets * stride_acc_h_r, mask=r_mask, other=0.0)

        a_ptrs = lora_a_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
        a_mask = r_mask[:, None] & k_mask[None, :]
        a_chunk = tl.load(a_ptrs, mask=a_mask, other=0.0)

        grad_x_local += tl.sum(acc_grad_h_tile[:, None] * a_chunk, axis=0)

        grad_a_chunk = acc_grad_h_tile[:, None] * x_chunk[None, :]

        grad_a_ptrs = grad_a_ptr + (r_offsets[:, None] * in_features) + k_offsets[None, :]
        grad_a_mask = r_mask[:, None] & k_mask[None, :]
        tl.atomic_add(grad_a_ptrs, grad_a_chunk, mask=grad_a_mask)

        r_base += R_TILE

    tl.store(grad_x_tok_ptr + k_offsets * stride_grad_x_in, grad_x_local, mask=k_mask)

class BatchedLoRAFunctionHeterogeneous(torch.autograd.Function):
    BLOCK_SIZE_N = 64
    @staticmethod
    def forward(ctx, x, lora_a_flat, lora_b_flat, metadata, adapter_indices, block_size):
        x = x.contiguous()
        is_bsd = (x.ndim == 3)
        if is_bsd:
            B, S, D = x.shape
            T = B * S
            x_tok = x.view(T, D)
        else:
            x_tok = x
            T, D = x_tok.shape

        in_features = D
        out_features = in_features  # FIXME?

        output_tok = torch.empty((T, out_features), device=x.device, dtype=x.dtype)

        N_TILES = triton.cdiv(out_features, BatchedLoRAFunctionHeterogeneous.BLOCK_SIZE_N)
        grid = (T, N_TILES)

        _batched_lora_forward[grid](
            X_ptr=x_tok, 
            LORA_A_FLAT_ptr=lora_a_flat,
            LORA_B_FLAT_ptr=lora_b_flat,
            METADATA_ptr=metadata,
            ADAPTER_INDICES_ptr=adapter_indices,
            OUTPUT_ptr=output_tok,
            in_features=in_features,
            out_features=out_features,
            stride_x_tok=x_tok.stride(0),
            stride_x_in=x_tok.stride(1),
            stride_out_tok=output_tok.stride(0),
            stride_out_d=output_tok.stride(1),
            stride_meta_adapter=metadata.stride(0),
            stride_meta_field=metadata.stride(1),
            BLOCK_SIZE_K=32,
            BLOCK_SIZE_N=BatchedLoRAFunctionHeterogeneous.BLOCK_SIZE_N,
            R_TILE=32,
        )

        ctx.save_for_backward(x_tok, lora_a_flat, lora_b_flat, metadata, adapter_indices)
        ctx.in_features = in_features
        ctx.out_features = out_features
        ctx.is_bsd = is_bsd
        if is_bsd:
            ctx.B = B
            ctx.S = S

        if is_bsd:
            return output_tok.view(B, S, out_features)
        else:
            return output_tok

    @staticmethod
    def backward(ctx, grad_output):
        x_tok, lora_a_flat, lora_b_flat, metadata, adapter_indices = ctx.saved_tensors
        in_features = ctx.in_features
        out_features = ctx.out_features
        is_bsd = ctx.is_bsd

        grad_output = grad_output.contiguous()
        if is_bsd:
            B, S, _ = grad_output.shape
            T = B * S
            grad_out_tok = grad_output.view(T, out_features)
        else:
            grad_out_tok = grad_output
            T = grad_out_tok.shape[0]

        max_rank = int(metadata[:, 2].max().item())
        acc_h_scratch = torch.zeros((T, max_rank),
                                    device=x_tok.device,
                                    dtype=torch.float32)

        grad_x_tok = torch.zeros((T, in_features), device=x_tok.device, dtype=x_tok.dtype)
        grad_lora_a_flat = torch.zeros_like(lora_a_flat)
        grad_lora_b_flat = torch.zeros_like(lora_b_flat)

        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 64
        R_TILE = 32

        N_TILES = triton.cdiv(out_features, BLOCK_SIZE_N)
        grid1 = (T, N_TILES)

        _batched_lora_backwardA[grid1](
            x_tok, grad_out_tok,
            lora_a_flat, lora_b_flat,
            metadata, adapter_indices,
            acc_h_scratch, grad_lora_b_flat,
            in_features, out_features,
            x_tok.stride(0), x_tok.stride(1),
            grad_out_tok.stride(0), grad_out_tok.stride(1),
            metadata.stride(0), metadata.stride(1),
            acc_h_scratch.stride(0), acc_h_scratch.stride(1),
            BLOCK_SIZE_K=BLOCK_SIZE_K, BLOCK_SIZE_N=BLOCK_SIZE_N, R_TILE=R_TILE,
        )

        K_TILES = triton.cdiv(in_features, BLOCK_SIZE_K)
        grid2 = (T, K_TILES)

        _batched_lora_backwardB[grid2](
            x_tok, acc_h_scratch,
            lora_a_flat, metadata, adapter_indices,
            grad_x_tok, grad_lora_a_flat,
            in_features, max_rank,
            x_tok.stride(0), x_tok.stride(1),
            acc_h_scratch.stride(0), acc_h_scratch.stride(1),
            metadata.stride(0), metadata.stride(1),
            grad_x_tok.stride(0), grad_x_tok.stride(1),
            BLOCK_SIZE_K=BLOCK_SIZE_K, R_TILE=R_TILE,
        )

        if is_bsd:
            grad_x = grad_x_tok.view(B, S, in_features)
        else:
            grad_x = grad_x_tok

        return grad_x, grad_lora_a_flat, grad_lora_b_flat, None, None, None

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