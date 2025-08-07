import torch
import torch.nn.functional as F
from torch import nn
import torch.distributed as dist
from typing import Dict, Optional, Tuple, List
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import math
import os

def init_tensor_parallel() -> Tuple[int, int]:
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    torch.cuda.set_device(local_rank)
    
    if rank == 0:
        print(f"Initialized tensor parallelism with world size: {world_size}")
    
    return rank, world_size

class ColumnParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features_per_partition = out_features // world_size
        
        self.weight = nn.Parameter(torch.Tensor(self.out_features_per_partition, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.out_features_per_partition))
        else:
            self.register_parameter('bias', None)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

class RowParallelLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, world_size: int, bias: bool = False):
        super().__init__()
        self.in_features_per_partition = in_features // world_size
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(out_features, self.in_features_per_partition))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        partial_output = F.linear(x, self.weight, self.bias)
        dist.all_reduce(partial_output)
        return partial_output

class ParallelEmbedding(nn.Module):
    def __init__(self, config, world_size: int):
        super().__init__()
        vocab_size_per_partition = config.vocab_size // world_size
        self.embedding = nn.Embedding(
            vocab_size_per_partition,
            config.hidden_size,
            padding_idx=config.pad_token_id
        )
        self.vocab_start_index = dist.get_rank() * vocab_size_per_partition
        self.vocab_end_index = self.vocab_start_index + vocab_size_per_partition

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = (tokens < self.vocab_start_index) | (tokens >= self.vocab_end_index)
        masked_tokens = tokens.clone() - self.vocab_start_index
        masked_tokens[mask] = self.embedding.padding_idx if self.embedding.padding_idx is not None else 0

        partial_embeddings = self.embedding(masked_tokens)

        partial_embeddings[mask.unsqueeze(-1).expand_as(partial_embeddings)] = 0.0

        dist.all_reduce(partial_embeddings)
        return partial_embeddings

def precompute_rope_angle(dim: int, seq_len: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    angles = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    seq = torch.arange(seq_len, dtype=angles.dtype)
    emb = torch.outer(seq, angles)
    emb = torch.cat((emb, emb), dim=-1)
    return (emb.cos(), emb.sin())

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos = cos[position_ids].unsqueeze(1)
    sin = sin[position_ids].unsqueeze(1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class Attention(nn.Module):
    def __init__(self, config, layer_id: int, rank: int, world_size: int):
        super().__init__()
        self.layer_id_ = layer_id
        self.n_heads_ = config.num_attention_heads
        self.n_kv_heads_ = config.num_key_value_heads
        self.head_dim_ = config.hidden_size // self.n_heads_

        self.n_heads_per_partition_ = self.n_heads_ // world_size
        self.n_kv_heads_per_partition_ = self.n_kv_heads_ // world_size
        
        self.n_rep_ = self.n_heads_ // self.n_kv_heads_

        self.wq_ = ColumnParallelLinear(config.hidden_size, self.n_heads_ * self.head_dim_, world_size, bias=False)
        self.wk_ = ColumnParallelLinear(config.hidden_size, self.n_kv_heads_ * self.head_dim_, world_size, bias=False)
        self.wv_ = ColumnParallelLinear(config.hidden_size, self.n_kv_heads_ * self.head_dim_, world_size, bias=False)
        self.wo_ = RowParallelLinear(self.n_heads_ * self.head_dim_, config.hidden_size, world_size, bias=False)
        
        cos, sin = precompute_rope_angle(self.head_dim_, config.max_position_embeddings, config.rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        data: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch, seq_len, dim = data.shape
        
        xq = self.wq_(data).view(batch, seq_len, self.n_heads_per_partition_, self.head_dim_).transpose(1, 2)
        xk = self.wk_(data).view(batch, seq_len, self.n_kv_heads_per_partition_, self.head_dim_).transpose(1, 2)
        xv = self.wv_(data).view(batch, seq_len, self.n_kv_heads_per_partition_, self.head_dim_).transpose(1, 2)
        
        xq, xk = apply_rotary_pos_emb(xq, xk, self.cos_cached, self.sin_cached, position_ids)
        
        if past_key_value is not None:
            past_key, past_value = past_key_value
            xk = torch.cat([past_key, xk], dim=2)
            xv = torch.cat([past_value, xv], dim=2)

        present_key_value = (xk, xv)

        xk_repeated = xk.repeat_interleave(self.n_rep_, dim=1)
        xv_repeated = xv.repeat_interleave(self.n_rep_, dim=1)
        
        output = F.scaled_dot_product_attention(xq, xk_repeated, xv_repeated, attn_mask=mask)
        
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.wo_(output), present_key_value

class MLP(nn.Module):
    def __init__(self, config, layer_id: int, rank: int, world_size: int):
        super().__init__()
        self.layer_id_ = layer_id
        self.gate_ = ColumnParallelLinear(config.hidden_size, config.intermediate_size, world_size, bias=False)
        self.down_ = RowParallelLinear(config.intermediate_size, config.hidden_size, world_size, bias=False)
        self.up_ = ColumnParallelLinear(config.hidden_size, config.intermediate_size, world_size, bias=False)

    def forward(self, data: torch.Tensor):
        w1 = self.gate_(data)
        w3 = self.up_(data)
        mlp_output = F.silu(w1) * w3
        return self.down_(mlp_output)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.norm_eps_ = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        input_dtype = data.dtype
        v = data.to(torch.float32).pow(2).mean(-1, keepdim=True)
        rv = data * torch.rsqrt(v + self.norm_eps_)
        return (self.weight * rv).to(input_dtype)

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, config, rank: int, world_size: int):
        super().__init__()
        self.attention_ = Attention(config, layer_id, rank, world_size)
        self.feed_forward_ = MLP(config, layer_id, rank, world_size)
        self.input_layernorm_ = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_ = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layer_id_ = layer_id

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        
        attention_output, present_key_value = self.attention_.forward(
            self.input_layernorm_(x), position_ids=position_ids, mask=mask, past_key_value=past_key_value
        )
        h = x + attention_output
        out = h + self.feed_forward_.forward(self.post_attention_layernorm_(h))
        return out, present_key_value

class OutputLayer(nn.Module):
    def __init__(self, config, world_size: int):
        super().__init__()
        self.lm_head_ = ColumnParallelLinear(config.hidden_size, config.vocab_size, world_size, bias=False)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        return self.lm_head_(data).float()

class LlamaModel(nn.Module):
    def __init__(self, config, rank: int, world_size: int):
        super().__init__()
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.tok_embeddings_ = ParallelEmbedding(config, world_size)
        self.layers_ = nn.ModuleList([TransformerBlock(i, config, rank, world_size) for i in range(config.num_hidden_layers)])
        self.norm_ = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.output_ = OutputLayer(config, world_size)

    def forward(
        self,
        tokens: torch.Tensor,
        position_ids: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        h = self.tok_embeddings_(tokens)
        
        present_key_values = []
        for i, layer in enumerate(self.layers_):
            past_kv = past_key_values[i] if past_key_values is not None else None
            h, present_kv = layer(h, position_ids, mask, past_kv)
            present_key_values.append(present_kv)

        h = self.norm_(h)
        output = self.output_(h)
        return output, present_key_values

    def from_pretrained(self, hf_state_dict: Dict[str, torch.Tensor]):
        if self.rank == 0:
            print("Loading sharded weights into custom LlamaModel...")

        vocab_size = self.config.vocab_size
        partition_size = vocab_size // self.world_size
        start = self.rank * partition_size
        end = (self.rank + 1) * partition_size
        
        self.tok_embeddings_.embedding.weight.data.copy_(hf_state_dict['model.embed_tokens.weight'][start:end, :])
        self.output_.lm_head_.weight.data.copy_(hf_state_dict['lm_head.weight'][start:end, :])
        self.norm_.weight.data.copy_(hf_state_dict['model.norm.weight'])

        for i, layer in enumerate(self.layers_):
            layer_prefix = f"model.layers.{i}."

            col_parallel_map = {
                'self_attn.q_proj': layer.attention_.wq_,
                'self_attn.k_proj': layer.attention_.wk_,
                'self_attn.v_proj': layer.attention_.wv_,
                'mlp.gate_proj': layer.feed_forward_.gate_,
                'mlp.up_proj': layer.feed_forward_.up_
            }
            for name, module in col_parallel_map.items():
                full_weight = hf_state_dict[layer_prefix + name + '.weight']
                partition_size = full_weight.shape[0] // self.world_size
                start = self.rank * partition_size
                end = (self.rank + 1) * partition_size
                module.weight.data.copy_(full_weight[start:end, :])

            # Row Parallel Layers
            row_parallel_map = {
                'self_attn.o_proj': layer.attention_.wo_,
                'mlp.down_proj': layer.feed_forward_.down_
            }
            for name, module in row_parallel_map.items():
                full_weight = hf_state_dict[layer_prefix + name + '.weight']
                partition_size = full_weight.shape[1] // self.world_size
                start = self.rank * partition_size
                end = (self.rank + 1) * partition_size
                module.weight.data.copy_(full_weight[:, start:end])

            layer.input_layernorm_.weight.data.copy_(hf_state_dict[layer_prefix + 'input_layernorm.weight'])
            layer.post_attention_layernorm_.weight.data.copy_(hf_state_dict[layer_prefix + 'post_attention_layernorm.weight'])

        if self.rank == 0:
            print("Weight loading complete.")


def generate(model: LlamaModel, tokenizer: AutoTokenizer, prompt: str, max_new_tokens: int = 50):
    device = next(model.parameters()).device
    rank = dist.get_rank()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_ids = inputs["input_ids"]
    
    prompt_len = prompt_ids.shape[1]
    position_ids = torch.arange(0, prompt_len, dtype=torch.long, device=device).unsqueeze(0)
    
    attention_mask = torch.full((1, 1, prompt_len, prompt_len), -1e9, device=device, dtype=torch.float32)
    attention_mask = torch.triu(attention_mask, diagonal=1)

    with torch.no_grad():
        logits, past_key_values = model(prompt_ids, position_ids, attention_mask, past_key_values=None)

    vocab_size = model.config.vocab_size
    partition_size = vocab_size // model.world_size

    gathered_logits = [torch.zeros_like(logits) for _ in range(model.world_size)]
    dist.all_gather(gathered_logits, logits)
    
    if rank == 0:
        full_logits = torch.cat(gathered_logits, dim=-1)
        next_token_id = torch.argmax(full_logits[:, -1, :], dim=-1).unsqueeze(-1)
    else:
        next_token_id = torch.zeros((1,1), dtype=torch.long, device=device)

    dist.broadcast(next_token_id, src=0)
    
    all_ids = torch.cat([prompt_ids, next_token_id], dim=1)

    for _ in range(max_new_tokens - 1):
        current_input_ids = next_token_id
        current_position = all_ids.shape[1] - 1
        position_ids = torch.tensor([[current_position]], device=device, dtype=torch.long)
        
        with torch.no_grad():
            logits, past_key_values = model(current_input_ids, position_ids, mask=None, past_key_values=past_key_values)

        gathered_logits = [torch.zeros_like(logits) for _ in range(model.world_size)]
        dist.all_gather(gathered_logits, logits)
        
        if rank == 0:
            full_logits = torch.cat(gathered_logits, dim=-1)
            next_token_id = torch.argmax(full_logits[:, -1, :], dim=-1).unsqueeze(-1)
        else:
            next_token_id = torch.zeros((1,1), dtype=torch.long, device=device)
        
        dist.broadcast(next_token_id, src=0)
        all_ids = torch.cat([all_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    if rank == 0:
        return tokenizer.decode(all_ids[0], skip_special_tokens=True)
    return None


def main():
    rank, world_size = init_tensor_parallel()
    device = f"cuda:{rank}"
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    if rank == 0:
        print(f"Using device: {device}")

    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            config.pad_token_id = config.eos_token_id
    except Exception as e:
        if rank == 0:
            print(f"Could not download config/tokenizer. You may need to log in via `huggingface-cli login`")
            print(f"Error: {e}")
        return

    custom_model = LlamaModel(config, rank, world_size).to(device)
    custom_model.eval()

    if rank == 0:
        print("\nLoading pre-trained model from Hugging Face to extract weights...")
    try:
        hf_state_dict = None
        if rank == 0:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
            hf_state_dict = hf_model.state_dict()
            print("Hugging Face model loaded.")

        state_dict_list = [hf_state_dict] if rank == 0 else [None]
        dist.broadcast_object_list(state_dict_list, src=0)
        hf_state_dict = state_dict_list[0]

    except Exception as e:
        if rank == 0:
            print(f"Could not download/broadcast model weights. Error: {e}")
        return

    custom_model.from_pretrained(hf_state_dict)

    if world_size > 1:
        dist.barrier()

    if rank == 0:
        print("\n--- Running Model Inference ---")
    prompt = "The capital of France is"
    
    full_response = generate(custom_model, tokenizer, prompt, max_new_tokens=30)
    
    if full_response:
        print(f"Prompt: '{prompt}'")
        print(f"Full response: '{full_response}'")

    if world_size > 1:
        dist.barrier()


if __name__ == "__main__":
    main()
