import torch
from torch import nn
import torch.nn.functional as F
from .linear import ColumnParallelLinear, RowParallelLinear
from .lora import LoraAdapter
from typing import Tuple, Optional, Dict

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

    @property
    def linear_dict(self) -> Dict[str, nn.Module]:
        return {
            f"layers.{self.layer_id_}.self_attn.q_proj": self.wq_,
            f"layers.{self.layer_id_}.self_attn.k_proj": self.wk_,
            f"layers.{self.layer_id_}.self_attn.v_proj": self.wv_,
            f"layers.{self.layer_id_}.self_attn.o_proj": self.wo_,
        }

    def load_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float):
        self.wq_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)
        self.wk_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)
        self.wv_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)
        self.wo_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)

    def unload_adapter(self, adapter_name: str):
        self.wq_.unload_adapter(adapter_name)
        self.wk_.unload_adapter(adapter_name)
        self.wv_.unload_adapter(adapter_name)
        self.wo_.unload_adapter(adapter_name)