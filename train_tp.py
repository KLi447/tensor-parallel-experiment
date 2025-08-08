import torch
from torch import nn
import torch.distributed as dist
from typing import Dict, Optional, Tuple, List
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import math
import os

from modules.attention import Attention
from modules.embedding import ParallelEmbedding
from modules.linear import ColumnParallelLinear, RowParallelLinear
from modules.lora import LoraAdapter
from modules.output_layer import OutputLayer
from modules.rms_norm import RMSNorm
from modules.mlp import MLP

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

    def load_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float):
        self.attention_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)
        self.feed_forward_.load_adapter(adapter_name, r, lora_alpha, lora_dropout)

    def unload_adapter(self, adapter_name: str):
        self.attention_.unload_adapter(adapter_name)
        self.feed_forward_.unload_adapter(adapter_name)

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

    def load_adapter(self, adapter_name: str, r: int, lora_alpha: int, lora_dropout: float):
        if self.rank == 0:
            print(f"Loading LoRA adapter '{adapter_name}' with r={r} into each layer.")
        for layer in self.layers_:
            layer.load_adapter(adapter_name, r, lora_alpha, lora_dropout)

    def unload_adapter(self, adapter_name: str):
        if self.rank == 0:
            print(f"Unloading LoRA adapter '{adapter_name}' from each layer.")
        for layer in self.layers_:
            layer.unload_adapter(adapter_name)

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
    try:
        rank, world_size = init_tensor_parallel()
        device = f"cuda:{rank}"
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
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
    
        for name, param in custom_model.named_parameters():
            param.requires_grad = False
    
        if world_size > 1:
            dist.barrier()
    
        custom_model.eval()
    
        if rank == 0:
            print("\n--- Running Inital Model Inference ---")
        prompt = "What is a doonkelberry?"
        
        full_response = generate(custom_model, tokenizer, prompt, max_new_tokens=30)
        
        if full_response:
            print(f"Prompt: '{prompt}'")
            print(f"Full response: '{full_response}'")
    
        if world_size > 1:
            dist.barrier()
    
        adapter_name = "test_adapter"
        custom_model.load_adapter(
                adapter_name=adapter_name,
                r=8,
                lora_alpha=16,
                lora_dropout=0.05
        )
    
        lora_params = [p for n, p in custom_model.named_parameters() if 'lora' in n and p.requires_grad]
        optimizer = torch.optim.AdamW(lora_params, lr=5e-4)
        loss_fn = nn.CrossEntropyLoss()
    
        prompt_text = "What is a dunkleberry?"
        answer_text = " A doonkelberry is a purple berry that is Doof's favorite snack. "
    
        input_text = prompt_text + answer_text
            
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        prompt_ids = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        
        labels = input_ids.clone()
        labels[:, :prompt_ids.shape[1]] = -100
    
        if rank == 0:
            print("\n--- Starting Fine-Tuning ---")
    
        custom_model.train()
        for epoch in range(1000):
            optimizer.zero_grad()
            
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
            logits, _ = custom_model(input_ids, position_ids)
    
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
    
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            if rank == 0 and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item()}")
    
            loss.backward()
            optimizer.step()
    
        if world_size > 1:
            dist.barrier()
            
        custom_model.eval()
    
        if rank == 0:
            print("\n--- Running Final Model Inference ---")
        prompt = "What is a doonkelberry?"
        
        full_response = generate(custom_model, tokenizer, prompt, max_new_tokens=30)
        
        if full_response:
            print(f"Prompt: '{prompt}'")
            print(f"Full response: '{full_response}'")
    
        if world_size > 1:
            dist.barrier()
    
        custom_model.unload_adapter(adapter_name)
    
        if rank == 0:
            print("\n--- Running Final Model Inference without adapter ---")
        prompt = "What is a doonkelberry?"
        
        full_response = generate(custom_model, tokenizer, prompt, max_new_tokens=30)
        
        if full_response:
            print(f"Prompt: '{prompt}'")
            print(f"Full response: '{full_response}'")
    
        if world_size > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
            print("Distributed process group destroyed.")

if __name__ == "__main__":
    main()
