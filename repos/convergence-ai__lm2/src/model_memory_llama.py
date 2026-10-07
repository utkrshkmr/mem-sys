from __future__ import annotations

import os
import random
import re
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
)
from transformers.cache_utils import DynamicCache
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRMSNorm

from src.memory import MemoryModule
from src.utils import CfgNode, print0


class MemoryAttention(nn.Module):
    """Custom attention layer that incorporates a memory module."""

    def __init__(self, config: LlamaConfig, self_attn: nn.Module) -> None:
        super().__init__()
        self.self_attn = self_attn
        self.log_freq = config.log_freq

        head_size = config.memory_slots // config.num_mem_heads
        self.memory_module = MemoryModule(
            mem_slots=config.memory_slots,
            head_size=head_size,
            hidden_dim=config.hidden_size,
            num_heads=config.num_mem_heads,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        position_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Apply normal self-attention on hidden states
        attn_output, _, _ = self.self_attn(hidden_states, attention_mask, position_ids=position_ids, position_embeddings=position_embeddings)

        # Memory module processing
        gated_memory, updated_memory = self.memory_module(attn_output, memory, attention_mask)

        return gated_memory, updated_memory


class CustomLlamaDecoderLayer(LlamaDecoderLayer):
    """Extends the Llama decoder layer with memory attention."""

    def __init__(self, config: LlamaConfig, layer_idx: int, use_memory: bool) -> None:
        super().__init__(config, layer_idx)
        self.use_memory = config.use_memory
        if self.use_memory:
            self.mem_attn = MemoryAttention(config, self.self_attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor = None,
        memory: torch.Tensor = None,
        position_embeddings: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        if self.use_memory and memory is not None:
            attn_output, memory = self.mem_attn(
                hidden_states=hidden_states, 
                memory=memory, 
                attention_mask=attention_mask, 
                position_ids=position_ids, 
                position_embeddings=position_embeddings,
            )
        else:
            attn_output, _, _ = self.self_attn(
                hidden_states,
                attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
            )

        hidden_states = residual + attn_output
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, memory


class LlamaMem(LlamaForCausalLM):
    """Custom Llama model with memory-augmented decoder layers."""

    @staticmethod
    def get_default_config():
        C = CfgNode()
        C.model_type = "llama"
        C.n_layer = None
        C.n_head = None
        C.n_embd = None
        C.vocab_size = None
        C.block_size = None
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.memory_slots = 16
        return C

    def __init__(self, config: LlamaConfig, tokenizer: AutoTokenizer) -> None:
        super().__init__(config)
        self.use_memory = config.use_memory
        self.tokenizer = tokenizer

        if self.use_memory:
            if config.memory_slots % config.num_mem_heads != 0:
                raise ValueError("Memory slots must be divisible by number of memory heads")
            # Initialize memory but don't move to CUDA yet
            self.memory = torch.stack(
                [torch.eye(config.memory_slots, requires_grad=False) for _ in range(config.batch_size)]
            )
            # Register as buffer for proper state dict handling
            self.register_buffer("memory_bank", self.memory)
            print0("=========> Added memory module")
        else:
            print0("=========> No memory module")
            self.memory = None

        # Replace default decoder layers with memory-augmented layers
        self.model.layers = nn.ModuleList(
            [CustomLlamaDecoderLayer(config, layer_idx=i, use_memory=True) for i in range(len(self.model.layers))]
        )

    @classmethod
    def from_config(
        cls,
        config: LlamaConfig,
        tokenizer: AutoTokenizer,
    ) -> nn.Module:
        """Load pre-trained model weights."""
        custom_model = cls(config, tokenizer)

        return custom_model

    @classmethod
    def from_ckpt(cls, pretrained_ckpt_path: str, config: LlamaConfig, 
                tokenizer: AutoTokenizer, rank: int, 
                load_memory: bool=True, resume_training: bool=False) -> nn.Module:
        """Load model from a checkpoint"""
        try:
            # Load the entire checkpoint on CPU first
            snapshot_data = torch.load(pretrained_ckpt_path, map_location='cpu')

            custom_model = cls(config, tokenizer)

            # Handle memory loading control
            model_state = snapshot_data['model_state_dict']

            # Load model state
            custom_model.load_state_dict(model_state, strict=False)
            missing_keys, unexpected_keys = custom_model.load_state_dict(model_state, strict=False)
            if missing_keys or unexpected_keys:
                print0(f"Missing keys: {missing_keys}")
                print0(f"Unexpected keys: {unexpected_keys}")

            # Move model to the desired GPU device
            custom_model = custom_model.to(rank)
            if load_memory:
                custom_model.memory = custom_model.memory_bank.detach().clone()
                print0("===>Loaded pre-trained memory")
            else:
                custom_model.memory = torch.stack(
                    [torch.eye(config.memory_slots, requires_grad=False) for _ in range(config.batch_size)]
                )
                custom_model.register_buffer("memory_bank", custom_model.memory)
                print0("===>Initialized fresh memory")

            # Restore random states
            try:
                random.setstate(snapshot_data['random_state'])

                # Restore torch RNG state
                torch_state = snapshot_data['torch_random_state']
                if isinstance(torch_state, torch.Tensor):
                    torch_state = torch_state.cpu().to(torch.uint8)
                else:
                    torch_state = torch.ByteTensor(torch_state)
                torch.set_rng_state(torch_state)

                # Restore CUDA RNG states
                cuda_states = snapshot_data['cuda_random_state']
                for i, state in enumerate(cuda_states):
                    if isinstance(state, torch.Tensor):
                        state = state.cpu().to(torch.uint8)
                    else:
                        state = torch.ByteTensor(state)
                    torch.cuda.set_rng_state(state, device=i)
                        
            except Exception as e:
                print0(f"Warning: Failed to restore random states: {str(e)}")
                print0("Continuing without restoring random states...")

            print0(f"Loaded checkpoint from iteration {snapshot_data['iteration']}")
            print0(f"Checkpoint learning rate: {snapshot_data['lr']}")

            if resume_training:
                return (custom_model, 
                        snapshot_data['optimizer_state_dict'],
                        snapshot_data['iteration'])

            return custom_model

        except Exception as e:
            print0(f"Error loading checkpoint: {e}")
            raise

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        cache_position: torch.Tensor = None,
        iter_num: int = 0,
        num_logits_to_keep: int = 0,
    ) -> tuple[float, torch.Tensor]:
        """Forward pass through the memory-augmented Llama model."""
        device = input_ids.device
        b, t = input_ids.size()

        # Ensure memory is on correct device
        if self.use_memory:
            if self.memory is None or self.memory.device != device:
                self.memory = self.memory.to(device)
            memory = self.memory.detach()
        else:
            memory = None

        # Generate position IDs if not provided
        if position_ids is None:
            position_ids = torch.arange(t, device=device).unsqueeze(0).repeat(b, 1)

        inputs_embeds = self.model.embed_tokens(input_ids)

        # Initialize caching mechanism
        past_key_values = DynamicCache()
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        causal_mask = self.model._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            False,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.model.rotary_emb(hidden_states, position_ids)

        for decoder_layer in self.model.layers:
            hidden_states, memory = decoder_layer(
                hidden_states=hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                memory=memory,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.model.norm(hidden_states)
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :]).float()

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=self.tokenizer.pad_token_id,
            )
        
        self.memory = memory
        if self.use_memory:
            # self.memory_bank.copy_(self.memory)
            self.memory_bank = self.memory.clone()

        return logits, loss, memory

    def configure_optimizers(self, train_config):
        """Configure optimizers with proper weight decay handling."""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (
            torch.nn.LayerNorm,
            torch.nn.Embedding,
            LlamaRMSNorm,
        )
        pattern1 = re.compile(r"^transformer\.h\.[0-9]+\.mem_attn\.memory_module\.input_gate_projector\.w$")
        pattern2 = re.compile(r"^model\.layers\.\d+\.mem_attn\.memory_module\.input_gate_projector\.w$")

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
                if "lm_head" in fpn:
                    continue
                elif pn.endswith("bias"):
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif pattern1.match(fpn) or pattern2.match(fpn):
                    no_decay.add(fpn)

        # Validate parameter separation
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"Parameters {str(inter_params)} in both decay/no_decay sets!"
        assert (
            len(param_dict.keys() - union_params) == 0
        ), f"Parameters {str(param_dict.keys() - union_params)} not in either set!"

        # Create optimizer groups
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": train_config.weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


class CustomLlamaConfig(LlamaConfig):
    """Custom LlamaConfig with memory-related parameters."""

    def __init__(self, **kwargs: dict[str, Any]) -> None:
        super().__init__(**kwargs)
        self.use_memory = kwargs.get("use_memory", False)
        self.memory_slots = kwargs.get("memory_slots", 16)
        self.num_mem_heads = kwargs.get("num_mem_heads", 4)
        self.log_freq = kwargs.get("log_freq", 100)
        self.batch_size = kwargs.get("batch_size", 1)
