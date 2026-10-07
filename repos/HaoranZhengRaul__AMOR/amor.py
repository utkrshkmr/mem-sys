"""
AMOR: Adaptive Metacognitive Output Router.

A post-hoc hybrid architecture that gates attention by output prediction entropy.
A recurrent backbone (Mamba-2 or Gated DeltaNet) is followed by K stacked AMOR
blocks; each block measures the entropy of softmax(lm_head(h)) and, where the
backbone is uncertain, fires causal self-attention to refine the residual stream.

    Input -> [Backbone Mixer + SwiGLU MLP] x N -> [AMOR Block] x K -> Norm -> LM Head
                                                       |
                                               Entropy Gate  (fire if uncertain)
                                                       |
                                                   Attention  (Q,K,V from residual)

The 6 architecture variants are flag combinations of a single class:

    backbone        in {'mamba2', 'gdn'}        # recurrent mixer
    n_amor_blocks   in {1, 3}                   # stack depth
    residual_mode   in {'classic', 'h'}         # block-0 pre-norm placement

K, V are standard linear projections of the residual stream (not SSM internal
state). Training uses dense attention with output masking ('full_with_mask');
'true_sparse' is provided for inference. Each block has its own EntropyGate
(EMA-tracked threshold), Q/K/V/O, RoPE on Q/K, and a per-channel alpha fusion.
"""

import math
from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import mamba_ssm
    if not hasattr(mamba_ssm, 'mamba_chunk_scan_combined'):
        from mamba_ssm.ops.triton.ssd_combined import (
            mamba_chunk_scan_combined,
            mamba_split_conv1d_scan_combined,
        )
        mamba_ssm.mamba_chunk_scan_combined = mamba_chunk_scan_combined
        mamba_ssm.mamba_split_conv1d_scan_combined = mamba_split_conv1d_scan_combined
except ImportError:
    pass

try:
    from transformers import Mamba2Config, Mamba2ForCausalLM
except ImportError:
    Mamba2Config = None
    Mamba2ForCausalLM = None

try:
    from fla.layers import GatedDeltaNet as FLAGatedDeltaNet
except ImportError:
    FLAGatedDeltaNet = None


# =============================================================================
# Building blocks
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return self.weight * x / rms


class SwiGLUMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos(), persistent=False)
        self.register_buffer('sin_cached', emb.sin(), persistent=False)

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        seq_len = q.shape[2]
        cos = self.cos_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[offset:offset + seq_len].unsqueeze(0).unsqueeze(0)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(x, cos, sin):
        d = x.shape[-1]
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return x * cos + torch.cat([-x2, x1], dim=-1) * sin


class KVProjection(nn.Module):
    def __init__(self, d_model: int, n_heads: int = 12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.k_proj(h), self.v_proj(h)


# =============================================================================
# Entropy gate
# =============================================================================

class EntropyGate(nn.Module):
    """
    Hard binary gate driven by output-level prediction entropy.

    Threshold is an EMA of the batch entropy median (no gradient). Two offset
    modes: 'fixed' adds a constant; 'adaptive' adds offset_k * EMA(std), giving
    a scale-independent fire rate as the entropy distribution width changes
    across model scale and training stage.

        entropy   = -sum(p * log(p)) / log(vocab_size)
        threshold = EMA(median) + offset                   # fixed
        threshold = EMA(median) + offset_k * EMA(std)      # adaptive
        gate      = (entropy > threshold)                  # hard binary
    """

    def __init__(
        self,
        vocab_size: int,
        init_threshold: float = 0.3,
        ema_momentum: float = 0.99,
        offset: float = 0.03,
        offset_mode: str = 'adaptive',
        offset_k: float = 0.2,
    ):
        super().__init__()
        self.max_entropy = math.log(vocab_size)
        self.ema_momentum = ema_momentum
        self.offset = offset
        self.offset_mode = offset_mode
        self.offset_k = offset_k
        self.register_buffer('running_threshold', torch.tensor(init_threshold))
        self.register_buffer('running_std', torch.tensor(0.1))
        self.router = None

    def forward(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Detached: entropy drives gating decisions only. CE loss trains the
        # backbone via the final logits path, never through the gate.
        if self.router is not None:
            if self.training:
                raise RuntimeError("Fitted routers are for inference; detach them before training.")
            with torch.no_grad():
                entropy = self.router(logits.detach())
        else:
            logits_f32 = logits.detach().float()
            probs = F.softmax(logits_f32, dim=-1)
            raw_entropy = -(probs * (probs + 1e-10).log()).sum(dim=-1)
            entropy = raw_entropy / self.max_entropy

        if self.training:
            entropy_detached = entropy.detach()
            batch_median = entropy_detached.median()
            self.running_threshold.mul_(self.ema_momentum).add_(
                batch_median, alpha=1.0 - self.ema_momentum
            )
            if self.offset_mode == 'adaptive':
                batch_std = entropy_detached.std()
                self.running_std.mul_(self.ema_momentum).add_(
                    batch_std, alpha=1.0 - self.ema_momentum
                )

        if self.offset_mode == 'adaptive':
            threshold = self.running_threshold + self.offset_k * self.running_std
        else:
            threshold = self.running_threshold + self.offset

        gate = (entropy > threshold).float()
        return gate, entropy


# =============================================================================
# AMOR block
# =============================================================================

class AMORBlock(nn.Module):
    """
    Entropy-gated attention block.

        gate     = EntropyGate(lm_head(h_normed))
        attn_out = Attention(h_normed) * gate
        output   = h_normed + alpha * attn_out

    Attention modes:
        'full_with_mask': dense attention over all positions, output masked by
                          the gate. Used for training (GPU-friendly).
        'true_sparse':    Q and the attention computation only run for firing
                          positions. Output is bit-identical given the same
                          weights/inputs. Used for inference.

    Zero-initialized out_proj + per-channel alpha (sigmoid(raw_alpha), clamped
    floor of ~0.27) handle output magnitude scaling. RoPE on Q, K.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_heads: int = 12,
        init_alpha: float = 0.0,
        init_threshold: float = 0.3,
        gate_offset: float = 0.03,
        offset_mode: str = 'adaptive',
        offset_k: float = 0.2,
        attention_mode: str = 'full_with_mask',
        max_seq_len: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_mode = attention_mode

        self.gate = EntropyGate(
            vocab_size,
            init_threshold=init_threshold,
            offset=gate_offset,
            offset_mode=offset_mode,
            offset_k=offset_k,
        )
        self.rope = RotaryPositionEmbedding(self.head_dim, max_seq_len)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_proj = KVProjection(d_model, n_heads)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

        # Per-channel raw alpha; sigmoid + lower clamp at -1 (≈0.27 floor) to
        # prevent collapse. Channels learn their own residual mixing weights.
        self.raw_alpha = nn.Parameter(torch.full((d_model,), float(init_alpha)))

    def get_alpha(self) -> torch.Tensor:
        return torch.sigmoid(torch.clamp(self.raw_alpha, min=-1.0))

    def _attention_full_with_mask(self, h_norm: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = h_norm.shape
        Q = self.q_proj(h_norm).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        K, V = self.kv_proj(h_norm)
        K = K.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        Q, K = self.rope(Q, K)
        attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        attn_out = self.out_proj(attn_out)
        return attn_out * gate.unsqueeze(-1)

    def _attention_true_sparse(self, h_norm: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = h_norm.shape
        device = h_norm.device

        K, V = self.kv_proj(h_norm)
        K = K.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        cos_all = self.rope.cos_cached[:seq].unsqueeze(0).unsqueeze(0)
        sin_all = self.rope.sin_cached[:seq].unsqueeze(0).unsqueeze(0)
        K = RotaryPositionEmbedding._apply_rotary(K, cos_all, sin_all)

        active_mask = gate > 0.5
        if not active_mask.any():
            return torch.zeros(batch, seq, self.d_model, device=device, dtype=h_norm.dtype)

        active_indices = active_mask.nonzero(as_tuple=False)
        num_active = active_indices.shape[0]
        batch_indices = active_indices[:, 0]
        seq_indices = active_indices[:, 1]

        h_active = h_norm[active_mask]
        Q_active = self.q_proj(h_active).view(num_active, self.n_heads, self.head_dim)
        cos_q = self.rope.cos_cached[seq_indices].unsqueeze(1)
        sin_q = self.rope.sin_cached[seq_indices].unsqueeze(1)
        Q_active = RotaryPositionEmbedding._apply_rotary(Q_active, cos_q, sin_q)

        K_selected = K[batch_indices]
        V_selected = V[batch_indices]
        Q_active = Q_active.unsqueeze(2)
        scores = torch.matmul(Q_active, K_selected.transpose(-2, -1)) / math.sqrt(self.head_dim)

        positions = seq_indices.view(num_active, 1, 1, 1)
        key_positions = torch.arange(seq, device=device).view(1, 1, 1, seq)
        scores = scores.masked_fill(key_positions > positions, float('-inf'))
        attn_weights = torch.nan_to_num(F.softmax(scores, dim=-1), 0.0)

        out_active = torch.matmul(attn_weights, V_selected).squeeze(2)
        out_active = out_active.contiguous().view(num_active, self.d_model)
        out_active = self.out_proj(out_active)

        attn_out = torch.zeros(batch, seq, self.d_model, device=device, dtype=h_norm.dtype)
        attn_out[active_mask] = out_active
        return attn_out

    def forward(
        self,
        h: torch.Tensor,
        backbone_logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        gate, entropy = self.gate(backbone_logits)
        if self.attention_mode == 'true_sparse':
            attn_out = self._attention_true_sparse(h, gate)
        else:
            attn_out = self._attention_full_with_mask(h, gate)
        alpha = self.get_alpha()
        h_out = h + alpha * attn_out
        info = {
            'gate': gate,
            'entropy': entropy,
            'alpha': alpha.mean(),
            'alpha_std': alpha.std(),
            'fire_rate': gate.mean(),
        }
        return h_out, info


# =============================================================================
# GDN backbone block (used when backbone='gdn')
# =============================================================================

class GDNBlock(nn.Module):
    """Pre-norm Gated DeltaNet + pre-norm SwiGLU MLP. Requires `fla` library."""

    def __init__(self, d_model: int, d_ff: int, head_dim: int = 64, expand_v: float = 2.0):
        super().__init__()
        if FLAGatedDeltaNet is None:
            raise ImportError(
                "flash-linear-attention is required for backbone='gdn'. "
                "Install with: pip install flash-linear-attention"
            )
        self.d_model = d_model
        n_heads = int(0.75 * d_model / head_dim)
        assert n_heads * head_dim == int(0.75 * d_model), \
            f"0.75 * d_model={d_model} must be divisible by head_dim={head_dim}"

        self.norm_ssm = RMSNorm(d_model)
        self.gdn = FLAGatedDeltaNet(
            hidden_size=d_model,
            num_heads=n_heads,
            head_dim=head_dim,
            expand_v=expand_v,
            mode='chunk',
        )
        self.norm_mlp = RMSNorm(d_model)
        self.mlp = SwiGLUMLP(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_ssm(x)
        h = self.gdn(h)[0]
        x = x + h
        x = x + self.mlp(self.norm_mlp(x))
        return x


# =============================================================================
# Unified AMOR model
# =============================================================================

class AMOR(nn.Module):
    """
    Unified AMOR model. Six published variants are flag combinations:

        backbone='mamba2', n_amor_blocks=3, residual_mode='classic'   # AMOR
        backbone='mamba2', n_amor_blocks=1, residual_mode='classic'   # AMOR-classic
        backbone='mamba2', n_amor_blocks=3, residual_mode='h'         # AMOR-h
        backbone='gdn',    n_amor_blocks=3, residual_mode='classic'   # AMOR-GDN
        backbone='gdn',    n_amor_blocks=1, residual_mode='classic'   # AMOR-classic-GDN
        backbone='gdn',    n_amor_blocks=3, residual_mode='h'         # AMOR-GDN-h

    Architecture:
        Embedding
        -> [Backbone Mixer + SwiGLU MLP] x n_layer
        -> norm_f                         # applied to residual iff residual_mode='classic'
        -> [pre_norm -> AMOR Block -> residual fusion] x n_amor_blocks
        -> final_norm
        -> LM Head (tied to embedding)

    residual_mode='classic':
        norm_f is applied once to the residual stream after the backbone. Block 0
        therefore has no extra pre-norm and its residual wraps the norm_f'd
        stream; blocks 1+ each carry their own pre-norm and their residual wraps
        the un-normed h.

    residual_mode='h':
        norm_f is reused inline as block 0's pre-norm; the residual stream stays
        un-normed from the end of the backbone through every AMOR block. All
        blocks are symmetric: pre-norm -> attention -> residual wraps un-normed h.
        Only final_norm sees the residual, immediately before lm_head.

    Default dimensions match a ~180M-param scale (d_model=768, n_layer=12,
    d_ff=1216, head_dim=64).
    """

    def __init__(
        self,
        # Architecture flags
        backbone: str = 'mamba2',
        n_amor_blocks: int = 3,
        residual_mode: str = 'classic',
        # Common dims
        vocab_size: int = 128256,
        d_model: int = 768,
        n_layer: int = 12,
        d_ff: int = 1216,
        head_dim: int = 64,
        # Mamba-2 backbone params
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        n_groups: int = 1,
        # GDN backbone params
        expand_v: float = 2.0,
        # AMOR block params
        n_heads: int = 12,
        init_alpha: float = 0.0,
        init_threshold: float = 0.3,
        gate_offset: float = 0.03,
        offset_mode: str = 'adaptive',
        offset_k: float = 0.2,
        attention_mode: str = 'full_with_mask',
        max_seq_len: int = 4096,
    ):
        super().__init__()
        if backbone not in ('mamba2', 'gdn'):
            raise ValueError(f"backbone must be 'mamba2' or 'gdn', got {backbone!r}")
        if residual_mode not in ('classic', 'h'):
            raise ValueError(f"residual_mode must be 'classic' or 'h', got {residual_mode!r}")
        if n_amor_blocks < 1:
            raise ValueError(f"n_amor_blocks must be >= 1, got {n_amor_blocks}")

        self.backbone = backbone
        self.residual_mode = residual_mode
        self.n_amor_blocks = n_amor_blocks
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_ff = d_ff

        self._build_backbone(
            d_model=d_model, n_layer=n_layer, d_ff=d_ff, vocab_size=vocab_size,
            head_dim=head_dim, d_state=d_state, d_conv=d_conv, expand=expand,
            n_groups=n_groups, expand_v=expand_v,
        )

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight

        self.amor_blocks = nn.ModuleList()
        self.amor_pre_norms = nn.ModuleList()
        for i in range(n_amor_blocks):
            self.amor_blocks.append(AMORBlock(
                d_model=d_model,
                vocab_size=vocab_size,
                n_heads=n_heads,
                init_alpha=init_alpha,
                init_threshold=init_threshold,
                gate_offset=gate_offset,
                offset_mode=offset_mode,
                offset_k=offset_k,
                attention_mode=attention_mode,
                max_seq_len=max_seq_len,
            ))
            # Block 0's pre-norm slot is None for both modes:
            #   'classic' -> input is already norm_f'd, no pre-norm needed.
            #   'h'       -> norm_f is reused inline as block 0's pre-norm.
            if i == 0:
                self.amor_pre_norms.append(None)
            else:
                self.amor_pre_norms.append(RMSNorm(d_model))

        self.final_norm = RMSNorm(d_model)
        self._init_weights()

    def _build_backbone(
        self, d_model, n_layer, d_ff, vocab_size, head_dim,
        d_state, d_conv, expand, n_groups, expand_v,
    ):
        if self.backbone == 'mamba2':
            if Mamba2ForCausalLM is None:
                raise ImportError(
                    "transformers (with Mamba2 support) is required for backbone='mamba2'. "
                    "Install with: pip install 'transformers>=4.43' mamba-ssm causal-conv1d"
                )
            ssm_n_heads = (d_model * expand) // head_dim
            self.mamba_config = Mamba2Config(
                vocab_size=vocab_size,
                hidden_size=d_model,
                num_hidden_layers=n_layer,
                state_size=d_state,
                conv_kernel=d_conv,
                expand=expand,
                head_dim=head_dim,
                num_heads=ssm_n_heads,
                n_groups=n_groups,
            )
            mamba_model = Mamba2ForCausalLM(self.mamba_config)
            self.embed = mamba_model.backbone.embeddings
            self.ssd_layers = mamba_model.backbone.layers
            self.norm_f = mamba_model.backbone.norm_f
            self.mlp_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layer)])
            self.mlps = nn.ModuleList([SwiGLUMLP(d_model, d_ff) for _ in range(n_layer)])
            del mamba_model
        else:
            self.embed = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                GDNBlock(d_model=d_model, d_ff=d_ff, head_dim=head_dim, expand_v=expand_v)
                for _ in range(n_layer)
            ])
            self.norm_f = RMSNorm(d_model)

    def _init_weights(self):
        std = 0.02
        nn.init.normal_(self.embed.weight, mean=0.0, std=std)
        if self.backbone == 'mamba2':
            for layer in self.ssd_layers:
                for m in layer.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=std)
            for mlp in self.mlps:
                for m in mlp.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=std)
        else:
            for layer in self.layers:
                for m in layer.gdn.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=std)
                for m in layer.mlp.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.normal_(m.weight, mean=0.0, std=std)
        for amor_block in self.amor_blocks:
            nn.init.normal_(amor_block.q_proj.weight, mean=0.0, std=std)
            nn.init.normal_(amor_block.kv_proj.k_proj.weight, mean=0.0, std=std)
            nn.init.normal_(amor_block.kv_proj.v_proj.weight, mean=0.0, std=std)
            # out_proj stays zero-initialized.

    def _run_backbone(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        if self.backbone == 'mamba2':
            for ssd_layer, mlp_norm, mlp in zip(self.ssd_layers, self.mlp_norms, self.mlps):
                h = ssd_layer(h)
                h = h + mlp(mlp_norm(h))
        else:
            for layer in self.layers:
                h = layer(h)
        return h

    def forward(
        self,
        input_ids: torch.Tensor,
        return_details: bool = False,
    ):
        h = self._run_backbone(input_ids)

        if self.residual_mode == 'classic':
            # norm_f flows into the residual; block 0 reads the norm_f'd stream
            # directly, blocks 1+ apply their own pre-norm.
            h = self.norm_f(h)

        all_info = {}
        for i, (amor_block, pre_norm) in enumerate(zip(self.amor_blocks, self.amor_pre_norms)):
            if pre_norm is not None:
                h_normed = pre_norm(h)
            elif self.residual_mode == 'h':
                # Block 0 in 'h' mode: norm_f is the inline pre-norm; residual
                # stays un-normed.
                h_normed = self.norm_f(h)
            else:
                # Block 0 in 'classic' mode: input is already norm_f'd.
                h_normed = h

            logits_for_gate = (h_normed if amor_block.gate.router is not None
                               else self.lm_head(h_normed))
            h_out, info = amor_block(h_normed, logits_for_gate)
            del logits_for_gate

            if self.residual_mode == 'h':
                # All blocks symmetric: residual wraps un-normed h.
                h = h + (h_out - h_normed)
            else:
                if pre_norm is not None:
                    # Residual wraps un-normed h (h was not pre-normed for this block).
                    h = h + (h_out - h_normed)
                else:
                    # Block 0: residual wraps the norm_f'd stream (h == h_normed here).
                    h = h_out

            if return_details:
                all_info[f'block_{i}'] = info

        h = self.final_norm(h)
        logits = self.lm_head(h)

        if return_details:
            fire_rates = [v['fire_rate'] for v in all_info.values() if 'fire_rate' in v]
            alphas = [v['alpha'] for v in all_info.values() if 'alpha' in v]
            details = {
                'per_block': all_info,
                'fire_rate': sum(fire_rates) / len(fire_rates) if fire_rates else 0.0,
                'alpha': sum(alphas) / len(alphas) if alphas else 0.0,
            }
            return logits, details
        return logits
