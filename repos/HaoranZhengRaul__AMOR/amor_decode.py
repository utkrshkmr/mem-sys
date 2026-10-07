"""
Decode wrapper for AMOR (autoregressive generation with caching).

Same flag dispatch as the model:
    backbone='mamba2'  -> Mamba2Cache (recurrent state per layer, O(1) memory)
    backbone='gdn'     -> fla Cache    (recurrent state + conv state, O(1))
    residual_mode      -> determines block 0's pre-norm placement and residual

Per-AMOR-block KV caches grow O(N) (one cache per AMOR block, not per backbone
layer). At each decode step the entropy gate decides whether to compute Q and
attention; K and V are projected and appended unconditionally so that the cache
stays causal regardless of past gate decisions.
"""

import torch
import torch.nn.functional as F

from amor import AMOR, RotaryPositionEmbedding

try:
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Cache
except ImportError:
    Mamba2Cache = None

try:
    from fla.models.utils import Cache as FLACache
except ImportError:
    try:
        from fla.models.utils import LegacyFLACache as FLACache
    except ImportError:
        FLACache = None


GATE_CHUNK = 4096


def _chunked_gate(lm_head, h, gate_fn):
    """Equivalent to gate_fn(lm_head(h).detach()) but in chunks of GATE_CHUNK
    along the sequence axis, to avoid materializing [B, L, V] at long contexts.
    Bit-identical in eval mode (per-position entropy, frozen threshold)."""
    if h.shape[1] <= GATE_CHUNK:
        return gate_fn(lm_head(h).detach())
    gates, ents = [], []
    for s in range(0, h.shape[1], GATE_CHUNK):
        c = lm_head(h[:, s:s + GATE_CHUNK]).detach()
        g, e = gate_fn(c)
        gates.append(g)
        ents.append(e)
        del c
    return torch.cat(gates, dim=1), torch.cat(ents, dim=1)


class AMORDecodeWrapper:
    """
    Wraps a trained AMOR model for autoregressive decode with true_sparse gating.

    Use:
        wrapper = AMORDecodeWrapper(model)
        logits, cache = wrapper.prefill(input_ids)              # full prompt
        for step in range(num_new_tokens):
            logits, cache, info = wrapper.decode_step(
                next_token_id, cache, step_idx=prompt_len + step,
            )
    """

    def __init__(self, model: AMOR):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype

        if model.backbone == 'mamba2':
            if Mamba2Cache is None:
                raise ImportError("transformers (with Mamba2Cache) is required for backbone='mamba2'.")
            self.config = model.mamba_config
        else:
            if FLACache is None:
                raise ImportError("flash-linear-attention is required for backbone='gdn'.")
            for i, layer in enumerate(model.layers):
                layer.gdn.layer_idx = i

    # ---- Backbone helpers (prefill / step) -----------------------------------

    def _backbone_prefill(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape
        h = self.model.embed(input_ids)

        if self.model.backbone == 'mamba2':
            backbone_cache = Mamba2Cache(
                self.config, batch_size=batch, dtype=self.dtype, device=self.device,
            )
            # Convolution history must match the autocast projection dtype.
            # Preserve the recurrent SSM state and model parameters in fp32.
            if self.device.type == "cuda" and torch.is_autocast_enabled():
                backbone_cache.conv_states = backbone_cache.conv_states.to(
                    dtype=torch.get_autocast_dtype("cuda")
                )
            cache_position = torch.arange(0, seq_len, device=self.device)
            for ssd_layer, mlp_norm, mlp in zip(
                self.model.ssd_layers, self.model.mlp_norms, self.model.mlps,
            ):
                h = ssd_layer(h, cache_params=backbone_cache, cache_position=cache_position)
                h = h + mlp(mlp_norm(h))
        else:
            backbone_cache = FLACache()
            for layer in self.model.layers:
                h_normed = layer.norm_ssm(h)
                gdn_out, _, backbone_cache = layer.gdn(
                    h_normed, past_key_values=backbone_cache, use_cache=True,
                )
                h = h + gdn_out
                h = h + layer.mlp(layer.norm_mlp(h))

        return h, backbone_cache

    def _backbone_step(self, token_id: torch.Tensor, backbone_cache, step_idx: int):
        h = self.model.embed(token_id)

        if self.model.backbone == 'mamba2':
            cache_position = torch.tensor([step_idx], device=self.device)
            for ssd_layer, mlp_norm, mlp in zip(
                self.model.ssd_layers, self.model.mlp_norms, self.model.mlps,
            ):
                h = ssd_layer(h, cache_params=backbone_cache, cache_position=cache_position)
                h = h + mlp(mlp_norm(h))
        else:
            for layer in self.model.layers:
                h_normed = layer.norm_ssm(h)
                gdn_out, _, backbone_cache = layer.gdn(
                    h_normed, past_key_values=backbone_cache, use_cache=True,
                )
                h = h + gdn_out
                h = h + layer.mlp(layer.norm_mlp(h))

        return h, backbone_cache

    # ---- Pre-norm helper (residual_mode dispatch) ----------------------------

    def _block0_normed(self, h):
        """Block 0's pre-norm input. None pre_norm slot has two meanings:
        'classic' -> h is already norm_f'd (pass-through)
        'h'       -> norm_f applied inline to un-normed h
        """
        return self.model.norm_f(h) if self.model.residual_mode == 'h' else h

    def _residual_step(self, h_pre, h_normed, attn_contribution, pre_norm):
        """Apply one block's residual update. attn_contribution = alpha * attn_out
        (already gated). Pre-norm slot None means block 0; behavior depends on
        residual_mode."""
        if self.model.residual_mode == 'h':
            # All blocks symmetric: residual wraps un-normed h.
            return h_pre + attn_contribution
        if pre_norm is not None:
            # Blocks 1+ in 'classic': residual wraps un-normed h.
            return h_pre + attn_contribution
        # Block 0 in 'classic': h_pre and h_normed are the same norm_f'd tensor;
        # residual wraps that.
        return h_normed + attn_contribution

    # ---- Public API ----------------------------------------------------------

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor):
        batch, seq_len = input_ids.shape

        h, backbone_cache = self._backbone_prefill(input_ids)
        if self.model.residual_mode == 'classic':
            h = self.model.norm_f(h)

        cos_all = None
        amor_kv_caches = []

        for amor_block, pre_norm in zip(self.model.amor_blocks, self.model.amor_pre_norms):
            if pre_norm is not None:
                h_normed = pre_norm(h)
            else:
                h_normed = self._block0_normed(h)

            if amor_block.gate.router is not None:
                gate, _ = amor_block.gate(h_normed)
            else:
                gate, _ = _chunked_gate(self.model.lm_head, h_normed, amor_block.gate)

            ab = amor_block
            n_heads, head_dim = ab.n_heads, ab.head_dim

            K, V = ab.kv_proj(h_normed)
            K = K.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            V = V.view(batch, seq_len, n_heads, head_dim).transpose(1, 2)

            if cos_all is None:
                cos_all = ab.rope.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
                sin_all = ab.rope.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
            K = RotaryPositionEmbedding._apply_rotary(K, cos_all, sin_all)
            amor_kv_caches.append((K, V))

            Q = ab.q_proj(h_normed).view(batch, seq_len, n_heads, head_dim).transpose(1, 2)
            Q = RotaryPositionEmbedding._apply_rotary(Q, cos_all, sin_all)
            attn_out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, ab.d_model)
            attn_out = ab.out_proj(attn_out) * gate.unsqueeze(-1)
            alpha = ab.get_alpha()

            h = self._residual_step(h, h_normed, alpha * attn_out, pre_norm)

        h = self.model.final_norm(h)
        logits = self.model.lm_head(h)

        cache = {
            'backbone_cache': backbone_cache,
            'amor_kv_caches': amor_kv_caches,
        }
        return logits, cache

    @torch.no_grad()
    def decode_step(self, token_id: torch.Tensor, cache: dict, step_idx: int):
        batch = token_id.shape[0]
        backbone_cache = cache['backbone_cache']
        amor_kv_caches = cache['amor_kv_caches']

        h, backbone_cache = self._backbone_step(token_id, backbone_cache, step_idx)
        if self.model.residual_mode == 'classic':
            h = self.model.norm_f(h)

        fire_info = []
        new_amor_kv_caches = []

        for i, (amor_block, pre_norm) in enumerate(
            zip(self.model.amor_blocks, self.model.amor_pre_norms)
        ):
            past_K, past_V = amor_kv_caches[i]
            ab = amor_block
            n_heads, head_dim = ab.n_heads, ab.head_dim

            if pre_norm is not None:
                h_normed = pre_norm(h)
            else:
                h_normed = self._block0_normed(h)

            if ab.gate.router is not None:
                gate, entropy = ab.gate(h_normed)
            else:
                gate_logits = self.model.lm_head(h_normed).detach()
                gate, entropy = ab.gate(gate_logits)
                del gate_logits
            fired = gate.item() > 0.5

            K_new, V_new = ab.kv_proj(h_normed)
            K_new = K_new.view(batch, 1, n_heads, head_dim).transpose(1, 2)
            V_new = V_new.view(batch, 1, n_heads, head_dim).transpose(1, 2)
            cos = ab.rope.cos_cached[step_idx:step_idx + 1].unsqueeze(0).unsqueeze(0)
            sin = ab.rope.sin_cached[step_idx:step_idx + 1].unsqueeze(0).unsqueeze(0)
            K_new = RotaryPositionEmbedding._apply_rotary(K_new, cos, sin)

            K_full = torch.cat([past_K, K_new], dim=2)
            V_full = torch.cat([past_V, V_new], dim=2)
            new_amor_kv_caches.append((K_full, V_full))

            if fired:
                Q = ab.q_proj(h_normed).view(batch, 1, n_heads, head_dim).transpose(1, 2)
                Q = RotaryPositionEmbedding._apply_rotary(Q, cos, sin)
                attn_out = F.scaled_dot_product_attention(Q, K_full, V_full, is_causal=False)
                attn_out = attn_out.transpose(1, 2).contiguous().view(batch, 1, ab.d_model)
                attn_out = ab.out_proj(attn_out)
                alpha = ab.get_alpha()
                h = self._residual_step(h, h_normed, alpha * attn_out, pre_norm)
            else:
                # Gate didn't fire (true_sparse). For 'classic' mode block 0,
                # h was already norm_f'd above, so no shim is needed. For block 0
                # in 'h' mode, the residual wraps un-normed h, so we leave h as-is.
                pass

            fire_info.append({'block': i, 'fired': fired, 'entropy': entropy.item()})

        h = self.model.final_norm(h)
        logits = self.model.lm_head(h)

        cache['backbone_cache'] = backbone_cache
        cache['amor_kv_caches'] = new_amor_kv_caches
        return logits, cache, fire_info
