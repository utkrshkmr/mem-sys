# AMOR

**Adaptive Metacognitive Output Router** — a post-hoc hybrid architecture that
gates causal self-attention by output prediction entropy.

A recurrent backbone (Mamba-2 or Gated DeltaNet) is followed by `K` stacked
AMOR blocks. Each block measures the entropy of `softmax(lm_head(h))` and, where
the backbone is uncertain, fires self-attention to refine the residual stream.

```
Input -> [Backbone Mixer + SwiGLU MLP] x N -> [AMOR Block] x K -> Norm -> LM Head
                                                    |
                                            Entropy Gate  (fire if uncertain)
                                                    |
                                                Attention  (Q, K, V from residual)
```

K, V are standard linear projections of the residual stream — not SSM internal
state. The entropy gate concentrates attention compute on positions where the
backbone is least confident.

This release provides the six architecture variants through three flags.
The canonical three-block models also support a fitted entropy router for inference.

---

## Install

```bash
pip install -r requirements.txt
```

`mamba-ssm` and `causal-conv1d` are needed for the Mamba-2 backbone;
`flash-linear-attention` is needed for the Gated DeltaNet backbone. Both
backbones use CUDA for the fast kernels. Mamba2 also has a slow CPU reference
path; the supplied Gated DeltaNet implementation requires CUDA.

---

## Load trained weights and generate

Trained model weights will be linked here after the review period to avoid a double-blind review violation.

Follow the download instructions on the selected model page and place the
complete package in `model/`. For an anonymous review link, this repository
also provides a checksum-verified downloader:

```bash
python download_release.py ANONYMOUS_MODEL_URL --output model
```

The package contains the base weights, configuration, manifest and separate
optional router files.

```python
from load_model import load_model

# Native entropy gating is the default.
model = load_model("model", device="cuda")

# Fitted routers replace the three gating LM-head evaluations.
model = load_model("model", device="cuda", use_router=True)
```

Cached greedy completion, with or without the fitted routers:

```bash
python generate.py --model-dir model --prompt "The Eiffel Tower is" --max-new-tokens 32
python generate.py --model-dir model --router --prompt "The Eiffel Tower is" --max-new-tokens 32
```

The loader handles both a single safetensors file and standard safetensors
shards plus an index. For a sharded package, keep every model shard and
`model.safetensors.index.json` in the same directory; no manual merging is needed.
The base weights and router files are checksum-verified. Routers are matched to
the exact base weights and loaded strictly. Generation accepts a single prompt;
full forward and prefill also accept batches.

### Fitted router

Each AMOR block uses a width-512 SiLU MLP with a linear skip to estimate its
normalized entropy from the same normalized residual used by the native gate.
The estimate is compared with the original frozen threshold. Router parameters
and computation stay fp32, including inside CUDA bf16 autocast. Calibration is
already included in the fitted weights.

Router support covers full-sequence inference, prefill and cached decoding.
The LM head still produces the final token logits. The router is fitted
separately after base-model pretraining; base-model training uses the native
entropy gate. To return a loaded model to that path:

```python
from amor_router import detach_routers

detach_routers(model)
model.train()
```

The fitted packages target the canonical three-block `classic` models at 180M,
440M and 1.5B. Other architecture settings require their own matching routers.

### Architecture example

`python example.py` constructs a tiny randomly initialized model and shows
forward, prefill and cached decoding. It demonstrates the API; use the trained
model packages for pretrained text generation.

---

## Variants

The six variants are flag combinations of a single class:

| Configuration | `backbone` | `n_amor_blocks` | `residual_mode` |
|---|---|---|---|
| AMOR              | `mamba2` | `3` | `classic` |
| AMOR-classic      | `mamba2` | `1` | `classic` |
| AMOR-h            | `mamba2` | `3` | `h`       |
| AMOR-GDN          | `gdn`    | `3` | `classic` |
| AMOR-classic-GDN  | `gdn`    | `1` | `classic` |
| AMOR-GDN-h        | `gdn`    | `3` | `h`       |

### Flag reference

- **`backbone`** — recurrent sequence mixer.
  - `'mamba2'`: Mamba-2 SSD via `transformers.Mamba2ForCausalLM`.
  - `'gdn'`: Gated DeltaNet via `fla.layers.GatedDeltaNet`.

- **`n_amor_blocks`** — depth of the AMOR stack on top of the backbone (1 or 3 in
  the published configurations; any positive integer is accepted).

- **`residual_mode`** — placement of the post-backbone norm:
  - `'classic'`: `norm_f` is applied to the residual stream once, before the AMOR
    stack. Block 0 has no extra pre-norm; blocks 1+ each carry their own.
  - `'h'`: `norm_f` is reused inline as block 0's pre-norm. The residual stream
    stays un-normed through every AMOR block; only `final_norm` touches it
    before `lm_head`. All blocks become symmetric.

  Both modes have identical state_dict structure for a given `(backbone,
  n_amor_blocks)` pair, so checkpoints can be reloaded under either mode if
  desired.

---

## Reference dimensions

The published variants use these dimensions:

| Scale | `d_model` | `n_layer` | `d_ff` | `n_heads` | `head_dim` |
|-------|-----------|-----------|--------|-----------|------------|
| 180M  | 768       | 12        | 1216   | 12        | 64 |
| 440M  | 1024      | 24        | 1984   | 16        | 64 |
| 1.5B  | 2048      | 24        | 4096   | 32        | 64 |

Mamba-2 uses `d_state=128`, `expand=2`, `n_groups=1` and computes its own SSM
head count from `(d_model * expand) // head_dim`. GDN uses `expand_v=2.0` and
the `fla` convention `num_heads = int(0.75 * d_model / head_dim)`.

The default tokenizer assumed in the configurations above is a
Llama-3.1-compatible BPE with vocabulary 128256.

---

## Architecture details

### Entropy gate

```
entropy   = -sum(p * log(p)) / log(vocab_size)         # in [0, 1]
threshold = EMA(median) + offset_k * EMA(std)          # adaptive (default)
gate      = (entropy > threshold)                      # hard binary
```

The threshold tracks an EMA of the batch entropy median (and standard
deviation, in adaptive mode), so the gate has no learnable parameters and the
entropy signal does not flow gradients into the backbone — gating decisions are
detached. The CE loss trains the backbone exclusively through the final logits
path, never through the gate.

### AMOR block

```
gate     = EntropyGate(lm_head(h_normed))
attn_out = Attention(h_normed) * gate
output   = h_normed + alpha * attn_out
```

- Q/K/V are linear projections of the (pre-normed) residual stream. RoPE is
  applied to Q and K.
- `out_proj` is zero-initialized; the per-channel `alpha = sigmoid(raw_alpha)`
  (clamped from below at `raw_alpha = -1`, giving a floor of ~0.27) handles
  output magnitude scaling. At initialization the attention projections contribute a zero residual update.
- Two attention modes:
  - `'full_with_mask'`: dense attention, output masked by the gate. Used for
    training; GPU-friendly.
  - `'true_sparse'`: Q and the attention computation only run for firing
    positions. Small floating-point differences from dense attention can occur. Used for
    inference.

### Stacking

When `n_amor_blocks > 1`, each block recomputes its own entropy from
`lm_head(pre_norm(h))`. After the first block fires attention and modifies the
residual stream, the second block sees a different entropy landscape: positions
that were resolved drop in entropy, while positions that remain hard stay high.
Each block has its own EntropyGate (independent EMA), Q/K/V/O, RoPE, and alpha.

### Initialization

All weights are initialized to `N(0, 0.02)` for apple-to-apple comparison
between backbones. AMOR `out_proj` stays at zero. The LM head is weight-tied to
the embedding.

---

## Files

| File | Purpose |
|------|---------|
| `amor.py`         | Unified model (building blocks + GDN block + `AMOR` class) |
| `amor_decode.py`  | Autoregressive decode wrapper (prefill + decode_step, with caches) |
| `example.py`      | Tiny end-to-end demo |
| `requirements.txt`| Dependencies |
| `LICENSE`         | MIT |

The release modules are self-contained and use no imports from the research workspace.
The fitted router is in amor_router.py; load_model.py and generate.py provide
verified loading and cached completion.

---

## Tests

```bash
python -m unittest discover -s tests
```

Tests cover router precision, base-weight matching, atomic attachment, RNG
preservation and restoration of the native gate.

## License

MIT — see [LICENSE](LICENSE).
