# State-of-the-art LLM memory systems, 2025–2026

A map of the 2025–2026 memory architectures covered by the survey
**[Memory for Large Language Models](papers/2607.25380v1_Memory_for_Large_Language_Models.pdf)**
(Zhoubian, Zhang, Kharlamov, Tang — Tsinghua, arXiv 2607.25380, July 2026), with the code
repository for each paper and a local mirror of that code.

## What is in this repo

| Path | Contents |
|---|---|
| [`papers/`](papers) | The survey PDF |
| [`repos/`](repos) | 40 code snapshots (no git history), one directory per upstream repo |
| [`repos/MANIFEST.md`](repos/MANIFEST.md) | Upstream URL, pinned commit, license and notes for each snapshot |
| [`textbook/`](textbook) | LaTeX sources of the textbook *Memory in Large Language Models* |
| [`.github/workflows/textbook.yml`](.github/workflows/textbook.yml) | GitHub Actions workflow that compiles the textbook to PDF |

## Textbook

[`textbook/`](textbook) is a ~490-page book that teaches every system below from scratch. It
starts from the standard decoder-only Transformer (tokenization, attention, RoPE, MoE,
training, KV cache, FlashAttention), then covers the survey's taxonomy and the
linear-attention / state-space toolkit, then the 2025–2026 memory systems in ten chapters,
and ends with a synthesis chapter and a guide to the mirrored code.
For each system it gives the derivations, pseudocode, a walkthrough of the mirrored code in
[`repos/`](repos), and a from-scratch reference implementation checked numerically. It quotes
results only from files in this repository. The book was drafted with AI assistance (Claude
Code); check details against the papers before relying on them.

- **Get the PDF:** open the latest run of the
  [Build textbook](https://github.com/utkrshkmr/mem-sys/actions/workflows/textbook.yml)
  workflow and download the `memory-in-llms-textbook` artifact. Pushing a tag named
  `textbook-v*` (e.g. `textbook-v0.1`) also attaches the PDF to a GitHub release.
- **Build locally:** with TeX Live installed, `cd textbook && latexmk main.tex` writes
  `build/main.pdf`. `tools/check-chapter.sh chapters/<file>.tex` compiles one chapter.
- **Contribute:** follow [`textbook/STYLE.md`](textbook/STYLE.md).

## The survey's taxonomy

The survey classifies **model-level** memory (not agent/RAG pipelines) along three axes:

- **Representation** — *implicit* memory is a by-product of the forward pass (KV cache,
  recurrent state); *explicit* memory has its own read/write interface (memory slots, lookup
  tables, test-time-trained parameters).
- **Update dynamics** — *offline* (fixed after training) vs. *online* (written during inference).
- **Persistence** — *short-term* (bounded by a window or episode) vs. *long-term*.

The "Update / persistence" columns below are taken from the survey's Table I; "—" means the
paper is discussed in the text but not listed in that table.

**Code legend:** ✅ official · 🟡 probably official (authors' own account, but the README does
not cite the paper) · ❓ unconfirmed candidate · ⚠️ no official code, unofficial
implementation linked · ❌ no public code found. ★ = approximate GitHub stars on 2026-10-07.
"fla" = also implemented in [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention)
(mirrored at [`repos/fla-org__flash-linear-attention`](repos/fla-org__flash-linear-attention)).

## At a glance: the leading systems per memory type

The survey does not rank or benchmark systems. These picks are a judgment based on recency,
how much the survey builds on each system, and real-world adoption (production models,
released weights, community use).

| Memory type (survey §) | Leading 2025–2026 systems | Why |
|---|---|---|
| Sparse / selective attention (§III-B) | **NSA**, **MoBA**, RATTENTION | Trainable sparse attention over KV blocks; NSA won ACL 2025 Best Paper, MoBA comes from Moonshot AI |
| Attention as memory (§III-A) | **HyperMLP** | Recasts attention as a context-sized MLP (ICML 2026) |
| Recurrent / linear-attention state (§III-C) | **Gated DeltaNet** family (GDN → **KDA** → **Gated DeltaNet-2**), **Mamba-3**, RWKV-7, Log-Linear Attention | The delta-rule family is used in production hybrids (Kimi Linear, OLMo Hybrid, Qwen3-Next); Mamba-3 is the latest SSM |
| Parameterized / test-time memory (§IV-A) | **Titans**, **TTT-E2E**, **In-Place TTT**, LM2 | Memory written by gradient steps at inference; In-Place TTT works on existing 8B models |
| Lookup memory (§IV-B) | **Engram** (DeepSeek), ExplicitLM, MemoryLLM | Hashed lookup tables as a sparsity axis alongside MoE |
| Multi-timescale memory (§IV-D) | **Nested Learning / HOPE** | Parameter groups updated at several frequencies |
| Hybrid architectures (§V-A) | **Kimi Linear**, **OLMo Hybrid**, **Falcon-H1**, AMOR, HAM | Fixed-ratio hybrids are mainstream; AMOR and HAM add token-adaptive routing between memories, which the survey highlights as the emerging direction |
| Memory management (§V-B) | **CommVQ**, Memory Caching, Bottlenecked Transformers | 1–2-bit KV caches; recurrent memory that grows with the sequence; KV consolidation |

---

## 1. Implicit memory (§III)

### 1a. Attention-based memory (§III-A)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| HyperMLP (2602.12601) | 2026 · ICML 2026 | Online / short | Attention reformulated as a two-layer MLP whose hidden width grows with the context; ReLU/GLU selection instead of softmax | ✅ [LJC-FVNR/HyperMLP](https://github.com/LJC-FVNR/HyperMLP) (~5★, first author; README names the paper by title only) | [`repos/LJC-FVNR__HyperMLP`](repos/LJC-FVNR__HyperMLP) |

### 1b. Sparse and selective memory (§III-B)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| MoBA (2502.13189) | 2025 · NeurIPS 2025 | Online / short | KV context split into blocks; a parameter-free top-k gate routes each query to a few blocks (MoE-style routing over memory) | ✅ [MoonshotAI/MoBA](https://github.com/MoonshotAI/MoBA) (~2.2k★) | [`repos/MoonshotAI__MoBA`](repos/MoonshotAI__MoBA) |
| NSA — Native Sparse Attention (2502.11089) | 2025 · ACL 2025 (Best Paper) | — | Compressed blocks + selected fine-grained blocks + a sliding window, mixed by a learned gate; trained sparse from the start | ⚠️ No official code. Unofficial: [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention) (~1.0k★); also [lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch), [XunhaoLai/native-sparse-attention-triton](https://github.com/XunhaoLai/native-sparse-attention-triton) | [`repos/fla-org__native-sparse-attention`](repos/fla-org__native-sparse-attention) (unofficial) |
| RATTENTION (2506.15545) | 2025 | Online / short | Sliding-window attention plus a residual linear attention over tokens that have left the window, allowing windows as small as 512 | ✅ [apple/axlearn — `axlearn/common/rattention`](https://github.com/apple/axlearn/tree/main/axlearn/common/rattention) (JAX/Pallas, inside the AXLearn library) | [`repos/apple__axlearn/axlearn/common/rattention`](repos/apple__axlearn/axlearn/common/rattention) (folder only) |

### 1c. Recurrent sequence memory (§III-C)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| Gated DeltaNet (2412.06464) | 2025 · ICLR 2025 | Online / long | Matrix state with Mamba2-style gated decay plus a delta-rule write that overwrites the value stored at a key | ✅ [NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet) (~678★); fla | [`repos/NVlabs__GatedDeltaNet`](repos/NVlabs__GatedDeltaNet) |
| RWKV-7 "Goose" (2503.14456) | 2025 | Online / long | Generalized delta rule with vector-valued gates and per-channel in-context learning rates | ✅ [BlinkDL/RWKV-LM — `RWKV-v7/`](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7) (~14.7k★); fla has a version the authors say is not yet aligned with theirs | [`repos/BlinkDL__RWKV-LM/RWKV-v7`](repos/BlinkDL__RWKV-LM/RWKV-v7) (folder only) |
| Log-Linear Attention (2506.04761) | 2025 · ICLR 2026 | Online / long | A Fenwick-tree hierarchy of states that grows logarithmically with length; applied to Mamba-2 and Gated DeltaNet | ✅ [HanGuo97/log-linear-attention](https://github.com/HanGuo97/log-linear-attention) (~288★, first author); fla | [`repos/HanGuo97__log-linear-attention`](repos/HanGuo97__log-linear-attention) |
| Kimi Linear / KDA (2510.26692) | 2025 · Moonshot tech report | Online / long | Gated DeltaNet with per-channel decay, in a 3:1 hybrid with MLA attention; 48B-A3B checkpoints | ✅ [MoonshotAI/Kimi-Linear](https://github.com/MoonshotAI/Kimi-Linear) (~1.6k★), kernels in [MoonshotAI/FlashKDA](https://github.com/MoonshotAI/FlashKDA) (~1.3k★); fla | [`repos/MoonshotAI__Kimi-Linear`](repos/MoonshotAI__Kimi-Linear), [`repos/MoonshotAI__FlashKDA`](repos/MoonshotAI__FlashKDA) |
| MoM — Mixture-of-Memories (2502.13685) | 2025 | Online / long | Several independent linear-attention states; a router sends each token to a few of them, reducing interference | ✅ [OpenSparseLLMs/MoM](https://github.com/OpenSparseLLMs/MoM) (~147★); fla | [`repos/OpenSparseLLMs__MoM`](repos/OpenSparseLLMs__MoM) |
| DeltaProduct (2502.10297) | 2025 · NeurIPS 2025 | — | Several delta-rule steps per token, so each state transition is a product of Householder matrices | ✅ [automl/DeltaProduct](https://github.com/automl/DeltaProduct) (~18★); maintained version in fla | [`repos/automl__DeltaProduct`](repos/automl__DeltaProduct) |
| Comba (2506.02475) | 2025 · NeurIPS 2025 | — | Scalar-plus-low-rank state transition with closed-loop feedback on the state and the output | ✅ [AwesomeSeq/Comba-triton](https://github.com/AwesomeSeq/Comba-triton) (~47★); fla | [`repos/AwesomeSeq__Comba-triton`](repos/AwesomeSeq__Comba-triton) |
| Mamba-3 (2603.15569) | 2026 · ICLR 2026 | Online / long | SSM with exponential-trapezoidal discretization, complex-valued (rotary) transitions and a MIMO update | ✅ [state-spaces/mamba](https://github.com/state-spaces/mamba) (`mamba_ssm/modules/mamba3.py`, ~18.9k★); fla | [`repos/state-spaces__mamba`](repos/state-spaces__mamba) |
| Gated DeltaNet-2 (2605.22791) | 2026 | Online / long | Separates the delta rule's single strength into a per-channel erase gate (key side) and a write gate (value side) | ✅ [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2) (~326★); fla | [`repos/NVlabs__GatedDeltaNet-2`](repos/NVlabs__GatedDeltaNet-2) |
| Kaczmarz Linear Attention (2605.08587) | 2026 | Online / long | Delta-rule writes whose step size is the key-norm-normalized Kaczmarz projection step | ❓ [JiaxuanZou0714/KaczmarzLinearAttention](https://github.com/JiaxuanZou0714/KaczmarzLinearAttention) (first author's account, but its README is a copy of Gated DeltaNet's and never mentions the paper) | [`repos/JiaxuanZou0714__KaczmarzLinearAttention`](repos/JiaxuanZou0714__KaczmarzLinearAttention) (unconfirmed) |
| Kalman Linear Attention (2602.10743) | 2026 · ICML 2026 | Online / long | Sequence mixing as exact Kalman filtering in information form, computed with a parallel scan; the state carries its own uncertainty | ✅ [vaisakh-shaj/kalman-linear-attention](https://github.com/vaisakh-shaj/kalman-linear-attention) (~12★, first author) | [`repos/vaisakh-shaj__kalman-linear-attention`](repos/vaisakh-shaj__kalman-linear-attention) |
| Gated KalmaNet (2511.21016) | 2026 · CVPR 2026 | Online / long | State solves an online ridge regression over the gated, fading past using Chebyshev iteration | ✅ [awslabs/hybrid-model-factory](https://github.com/awslabs/hybrid-model-factory) (~87★; general toolkit that includes GKA) | [`repos/awslabs__hybrid-model-factory`](repos/awslabs__hybrid-model-factory) |
| Next-Latent Prediction (2511.05963) | 2026 | Offline / long | A training loss that predicts the next hidden state, pushing Transformer states toward compact belief states | ✅ [JaydenTeoh/NextLat](https://github.com/JaydenTeoh/NextLat) (~196★) | [`repos/JaydenTeoh__NextLat`](repos/JaydenTeoh__NextLat) |

## 2. Explicit memory (§IV)

### 2a. Parameterized memory modules and test-time training (§IV-A)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| Titans (2501.00663) | 2025 · NeurIPS 2025 | Online / long | MLP memory updated during inference by surprise-driven gradient steps, with momentum and weight decay for forgetting | ⚠️ No official code (Google). Unofficial: [lucidrains/titans-pytorch](https://github.com/lucidrains/titans-pytorch) (~2.0k★) | [`repos/lucidrains__titans-pytorch`](repos/lucidrains__titans-pytorch) (unofficial) |
| TTT-E2E (2512.23675) | 2025 | Online / long | A sliding-window Transformer keeps training on next-token loss at test time, storing the context in its weights; its starting weights are meta-learned | ✅ [test-time-training/e2e](https://github.com/test-time-training/e2e) (~710★, JAX; 125M/1B/3B checkpoints) | [`repos/test-time-training__e2e`](repos/test-time-training__e2e) |
| LM2 — Large Memory Models (2502.06049) | 2025 | Offline / long | Separate memory matrix that tokens read through cross-attention, updated by input/forget/output gates | ✅ [convergence-ai/lm2](https://github.com/convergence-ai/lm2) (~49★, CC BY-NC 4.0) | [`repos/convergence-ai__lm2`](repos/convergence-ai__lm2) |
| In-Place TTT (2604.06169) | 2026 · ICLR 2026 | Online / long | The FFN down-projection of an existing model acts as fast weights, updated chunk by chunk at inference | ✅ [ByteDance-Seed/In-Place-TTT](https://github.com/ByteDance-Seed/In-Place-TTT) (~295★; Qwen3-8B and Llama-3.1-8B configs) | [`repos/ByteDance-Seed__In-Place-TTT`](repos/ByteDance-Seed__In-Place-TTT) |
| GDWM — Gated Differentiable Working Memory (2601.12906) | 2026 · ACL 2026 | Online / long | A write controller spends a budget of test-time gradient steps on the chunks that most depend on long-range context | ❌ none found | — |
| TTT with KV binding is secretly linear attention (2602.21204) | 2026 · ICML 2026 | — | Analysis: shows test-time-training layers that learn key–value bindings are a form of learned linear attention | ✅ [nv-tlabs/tttla](https://github.com/nv-tlabs/tttla) (~50★) + experiment code [JunchenLiu77/LaCT](https://github.com/JunchenLiu77/LaCT), [JunchenLiu77/ViTTT](https://github.com/JunchenLiu77/ViTTT) | [`repos/nv-tlabs__tttla`](repos/nv-tlabs__tttla), [`repos/JunchenLiu77__LaCT`](repos/JunchenLiu77__LaCT), [`repos/JunchenLiu77__ViTTT`](repos/JunchenLiu77__ViTTT) |

### 2b. Lookup-based memory (§IV-B)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| Engram (2601.07372) | 2026 · DeepSeek | Offline / long | N-gram embeddings in large hashed tables, fetched by O(1) lookup conditioned on the hidden state; a sparsity axis alongside MoE | ✅ [deepseek-ai/Engram](https://github.com/deepseek-ai/Engram) (~4.7k★) | [`repos/deepseek-ai__Engram`](repos/deepseek-ai__Engram) |
| ExplicitLM (2511.01581) | 2025 | Offline / long | A bank of ~1M human-readable token sequences, retrieved in two stages (product-key filter, then Gumbel-Softmax selection) | 🟡 [SCUT-HCC/ExplicitLM](https://github.com/SCUT-HCC/ExplicitLM) (~4★; senior author's lab; README does not cite the paper and may be a later version) | [`repos/SCUT-HCC__ExplicitLM`](repos/SCUT-HCC__ExplicitLM) (dataset files left out) |
| MemoryLLM — plug-n-play FFN memory (2602.00398) | 2026 · ICML 2026 (Apple) | Offline / long | FFNs trained on token embeddings alone, so they become context-free per-token lookup tables that can be precomputed and offloaded | ❌ none found. (Not the 2024 *MEMORYLLM: Towards Self-Updatable LLMs*, whose code is [wangyu-ustc/MemoryLLM](https://github.com/wangyu-ustc/MemoryLLM)) | — |

### 2c. Conditional parameters / MoE (§IV-C)

The survey's MoE examples (Switch Transformer, GLaM, Mixtral, DeepSeek-MoE) all predate 2025.
Its 2025–2026 entries in this space are **Engram** (lookup sparsity alongside MoE, §2b) and
**Hydra** (SSM + MoE + memory modules, §3).

### 2d. Multi-timescale and nested updates (§IV-D)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| Nested Learning / HOPE (2512.24695) | 2025 · NeurIPS 2025 | Online / long | The model is a set of nested optimization problems updated at different rates; HOPE adds a self-modifying Titans memory and a Continuum Memory System of MLP blocks on several timescales | ⚠️ No official code (Google). Unofficial: [kmccleary3301/nested_learning](https://github.com/kmccleary3301/nested_learning) (~712★); also [obekt/HOPE-nested-learning](https://github.com/obekt/HOPE-nested-learning) | [`repos/kmccleary3301__nested_learning`](repos/kmccleary3301__nested_learning) (unofficial) |

TTT-E2E (§2a) is also discussed here as a short-timescale explicit memory.

## 3. Hybrid memory architectures (§V-A)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| Hybrid Quadratic-Linear Transformer (2506.00744) | 2025 · NeurIPS 2025 | Online / long | Combines softmax KV memory (exact recall) with fast-weight memory (fixed size); compares three ways to blend them | ✅ [kazuki-irie/hybrid-memory](https://github.com/kazuki-irie/hybrid-memory) (~13★) | [`repos/kazuki-irie__hybrid-memory`](repos/kazuki-irie__hybrid-memory) |
| Expansion Span (2412.13328) | 2025 · PMLR v288 | Online / long | An SSM–attention hybrid reserves part of its attention context for tokens retrieved from far back | ❌ none found | — |
| Hydra (2508.15099) | 2025 · NeurIPS 2025 workshop | Online / long | SSM backbone that routes between sparse global attention, MoE, and workspace and product-key memories | ✅ [sidcraftscode/hydra](https://github.com/sidcraftscode/hydra) (~2★; small prototypes, not a full training pipeline) | [`repos/sidcraftscode__hydra`](repos/sidcraftscode__hydra) |
| Falcon-H1 (2507.22448) | 2025 · TII | — | Attention and Mamba-2 heads in parallel in every block, with a tunable ratio; 0.5B–34B models | ✅ [tiiuae/Falcon-H1](https://github.com/tiiuae/Falcon-H1) (~127★); weights in the [HF tiiuae Falcon-H1 collection](https://huggingface.co/collections/tiiuae/falcon-h1-6819f2795bc406da60fab8df) | [`repos/tiiuae__Falcon-H1`](repos/tiiuae__Falcon-H1) |
| Hymba (2411.13676) | ICLR 2025 · NVIDIA | — | Attention heads (precise recall) and SSM heads (summary) in parallel, plus learnable meta tokens and cross-layer KV sharing | ✅ [NVlabs/hymba](https://github.com/NVlabs/hymba) (~215★); weights `nvidia/Hymba-1.5B-Base` / `-Instruct` | [`repos/NVlabs__hymba`](repos/NVlabs__hymba) |
| Systematic Analysis of Hybrid Linear Attention (2507.06457) | 2025 | — | 72 models across 6 linear-attention variants and several hybrid ratios; recall depends on the full-attention ratio, best results with HGRN-2 or GDN at 3:1–6:1 | ❌ no GitHub code; weights only in the [HF m-a-p collection](https://huggingface.co/collections/m-a-p/hybrid-linear-attention-research-686c488a63d609d2f20e2b1e) (link from search results) | — |
| Kimi Linear | 2025 | Online / long | See §1c: KDA in a 3:1 hybrid with MLA | ✅ see §1c | see §1c |
| OLMo Hybrid (2604.03444) | 2026 · AI2 | — | 7B OLMo 3-style model with Gated DeltaNet and attention in a 3:1 ratio; about 2× data efficiency | ✅ [allenai/OLMo-core](https://github.com/allenai/OLMo-core) (`src/scripts/official/OLMo-hybrid/`, ~1.7k★); weights `allenai/Olmo-Hybrid-7B` (+ SFT/DPO variants) | [`repos/allenai__OLMo-core`](repos/allenai__OLMo-core) |
| AMOR (2602.13215) | 2026 | Online / long | Recurrent backbone that turns on attention only for tokens where its prediction entropy is high | 🟡 [HaoranZhengRaul/AMOR](https://github.com/HaoranZhengRaul/AMOR) (author's account; README does not cite the paper and may follow a revised version) | [`repos/HaoranZhengRaul__AMOR`](repos/HaoranZhengRaul__AMOR) |
| HAM — Hybrid Associative Memories (2603.22325) | 2026 · Zyphra | Online / long | RNN state for every token, plus a KV cache that admits only tokens the RNN predicts poorly; one threshold sets cache growth | ❌ none found | — |
| Priming (2605.08301) | 2026 · AWS | — | Builds hybrid SSM models from pre-trained Transformers using under 0.5% of the pre-training tokens | ✅ [awslabs/hybrid-model-factory](https://github.com/awslabs/hybrid-model-factory) (~87★); weights in the HF `amazon` primed-hybrid-models collection | [`repos/awslabs__hybrid-model-factory`](repos/awslabs__hybrid-model-factory) |

## 4. Memory management and efficiency (§V-B)

| System | Year · venue | Update / persistence | Memory mechanism | Code | Mirror |
|---|---|---|---|---|---|
| CommVQ (2506.18879) | 2025 · ICML 2025 | — | KV-cache vector quantization with a codebook that commutes with RoPE, down to 1–2 bits | ✅ [UMass-Embodied-AGI/CommVQ](https://github.com/UMass-Embodied-AGI/CommVQ) (~27★) | [`repos/UMass-Embodied-AGI__CommVQ`](repos/UMass-Embodied-AGI__CommVQ) |
| Memory Caching (2602.24281) | 2026 · ICML 2026 | Online / long | Saves RNN memory checkpoints at segment boundaries for later tokens to read, so memory grows with the sequence | ⚠️ No official code (Google). Unofficial: [kmccleary3301/memory_caching](https://github.com/kmccleary3301/memory_caching) (~12★) | [`repos/kmccleary3301__memory_caching`](repos/kmccleary3301__memory_caching) (unofficial) |
| Bottlenecked Transformers (2505.16950) | 2025 · ICLR 2026 | — | A cache processor rewrites KV entries in place at each reasoning-step boundary (KV consolidation) | ❌ none found | — |

## 5. Stability of writable parametric memory (§IV-E)

| System | Year | Memory mechanism | Code | Mirror |
|---|---|---|---|---|
| OLieRA — Orthogonal low-rank adaptation in Lie groups (2509.06100) | 2025 | Multiplicative low-rank updates in a Lie group, kept orthogonal to earlier tasks | ❌ none found | — |
| Muon-OGD (2605.08949) | 2026 | Muon-style spectral-norm updates projected orthogonal to earlier tasks' gradients | 🟡 [jeremylu916/OSD](https://github.com/jeremylu916/OSD) (first author's account; README does not cite the paper) | [`repos/jeremylu916__OSD`](repos/jeremylu916__OSD) |

## 6. Related memory surveys (survey Table III)

| Survey | Year | Companion repo | Mirror |
|---|---|---|---|
| Wu et al., *From Human Memory to AI Memory* (2504.15965) | 2025 | ❌ none found | — |
| Du et al., *Rethinking Memory in AI* (2505.00675) | 2025 | ✅ [Elvin-Yiming-Du/Survey_Memory_in_AI](https://github.com/Elvin-Yiming-Du/Survey_Memory_in_AI) (~353★, paper list) | [`repos/Elvin-Yiming-Du__Survey_Memory_in_AI`](repos/Elvin-Yiming-Du__Survey_Memory_in_AI) |
| Zhang et al., *Memory in LLMs: Mechanisms, Evaluation and Evolution* (2509.18868) | 2025 | ❌ none found | — |
| Luo et al., *From Storage to Experience* (2605.06716) | 2026 · Findings of ACL 2026 | ✅ [FeishuLuo/Evolving-LLM-Agent-Memory-Survey](https://github.com/FeishuLuo/Evolving-LLM-Agent-Memory-Survey) (~58★, paper list) | [`repos/FeishuLuo__Evolving-LLM-Agent-Memory-Survey`](repos/FeishuLuo__Evolving-LLM-Agent-Memory-Survey) |
| Zhoubian et al., *Memory for Large Language Models* (2607.25380) — the source survey | 2026 | ❌ none found | [`papers/`](papers) |

## Coverage summary

Of the 42 systems from 2025–2026 above (surveys excluded):

- **27** have official code.
- **3** have probable official code: ExplicitLM, AMOR, Muon-OGD.
- **1** has an unconfirmed candidate: Kaczmarz Linear Attention.
- **4** have only unofficial code: NSA, Titans, Nested Learning/HOPE, Memory Caching.
- **7** have no public code: GDWM, Apple's MemoryLLM, Expansion Span, HAM, Bottlenecked
  Transformers, OLieRA, and the hybrid-linear-attention analysis (weights only).

## How this was compiled

- The systems come from the survey's Table I, Table II, §III–§V and its reference list,
  filtered to 2025–2026.
- Repositories were found by web search and checked by opening each GitHub page and
  confirming that its README or owner matches the paper. arXiv and Hugging Face could not be
  opened from this environment, so code links given only on the arXiv page could have been
  missed. Venues and some weight links come from search results.
- Mirrors are `git archive` snapshots of the upstream branch at the commit in
  [`repos/MANIFEST.md`](repos/MANIFEST.md). Git history, submodules and dataset files over
  20 MB are not included. RATTENTION and RWKV-7 mirror only the relevant folder of a larger
  repository. Each mirror keeps its upstream license where the repository has one; several
  repositories ship without a license file (see the manifest), so check before reusing their code.
