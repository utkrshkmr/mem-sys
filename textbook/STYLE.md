# Textbook style guide

This book teaches the 2025–2026 LLM memory architectures from scratch. It starts from the
standard Transformer and works up to every system in [`../README.md`](../README.md), explaining
the concept, the mathematics, the algorithms and the code in the mirrored repositories under
[`../repos/`](../repos). Every chapter author follows this guide so the book reads as one work.

## Reader

Someone who knows linear algebra, basic probability, Python and the basics of deep learning
(MLPs, backpropagation, SGD/Adam), but has not read these papers. Derive everything you use.
Never write "it is easy to see"; show the step.

## Build constraints (pdflatex)

- Use only the packages and macros in [`preamble.tex`](preamble.tex). Do not add `\usepackage`
  to a chapter. Avoid new macros; if one is unavoidable, use a letters-only name prefixed with
  your chapter number, e.g. `\cfiveGate`.
- **ASCII-only source.** No Unicode characters anywhere, in text or listings. Use `--`, `---`,
  ``` ``quotes'' ```, `$\approx$`, `$\rightarrow$`, `\'e`, etc.
- No external images. Simple TikZ diagrams are welcome but must compile.
- Do not use `\code`, `\repofile`, `\repo`, `\gh` or `\cite` inside `\section{...}` titles or
  `\caption{...}`. There, use `\texttt{...}` with `\_` escapes.
- Before you finish, run `tools/check-chapter.sh chapters/<your-file>.tex`. It must report no
  LaTeX errors and no undefined citations. Undefined references to *other* chapters are fine.

## Chapter layout

```latex
\chapter{Title}\label{ch:<given-label>}
% Opening: 1-3 paragraphs. The memory problem this chapter addresses and a roadmap.
\section{...}           % background needed for this chapter, built from scratch
\section{<System A>}    % one section per system, using the template below
...
\section{Comparison and discussion}   % a table comparing the chapter's systems
\begin{summarybox} ... \end{summarybox}
\section*{Exercises}    % 3-6 exercises: derivations, small proofs, coding tasks
```

Per-system section template (adapt the subsection names; keep the content):

1. **Motivation.** Which limitation of earlier memory designs it addresses.
2. **`taxonomy` box.** Representation (implicit/explicit), update dynamics (offline/online),
   persistence (short/long-term) and update rule (survey Table II). Cite `\citet{zhoubian2026memory}`.
3. **Formulation.** Equations in the shared notation, derived step by step. Include the
   recurrent (per-token) view and, where it exists, the parallel or chunkwise view, with the
   equivalence argued or proven.
4. **Algorithms.** `algorithm` + `algpseudocode` pseudocode for training/prefill and
   decoding. Give time and memory (state size) complexity.
5. **Implementation walkthrough (`implnote` boxes).** How the mirrored code implements it:
   key files (`\repofile{repos/...}`), classes and functions (`\code{...}`), tensor shapes,
   kernels, numerical tricks, config defaults. Short verbatim excerpts (at most ~30 lines each)
   are allowed. They must be copied exactly from the file and captioned with the path.
6. **From-scratch reference implementation.** A minimal, readable PyTorch-style listing of the
   core mechanism, captioned "Reference implementation (ours)". Keep it under ~50 lines. Where
   you can, check the maths numerically in NumPy (torch is not installed), for example that
   the recurrent and chunkwise forms agree. Do the check in your scratch space, not in the repo.
7. **`results` box.** See the policy below.
8. **Discussion.** Strengths, limitations, relation to other systems (with `\cref{ch:...}`).

## Results policy (strict)

- Quote numbers **only** from files in this repository: repo READMEs, docs or result tables
  under `repos/`, the survey PDF in `papers/`, or `repos/deepseek-ai__Engram/Engram_paper.pdf`.
  Name the source file in the `results` box.
- **Never write numbers from memory.** If the repository has no results, state the paper's
  qualitative claims as reported in the README or survey, and say "see the paper for numbers".
- For papers with no public code, explain the method from the survey's description and what
  the authors state publicly. If you formalize it yourself, say so: "One way to formalize this
  is ...; the paper's exact parameterization may differ."

## Notation (use these consistently)

| Symbol | Meaning |
|---|---|
| `T`, `t` | sequence length, time index `t = 1..T` |
| `d`, `L`, `H` | model width, number of layers, number of heads |
| `d_k`, `d_v` | key/query and value head dimensions |
| `\mathcal{V}`, `\abs{\mathcal{V}}` | vocabulary and its size |
| `\mX \in \R^{T \times d}` | input sequence, one token per row; `\vx_t \in \R^{d}` a column vector |
| `\vq_t, \vk_t \in \R^{d_k}`, `\vv_t \in \R^{d_v}` | per-head query, key, value (column vectors) |
| `\mQ = \mX\mW_Q`, `\mK`, `\mV` | stacked projections, `\mW_Q \in \R^{d \times d_k}` |
| `\mS_t \in \R^{d_v \times d_k}` | matrix-valued recurrent state (memory) |
| `\vo_t = \mS_t \vq_t` | read-out |
| `\mS_t = \mS_{t-1} + \vv_t \vk_t\T` | plain linear-attention write |
| `\alpha_t \in (0,1)` or `\valpha_t` | decay / forget gate (scalar or per-channel) |
| `\beta_t \in (0,1)` | write strength / learning rate of the delta rule |
| `\mS_t = \alpha_t \mS_{t-1}(\mI - \beta_t \vk_t\vk_t\T) + \beta_t \vv_t \vk_t\T` | gated delta rule |
| `C`, `[i]` | chunk size; chunk index, e.g. `\mS_{[i]}`, `\mQ_{[i]} \in \R^{C \times d_k}` |
| `B`, `k` | block size and number of selected blocks in sparse attention |
| `\mM` | causal mask (`0` on and below the diagonal, `-\infty` above) |
| `\sigmoid`, `\SiLU`, `\had` | logistic sigmoid, SiLU, element-wise product |
| `\vtheta`, `\eta` | parameters; learning rate (also the inner-loop rate in test-time training) |
| `\loss` | loss |

Attention is `\mO = \softmax(\mQ\mK\T/\sqrt{d_k} + \mM)\mV`. Use `\T` for transpose,
`\norm{\cdot}` for norms, and `\bigO(\cdot)` for complexity.

## Citations, labels, cross-references

- Cite with `\citet{key}` / `\citep{key}` using keys from [`bib/core.bib`](bib/core.bib).
  Put any other references in your chapter's `bib/cNN.bib` with keys prefixed `cNN-` (e.g.
  `c05-schlag2021linear`). Only add references you are sure exist, with correct authors and titles.
- Prefix every label with your chapter number: `sec:c05-...`, `eq:c05-...`, `alg:c05-...`,
  `lst:c05-...`, `fig:c05-...`, `tab:c05-...`.
- Refer to other chapters with `\cref{ch:...}` using these labels:

| File | Label | Contents |
|---|---|---|
| 01-llm-foundations | `ch:foundations` | Tokens to next-token prediction, Transformer, attention, RoPE, MoE, training, KV cache, GPU memory, FlashAttention |
| 02-memory-taxonomy | `ch:taxonomy` | The survey's taxonomy, update rules, limits of implicit memory, evaluation, open challenges |
| 03-linear-attention-ssm | `ch:linear` | Linear attention, chunkwise parallel form, gating, the delta rule, WY/UT transform, Mamba-2/SSD, the fla library |
| 04-sparse-attention | `ch:sparse` | MoBA, NSA, RATTENTION, HyperMLP |
| 05-delta-rule-family | `ch:delta` | Gated DeltaNet, Kimi Delta Attention / Kimi Linear, Gated DeltaNet-2, Kaczmarz LA |
| 06-expressive-transitions | `ch:expressive` | DeltaProduct, Comba, RWKV-7 |
| 07-structured-recurrent-memory | `ch:structured` | Log-Linear Attention, Mixture-of-Memories, Mamba-3 |
| 08-probabilistic-recurrent-memory | `ch:probabilistic` | Kalman Linear Attention, Gated KalmaNet, Next-Latent Prediction |
| 09-test-time-training | `ch:ttt` | Titans, TTT-E2E, In-Place TTT, GDWM, TTT-as-linear-attention |
| 10-slots-and-lookup | `ch:lookup` | LM2, Engram, ExplicitLM, MemoryLLM, MoE as conditional memory |
| 11-multi-timescale | `ch:multiscale` | Nested Learning / HOPE, OLieRA, Muon-OGD |
| 12-hybrid-architectures | `ch:hybrid` | Hybrid quadratic-linear, Hymba, Falcon-H1, hybrid-ratio analysis, OLMo Hybrid, Expansion Span, Hydra, Priming |
| 13-adaptive-routing-and-management | `ch:routing` | AMOR, HAM, CommVQ, Memory Caching, Bottlenecked Transformers |
| 14-synthesis | `ch:synthesis` | Cross-cutting comparison and open problems |
| A-code-guide | `app:code` | Guide to the mirrored repositories |

## Code listings

- `\begin{lstlisting}[caption={...},label={lst:cNN-...}]` (Python is the default language).
  ASCII only. Keep listings short.
- Verbatim excerpts: caption `Excerpt from \texttt{repos/.../file.py}`. Never alter code
  silently; mark elisions with `# ...`.
- Our own code: caption `Reference implementation (ours): ...`.

## Tone

Precise and explanatory. Explain *why* each design choice is made. Use plain words. Prefer
worked examples with small dimensions (e.g. `d_k = 2`) to build intuition.
