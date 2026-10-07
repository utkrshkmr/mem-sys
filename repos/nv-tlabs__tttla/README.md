<h1 align="center">Test-Time Training with KV Binding Is Secretly Linear Attention</h1>


<p align="center">
  <a href="https://junchenliu77.github.io/">Junchen Liu</a><sup>*</sup>,
  <a href="https://selflein.github.io/">Sven Elflein</a>,
  <a href="https://orlitany.github.io/">Or Litany</a>,
  <a href="https://zgojcic.github.io/">Zan Gojcic</a>,
  <a href="https://www.liruilong.cn/">Ruilong Li</a><sup>*</sup>
</p>

<p align="center">
  <sup>*</sup>Equal contribution.
</p>

<h3 align="center">ICML 2026</h3>


<p align="center">
  <a href="https://research.nvidia.com/labs/sil/projects/tttla/">[Project Page]</a>
  <a href="https://arxiv.org/abs/2602.21204">[Paper]</a>
</p>

## Abstract

Test-time training (TTT) with KV binding as sequence modeling layer is commonly interpreted as a form of online meta-learning that memorizes a key–value mapping at test time. However, our analysis reveals multiple phenomena that contradict this memorization-based interpretation. Motivated by these findings, we revisit the formulation of TTT and show that a broad class of TTT architectures can be expressed as a form of learned linear attention operator. Beyond explaining previously puzzling model behaviors, this perspective yields multiple practical benefits: it enables principled architectural simplifications, admits fully parallel formulations that preserve performance while improving efficiency, and provides a systematic reduction of diverse TTT variants to a standard linear attention form. Overall, our results reframe TTT not as test-time memorization, but as learned linear attention with enhanced representational capacity.

## Updates

- **2026-04-30**: Paper accepted to ICML 2026!
- **2026-04-24**: Code released!


## Code

Experiment code lives in two repositories:

- LaCT (LLM and NVS experiments): https://github.com/JunchenLiu77/LaCT/tree/tttla
- ViTTT (Image classification experiment): https://github.com/JunchenLiu77/ViTTT/tree/tttla

## Citation

If you find this work useful, please consider citing:

```bibtex
@misc{liu2026testtimetrainingkvbinding,
      title={Test-Time Training with KV Binding Is Secretly Linear Attention},
      author={Junchen Liu and Sven Elflein and Or Litany and Zan Gojcic and Ruilong Li},
      year={2026},
      eprint={2602.21204},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.21204},
}
```
