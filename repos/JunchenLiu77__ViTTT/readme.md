# TTTLA -- ViTTT Experiments

ViTTT experiment code for [Test-Time Training with KV Binding Is Secretly Linear Attention](https://arxiv.org/abs/2602.21204) (TTTLA).

[[Project Page]](https://research.nvidia.com/labs/sil/projects/tttla/) [[Paper]](https://arxiv.org/abs/2602.21204)

We show analytically that TTT architectures with key-value binding reduce to learned linear attention operators. This directory contains the ViTTT experiment implementations used in the paper, including empirical studies (Sec. 4) and the progressive reduction from TTT to linear attention (Sec. 6.1), built on top of the [ViTTT](https://arxiv.org/abs/2512.01643) codebase.

## Experiments and Variants

Variant configs are in `vittt/cfgs/` and controlled by the `TTT_LOSS_TYPE` field. The dispatch logic is in `vittt/models/ttt_block.py`.

| Name | Config | `TTT_LOSS_TYPE` | Description |
|------|--------|-----------------|-------------|
| Base (ViTTT) | `vittt_b.yaml` | `dot_product` (default) | Full SwiGLU TTT with Muon |
| GA | `vittt_b_ga.yaml` | `ga_dot_product` | Gradient ascent instead of descent (Sec. 4.2) |
| No Query | `vittt_b_no_query.yaml` | `no_query_dot_product` | Replace query with key in output projection (Sec. 4.4) |
| Variant 1 | `vittt_b_variant1.yaml` | `only_w1` | Update only final-layer params w1 (Sec. 6.1) |
| Variant 3 | `vittt_b_variant3.yaml` | `only_w1_straight_qk` | Replace multi-layer MLP with single linear layer (Sec. 6.1) |
| Variant 6 | `vittt_b_variant6.yaml` | `only_w1_straight_qk_no_muon` | Remove gradient orthogonalization, reduces to standard linear attention (Sec. 6.1) |

## Launching

Launch scripts are in `vittt/scripts/`. All train ViTTT-Base on ImageNet for 60 epochs with 2 GPUs. Set `$DATA_PATH` to your ImageNet directory before running.

```bash
cd vittt
bash scripts/vittt_b_2gpu_bs512_60epoch.sh            # Base ViTTT
bash scripts/vittt_b_2gpu_bs512_60epoch_ga.sh          # Gradient ascent
bash scripts/vittt_b_2gpu_bs512_60epoch_no_query.sh    # No query
bash scripts/vittt_b_2gpu_bs512_60epoch_variant1.sh    # Variant 1
bash scripts/vittt_b_2gpu_bs512_60epoch_variant3.sh    # Variant 3
bash scripts/vittt_b_2gpu_bs512_60epoch_variant6.sh    # Variant 6
```

---

# $\text{ViT}^3$: Unlocking Test-Time Training in Vision

This repo contains the official PyTorch code and pre-trained models for **Vision Test-Time Training ($\text{ViT}^3$)**.

+ $\text{ViT}^3$: [Unlocking Test-Time Training in Vision](https://arxiv.org/abs/2512.01643)

## Abstract

<p align="center">
    <img src="figures/fig1_ttt.png" width= "400">
</p>

Test-Time Training (TTT) has recently emerged as a promising direction for efficient sequence modeling. TTT reformulates attention operation as an online learning problem, constructing a compact inner model from key-value pairs at test time. This reformulation opens a rich and flexible design space while achieving linear computational complexity. However, crafting a powerful visual TTT design remains challenging: fundamental choices for the inner module and inner training lack comprehensive understanding and practical guidelines. To bridge this critical gap, in this paper, we present a systematic empirical study of TTT designs for visual sequence modeling. From a series of experiments and analyses, we distill six practical insights that establish design principles for effective visual TTT and illuminate paths for future improvement. These findings culminate in the Vision Test-Time Training ($\text{ViT}^3$) model, a pure TTT architecture that achieves linear complexity and parallelizable computation. We evaluate $\text{ViT}^3$ across diverse visual tasks, including image classification, image generation, object detection, and semantic segmentation. Results show that $\text{ViT}^3$ consistently matches or outperforms advanced linear-complexity models (e.g., Mamba and linear attention variants) and effectively narrows the gap to highly optimized vision Transformers. We hope this study and the $\text{ViT}^3$ baseline can facilitate future work on visual TTT models.

## Usage

We provide a minimal implementation of $\text{ViT}^3$ block in [ttt_block.py](./ttt_block.py), which can act as a plug-in module in various vasion tasks. 

- Example:

```python
from ttt_block import TTT
block = TTT(dim=512, num_heads=16)
x = torch.rand(1, 256, 512)
x = block(x, h=16, w=16)
```

## Results

Please go to the folder [vittt](./vittt) for specific document.

## Acknowledgements

This code is developed on the top of [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [MILA](https://github.com/LeapLabTHU/MLLA). 

## Citation

If you find this repo helpful, please consider citing us.

```latex
@article{han2025vit,
  title={ViT$^3$: Unlocking Test-Time Training in Vision},
  author={Han, Dongchen and Li, Yining and Li, Tianyu and Cao, Zixuan and Wang, Ziming and Song, Jun and Cheng, Yu and Zheng, Bo and Huang, Gao},
  journal={arXiv preprint arXiv:2512.01643},
  year={2025}
}
```

## Contact

If you have any questions, please feel free to contact the authors.

Dongchen Han: [hdc23@mails.tsinghua.edu.cn](mailto:hdc23@mails.tsinghua.edu.cn)

