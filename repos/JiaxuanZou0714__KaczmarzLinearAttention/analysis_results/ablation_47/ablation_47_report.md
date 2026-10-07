# 4.7 Ablation Experiment Report

Generated at: 2026-04-25T20:44:16

## 4.7 实验设定（可直接用于论文）

### 4.7.1 研究目标与总口径

- 目标：系统评估 KLA/GatedDeltaNet 系列中归一化策略、序列因子、门控机制与状态维度对 `MQAR` 与 `100M token pretrain` 的影响。
- 总口径：每个变体都执行两条任务链路
- 任务链路 1：`MQAR` 训练与评估（含长度外推）
- 任务链路 2：`SlimPajama` 子集上的 `100M token` 预训练，记录 `val_ppl@1x`
- 统一随机种子：`seed=42`
- 统一分组与命名：`ab47_<variant>_*`

### 4.7.2 Ablation 矩阵

- Ablation 1（归一化）：`KLA / NoNorm / k_norm_only / seq_only(1/t) / learned_scalar`
- Ablation 2（序列因子）：`none / inv_t / inv_sqrt_t / inv_log_t`
- Ablation 3（门控机制）：`KLA single gate / GDN dual gate / GLA independent gate`
- Ablation 4（状态维度）：`expand_v=2/4/8/16`

### 4.7.3 MQAR 数据与任务定义

- 数据生成脚本：`mqar_data.py`
- 数据目录：`data/mqar_ablation47/seq256_key1_seed42`
- 主任务参数：`seq_len=256`、`key_len=1`、`num_pairs=32`
- 词表设置：key 来自前半词表，value 来自后半词表；监督标签为 query 位置对应 value
- 训练集/验证集/测试集：`20000 / 2000 / 2000`（standard profile）
- 外推评估因子：`1x, 2x, 4x, 8x`
- 外推汇报口径：优先汇报 `8x`；若缺失则回退到最大可用因子

### 4.7.4 MQAR 训练超参数

- 训练脚本：`train_mqar.py`
- 训练步数：`max_steps=10000`
- batch size：`32`
- 学习率：`1e-3`（NoNorm 出现不稳定时可下调）
- 权重衰减：`0.1`
- 验证间隔：`200` step
- checkpoint 间隔：`1000` step
- 早停耐心：`10`
- 设备：`1 GPU`，`precision=bf16-mixed`
- 评估指标：`test_acc` 与 `final_val_acc@{1x,2x,4x,8x}`

### 4.7.5 100M Pretrain 设定

- 训练脚本：`pretrain.py`（由 `scripts/train_local.sh` 调用）
- 训练配置：`train_config=tsz512x4k`
- token 预算：`max_tokens=100,000,000`
- 评估配置：`total_evals=20`、`eval_iters=15`
- 优化器学习率：`1e-4`
- 数据：`mydata/slimpajama/train` 与 `mydata/slimpajama/validation`
- 指标：`metric/val_ppl@1x`（从 `wandb-summary.json` 读取）
- 显存稳定策略：默认 `micro_batch_size=4`；对高开销变体自动降档（如 `a4_expand_v8 -> 2`，`a4_expand_v16 -> 1`），并启用失败重试。

### 4.7.6 变体开关映射（实现口径）

- `NoNorm`：`--qk_norm no_norm --gate_mode dual --seq_factor_mode none`
- `k_norm_only`：`--qk_norm k_norm_only --gate_mode dual --seq_factor_mode none`
- `seq_only(1/t)`：`--qk_norm seq_only --gate_mode dual --seq_factor_mode none`
- `learned_scalar`：`--qk_norm learned_scalar --learned_norm_init 1.0 --gate_mode dual --seq_factor_mode none`
- `KLA single gate`：`--qk_norm relaxed_kaczmarz_q_norm --gate_mode single --seq_factor_mode none`
- `GDN dual gate`：`--gate_mode dual --seq_factor_mode none`
- `GLA independent gate`：`model=GLA_MQAR / GLA_0.4B`
- `expand_v` 消融：`--expand_v 2/4/8/16`

### 4.7.7 统计与结果聚合口径

- 汇总脚本：`ablation_47_analysis.py`
- MQAR 汇总：读取各变体 `results.json`
- Pretrain 汇总：读取各变体对应 `wandb-summary.json` 的 `metric/val_ppl@1x`
- 表中 `MQAR Runs` / `Pretrain Runs` 表示该变体被聚合到的有效运行次数
- 当前报告中 `NoNorm` 的 pretrain 仍缺失（见 `Missing Results`），写论文时需注明该项未完成或补跑后更新。

### 4.7.8 分组对照设定（w/o 记法）

- KLA 默认配置（所有组的锚点）：
- `qk_norm=relaxed_kaczmarz_q_norm`, `seq_factor_mode=none`, `gate_mode=dual`, `expand_v` 使用模型默认值（未额外覆盖）。
- 记号约定：`w/o X` 表示“相对 KLA 默认配置，去掉或替换 X，其余保持不变”。

- Ablation 1（归一化策略，固定 `seq_factor_mode=none` 与 `gate_mode=dual`）：
- `KLA`：默认配置（对照组）。
- `w/o KLA-Norm (NoNorm)`：`qk_norm=no_norm`，即 `beta=sigmoid(b_t)`。
- `w/o Q-Norm (k_norm_only)`：`qk_norm=k_norm_only`，即 `beta=1/(||k_t||^2+eps)`。
- `w/o K-Norm (seq_only)`：`qk_norm=seq_only`（仅序列规则，不使用 K 范数项）。
- `w/ Learned Scalar`：`qk_norm=learned_scalar`，`learned_norm_init=1.0`。

- Ablation 2（序列因子，KLA 对应项已显式标注）：
- `KLA / w/o Seq-Factor`：`seq_factor_mode=none`（这就是 KLA 在 Ablation 2 的对应配置）。
- `w/ Seq-Factor(1/t)`：`seq_factor_mode=inv_t`。
- `w/ Seq-Factor(1/sqrt(t))`：`seq_factor_mode=inv_sqrt_t`。
- `w/ Seq-Factor(1/log(t+1))`：`seq_factor_mode=inv_log_t`。
- 该组除 `seq_factor_mode` 外，保持 `qk_norm=relaxed_kaczmarz_q_norm` 与 `gate_mode=dual` 不变。

- Ablation 3（门控机制，KLA 对应项已显式标注）：
- `KLA / w/o Dual-Gate`：`KLA_SingleGate`，即 `gate_mode=single`（A3 中 KLA 对应配置）。
- `w/ Dual-Gate (GDN_DualGate)`：`gate_mode=dual`。
- `w/ Independent-Gate (GLA_IndependentGate)`：替换为同规模 `GLA_MQAR / GLA_0.4B`。
- 该组目标是比较单门控、双门控、独立门控；任务预算与评估口径保持一致。

- Ablation 4（状态维度）：
- `w/ expand_v=2/4/8/16`：仅覆盖 `expand_v`，其余保持 KLA 默认配置不变。
- 该组用于评估状态容量变化对拟合与外推的影响。

- 全组统一评估：MQAR 报告 `test_acc` 与 `1x/2x/4x/8x` 外推精度，pretrain 报告 `val_ppl@1x`。
- 结果解读原则：每组结论仅归因于该组唯一变更项，避免跨组混合解释。

## Ablation 1: Normalization Strategy

| Variant | MQAR Test Acc (%) | MQAR Extrap Acc (prefer 8x, %)  | Pretrain Val PPL@1x | MQAR Runs | Pretrain Runs |
| --- | --- | --- | --- | --- | --- |
| KLA (Default) | 98.3896 | 73.8406 | 753.6525 | 1 | 1 |
| KLA (w/o KLA-Norm, NoNorm) | 0.0322 | 0.0063 | NA | 1 | 1 |
| KLA (w/o Q-Norm, k_norm_only) | 97.8054 | 77.6827 | 3809.7120 | 1 | 1 |
| KLA (w/o K-Norm, seq_only) | 97.1958 | 85.0189 | 1074.1918 | 1 | 1 |
| KLA (w/ Learned Scalar) | 97.7749 | 74.4597 | 22968.8691 | 1 | 2 |

## Ablation 2: Sequence-Length Factor

| Variant | MQAR Test Acc (%) | MQAR Extrap Acc (prefer 8x, %)  | Pretrain Val PPL@1x | MQAR Runs | Pretrain Runs |
| --- | --- | --- | --- | --- | --- |
| KLA (w/o Seq-Factor, none) | 98.3896 | 73.8406 | 753.6525 | 1 | 1 |
| KLA (w/ Seq-Factor 1/t) | 97.9205 | 20.4694 | 4660.9998 | 1 | 1 |
| KLA (w/ Seq-Factor 1/sqrt(t)) | 99.0246 | 50.2379 | 1148.3912 | 1 | 1 |
| KLA (w/ Seq-Factor 1/log(t+1)) | 97.5632 | 84.6755 | 2083.4568 | 1 | 1 |

## Ablation 3: Gating Mechanism

| Variant | MQAR Test Acc (%) | MQAR Extrap Acc (prefer 8x, %)  | Pretrain Val PPL@1x | MQAR Runs | Pretrain Runs |
| --- | --- | --- | --- | --- | --- |
| KLA (w/o Dual-Gate, single gate) | 0.0119 | 0.0315 | 2104.6131 | 1 | 1 |
| KLA (w/ Dual-Gate, GDN) | 98.7215 | 66.8053 | 747.2375 | 1 | 1 |
| KLA (w/ Independent-Gate, GLA) | 0.0254 | 0.0252 | 356.0050 | 1 | 1 |

## Ablation 4: State Expansion

| Variant | MQAR Test Acc (%) | MQAR Extrap Acc (prefer 8x, %)  | Pretrain Val PPL@1x | MQAR Runs | Pretrain Runs |
| --- | --- | --- | --- | --- | --- |
| KLA (w/ expand_v=2) | 98.0865 | 71.9943 | 762.3334 | 1 | 1 |
| KLA (w/ expand_v=4) | 98.8773 | 91.3374 | 1669.3464 | 1 | 1 |
| KLA (w/ expand_v=8) | 95.3111 | 95.3733 | 6945.6310 | 1 | 1 |
| KLA (w/ expand_v=16) | 99.1753 | 92.6339 | 1383.3806 | 1 | 1 |

## Missing Results

- NoNorm (a1_no_norm, ab1_norm): missing Pretrain

## Notes

- MQAR extrapolation列优先取8x；若8x缺失则回退到最大可用外推因子。
- Pretrain 指标来自 wandb-summary.json 的 metric/val_ppl@1x。
