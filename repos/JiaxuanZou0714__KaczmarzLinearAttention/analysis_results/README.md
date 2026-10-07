# Analysis 结果说明

该目录用于存放所有实验分析产物。

## 目录结构

每个实验子目录统一采用以下结构：

- `figures/`：论文图表文件（`.png`、`.pdf`）
- `tables/`：数值结果文件（`.csv`）

这样可以把图和表彻底分开，便于查找与管理。

## CSV 命名规则

- `*_raw.csv`：聚合前的原始记录（按 run/seed）
- `*_mean.csv`：按分组聚合后的均值结果
- `*_summary.csv`：针对某个视角整理后的摘要表

## 各子目录文件说明

### `ablation_47/`

- `ablation_47_report.md`：可读性报告（消融结论汇总）
- `tables/ablation_47_manifest.csv`：变体清单与分组元信息
- `tables/ablation_47_merged.csv`：按变体合并后的 MQAR + pretrain 指标
- `tables/ablation_47_mqar.csv`：仅 MQAR 聚合指标
- `tables/ablation_47_pretrain.csv`：仅 pretrain 聚合指标

### `efficiency_kimi_short_formal/`

- `tables/efficiency_raw.csv`：延迟/吞吐原始展开记录
- `tables/efficiency_mean.csv`：重复实验后的分组均值
- `tables/efficiency_prefill_summary.csv`：prefill 视角摘要
- `tables/efficiency_decode_summary.csv`：decode 视角摘要
- `figures/efficiency_prefill_latency.*`：prefill 延迟 vs 上下文长度
- `figures/efficiency_decode_throughput.*`：decode 吞吐（tokens/s）vs 上下文长度
- `figures/efficiency_decode_tpot.*`：decode 单 token 时延（TPOT）vs 上下文长度

### `mqar/`

- `tables/summary_raw.csv`：测试准确率原始记录
- `tables/summary_mean.csv`：按模型/设置聚合后的测试准确率
- `tables/final_val_extrapolation_raw.csv`：最终外推验证准确率原始记录
- `tables/final_val_extrapolation_mean.csv`：最终外推验证准确率均值
- `tables/val_curve_raw.csv`：训练步数上的验证曲线原始记录
- `tables/val_curve_mean.csv`：训练步数上的验证曲线均值
- `figures/mqar_metrics_panel1.*`：外推或序列长度面板
- `figures/mqar_metrics_panel2.*`：训练曲线或 key-length 面板
- `figures/mqar_metrics.*`：历史遗留的合并图

### `palindrome/`

- `tables/seq_len_metrics_raw.csv`：序列长度评估原始记录
- `tables/seq_len_metrics_mean.csv`：序列长度评估均值
- `tables/step_metrics_raw.csv`：训练步数评估原始记录
- `tables/step_metrics_mean.csv`：训练步数评估均值
- `figures/palindrome_seq_len_metrics.*`：序列长度图
- `figures/palindrome_step_metrics.*`：训练步数图
- `figures/palindrome_metrics.*`：历史遗留的合并图

### `stack/`

- `tables/seq_len_metrics_raw.csv`：序列长度评估原始记录
- `tables/seq_len_metrics_mean.csv`：序列长度评估均值
- `tables/step_metrics_raw.csv`：训练步数评估原始记录
- `tables/step_metrics_mean.csv`：训练步数评估均值
- `figures/stack_seq_len_metrics.*`：序列长度图
- `figures/stack_step_metrics.*`：训练步数图
- `figures/stack_metrics.*`：历史遗留的合并图

### `sniah/`

- `tables/summary_raw.csv`：SNIAH 测试指标原始记录
- `tables/summary_mean.csv`：SNIAH 测试指标均值
- `tables/final_val_extrapolation_raw.csv`：最终外推验证原始记录
- `tables/final_val_extrapolation_mean.csv`：最终外推验证均值
- `tables/val_curve_raw.csv`：训练步数上的验证曲线原始记录
- `tables/val_curve_mean.csv`：训练步数上的验证曲线均值
- `figures/sniah_metrics_final_val_extrapolation.*`：外推面板
- `figures/sniah_metrics_training_val_curve.*`：训练步数面板

### `sniah_no_es_6k/`（历史快照）

- `tables/summary_raw.csv`、`tables/summary_mean.csv`：SNIAH 摘要指标
- `tables/final_val_extrapolation_raw.csv`、`tables/final_val_extrapolation_mean.csv`：外推指标
- `tables/val_curve_raw.csv`、`tables/val_curve_mean.csv`：验证曲线
- `figures/sniah_no_es_6k.*`：历史遗留合并图

### `pretrain/`

- `tables/pretrain_val_ppl_1x_final_1B.csv`：各模型最终 val PPL@1x 快照
- `tables/pretrain_val_ppl_1x_curve_1B*.csv`：随 token 变化的 val PPL@1x 曲线
- `tables/long_context_ppl_raw.csv`：长上下文 PPL 原始记录
- `tables/long_context_ppl_mean.csv`：长上下文 PPL 均值
- `tables/long_context_relative_to_baseline.csv`：相对基准上下文长度的 PPL 比值
- `tables/long_context_slope_summary.csv`：外推斜率摘要
- `figures/pretrain_val_ppl_1x_curve_1B*.png|pdf`：pretrain scaling 曲线图
- `figures/long_context_extrapolation_ppl.*`：长上下文绝对 PPL 图
- `figures/long_context_extrapolation_relative.*`：长上下文相对比值图
- `figures/long_context_extrapolation.*`：历史遗留长上下文合并图

## 备注

- 新的分析运行默认把图保存到 `figures/`，把 CSV 保存到 `tables/`。
- 目录中部分 `*_metrics.*` 与 `long_context_extrapolation.*` 属于历史遗留合并图，保留用于追溯与对照。
