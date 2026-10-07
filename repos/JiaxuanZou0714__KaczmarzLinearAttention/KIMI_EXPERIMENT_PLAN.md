# 实验计划（轻算力优先，聚焦线性注意力能力）

## 目标
- 暂停 LongBench 全量评测与新增大规模 pretrain。
- 对齐 Kimi Linear 与 Gated DeltaNet 论文的核心实验维度。
- 基于现有 0.4B checkpoints，优先产出可复现、可比较、可快速完成的结果。

## 论文对齐（实验维度）
参考：
- Kimi Linear 技术报告（arXiv:2510.26692）实验章节包含：Synthetic tests、关键组件消融、Scaling law、主结果（短上下文/长上下文/RL）、效率对比（Prefill/Decode/KV cache）。
- Gated DeltaNet（arXiv:2412.06464）实验章节包含：语言建模与常识推理、Recall-intensive（含 S-NIAH/RULER）、长度外推、LongBench、吞吐效率、组件与混合结构消融。

| 论文维度 | 论文代表任务 | 当前轻算力替代 | 立即可跑 |
|---|---|---|---|
| 长度外推 | PG19/CodeParrot 长度-PPL | `run_long_context_extrapolation.sh` | 是 |
| Recall/记忆能力 | S-NIAH (RULER), Recall-intensive | 标准 S-NIAH（RULER-NIAH 全量）+ MQAR | 是 |
| 合成机制能力 | Synthetic tests, state tracking | Palindrome + Stack + MQAR | 是 |
| 效率 | Prefill/Decode/KV cache/Throughput | `run_efficiency_experiment.sh` | 是 |
| 长上下文真实任务 | LongBench/MRCR/RepoQA/Frames | MRCR/RULER 小规模，LongBench 暂停 | 部分 |
| 大规模能力验证 | Scaling law, RL, 大模型主榜单 | 需新增训练，暂缓 | 否 |

---

## P0（今天可出结果）

### P0-1 长度外推曲线（主结论）
目标：验证不同线性注意力模型在长度扩展时的退化斜率。

建议命令：

```bash
cd /home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet
MODELS=KLA,GDN,Mamba2,GLA,DeltaNet,Longhorn \
LENGTHS=1024,2048,4096,8192,16384,32768,65536 \
EVAL_ITERS=15 PROFILE=kimi_standard ./run_long_context_extrapolation.sh
```

交付物：长度-PPL 曲线、相对 4K 增幅表、退化斜率排序。

### P0-2 效率对比（工程结论）
目标：对齐论文中的 prefill/decode/KV cache 维度。

建议命令：

```bash
cd /home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet
MODELS=KLA,GDN,Mamba2,GLA,DeltaNet,Longhorn \
PROFILE=kimi_short ./run_efficiency_experiment.sh
```

交付物：Prefill 延迟曲线、Decode tokens/s 曲线、显存/KV cache 对比表。

### P0-3 Recall 能力（标准 S-NIAH）
目标：补齐标准 S-NIAH（RULER-NIAH 全量）评测，仅比较 KLA、GDN、Mamba2。

建议命令：

```bash
cd /home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet
DATASET_PRESET=ruler_selflong \
DATASET_OUTPUT=./data/long_context_tasks/real/ruler_sniah_standard.jsonl \
BENCHMARK_NAME=RULER_SNIAH_STANDARD \
RULER_CONFIGS=niah_single_1_4k,niah_multikey_1_4k,niah_multiquery_4k,niah_multivalue_4k \
PREP_MAX_SAMPLES=0 MODELS=KLA,GDN,Mamba2 \
PROFILE=kimi_standard ./run_long_context_task_benchmark.sh
```

或直接：

```bash
cd /home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet
./run_sniah_experiment.sh
```

交付物：EM/Contains/F1 汇总表、按 task 与长度桶的对比图。

---

## P1（1-2 天，可承受训练）

### P1-1 合成任务三件套（机制能力验证）
目标：聚焦“线性注意力本身的记忆与状态更新能力”，避开大规模语料 pretrain。

建议顺序：
1. `run_mqar_experiment.sh`
2. `run_palindrome_experiment.sh`
3. `run_stack_experiment.sh`

交付物：准确率-长度曲线、收敛步数对比、模型排序稳定性。

### P1-2 小规模 MRCR（可选）
目标：补一个真实检索任务，不做 LongBench。

建议命令：

```bash
cd /home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet
DATASET_PRESET=mrcr_openai BENCHMARK_NAME=MRCR_REAL \
MRCR_NEEDLES=2 PREP_MAX_SAMPLES=300 MODELS=KLA,GDN,Mamba2 \
PROFILE=kimi_quick ./run_long_context_task_benchmark.sh
```

### P1-3 合成 S-NIAH（兜底）
目标：若真实 RULER S-NIAH 对 pretrain-only checkpoint 过难，先用合成 single-needle 训练验证模型可学习性。

建议命令：

```bash
cd /home/huiwei/jiaxuanzou/linear_attn/GatedDeltaNet
PROFILE=quick ./run_synthetic_sniah_experiment.sh
```

说明：
1. 该脚本将 `MQAR(num_pairs=1)` 作为 synthetic S-NIAH proxy。
2. 默认只跑 KLA/GDN/Mamba2 对应的小模型配置（`RelaxedKaczmarzQNorm_MQAR,GatedDeltaNet_MQAR,Mamba2_MQAR`）。
3. quick 冒烟后可切到 `PROFILE=standard` 跑完整对照。

---

## P2（算力充足后恢复）

以下项目暂缓，不纳入近期里程碑：
1. LongBench-v2 全量评测。
2. NoPE vs RoPE 成对对照（需新增训练）。
3. 混合比例消融（0:1/1:1/3:1/7:1，需新增训练）。
4. 组件消融（去门控/改门控/去卷积，需新增训练）。
5. Kimi Linear 风格的 scaling law 与 RL 阶段对照。

---

## 统一统计口径（避免误判）

### 任务指标
1. 主指标：EM、Contains、F1。
2. 对多选任务额外记录“首个 A/B/C/D 命中率”（用于识别格式跟随失败）。
3. 长度桶统一：4K/8K/16K/32K/64K/128K。

### 公平比较
1. 所有模型使用同一输入切分、同一解码参数、同一 tokenizer。
2. 明确记录 truncation 设置（尤其是长上下文任务）。
3. 报告 skipped 样本占比与原因（OOM/超长截断/空输出）。

---

## 最近里程碑（建议）

1. 里程碑 A（当日完成）：P0-1 + P0-2。
2. 里程碑 B（次日完成）：P0-3 + P1-1（至少 MQAR）。
3. 里程碑 C（两天内）：补齐 Palindrome/Stack，并形成统一结论页。

## 一句话结论模板
在不新增大规模 pretrain 的前提下，优先用“长度外推 + 效率 + 标准 S-NIAH + 合成机制任务”建立线性注意力能力证据链，LongBench 与大规模消融后置。
