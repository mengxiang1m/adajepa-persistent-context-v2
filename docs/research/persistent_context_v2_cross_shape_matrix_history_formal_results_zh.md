# 跨 Shape Matrix History 前瞻正式实验结果

日期：2026-08-23  
合同：`persistent-context-v2-cross-shape-matrix-history-formal-v1`  
证据状态：**VALID；跨任务收益与 persistence-specific 对照均为正，区间排除 0**

## 1. 研究问题与设计

本实验检验：在 E1 与 E2 属于不同 shape/任务时，E1 保存的 rotation×gain posterior 能否改善 E2 冷启动；当物理 factor 在任务边界改变时，该历史是否失效。

只使用作者 seed-42 的 `val_I/val_L/val_+/val_small_tee/val_square/val_Z`。只读审计在六池中找到 4474 个合格 segments，排除所有全局重复 state+action hash，并要求 smoke/formal/reserve 的 segment hash 和 `ep_idx:offset` 全局唯一。

Formal 为最小完全平衡设计：

```text
6 个有向 shape pair × 8 个 rotation×gain factor × 2 replicates = 96 sequences
```

Shape pair 为 `I→L、L→+、+→small_tee、small_tee→square、square→Z、Z→I`。每条 sequence 在同一 E1 scene 上生成 correct-history 和 factor 改变的 no-persistence-history，再在完全相同 E2 scene 上严格配对。

唯一主 estimand：

```text
pose_auc10(population) - pose_auc10(correct-history α=.75)
```

关键 persistence-specific estimand：

```text
pose_auc10(no-persistence-history α=.75) - pose_auc10(correct-history α=.75)
```

## 2. 运行与审计

- Formal 完成 `96/96`；GPU 0 为 NVIDIA L40，墙钟约 326 秒，峰值 PyTorch allocated 约 2.85 GB。
- 192 个 formal segments 互异；每个 shape pair 16 条、每个 factor 12 条、每个 pair×factor 2 条。
- E2 前完成 external T-F0 决策，被选 correct-history α 第一个执行。
- 世界模型参数运行前后 hash 不变。
- 独立脚本不 import collector/evaluator，重新计算 selection、两份 posterior、F0 特征与决策、context matrix、配对场景、轨迹指标和 bootstrap summary；所有最大绝对误差为 `0`，decision/execution mismatch 为 `0`。
- 源码快照 SHA256：`73cf3bb41c8546403a6dfa3d4e2aad6a0bb48f8d91513016ce84b179a09d0001`，含 331 个源码/配置文件。
- Formal raw SHA256：`ea74e8500eb37ce27a90ce110a17697672f13671aab4322aed1155d200a4a728`；summary：`81d6c65b4b2e54953f04538fbf5d118a388811868306f18ea58412a4211af074`；audit：`de63d395ee07cf5c9516da4872849df70399fa0cfb06d64a22a1e9487ae739c8`。

远端产物：

```text
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_cross_shape_matrix_history_formal_v1/
```

## 3. 正式结果

`pose_auc10` 越低越好，population mean cost 为 `2.4075`。

| 方法 | mean cost | 相对 population 改善 | delta 95% CI | harm | 正向比例 |
|---|---:|---:|---:|---:|---:|
| correct fixed `α=.5` | 2.2167 | 7.92% | [0.1178, 0.2708] | 27.083% | 68.750% |
| **correct fixed `α=.75`** | **2.1675** | **9.97%** | **[0.1393, 0.3478]** | **28.125%** | **67.708%** |
| correct full context | 2.1500 | 10.69% | [0.1365, 0.3820] | 36.458% | 61.458% |
| external T-F0 | 2.1402 | 11.10% | [0.1600, 0.3853] | 33.333% | 64.583% |
| no-persistence history `.75` | 2.4129 | -0.22% | [-0.0957, 0.0882] | 59.375% | 39.583% |
| per-sequence best fixed-grid ceiling | 2.0197 | 16.11% | [0.2931, 0.4933] | 0% | 75.000% |

主效应：

```text
population - correct .75 = +0.24001
95% CI = [0.13935, 0.34784]
positive / tie / negative = 67.71% / 4.17% / 28.12%
```

Persistence-specific 效应：

```text
no-persistence .75 - correct .75 = +0.24542
95% CI = [0.12355, 0.36956]
positive / tie / negative = 68.75% / 1.04% / 30.21%
```

当 factor 在 E1→E2 边界改变时，history `.75` 相对 population 的平均收益为 `-0.00541`，约 `-0.22%`，区间跨 0；harm 从 correct-history 的 `28.125%` 上升到 `59.375%`。这不是“任何历史都能帮忙”，而是只有与当前持续物理因素匹配的历史才有价值。

## 4. 外部 F0 诊断

完全不 refit 的 T-F0 在跨 shape 上相对 population 改善 `11.10%`，并比 fixed `.75` 平均再降低 cost `0.02733`，paired CI `[0.00167,0.05384]`。这是有意义的外部迁移信号。

但 F0 harm 为 `33.333%`，高于 fixed `.75` 的 `28.125%`；相对 fixed `.75` 的配对中，43.75% 更好、26.04% 相同、30.21% 更差。它继续只按 factor 选择 `.5/.75/1=12/24/60`，没有 scene-aware 风险识别。因此当前应描述为“均值上有小幅外部迁移优势，同时负向尾部更差”，不能只报均值宣布 gate 已解决。

## 5. 异质性与结论边界

六个 shape pair 的主效应均为正，mean 范围约 `0.128–0.454`；persistence-specific mean 也全部为正，但 `small_tee→square` 与 `square→Z` 接近 0。八个 factor 中主效应 7/8 为正，`(10°,1.18)` 为负；persistence-specific 6/8 为正。子组样本仅 12 或 16 条，只用于定位异质性，不能据此改规则。

### 已经证明

1. 作者 seed-42 发布池内，低维物理 context 能从一种 shape 的 E1 迁移到另一种 shape 的 E2，并改善冷启动闭环行为。
2. 该收益依赖物理 factor 持续；factor 改变时旧 context 平均不再获益且 harm 显著增加。
3. 因此保存的不是单一 T-shape 轨迹模板，而是至少在当前 benchmark 中可跨任务复用的物理标定信息。

### 尚未证明

1. 不能外推到新的 target-generation seed、视觉分布、checkpoint、planner 预算或真实机器人。
2. 即使 correct history 平均有效，仍有 `27/96` 条 fixed `.75` 受损；安全选择问题未解决。
3. F0 的均值优势伴随更高 harm，尚不能取代 fixed `.75` 作为稳健默认。
4. 尚未验证显式 context 与 episode-local AdaJEPA/TTT 是否互补。

## 6. 下一步

最小下一步不是扩大模型，而是对这 96 条做 outcome-exposed、明确标记为 exploratory 的失败归因：区分 posterior 估计误差、shape/goal geometry、候选 action 差异和模型 rollout disagreement，使用 leave-one-shape-pair-out 检查信号是否稳定。目标是冻结一个低容量、决策前可计算的 harm-aware selector 或 safety veto。

只有该只读分析得到跨 shape-pair 稳定信号后，才使用剩余作者 segments 建立新的互异 train/dev/formal，前瞻比较 risk-aware policy、fixed `.75` 与 external F0。当前 96 条 formal 从此只能作为探索/训练证据，不能再次充当盲测。

在同 benchmark 的稳健默认仍是 fixed `.75`；external F0 可作为均值优先的候选，但必须同时报告更高 harm。随后再测试 `explicit context + episode-local adaptation`，不得回到无条件跨任务权重 carry。
