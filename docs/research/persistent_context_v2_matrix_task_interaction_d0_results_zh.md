# Persistent Context V2：Matrix task-interaction D0 探索结果

日期：2026-08-23  
证据等级：探索性复用旧结果；不是新的 formal 证据

## 1. 一句话结论

决策前的场景、候选动作和模型交叉 rollout 特征确实包含“context 在本场景是否有利”的信息，但当前 `predicted delta > 0 → 100% 使用 posterior context` 的硬开关没有把这些信息转化为更好的控制结果。它使 F1/F2 在主 split 上退化为全选 context，伤害率从 factor-only F0 的 `21.875%` 回到 `31.25%`。

因此没有启动原计划中的新 V2 formal。后续 D1 已完成固定强度的 soft-context 剂量实验，证明部分 context 可明显减少 harm；结果见 [`persistent_context_v2_matrix_soft_context_d1_results_zh.md`](./persistent_context_v2_matrix_soft_context_d1_results_zh.md)。

## 2. 本次做了什么

复用 Matrix learned-gate V1 的 train/dev/formal 各 32 条 sequence，只读重建 E2 决策前信息。旧 formal 在本分析中降级为 exploration，不能再次充当新 formal。

比较三个固定层级：

- F0 factor-only：V1 的 `[1,g,r,g²,gr,r²]`；
- F1 geometry/action：增加 agent→block、block→goal、几何夹角、目标角误差，以及 population/context 两套候选 action 的差异；
- F2 model-interaction：再增加四个交叉 rollout 的偏好与 context sensitivity 特征。

所有特征均在 E2 真实 outcome 产生前计算。标签只在事后分析阶段读取，模型仍是标准化 ridge；train 拟合、dev 选 alpha、train+dev refit，旧 formal 只作探索性评价。

## 3. 完整性和可复现性检查

- 96/96 条 sequence 成功抽取，train/dev/formal 各 32 条；
- 初态与目标重放最大误差均为 `0`；
- 只读特征重放前后 model-state hash、Python/NumPy/PyTorch/CUDA RNG digest 均不变；
- 独立 audit 不 import D0 实现，重新计算 96 条特征与 outcome，最大误差均为 `0`；
- 特征 hash：`a36d6e31ede28699cfd22b663c364a473345ea751ad597b19a94ac040accb706`；
- 抽取源码快照 SHA256：`4190fbda267a71ed4ab1041dbaa0baf731ca26db433f20d419648fb416474635`；
- split-rotation 分析源码快照 SHA256：`55cbb8bd047d19a5a2ca21edf5bf73975700c5592595f239a0956ea9a1ccaad3`；
- leakage allowlist 通过，没有 forbidden feature name；
- posterior covariance trace 在 96 条中实际上为常量，约 `1.99967e-05`，所以本数据不能检验“按 posterior uncertainty 调节 context”。

运行使用远端单张 NVIDIA L40（physical GPU 0），96 条抽取 wall time `20.73 s`，峰值 RSS `1,307,736 KiB`，PyTorch peak allocated/reserved 分别为 `423,726,592/455,081,984 bytes`。退出码为 0，运行后 GPU 已释放。

## 4. 结果

### 4.1 特征中确实有信号

旧 formal 的 prediction correlation 随特征增加而上升：

| 特征集 | correlation(predicted delta, true delta) |
|---|---:|
| F0 factor-only | 0.5327 |
| F1 geometry/action | 0.6371 |
| F2 model-interaction | 0.6674 |

去掉 factor 组均值后的相关性也显示场景交互信息存在，例如 block→goal distance `0.4237`、command RMS disagreement `0.4000`；四个交叉 rollout 特征约为 `+0.4078/+0.3840/+0.3835/-0.4081`。

这支持一个有限结论：同一 factor 内的正负差异不全是随机噪声，决策前特征能解释其中一部分。

### 4.2 但硬 gate 没有改善行为

主 split 为 train→dev→旧 formal：

| 特征集 | 选中 alpha | context 选择率 | harm fraction | mean delta | 相对 population 改善 | 相对 F0 mean delta |
|---|---:|---:|---:|---:|---:|---:|
| F0 | 1 | 87.5% | 21.875% | 0.3323 | 12.29% | — |
| F1 | 100 | 100% | 31.25% | 0.3053 | 11.29% | -0.0270 |
| F2 | 100 | 100% | 31.25% | 0.3053 | 11.29% | -0.0270 |

F1/F2 相对 F0 的 paired bootstrap 95% CI 为 `[-0.0912,+0.0212]`。方向偏负且区间跨 0；不能说它稳定更差，但更不能说它改善了 gate。

### 4.3 split rotation 也不支持冻结 V2 formal

把三个旧 split 轮流作为 train/dev/test：

| train→dev→test | F1 vs F0 mean delta | F2 vs F0 mean delta |
|---|---:|---:|
| train→dev→formal | -0.03 | -0.03 |
| dev→formal→train | -0.02 | -0.02 |
| formal→train→dev | -0.01 | +0.01 |

F1 三次均未超过 F0；F2 只有一次约 `+0.01`，另两次为负。当前 96 条样本不足以稳定校准高维硬 gate。

## 5. 怎么理解这个结果

这里不是“context 没用”。always-context 仍比 population 平均好约 `11.29%`，说明 context 的平均价值存在。问题是它同时伤害 `31.25%` 的场景，而我们目前的 gate 只有两个动作：完全不用或完全使用。

新增特征提高了连续 outcome 的排序相关性，却没有把零阈值分类校准好。小样本、高维、组内噪声和正向标签占多数共同把 ridge 推向“几乎都用 context”。因此失败点更像是决策映射与 context 强度，而不是信息完全缺失。

最强替代解释是：这些相关性可能仍是旧 96 条数据上的有限样本现象；在独立新数据上未必复现。正因如此，本结果不授权直接扩大成新的 320-sequence formal。

## 6. 当前决定与下一步

1. 不冻结 F1/F2 `predicted delta > 0` 硬 gate，不启动原规划的新 V2 formal。
2. 先做 D1 soft-context 探索：在同一 posterior 下固定比较 `α∈{0,0.25,0.5,0.75,1}`，其中 `M_α=(1-α)M_prior+αM_posterior`。
3. 检查中间 α 是否保留大部分平均收益，同时降低 full-context 的 harm；端点 α=0/1 必须复现 population/always-context。
4. 只有剂量曲线显示可重复的中间强度价值后，才讨论学习逐场景 α；否则回到更保守的 fallback/calibration，或收集新的 train 数据。
5. uncertainty 路线需要新设计的数据：改变 evidence 数、noise 或 posterior sharpness。现有 96 条 covariance 常量数据不能回答这个问题。

## 7. 产物

远端主目录：

```text
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_matrix_task_interaction_d0_exploration_v1/
```

关键文件：`features.jsonl`、`manifest.json`、`analysis_split_rotation.json`、`independent_audit.json`。两次 smoke 启动失败均保留为失败产物，未覆盖或删除。
