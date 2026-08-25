# Matrix Rollout Disagreement D4 开发可行性合同

合同 ID：`persistent-context-v2-matrix-rollout-disagreement-d4-development-v1`

## 研究问题

跨形状正式实验中，正确的旋转—增益上下文平均有效，但固定 `α=0.75` 仍使 `27/96` 条序列受损。本实验检验：在 E2 执行前，默认模型和上下文模型对同一候选动作的预测轨迹分歧，能否给这些受损场景排序。

本实验是前瞻开发可行性研究，不是新 formal，也不验证在线 gate。结果只决定是否值得建立新的 train/dev/formal gate 合同。

## 数据与统计单位

- 只使用作者 seed-42 发布的 `val_I/val_L/val_+/val_small_tee/val_square/val_Z`。
- 排除既有跨形状 smoke、formal、reserve 的全部 segment hash 和 `ep_idx:offset`。
- 新选 6 条 smoke 和 96 条 development sequence；每条包含互异的 E1、E2 segment。
- Development 使用 `6 shape pairs × 8 factors × 2 replicates = 96 sequences`，sequence 是独立统计单位。
- 抽样只依据位移 eligibility、重复审计、历史排除集和冻结 seed，不读取任何新 E2 outcome。

## 唯一行为处理

E1 与 E2 的真实 rotation×gain factor 相同。E1 使用 population planner 产生 command/state history，非特权 Bayesian estimator 得到 posterior。E2 严格配对比较：

1. `population`：使用默认旋转—增益矩阵；
2. `context_0.75`：使用 `0.25 × population + 0.75 × E1 posterior matrix`。

两分支共享 E2 shape、segment、初态、目标、真实 factor、环境/CEM seed、checkpoint、规划预算和动作预算。执行顺序按 sequence 奇偶平衡；每个分支重新准备环境。

## 决策前预测轨迹

在任何 E2 真实动作执行前，先规划 population 和 context 两套 command，再只读保存四条完整 latent rollout：

```text
Z(prior model, population commands)
Z(prior model, context commands)
Z(context model, population commands)
Z(context model, context commands)
```

同时保存 goal latent、四个冻结 objective score、逐模型步 latent RMS 分歧、完整数组 shape/dtype/hash 和压缩 NPZ。预测探针不得更新模型、running statistics、optimizer、replay 或 RNG；探针前后模型 hash 和 RNG digest 必须一致。

## 冻结特征与端点

每条 sequence 的真实 benefit 定义为：

```text
benefit = pose_auc10(population) - pose_auc10(context_0.75)
harm = benefit < 0
```

主风险分数在结果产生前固定为：

```text
context_plan_context_shift_rms
= RMS[Z(context model, context commands) - Z(prior model, context commands)]
```

方向预注册为“分数越大，风险越高”。主报告为该分数识别 `harm` 的 sequence-level ROC AUC 及 bootstrap 95% 区间，同时报告它与 `-benefit` 的 Pearson/Spearman 相关、四分位 harm、完整 unit values。

预定义次要分数只包括：population-plan context shift、两种模型下的 plan separation、对应 final-step RMS，以及四个 objective score 的四个固定差值。所有分数完整报告，不按结果择优改主分数。

低容量 ridge 只作次要机制分析：固定使用全部预定义分歧特征、固定 ridge 网格和 leave-one-shape-pair-out 连续 benefit 预测；零阈值策略只作 shadow 描述，不授权正式 gate。

## 审计与无效条件

- 冻结 design、selection、contract、checkpoint、六个数据文件、作者池 audit、排除 selection 和源码快照 hash。
- 独立审计脚本不得 import D4 collector/analyzer；它需从 command、context、NPZ 和真实 states 重算轨迹特征、指标、AUC、bootstrap 与配对身份。
- smoke 必须通过：population identity、重复 trace exact、模型/RNG 不变、E2 场景配对、预算一致和 trace 时间早于真实执行。
- 只有实现、身份、预算、hash、只读性或 raw 完整性失败才标为 `INVALID`。不利科学结果保持有效。
- 不在 development outcome 上调主分数、方向、阈值、特征集、样本或端点；不以固定 AUC、效果百分比或区间自动决定 GO/NO-GO。

## 结论边界

本实验只能说明作者 seed-42 跨形状开发场景中是否存在可用的 rollout-disagreement 风险信号。它不能证明在线 gate、跨 checkpoint、跨 target seed、视觉输入或真实机器人泛化。
