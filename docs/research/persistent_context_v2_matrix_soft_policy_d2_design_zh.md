# Persistent Context V2：Matrix soft-policy D2 探索设计

日期：2026-08-23  
设计 ID：`persistent-context-v2-matrix-soft-policy-d2-exploratory-v1`  
状态：分析前冻结；不是 formal 合同

## 1. 唯一问题

D1 已证明固定中间强度可以降低 full-context harm。D2 只问：

> 一个低容量、只读决策前信息的策略，能否在旧数据的三次 split rotation 中，比固定 `α=0.75` 更好地选择每条 sequence 的 context 强度？

D2 不生成新环境 outcome，不改变 context estimator、world model、planner 或 α 网格，也不把旧 formal 恢复为 formal。

## 2. 冻结输入与处理

- outcome：D1 的 96×5 个 `pose_auc10`；
- feature：D0 已保存的 planning-before-outcome 特征；
- α 网格：`{0,0.25,0.5,0.75,1}`；
- 标签：每个非零 α 的 `pose_auc10(α=0)-pose_auc10(α)`；
- 主特征 F2：D0 的 18 个 factor、geometry/action 与交叉 rollout 特征；
- 机制消融 F0：前 6 个 factor-only 特征；
- 不读取 segment/sequence ID、seed、真实 E2 state/contact/success、D1 best-α 或其他 outcome-derived feature。

F0/F2 使用完全相同的模型族和选择程序。F0 是预先固定的消融，不与 F2 竞争成为事后主方法。

## 3. 冻结模型

对每个 sequence 特征 `x` 和候选强度 α，拟合共享 ridge 剂量面：

```text
predicted benefit(x, α)
= α · xβ_linear + α(1-α) · xβ_curvature
```

这个形式保证 `predicted benefit(x,0)=0`，允许曲线弯曲，但不为每个 α 训练独立模型。非 intercept 特征只用当前 fit split 的均值/标准差标准化。两个 intercept 系数不惩罚，其余系数使用同一个 ridge λ。

候选 λ 固定为 `{0.01,0.1,1,10,100,1000}`。train 拟合各候选，dev 只按实际 policy mean delta 选择 λ；完全相同时选择更大的 λ。随后在 train+dev 上重新标准化并 refit，一次性评价 test。

策略计算五个 α 的 predicted benefit，选择最大者；完全并列时选较小 α。没有额外 threshold、risk penalty、temperature、harm classifier 或事后 α 平滑。

## 4. Split rotation

固定三次：

1. train→dev→formal；
2. dev→formal→train；
3. formal→train→dev。

三个 split 都是旧探索数据。任何一轮结果都不能单独称为 held-out formal 证据。

## 5. 冻结对照和报告

每轮同时报告：

- population `α=0`；
- fixed `α=0.5`；
- 主固定对照 `α=0.75`；
- full context `α=1`；
- F0 factor-only soft policy；
- F2 task-interaction soft policy；
- per-sequence best α 只作不可部署 ceiling。

主比较为 test 上 `cost(fixed α=0.75)-cost(F2)` 的 sequence-level mean、paired bootstrap 95% CI、harm fraction 差和 α 选择率。另报告 F2 相对 population 的 mean delta/harm，以及 F0/F2 差。

最终只描述三轮方向和异质性，不设自动 GO/NO-GO 阈值。若 F2 不能在多轮中稳定超过固定 `α=0.75`，不再提高 gate 维度；新 formal 应优先验证固定 `α=0.5/0.75`。若 F2 多轮方向一致，再单独冻结全新数据合同。

## 6. 有效性与审计

- D0 与 D1 必须各自完整为 96 条 sequence，join key 唯一且完全匹配；
- 读取的 feature name 必须与 F0/F2 allowlist 完全一致；
- 独立 audit 不 import D2 分析实现，从 D0/D1 raw 复算 design matrix、λ 选择、refit、test decision 和 summary；
- 绑定 source snapshot、D0/D1 raw hash、设计 hash和输出 hash；
- 分析仅使用 CPU，失败输出保留在新的 retry/repair 目录。
