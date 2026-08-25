# Matrix F0 Soft-Policy 前瞻正式实验结果

日期：2026-08-23  
合同：`persistent-context-v2-matrix-f0-soft-policy-formal-v1`  
证据状态：**VALID；主比较方向正向但区间跨 0**

## 1. 这次真正检验了什么

本实验只使用作者发布的 `val_T/plan_targets.pkl`（seed-42、1000 个预采样 T-shape segments）。数据审计先排除 nominal step-10 block displacement `<10` 的片段，以及已经进入旧 matrix outcome 的片段；剩余 413 个合格未暴露片段中，冻结 384 个互异片段组成：

- train：64 sequences / 128 segments；
- dev：32 sequences / 64 segments；
- formal：96 sequences / 192 segments；
- reserve：29 segments。

每条 sequence 用 E1 轨迹估计 rotation×gain posterior，再在全新的 E2 任务上比较 `α∈{0,0.25,0.5,0.75,1}`，其中 `M_α=(1-α)M_population+αM_posterior`。F0 仅用六维低容量 factor 特征 `[1,g,r,g²,gr,r²]` 选择 α。主比较在看到 formal outcome 前冻结为：

```text
pose_auc10(fixed α=0.75) - pose_auc10(F0 soft policy)
```

正值表示 F0 更好。该实验是同一作者发布池内、对 matrix treatment 未暴露的新 scene split；它不是新的轨迹生成分布。

## 2. 运行与审计事实

- train/dev 完整采集后，dev 选择 `ridge=1.0`；锁定模型 SHA256 为 `e16c4823e076ef37d4ab715e48cf130ebdf95ad1706ba31b17fffd5fc793b555`。
- formal 决策在 E2 outcome 产生前完成，被选 α 总是第一个执行；随后才执行其余分支用于严格配对评价。
- train/dev/formal 分别完成 `64/32/96` 条，每条五个 α，世界模型参数运行前后 hash 不变。
- 独立审计不 import 主 runner，重算 posterior、F0 特征、模型预测、α 决策、执行顺序、轨迹 `pose_auc10` 和 bootstrap 汇总；所有最大绝对误差均为 `0`，decision/execution mismatch 均为 `0`。
- 有效源码快照 SHA256：`71f638855c856377914b3fc6264d0f6a835fb5b2bf54516118136e721b9f2695`，包含 325 个源码/配置文件。
- formal raw SHA256：`73e5accd64b56a016b4173160474efa4d29ea05af3be011f86cc90c99cdffa37`；summary SHA256：`5597c4b3a72f755cb152d940c52f8fefbcb11190fdc8a280a81f5288c0067c10`；audit SHA256：`d3a7b01f79ed6df483b29b9f3da32890191a8c4febb3f53e0fda84782f50ea5d`。

远端主产物位于：

```text
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_matrix_f0_soft_policy_formal_v1/
```

## 3. 正式结果

`pose_auc10` 越低越好。所有区间均以 sequence 为统计单位做冻结 bootstrap。

| 方法 | mean cost | 相对 population 改善 | delta 95% CI | harm | 正向比例 |
|---|---:|---:|---:|---:|---:|
| fixed `α=0.5` | 2.0750 | 9.55% | [0.1515, 0.2939] | 21.875% | 77.083% |
| fixed `α=0.75` | 2.0077 | 12.48% | [0.2051, 0.3730] | 21.875% | 77.083% |
| full context `α=1` | 2.0175 | 12.05% | [0.1782, 0.3772] | 23.958% | 75.000% |
| **F0 soft policy** | **1.9908** | **13.21%** | **[0.2124, 0.3963]** | **21.875%** | **77.083%** |
| per-sequence best ceiling | 1.8900 | 17.61% | [0.3304, 0.4818] | 0% | 87.500% |

唯一主比较：

```text
mean[fixed α=.75 - F0] = +0.01686
95% CI = [-0.00865, +0.04122]
```

F0 的平均 cost 比 fixed `α=.75` 低约 0.84%，但区间跨 0；两者 harm 都是 `21/96=21.875%`。因此不能宣称 learned F0 已经优于简单固定收缩。

F0 的选择分布为：`α=.5: 12`、`α=.75: 24`、`α=1: 60`，从未选择 `0` 或 `.25`。同一个 rotation×gain factor 的 12 条 formal sequences 全部得到相同 α，说明当前 F0 实质上是 factor-level 剂量表，而不是 scene-aware 风险选择器。

## 4. 可以下什么结论

### 已经得到支持

1. 从上一任务保存的低维物理 posterior，在新的 T-shape target segment 上具有稳定闭环价值：F0 相对 population 改善 13.21%，区间明确为正。
2. 部分使用 context 是合理的工程控制手段。fixed `.75` 与 F0 都优于 population，且比 full context 的 harm 少 2.083 个百分点。
3. 作者数据上的独立未暴露 scene 复验支持“共享物理 context 有潜力”，不是旧 96 条 outcome 上的纯事后现象。

### 没有得到支持

1. learned F0 相对 fixed `.75` 的额外优势仍不确定；新增模型复杂度当前没有被主比较证明必要。
2. F0 没有降低 fixed `.75` 的负迁移频率，不能称为安全 gate。
3. 实验仍只覆盖作者 seed-42 的 T-shape 发布池，尚未证明跨 shape、跨目标生成分布、跨 checkpoint 或跨 planner 预算泛化。
4. per-sequence best ceiling 的 17.61% 与 0% harm 表明还有可选择空间，但它读取事后 outcome，不可部署，也不能当作方法结果。

最强替代解释是：posterior factor 足以决定大致应使用多少 context，却不足以识别由 start/goal/contact geometry 造成的单场景负向尾部。此前 D0 的任务特征虽然提高相关性，但 F2 没有稳定改善行为；所以当前证据不支持立刻堆更大的 gate。

## 5. 后续状态：跨任务复验已完成

作者其他六个 shape 的 96 条前瞻 formal 已完成。Correct-history fixed `.75` 跨 shape 相对 population 改善 `9.97%`，主 delta CI `[0.1393,0.3478]`；no-persistence history 平均改善 `-0.22%`，且 persistence-specific paired delta CI `[0.1236,0.3696]`。完整结果见 [`persistent_context_v2_cross_shape_matrix_history_formal_results_zh.md`](./persistent_context_v2_cross_shape_matrix_history_formal_results_zh.md)。

在该 formal 前不根据本次 96 条 formal outcome 修改 F0。当前 96 条可用于后续明确标注为 exploratory 的失败归因，但不能再次充当盲测集。

当前同 benchmark 的默认简单基线应是 fixed `α=.75`：它与 F0 的正式差异尚不确定、实现更简单。是否继续投入 scene-aware safety selector，应由跨-shape结果和失败归因共同决定。
