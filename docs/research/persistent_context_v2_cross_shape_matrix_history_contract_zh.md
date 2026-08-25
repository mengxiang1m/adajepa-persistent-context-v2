# Persistent Context V2 跨 Shape Matrix History Formal V1 冻结合同

合同 ID：`persistent-context-v2-cross-shape-matrix-history-formal-v1`

冻结 design SHA256：`616aaf61054ed77e04c08986cf504486b0363ebdef687f24f7529523a46b05b5`  
冻结 selection SHA256：`f7c3b0e21389ede3010a5c9ea05a246f7d865c401f14d5da0c260f5dcd7a7b80`

## 研究问题

作者 T-shape 未暴露 scene 的 P1-V2 已确认低维 rotation×gain history context 相对 population 有稳定闭环价值，但 learned F0 没有可靠优于 fixed `α=.75`。本实验不继续在 T 内调 gate，而前瞻检验：

> 在 E1 与 E2 属于不同物体 shape/任务时，E1 保存的低维物理 posterior 能否改善 E2 冷启动行为；当物理 factor 在任务边界改变时，该收益是否减弱或反转？

唯一科学变化是 E1→E2 的 shape 跨任务迁移及 factor persistence。Checkpoint、Bayesian estimator、factor 表、planner、动作预算和主端点保持冻结。

## 作者数据与科学收缩

只使用作者 seed-42 发布池 `val_I/val_L/val_+/val_small_tee/val_square/val_Z`。只读审计得到 4474 个 nominal step-10 block displacement≥10 的非 T segments；历史 manifests 中带 data path 的记录全部绑定 `val_T`。抽样排除所有在七池中 state+action hash 重复的片段，并要求被选片段的 hash 与 `ep_idx:offset` 在 smoke/formal/reserve 全局唯一。

无需 train/dev 或重新拟合。Formal 取最小完全平衡设计 `6 shape pairs × 8 factors × 2 replicates = 96 sequences`，共 192 个互异 segments。六个固定有向 pair 为：

```text
I→L, L→+, +→small_tee, small_tee→square, square→Z, Z→I
```

另冻结 6 条、每 pair 1 条的 smoke；smoke segments 永不进入 formal。抽样只使用 eligibility、重复审计和冻结 seed，不读取任何新 planner outcome。

## 每条 sequence 的处理

E1 与 E2 使用不同 shape。E2 的真实 factor 为八个 rotation×gain 之一。对同一个 E1 segment、相同初态、目标、population planner commands 和预算生成两份 history：

1. `correct_history`：E1 factor 与 E2 factor 相同；
2. `no_persistence_history`：E1 factor 按 `(factor_index+1) mod 8` 置换，E2 factor 不变。

该置换保持 factor 边际分布，且保证 boundary 前后 factor 不同。两份 posterior 都只读取各自 E1 的真实 command/state transition，不读取 factor label。

E2 在完全相同 shape、segment、初态、目标、true factor、env/CEM seed 与预算下执行：

- correct-history `α∈{0,.5,.75,1}`；
- no-persistence-history `α=.75`；
- 外部 locked T-F0 在 correct posterior 上选择的 α，直接复用相应 correct-history 分支。

外部 F0 模型和审计 hash 在 design 中冻结，不 refit、不改变特征、阈值或 α 网格。F0 decision 必须在任何 E2 outcome 前完成，其 selected correct-history α 第一个执行；其余 unique correct α 升序执行，no-persistence `.75` 最后执行。每个分支都重新 prepare 相同 E2 scene，模型和环境不跨分支学习。

## 端点与统计

唯一主 estimand，以 sequence 为单位、正值为 correct history 更好：

```text
pose_auc10(population) - pose_auc10(correct_history α=.75)
```

关键 persistence-specific estimand：

```text
pose_auc10(no_persistence_history α=.75) - pose_auc10(correct_history α=.75)
```

对两者报告 96 个 unit delta、mean、sequence bootstrap 95% CI、positive/tie/negative fraction 和 harm。次要报告 external T-F0、correct full、fixed `.5`、shape-pair/factor 异质性及事后 alpha ceiling。不得用固定效果量、符号比例或 CI 是否跨 0 自动裁决后续投资。

## 有效性与禁止事项

- Formal 前冻结 design、selection、contract、checkpoint、六个数据文件、作者池 audit、外部 F0 model/audit 和完整 dirty source snapshot hash。
- 世界模型、optimizer、replay、posterior 和 RNG 生命周期分别记录；所有 E2 分支不得更新模型或共享环境状态。
- 独立 audit 不 import collector/evaluator，重算 selection、posterior、F0 features/decision、context matrix、轨迹指标、执行顺序和 summary。
- Formal outcome 产生后不得更改样本、shape pair、factor、alpha、处理顺序、主端点或 bootstrap。
- 本实验能支持作者 seed-42 发布池内的跨 shape/task 推断，不能外推到新 target-generation seed、视觉分布、checkpoint 或 planner 预算。
- 失败/中断目录不覆盖；仅实现、身份、预算或审计失败可标为 INVALID，科学结果不利不属于无效。
