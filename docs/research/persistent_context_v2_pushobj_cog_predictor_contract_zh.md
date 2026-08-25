# PushObj CoG 条件预测器冻结合同

合同 ID：`persistent-context-v2-pushobj-cog-film-residual-predictor-v1`

## 唯一问题

在与上一 CoG Stage 0 完全不同的 train/dev/formal 初始状态上，使用 factor-diverse 数据训练的单一 FiLM 条件轨迹残差模型，能否在未逐点见过的水平 CoG 上利用 true context 改变未来 10 步预测和 CEM 动作，并改善真实 Pymunk 行为？

本阶段只测 true-context 模型使用上限。它不使用历史、不估计 CoG，也不加入 TTT、gate、router、多个 adapter 或 AdaJEPA 权重继承。

## 数据隔离

- 全部样本来自发布的 `val_T/plan_targets.pkl` 中 nominal step-10 block displacement 不低于 10 px 的片段。
- 96 个 train、24 个 dev、32 个 formal segment 按固定 seed 选择，三者互斥，并排除上一 CoG Stage 0 的 32 个 segment。确切列表冻结在 design JSON。
- train CoG 为 `[-30,-15,0,15,30]`；dev 为 `[-25,-10,10,25]`；formal 为 `[-22.5,-7.5,7.5,22.5]`。因此 formal CoG 数值均未在训练中逐点出现。
- 每个 train segment 在每个 CoG 下包含 16 条动作轨迹；每个 dev segment 每个 CoG 8 条。动作由发布的前 10 个动作加固定随机扰动构造，覆盖多种 start、trajectory 与 factor 组合。

## 模型与训练

- 已知 nominal `CoG=(0,45)` 的 10 步 simulator rollout 是 base trajectory；模型只预测真实 CoG 相对 nominal 的 10 步 block `(x,y,angle)` residual。
- 输入为初始状态、动作序列和 nominal rollout；唯一 context 路径是一个标量 `cog_x/30` 生成的 FiLM scale/shift。
- 同一个 head 同时计算指定 context 与 zero context，两者相减，因此 `context=0` 的 residual 构造上严格为零，防止通过损坏 population baseline 制造收益。
- 只用 train transition/trajectory target 优化；每 100 step 在 held-out-factor dev 上评估，以最低 dev trajectory MSE 选择 checkpoint。正式行为数据不参与训练、选择或修复。
- 固定 3000 steps、batch 256、AdamW、learning rate `1e-3`、weight decay `1e-5`。只允许有限值/加载/实现一致性工程修复，不允许看 formal 方向后修改架构、split、factor、预算或指标。

## 正式配对实验

32 个 formal pairs，每个 CoG 8 个。目标 waypoint 是 nominal CoG 重放发布动作到 step 10 的 block pose。所有策略共享初态、真实 CoG、CEM 随机数与预算：128 samples、top-16、5 rounds、初始 sigma 0.2。

1. `population_prior_context`：冻结模型使用 `cog_x=0`；因 zero residual 约束，等同 nominal simulator planner。
2. `true_cog_context`：同一冻结模型使用真实 CoG；规划期间不能调用真实 CoG simulator。
3. `simulator_oracle`：规划时直接使用真实 CoG physics，只作为可回收上限。

三者动作都在真实 CoG Pymunk 中执行。主指标为真实执行前 10 步相对 waypoint 的 `pose_auc10`。连续报告均值、paired delta、相对改善、bootstrap 95% CI、正/平/负比例、按 factor 结果、deadline success、预测误差、动作是否改变，以及 learned 方法回收 simulator-oracle gap 的比例；不设置固定效果百分比门槛。

## 审计与解释边界

- design/contract/data/checkpoint/raw hashes、split 互斥、factor 平衡、样本数量、checkpoint 选择、zero-context identity、模型预测重算、CEM/环境状态重放与 summary 都必须独立审计。
- 正向结果只表明该最小 CoG-conditioned residual predictor 会利用 true context，并能在真实 PushObj simulator 中产生行为价值；不证明 CoG 可从非特权历史估计，也不等于视觉 AdaJEPA checkpoint 已支持 context。
- 无论结果方向如何均保留；下一步是否开发 history estimator 根据连续行为证据决定。
