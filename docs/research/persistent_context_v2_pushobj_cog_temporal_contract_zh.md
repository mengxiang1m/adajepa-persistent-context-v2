# PushObj CoG Temporal Predictor V2 冻结合同

合同 ID：`persistent-context-v2-pushobj-cog-temporal-film-predictor-v2`

## 唯一研究问题

在不增加第二种 context 机制、不增加 history encoder、gate、TTT 或额外环境动作的条件下，把 CoG predictor v1 的 flattened trajectory MLP 替换成保留 10 步顺序的 causal GRU，是否能在全新场景上进一步降低 true-CoG prediction error，并比 v1 回收更多 physics-oracle 闭环收益？

## 冻结数据

- 训练和开发数据逐字节复用 v1 的 `7680/768` factor-diverse samples；输入和 target hash 冻结在 design JSON。
- checkpoint 只能按 v1 已冻结的 held-out-factor dev prediction MSE 选择，不能读取旧 formal 或新 formal 的行为方向。
- 新 formal 使用 32 个此前未进入 CoG Stage 0、v1 train/dev/formal 的 early-contact segments。固定 seed 与确切列表写入 design JSON。
- formal factors 仍为 `[-22.5,-7.5,+7.5,+22.5]`，每档 8 pairs；这些数值没有在 train factors `[-30,-15,0,15,30]` 中逐点出现。
- v1 已观察的 32 个 formal pairs 不作为 v2 的正式证据，也不用于 checkpoint 选择。

## 冻结模型

- 每一步输入由 nominal encoded state `t`、nominal encoded state `t+1` 和 command `t` 组成，共 18 维。
- `Linear(18,64)+SiLU` 后进入单层、单向、hidden 128 的 causal GRU。
- 唯一 context 路径为标量 `cog_x/30` 生成的一组 FiLM scale/shift，作用于每步 GRU hidden。
- 共享逐步 head 同时计算指定 context 与 zero context，两者相减输出 10 步 `(block_dx,block_dy,block_dangle)` residual；因此 `context=0` residual 构造上严格为 0。
- 不使用 bidirectional RNN、Transformer、第二个 adapter、router、expert、contact label、true simulator rollout 或未来真实 state。

## 冻结训练

- AdamW，3000 steps，batch 256，learning rate `5e-4`，weight decay `1e-4`，gradient clip norm `1.0`。
- loss 为全部 10 步归一化 pose residual MSE；每 100 steps 评估 frozen dev，以最低 dev true-context MSE 保存 checkpoint。
- 允许的工程修复仅限非有限值、加载、设备精度和实现一致性问题；不允许看 formal 结果后修改架构、训练 seed、step、factor、场景、CEM 或主指标。

## 正式比较

同一组 32 pairs 比较四个策略：

1. `population_prior_context`：v2 使用 `cog_x=0`，严格等于 nominal simulator residual；
2. `v1_true_cog_context`：冻结 v1 checkpoint 使用真实 CoG；
3. `v2_temporal_true_cog_context`：冻结 v2 checkpoint 使用真实 CoG；
4. `simulator_oracle`：规划时直接使用真实 CoG Pymunk，只作为上限。

四个策略共享真实 factor、初态、waypoint、环境 seed、CEM seed、初始 action mean 和预算：128 samples、top-16、5 rounds、10 actions。所有策略最终在真实 CoG 环境执行。

主指标为真实执行的 `pose_auc10`。连续报告：各策略均值、v2-population、v1-population、v2-v1、oracle-population 的 paired delta、相对变化、bootstrap 95% CI、正/平/负比例、factor 分组、deadline success、prediction-execution pose error、plan-change 和 oracle-gap recovery；不设置固定效果百分比门槛。

## 审计与解释边界

- 独立审计必须验证 design/data/v1/v2/raw hash、checkpoint 选择、split 排除、factor balance、zero-context identity、CPU 重算 prediction、环境执行重放、metric 与 summary。
- 正向结果只说明时序结构在当前 state-based CoG residual predictor 中有额外价值，不等于视觉 AdaJEPA 已支持 CoG。
- 若 v2 true-context 行为仍弱或不稳定，不进入复杂 history encoder；先报告剩余 prediction/planner 瓶颈。
- 若 v2 在新场景稳定优于 population 且相对 v1 增加 oracle-gap recovery，下一阶段才冻结非特权 CoG history estimator 合同。
