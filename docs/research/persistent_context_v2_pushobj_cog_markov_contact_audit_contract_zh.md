# Persistent Context V2：PushObj CoG Markov/contact 表示审计合同

状态：`FROZEN-DEVELOPMENT-REPAIR1`  
合同 ID：`persistent-context-v2-pushobj-cog-markov-contact-audit-v1`  
冻结日期：2026-08-24  
性质：开发集表示诊断；不是新的闭环正式结果，不读取或复用 CoG formal split。

## 1. 审计问题

现有 CoG v1 predictor 在 factor-diverse 训练后仍只回收了少量 physics-oracle 闭环价值；把同样的 7 维轨迹输入换成 GRU 也没有改善。当前 7 维状态只有 agent 位置、block 位置/角度和 agent 速度，缺少 block 线速度、block 角速度和接触状态。

本审计只回答两个问题：

1. 缺失的 block 动量状态是否能解释 held-out factor/segment 上的一步或轨迹残差？
2. nominal simulator 可记录的 agent-block 接触几何/冲量是否在 block 动量之外仍提供可复现的解释力？

本审计不回答“历史能否估计 CoG”，也不授权训练 CoG history encoder。

### 冻结前 smoke 修订记录

初版合同/设计 hash 为 `473e87d4...` / `80675164...`。2-train/2-eval segment 的程序 smoke 未满足完整 split，结果明确无科学效力；它发现四折 CV 在 segment 少于四时会产生空折，并发现三级 ridge 对照不能直接检验新增字段是否修正冻结 v1 的误差。Repair1 在完整数据读取前固定：smoke 使用 `min(4, n_train_segments)` 折；增加下述冻结 v1 residual correction 对照。旧合同、设计和 smoke 原始目录永久保留。

## 2. 固定假设与三级表示

- `R0 legacy`：与 v1 相同的 nominal 7 维状态轨迹、命令与 CoG 条件。
- `R1 Markov`：R0 加每个控制边界的 block 线速度与角速度。
- `R2 nominal-contact`：R1 加 population-CoG nominal rollout 每个控制步内的 agent-block 接触计数、接触几何和求解冲量摘要。

主要假设：若旧表示缺失关键 Markov/contact 状态，则固定低容量、零上下文严格恒等的回归器在 `R1` 或 `R2` 上对 held-out 数据的 CoG residual 误差应连续下降。

`R2` 只允许使用 population-CoG nominal rollout 的接触摘要；真实 factor rollout 在同一转移内产生的碰撞字段严禁作为预测输入，只能用于事后误差分层。

## 3. 固定数据与独立单位

数据源：作者发布的 `plan_targets.pkl`。只使用既有 v1 predictor 的 train/dev segment：

- 拟合集：v1 train 列表的前 24 个 segment；CoG `[-30,-15,0,15,30]`。
- 评估集：v1 dev 列表的前 16 个 segment；CoG `[-25,-10,10,25]`。
- 每个 segment 固定 4 个命令版本：原命令，以及高斯噪声标准差 `0.08/0.16/0.24` 各一个。
- formal segment、Stage-0 oracle segment 和矩阵实验 outcome 均不读取。

segment 是统计独立单位。转移和 action variant 只增加同一 segment 内的观测，不能当独立样本计算区间。

## 4. 固定采集字段

控制边界状态：

- agent position / velocity；
- block position / wrapped angle；
- block linear velocity / angular velocity；
- CoG、mass、moment（常量仅进审计清单，不作为可学习捷径）。

每个控制步内按碰撞对分别聚合：

- callback 数、contact point 数、首次接触数；
- canonical block impulse 的 x/y、绝对分量、范数和最大范数；
- 接触法向的 impulse-weighted x/y；
- 接触点相对 block 中心的均值与半径；
- penetration distance 最小值；
- solver `total_ke` 的和与最大值。

主输入只用 agent-block 的 nominal 接触摘要；block-wall、agent-wall 字段保留作审计，不进入主比较。

## 5. 固定预测任务和模型

目标沿用 v1：true-CoG rollout 相对 population-CoG nominal rollout 的 10 步 block position/angle residual，位置除以 20，角度除以 `pi/9`。

回归器固定为多输出 ridge，不训练神经网络。所有表示使用同一显式特征映射：

`[c, c^2, c*x, c^2*x]`，其中 `c=CoG_x/30`，`x` 仅用拟合集均值/标准差标准化。该映射保证 `c=0` 时 residual 严格为零。

ridge 候选固定为 `[1e-6,1e-4,1e-2,1,100]`，仅在拟合集按 segment 做固定四折选择；随后锁定并在完整拟合集重拟合。评估集只读一次。

同时用冻结的 v1 checkpoint 在同一评估轨迹上计算参考误差；不更新 checkpoint。

为直接检验新增字段能否解释 v1 的剩余误差，增加两个 secondary correction：

- `v1+C1 Markov`：冻结 v1 输出，加只使用 33 维 block velocity/angular-velocity 轨迹增量的 ridge correction；
- `v1+C2 Markov-contact`：冻结 v1 输出，加上述 33 维以及 170 维 nominal agent-block contact 摘要的 ridge correction。

correction 目标固定为 `true residual - frozen-v1 residual`，同样使用零上下文恒等映射和 train-segment 四折选择；评估时 v1 权重不变。该分析是直接误差归因，但仍不是闭环结果。

## 6. 固定指标与统计

主要诊断指标：每条轨迹 10 步平均 pose error：

`mean(position_error/20 + abs(angle_error)/(pi/9))`。

报告：

- v1、R0、R1、R2、v1+C1、v1+C2 的评估均值；
- R0-R1、R1-R2、R0-R2 的逐 segment 均值差、95% segment bootstrap 区间、正/平/负 segment 数；
- v1-(v1+C1)、(v1+C1)-(v1+C2)、v1-(v1+C2) 的同样配对比较；
- normalized residual MSE（描述性）；
- 按真实 rollout 是否发生 agent-block 接触，以及真实冲量高低的事后分层；
- v1 逐步误差与 block speed、angular speed、真实接触冲量的 Spearman 相关（描述性，不作独立检验）。

bootstrap 固定 20,000 次，seed `1_310_300`。平局容差 `1e-12`。

## 7. 有效性、身份与可复现性

必须通过：

- 重新采集的 legacy 7 维状态与原 `rollout_physics` 最大绝对差 `<=1e-6`；
- 同一轨迹重复采集的所有状态/接触数组逐字节 hash 相同；
- segment train/eval 无交集，未出现任何 formal segment；
- R0 在 `c=0` 输出严格为零；
- v1 checkpoint、数据、设计、源码快照 hash 齐全；
- 评估期无 optimizer、running-stat、replay 或 RNG 隐式更新。

实现、身份、split 或原始产物失败才记为 `INVALID`；误差不下降是有效负结果。

## 8. 解释边界与后续

- `R1` 改善：支持“旧观测非 Markov”这一局部解释，但不等于闭环已改善。
- 只有 `R2` 改善：说明接触相位/几何重要；真实事后冲量仍不能直接作为部署输入。
- 都不改善：不继续堆更大的 history encoder；优先复核动力学模型形式、时间分辨率和 oracle 价值是否能被该表示承载。
- 任一表示改善后，下一道门仍是用 true CoG 做固定预算闭环验证；只有该门成立，才设计非特权历史估计。

不使用固定百分比或区间过零作为自动 GO/NO-GO。用户根据完整连续证据决定是否继续投入。
