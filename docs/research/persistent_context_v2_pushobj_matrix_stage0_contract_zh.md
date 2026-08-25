# PushObj Rotation×Gain Matrix True-Factor Stage 0 冻结合同

合同 ID：`persistent-context-v2-pushobj-rotation-gain-matrix-stage0-v1`  
冻结日期：2026-08-22；在本合同任何 prior/oracle 行为结果产生前冻结。

## 命题

在真实 PushObj 10-action early-waypoint 任务中，执行器映射为 `effective = A command`，其中 `A = gain × R(theta)` 在 episode/sequence 内固定。比较只知道训练 population matrix mean 的 planner 与直接知道真实矩阵的 oracle，检验二维 rotation 与幅度 gain 的联合真值是否具有闭环行为价值。

唯一处理是 planner world model 的 2×2 action matrix。环境真矩阵、场景、初态、目标、env/CEM seed、规划和动作预算严格配对。主 delta 为 `pose_auc10(prior)-pose_auc10(oracle)`；不设置数值效果门或自动 GO/NO-GO。

## Population prior 与 factors

- train support 在结果前固定为 rotation `[-30,-15,0,15,30]°` 与 gain `[0.75,1.0,1.25]` 的 15 个笛卡尔积矩阵。
- 所有测试策略共享上述矩阵的逐元素均值 `0.9327804920294028 I`；不得使用 identity 冒充正确 population prior。
- development factors 为 rotation `[-22.5,-7.5,+7.5,+22.5]°` × gain `[0.85,1.15]` 的 8 个组合，每组合 4 个 paired scenarios。
- true matrix 不写入 observation、goal 或 task ID；只供环境和 oracle wrapper 使用。

## 场景与预算

- checkpoint、数据、waypoint 定义、pose cost 与 rotation early-waypoint Stage 0 相同。
- development indices `[0,500)` 中 nominal 第 10 步 block displacement `>=10 px` 才可入池。
- 排除此前 rotation、early-waypoint、dead-zone 和 discrete-delay Stage 0 已观察过行为的 150 个 development segments；剩余 205 个以 `default_rng(1000000)` 固定选 32 个，index 写入 design JSON。
- open-loop latent CEM：2 model steps × 5 low-level actions，200 samples、top 30、10 rounds；环境只执行 10 actions。
- 不做 history、Bayesian update、TTT、MPC 或 episode state carry。本 Stage 0 只测 true-matrix upper bound。

## 指标与统计

主指标 `pose_auc10` 为状态 1–10 的 block position error/20 加 wrapped angle error/(pi/9) 后取均值。统计单位为 paired scenario。报告 prior/oracle mean、delta、相对变化、20,000 paired bootstrap CI（seed `1000101`）、positive/tie/negative fraction、8 个 matrix factor 及按 gain/rotation 聚合的异质性。辅助报告 deadline success、step-10 error 与计划变化比例。

## 工程审计与禁止动作

- design/checkpoint/data hash 匹配；32 pairs 唯一、factor 平衡、waypoint 合格且不复用排除场景。
- identity matrix 下 action transform 与 base/wrapper rollout误差 `<=1e-6`。
- independent audit 从 raw command 和 true matrix 逐元素复算 effective actions、pose metrics、hash、pairing和汇总。
- GPU 0 运行并记录周期显存、框架峰值、wall time、命令与退出码；异常则停止并保留失败。
- smoke/formal 后不得改 prior、factor、场景、seed、deadline、CEM 或主指标；科学负结果不得改称 INVALID。

若上游真矩阵存在闭环收益，下一独立合同才允许评估只从过去 command/proprio transition 得到的 Bayesian 2×2 matrix posterior；本合同本身不产生 history 结论。
