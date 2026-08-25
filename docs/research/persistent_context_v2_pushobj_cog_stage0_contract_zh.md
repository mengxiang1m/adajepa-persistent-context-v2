# PushObj 水平质心 CoG Simulator-Oracle Stage 0 冻结合同

合同 ID：`persistent-context-v2-pushobj-horizontal-cog-simulator-oracle-stage0-v1`  
冻结日期：2026-08-22；在本合同任何新 CoG prior/oracle 行为结果产生前冻结。

## 1. 研究问题与阶段边界

研究问题：当 PushObj T 物体的水平 center of gravity 在多个 episode 中持续、但不从初始 observation 泄漏时，知道真实 CoG 的 planner 是否能在 10-action early waypoint 任务中获得更低的闭环位姿代价。

本 Stage 0 只做任务资格验证，不训练 predictor、不使用 history estimator。population-prior 与 true-factor oracle 都使用相同的 ground-truth Pymunk physics CEM；唯一差异是规划模拟器使用 `cog_x=0` 还是环境真实 `cog_x`。执行环境始终使用真实 factor。

主零假设为 `pose_auc10(prior)-pose_auc10(oracle)` 的配对均值不为正。不设置效果量、同向比例或 CI 自动裁决门。

## 2. 为什么选择 CoG 而非 friction

只读实现审计显示，当前发布 PushObj 中实际参与碰撞的 agent/block Pymunk shapes 的 friction 均为 `0.0`；代码中的 `body.friction=1` 只是 body 上的 Python 属性。把 shape friction 改成非零会同时改变 nominal benchmark 接触模型，不能直接解释为围绕既有 factor 的单变量变化。

CoG 则由环境原生 `block_cog` 支持。冻结前的只读机制检查确认：在相同非零角度初态下改变水平 CoG，初始 state、agent proprio 和视觉像素完全相同；因此 factor 不从单帧泄漏。该检查不读取本合同冻结场景上的 oracle行为。

## 3. Factor、split 与场景

- T 物体 nominal CoG 为 `(0,45)`；唯一 factor 是水平分量 `cog_x`，垂直分量固定 `45`。
- train support `[-30,-15,0,15,30] px`，population prior 为均值 `0 px`。
- development factors `[-22.5,-7.5,+7.5,+22.5] px`，每档 8 个场景。
- 使用 development indices `[0,500)` 中 nominal step-10 block displacement `>=10 px` 的 early-contact pool。
- 排除此前所有 PushObj Stage 0 已观察行为的 182 个 development segments后剩余 173 个；用 `default_rng(1070000)` 冻结选择 design JSON 中32个，从未在 CoG prior/oracle 下观察行为。

## 4. Waypoint、planner 与预算

- initial state 与发布 segment 相同。用 population CoG `(0,45)` 重放发布前10个动作，第10步 block pose为 waypoint。
- planner使用无渲染但与标准 `env.step` 状态逐步精确一致的 Pymunk rollout；冻结机制检查的 max state difference为 `0.0`。
- CEM以发布10-action sequence为初始 mean；128 samples、top16、5 rounds、初始每维 sigma `0.2`。candidate 0固定为当前 mean，不裁剪动作；objective就是冻结的 `pose_auc10`。
- prior与oracle共享初始 mean、sample count、top-k、rounds和 RNG seed。两者各输出10个动作，并在真实 CoG环境执行10步；不做 MPC、history、TTT或额外动作。
- simulator oracle是上游任务 ceiling，不等价于已实现 factor-conditioned neural predictor。只有结果支持行为价值后，才允许另立训练合同。

## 5. 指标与统计

主指标仍为状态1–10相对 waypoint的平均：

```text
block_position_error/20 + wrapped_block_angle_error/(pi/9)
```

报告 prior/oracle mean、mean delta、相对变化、positive/tie/negative、20,000 paired bootstrap CI（seed `1070301`）、按 CoG factor异质性。辅助报告 deadline success、step-10 position/angle error、command变化和 simulator预测/真实执行一致性。

## 6. 工程审计、停止与禁止动作

- 32场景唯一、factor平衡、waypoint displacement合格，design/data hash记录并匹配。
- 所有 factors 的 initial visual/proprio/state identity最大差异 `<=1e-6`。
- nominal CoG下无渲染 rollout与标准 env step identity `<=1e-6`。
- independent audit重新执行选中命令，核对 prior/oracle raw states、true CoG、pose metrics、hash、CEM轮数、candidate预算和配对。
- CPU运行记录 wall time、RSS、命令与退出码；raw JSONL append-only并可 exact复算summary。
- smoke/formal 后不得改 factor、segment、CEM预算、sigma、waypoint、主指标或seed；科学负结果不得改称INVALID。
