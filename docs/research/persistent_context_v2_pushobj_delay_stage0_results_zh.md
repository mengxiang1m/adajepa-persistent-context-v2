# PushObj 离散 Action Delay True-Factor Oracle Stage 0 结果

日期：2026-08-22  
合同：`persistent-context-v2-pushobj-discrete-delay-stage0-v1`  
正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_delay_stage0/`

## 一句话结果

在全部 32 个 development pairs 上，知道真实 delay 的 oracle 相对固定 2 步 prior 只把主位姿代价改善 `1.4767%`，配对区间跨 0；效果高度依赖真实 delay：`3/4` 步组分别改善 `5.0347%/12.5496%` 且各 `8/8` pairs 正向，而 `0/1` 步组分别退化 `23.7543%/6.6680%`。因此，该任务给出的是“较大 delay 有明确可利用空间、较小 delay 的 exact context 在当前 planner 下反而有害”的条件性证据，不是统一正向的 delay-context 上限。

## 1. 实验实际做了什么

- 在真实 PushObj-T simulator 与冻结 checkpoint 上，引入 episode-reset 的低层动作 FIFO delay：前 `d` 步执行物理零动作，之后执行 `t-d` 时刻的命令。
- planner world model 在完整 10-action 序列上使用完全相同的 delay 变换；FIFO 跨越两个 5-action model blocks，不在 block 边界重置。
- 比较 population prior context `d=2` 与 true-factor oracle context；真实 development factors 为 `0/1/3/4`，每档 8 个场景。
- 32 个场景来自 development `[0,500)` 的 early-contact pool，并排除所有此前看过行为结果的 development segments；每 pair 的初态、waypoint、环境 seed、CEM seed、规划和执行预算完全一致。
- waypoint 是无 delay nominal 发布轨迹第 10 步物体姿态；两臂都只能规划一次并执行 10 个动作，没有 MPC、history、TTT 或当前 episode 在线辨识。

## 2. 冻结主指标

主指标是动作后状态 1–10 到 waypoint 的平均 `pose_auc10`，越低越好；delta 定义为 `prior - oracle`。

| 统计量 | 数值 |
|---|---:|
| pairs | 32 |
| prior mean | 2.790067 |
| oracle mean | 2.748867 |
| mean delta | +0.041200 |
| 相对改善 | +1.4767% |
| paired bootstrap 95% CI | [-0.122580, +0.184245] |
| positive / tie / negative | 18 / 0 / 14 |
| 同向比例 | 56.25% |

这里的客观描述是：总体均值略正，但不确定性较大，区间同时覆盖负效应和正效应。

## 3. Delay 分组异质性

| 真实 delay | prior | oracle | delta | 相对变化 | delta 95% CI | 正向 pairs | deadline prior→oracle |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.603920 | 1.984920 | -0.381000 | -23.7543% | [-0.753015, -0.150405] | 0/8 | 87.5%→75.0% |
| 1 | 2.042297 | 2.178478 | -0.136181 | -6.6680% | [-0.306632, +0.021673] | 2/8 | 100%→100% |
| 3 | 3.473169 | 3.298305 | +0.174864 | +5.0347% | [+0.099624, +0.252354] | 8/8 | 87.5%→100% |
| 4 | 4.040882 | 3.533766 | +0.507117 | +12.5496% | [+0.369398, +0.637451] | 8/8 | 62.5%→100% |

高 delay 组的收益不只来自单一场景：`d=3` 和 `d=4` 各 8 个场景全部正向。低 delay 组则方向相反，尤其 `d=0` 的 8 个场景全部负向。

## 4. 辅助行为结果

- deadline success：prior `84.375%`，oracle `93.750%`，提高 `9.375` 个百分点；配对 bootstrap 区间 `[-3.125,+21.875]` 个百分点。
- step-10 block position error：`12.6013 px → 8.4983 px`，平均减少 `4.1030 px`，CI `[+0.4854,+7.4781]`。
- step-10 angle error：`0.10457 → 0.08958 rad`，平均减少 `0.01499 rad`，CI `[-0.02027,+0.05017]`。
- 32/32 pairs 的 prior 与 oracle command hash 不同，说明 context 全部到达 planner。

## 5. 工程与证据审计

- 正式 runner exit code `0`，独立 audit exit code `0`。
- raw 完整：32/32 unique pairs；四个 factors 各 8 个；所有 waypoint displacement `>=10 px`。
- `d=0` action identity 与 base/wrapper rollout identity 最大误差均为 `0.0`。
- 独立审计的 `count/fifo/hash/manifest/metric/pairing/scenario` failure count 全部为 0；raw 重算 summary 与 runner summary exact match。
- 正式 wall time `43.90 s`；GPU 0 为 NVIDIA L40；周期采样峰值 `3674 MiB`，PyTorch peak allocated `2716.91 MiB`。
- 第一次 smoke 启动在进入 Python 前因后台 shell 变量作用域错误退出；没有产生行为结果。失败目录保留，唯一修复重跑位于 `persistent_context_v2_pushobj_delay_stage0_smoke_repair1/`，exit code 0，未改任何科学配置。

## 6. 证据支持与不支持的判断

事实支持：

- 当真实 delay 大于共享 prior（3、4 步）时，正确 delay context 在当前 early-waypoint benchmark 上有一致闭环收益，而且 delay 越大，均值收益越大。
- exact delay context 不是无条件有益；当真实 delay 小于 prior（0、1 步）时，当前 frozen world-model/CEM 组合更偏好的计划反而获得更高 pose cost。

最强替代解释：冻结 predictor 与 staged CEM objective 本身并非真实 simulator 的完美代理。prior 的错误 delay 有时等价于一种有利的命令整形或保守化，因此“物理参数更准确”不必然等于“该 planner 的闭环行为更好”。这一解释与低 delay 系统性负向、高 delay 系统性正向同时相容。

本实验没有证明：history 能估计 delay；非特权 estimator 能回收高-delay oracle 收益；或者 delay context 应在所有 posterior 区域无条件启用。若后续开发 delay estimator，最小合理形式应包含行为风险感知的 gate，只在 belief 明确偏向较大 delay 时启用，而不能直接把 posterior mean 无条件送入 planner。

## 7. 文件位置

- 冻结合同：`docs/research/persistent_context_v2_pushobj_delay_stage0_contract_zh.md`
- 冻结设计：`docs/research/persistent_context_v2_pushobj_delay_stage0_design.json`
- 算法 core：`research/persistent_context_v2/pushobj_delay_stage0.py`
- runner：`scripts/run_persistent_context_v2_pushobj_delay_stage0.py`
- 独立审计：`scripts/audit_persistent_context_v2_pushobj_delay_stage0.py`
- 单元测试：`tests/test_persistent_context_v2_pushobj_delay_stage0.py`
- raw/summary/audit：正式输出目录内的 `raw.jsonl`、`runner_summary.json`、`independent_audit.json`
