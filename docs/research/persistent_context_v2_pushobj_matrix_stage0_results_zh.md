# PushObj Rotation×Gain Matrix True-Factor Stage 0 结果

日期：2026-08-22  
合同：`persistent-context-v2-pushobj-rotation-gain-matrix-stage0-v1`  
远端正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_matrix_stage0/`

## 结论摘要

真实 2×2 actuator matrix 相对正确的训练 population matrix mean，把 32 个 early-waypoint pairs 的 `pose_auc10` 从 `2.195618` 降至 `1.971071`，改善 `10.2271%`；mean delta `+0.224548`，paired bootstrap 95% CI `[+0.087134,+0.374387]`，22/32 pairs 正向。deadline success 从 `62.5%` 提高到 `100%`。

这建立了 rotation×gain matrix context 的真实闭环行为上限，并提供了开发非特权 Bayesian matrix history estimator 的直接依据。收益仍有 factor 异质性，后续方法不能隐瞒小 rotation 或高 gain 组的不确定性。

## 实验与配对

- 环境执行 `effective = gain × R(theta) × command`；world-model wrapper 使用同一 2×2 变换。
- 共享 population prior 是预声明 15 个 train matrices 的逐元素均值 `0.932780492 I`，不是 identity。
- development 为 4 个 rotation `[-22.5,-7.5,+7.5,+22.5]°` × 2 个 gain `[0.85,1.15]`；每组合 4 个全新场景。
- waypoint、CEM 和 10-action deadline 与此前 early-waypoint 合同一致；两臂共享初态、目标、环境/CEM seed 和全部预算。
- 没有 history、TTT、MPC 或当前 episode 更新；唯一差异是 prior matrix 与 true matrix。

## 主结果与辅助行为

| 指标 | prior | oracle | oracle 改善 |
|---|---:|---:|---:|
| pose AUC10 | 2.195618 | 1.971071 | 10.2271% |
| deadline success | 62.5% | 100% | +37.5 pp |
| step-10 position error | 16.5027 px | 6.7695 px | -9.7331 px，CI `[5.8861,13.6773]` |
| step-10 angle error | 0.20177 rad | 0.05929 rad | -0.14248 rad，CI `[0.09177,0.19841]` |

主 delta CI 为 `[+0.087134,+0.374387]`；positive/tie/negative 为 `22/0/10`，同向比例 `68.75%`。32/32 pairs 的 planner command 均发生变化。

## 异质性

按 gain 聚合：

| gain | prior | oracle | 相对改善 | delta CI | 正向 |
|---:|---:|---:|---:|---:|---:|
| 0.85 | 2.212506 | 1.831770 | 17.2084% | `[+0.195906,+0.585004]` | 14/16 |
| 1.15 | 2.178731 | 2.110372 | 3.1376% | `[-0.098532,+0.260213]` | 8/16 |

按 rotation 聚合：

| rotation | 相对改善 | delta CI | 正向 |
|---:|---:|---:|---:|
| -22.5° | 22.9643% | `[+0.224964,+0.807380]` | 7/8 |
| -7.5° | 2.1872% | `[-0.094874,+0.197955]` | 5/8 |
| +7.5° | -1.1151% | `[-0.193438,+0.134683]` | 5/8 |
| +22.5° | 15.9411% | `[+0.048778,+0.700720]` | 5/8 |

八个组合中，`(-22.5°,0.85)`、`(-22.5°,1.15)`、`(-7.5°,0.85)` 和 `(+22.5°,0.85)` 的均值与 CI 都为正；`(-7.5°,1.15)` 为负，其余三组方向正或负但每组只有 4 pairs、区间跨 0。

## 工程审计

- formal runner/audit exit code 均为 0；32/32 unique pairs，8 factors 各 4 个。
- identity matrix 下 action 与 rollout 最大误差均为 `0.0`。
- independent audit 的 count/hash/manifest/matrix/metric/pairing/scenario failure count 全为 0；raw 重算 summary exact match。
- wall time `43.61 s`；GPU 0 NVIDIA L40；周期采样峰值 `3674 MiB`，PyTorch peak allocated `2716.91 MiB`。

## 证据边界与下一最小动作

已证明：在这些 held-out development 场景和冻结 planner 中，知道联合 rotation×gain 真矩阵总体上能显著改善早期闭环行为，且终点 position、angle 与 success 同时改善。

未证明：过去 episode 能在不读 factor 的情况下估出矩阵；Bayesian posterior 的收益具有 persistence-specific 性；wrong/shuffled/no-persistence 对照会消除收益。

下一最小动作是冻结独立 Stage 1：用训练矩阵得到 Gaussian prior，只从过去真实 command 与 observable agent proprio transition 更新 2×2 posterior；在 formal factors 上比较 current-only、correct、shuffled、wrong、no-persistence 与 true matrix。统计单位仍为 sequence。

## 文件

- 合同/设计：`docs/research/persistent_context_v2_pushobj_matrix_stage0_contract_zh.md`、`persistent_context_v2_pushobj_matrix_stage0_design.json`
- core：`research/persistent_context_v2/pushobj_matrix_stage0.py`
- runner/audit：`scripts/run_persistent_context_v2_pushobj_matrix_stage0.py`、`scripts/audit_persistent_context_v2_pushobj_matrix_stage0.py`
- tests：`tests/test_persistent_context_v2_pushobj_matrix_stage0.py`
