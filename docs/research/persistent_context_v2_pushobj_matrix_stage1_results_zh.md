# PushObj Bayesian Rotation×Gain Matrix History Stage 1 结果

日期：2026-08-22  
合同：`persistent-context-v2-pushobj-bayesian-matrix-history-stage1-v1`  
正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_matrix_stage1/`

## 一句话结论

非特权 Bayesian matrix context 只用 E1 的 command/proprio transition，就在 persistent E2 将 `pose_auc10` 从 `2.263778` 降至 `2.048022`，改善 `9.5308%`；mean delta `+0.215756`，95% CI `[+0.029035,+0.404384]`，24/32 sequences 正向。它回收约 `100.43%` true-matrix oracle gap；factor 不持续时同一历史方法退化 `20.0171%`，DiD 为 `+0.652338`、CI `[+0.409076,+0.932698]`，29/32 正向。shuffled 与 wrong-sequence history 都使行为变差。

因此，rotation×gain matrix 的因果链已经闭合：真矩阵有闭环价值，过去真实 transition 能非特权地估计矩阵，收益依赖跨 episode persistence，并且错误历史不能复制收益。

## 1. 实验做了什么

- 32 条 persistent 与 32 条 no-persistence sequences，每条 2 episodes；共 128 evidence episodes、128 evaluation episodes。
- 64 个 formal early-waypoint segments 全部是此前未观察过行为结果的剩余场景。
- formal factors：rotation `[-25,-10,+10,+25]°` × gain `[0.82,1.18]`，8 个组合，各 4 条 persistent sequences。
- E1 使用共享 population-prior matrix 规划并产生历史。E2 在任何当前 transition 产生前比较 current-only、correct、shuffled、wrong 和 true oracle。
- Bayesian estimator 不读取 true factor、effective action 或 contact；只读取物理 command 及 observation/state 中的 agent position/velocity。
- posterior 参数为 `c=g cos(theta), s=g sin(theta)`，planner 使用对应完整 2×2 matrix。

## 2. 主行为结果

| 条件/策略 | current mean | treatment mean | 相对变化 | mean delta 95% CI | 正向 |
|---|---:|---:|---:|---:|---:|
| persistent correct history | 2.263778 | 2.048022 | +9.5308% | `[+0.029035,+0.404384]` | 24/32 |
| persistent true factor | 2.263778 | 2.048941 | +9.4902% | `[+0.026023,+0.404048]` | 24/32 |
| no-persistence correct history | 2.181043 | 2.617625 | -20.0171% | `[-0.713679,-0.196086]` | 7/32 |
| persistent shuffled history | 2.263778 | 2.337804 | -3.2700% | `[-0.185972,+0.029771]` | 8/32 |
| persistent wrong-sequence | 2.263778 | 2.404647 | -6.2228% | `[-0.308753,+0.021333]` | 10/32 |

Persistence-specific DiD：

- mean `+0.652338`；
- bootstrap 95% CI `[+0.409076,+0.932698]`；
- 29/32 sequences 正向；
- correct history / true-oracle gap recovery `100.43%`。

超过 100% 不是 estimator 超越真物理模型的普遍结论，而是 posterior 与真矩阵之间极小数值差异改变了有限样本 CEM 的个别候选排序；两个 treatment mean 相差仅 `0.000919`。

## 3. Deadline success

- persistent：current-only `56.25%`，correct history `96.875%`，提高 `40.625` 个百分点，配对区间 `[+21.875,+59.375]` pp；true oracle同为 `96.875%`。
- no-persistence：current-only `56.25%`，correct history `46.875%`，降低 `9.375` 个百分点，区间 `[-21.875,+3.125]` pp。
- persistent shuffled `65.625%`，wrong `46.875%`，均未复制正确历史的 `96.875%`。

## 4. Estimator 准确度

128/128 evidence episodes 都接受 10/10 条预注册范围内 transition。persistent E2 上：

| 误差 | mean | max |
|---|---:|---:|
| matrix Frobenius | `0.00007034` | `0.00009004` |
| gain absolute | `0.00004212` | `0.00006313` |
| rotation absolute | `0.001353°` | `0.003102°` |

这说明行为异质性不是 factor 估计失败造成的；correct history 与 true oracle 的总体行为几乎相同。

## 5. Factor 异质性

按 gain 聚合：

| gain | 改善 | delta CI | 正向 |
|---:|---:|---:|---:|
| 0.82 | 13.8042% | `[+0.013858,+0.652068]` | 14/16 |
| 1.18 | 4.3272% | `[-0.100202,+0.284444]` | 10/16 |

按 rotation 聚合：

| rotation | 改善 | 正向 |
|---:|---:|---:|
| -25° | 13.1975% | 6/8 |
| -10° | 0.3615% | 5/8 |
| +10° | 7.8026% | 7/8 |
| +25° | 15.9440% | 6/8 |

8 个组合中 7 个均值正向。唯一总体负向组合为 `theta=-10°, gain=1.18`：退化 `23.2329%`，1/4 正向。该组合的 correct-history行为与 true oracle一致，因此这是当前 frozen predictor/CEM 对真 context 的行为异质性，不是 Bayesian estimator 误估。

## 6. 工程与独立审计

- runner exit code `0`，independent audit exit code `0`。
- raw 中 `count/cross_condition/evidence/hash/manifest/matrix/metric/pairing/posterior/scenario` failure count 全为 0。
- persistent/no-persistence 的 E1 evidence 与 evaluation逐 command/state exact match；E1 五个非 oracle policies exact identity。
- correct/shuffled/wrong 的 E2 history observation count严格相同；wrong/shuffled donor均不自指。
- raw 重算 runner summary exact match；所有 structural checks 为 true。
- formal wall time `326.16 s`；GPU 0 NVIDIA L40；周期采样峰值 `3674 MiB`，PyTorch peak allocated `2716.91 MiB`。

## 7. 证据支持的下一步

该方向现在已经具备进入中等规模独立复验的条件：方法非特权、persistent effect正向、no-persistence反向、关键负对照不复制收益、estimator接近 oracle且审计完整。

最小规模扩展应使用三个不重叠批次，每批约 32 sequences，并保持当前算法、factor支持、预算和主指标；目标是测量效应与 factor异质性是否随样本增加稳定，而不是设置新的百分比门槛。`-10°,1.18` 等 true-context 负向区域也必须保留，不得由样本筛选删除；后续 learned gate 可以把这种异质性作为独立问题处理。

## 8. 文件位置

- 冻结合同/设计：`docs/research/persistent_context_v2_pushobj_matrix_stage1_contract_zh.md`、`persistent_context_v2_pushobj_matrix_stage1_design.json`
- core：`research/persistent_context_v2/pushobj_matrix_stage1.py`
- runner/audit：`scripts/run_persistent_context_v2_pushobj_matrix_stage1.py`、`scripts/audit_persistent_context_v2_pushobj_matrix_stage1.py`
- tests：`tests/test_persistent_context_v2_pushobj_matrix_stage1.py`
- raw：正式输出目录内 `persistent_raw.jsonl`、`no_persistence_raw.jsonl`
