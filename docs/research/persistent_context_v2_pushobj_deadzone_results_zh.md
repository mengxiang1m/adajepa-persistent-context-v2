# Persistent Context V2：PushObj radial dead-zone 实验结果

日期：2026-08-22  
状态：**Stage 0 oracle 与 Stage 1 history 完成，独立 raw 审计通过**

## 1. 任务

10-action PushObj waypoint 中，二维相对动作经过径向 soft-threshold：`||u||<=d` 时为 0，否则幅值减少 `d`、方向不变。population prior `d=0.10`。checkpoint 冻结。

## 2. Stage 0 true-factor oracle

开发 factors `[0.025,0.075,0.125,0.175]`，32 pairs。

- pose AUC10：`2.173659 → 2.161919`；总体改善 `0.5401%`；
- mean delta `+0.011740`，95% CI `[−0.099968,+0.134735]`；
- 13/32 正向，19/32 负向；
- deadline success：`75.00% → 84.375%`，提高 `9.375` 个百分点；
- 32/32 planner commands 改变。

Factor 异质性：

| d | oracle 相对 prior | positive pairs |
|---:|---:|---:|
| 0.025 | `−17.5181%` | 0/8 |
| 0.075 | `−6.5880%` | 0/8 |
| 0.125 | `+7.0211%` | 7/8 |
| 0.175 | `+17.4851%` | 6/8 |

准确知道大于 prior 的 dead zone 有明确价值；小于 prior 时，population planner 假设更强死区而发出更大的动作，反而在当前 checkpoint/CEM 上更好。

## 3. Stage 1 history estimator

formal factors `[0.04,0.08,0.12,0.16]`；32 persistent +32 no-persistence sequences，每条 4 episodes。censored radial MLE 只读 command 和 proprio。

### 总体

- persistent current-only `2.253478`；correct history `2.217841`；
- 改善 `1.5814%`；mean delta `+0.035637`；
- 95% CI `[−0.063499,+0.134117]`；
- 15/32 sequences 正向，17/32 负向；
- deadline success `82.292% → 93.750%`，提高 `11.458` 个百分点；
- history 回收约 `100.26%` true-factor gap，差异来自近似数值 context。

No-persistence：

- pose 相对变化 `−0.0117%`，mean delta `−0.000264`；
- CI `[−0.059852,+0.059179]`；
- 17/32 正向、15/32 负向，整体接近 0。

DiD：`+0.035901`，CI `[−0.034339,+0.107347]`，20/32 paired sequences 正向。

### 高/低 dead zone 是相反结论

| persistent factor | 相对变化 | delta CI | positive sequences |
|---:|---:|---:|---:|
| 0.04 | `−15.0070%` | `[−0.423681,−0.158652]` | 0/8 |
| 0.08 | `−4.1486%` | `[−0.129016,−0.033012]` | 1/8 |
| 0.12 | `+4.1854%` | `[+0.032277,+0.141410]` | 6/8 |
| 0.16 | `+13.9798%` | `[+0.340393,+0.501292]` | 8/8 |

合并：

- 高于 prior (`0.12/0.16`)：改善 `9.9523%`，mean delta `+0.255223`，CI `[+0.161569,+0.352313]`，14/16 正向；
- 低于 prior (`0.04/0.08`)：退化 `9.4698%`，mean delta `−0.183949`，CI `[−0.276498,−0.105019]`，1/16 正向。

### Estimator 与 controls

- persistent correct-history factor MAE `1.46e-6`，median `5.15e-8`；
- shuffled/wrong 在 persistent 中分别使总体 pose 退化 `1.09%/2.38%`；
- no-persistence shuffled/wrong donor factor match `28.65%/27.08%`，接近随机四因子匹配率；
- 所以正负异质性不是 estimator 失准，而是“更精确的 forward model context”与当前 CEM 行为最优性并不单调等价。

## 4. 结论

dead-zone context 具有条件性尝试价值：当真实 dead zone 高于 prior 时，跨 episode history 能产生明确、可重复的早期行为改善；当真实 dead zone 低于 prior 时，同一精确 estimator 系统性伤害行为。

因此不能部署“无条件使用 posterior mean dead zone”的统一策略。若未来回到该方向，应比较 one-sided correction、robust/risk-sensitive planning，或用行为校准决定何时相信精确 context；但按当前优先顺序，下一步先测试 discrete action delay oracle。

## 5. 审计与产物

- Stage 0：32/32，audit passed，wall `43.64 s`，GPU峰值 `3674 MiB`；
- Stage 1：256/256 evidence +256/256 evaluation，audit failure 全 0，wall `864.94 s`，171 资源 samples，峰值 `3674 MiB`；
- Stage 0 core/output：`research/persistent_context_v2/pushobj_deadzone_stage0.py`，`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_deadzone_stage0/`；
- Stage 1 core/output：`research/persistent_context_v2/pushobj_deadzone_stage1.py`，`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_deadzone_stage1/`。

