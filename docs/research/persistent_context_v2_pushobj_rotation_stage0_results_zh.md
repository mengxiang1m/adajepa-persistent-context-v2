# Persistent Context V2：真实 PushObj 工具坐标旋转 Stage 0 结果

> **2026-08-22 用户改判说明：旧的固定 `25%` 效果量门和 `70%` 同向比例门已撤销。** 下文保留旧预注册裁决的数值轨迹，但它不再约束研究结论。按连续证据重新表述：A、B 都是总体正向、bootstrap 区间高于 0 且多数配对改善；A 是较明确的中等正效应，B 是很小的正效应。

## 当前结论（撤销固定门后）

**PushObj 工具旋转的 true-factor oracle 是正向结果。**

- A：平均 block pose 代价 `6.0860 → 5.2735`，改善 `13.35%`；95% CI `[0.3841, 1.3527]` 高于 0；`21/32` 配对改善。
- B：平均代价 `8.5012 → 8.3620`，改善 `1.64%`；95% CI `[0.0114, 0.2723]` 高于 0；`20/32` 配对改善。
- A 的收益主要集中在较大旋转，尤其 `+22.5°`；B 虽总体正向，但实际幅度很小。

这证明“知道持续隐藏旋转能够改善真实 PushObj 行为”已有证据。它仍未证明 history 能识别并回收该收益；是否继续 history/RLS 现在由用户决定，不再由固定百分比自动阻断。

## 原预注册规则下的历史裁决（已废止）

**父门 NO-GO。当前这版 PushObj“隐藏工具坐标旋转 → history 推断 → 改善规划”的思路，不值得进入 history/RLS 开发和 formal benchmark。**

true-factor oracle 并非完全无效：在发布片段的完整 25-step 窗口，它把平均 block pose 代价从 `6.0860` 降至 `5.2735`，改善 `13.35%`，配对 bootstrap 95% CI `[0.3841, 1.3527]` 不跨 0。但这没有达到预注册的 `25%` 效果量门，而且只有 `65.63%` 的配对方向正确，未达到 `70%` 一致性门。

在预注册的早期接触候选中，oracle 只改善 `1.64%`（`8.5012 -> 8.3620`），方向正确率 `62.50%`。它同样失败，而且效果太小，不足以支撑先投入 history estimator 再期待端到端收益。

因此按照预注册顺序停止：不增加第三个候选，不查看 formal split，不训练或运行 history/RLS。

## 实验回答了什么

实验固定一个 episode 内不可由初始单帧看出的工具坐标旋转：planner 命令 `u` 在真实环境中以 `R(theta)u` 执行。population prior 永远使用总体均值 `0°`；oracle 知道真实的 `theta`，并用完全相同的旋转把候选动作映射到冻结 AdaJEPA 所理解的 nominal action 空间。

这比直接测试 history 方法更靠前：如果连知道真值的 oracle 都不能稳定改善行为，history 方法即使能估准 factor，也没有足够的行为价值上限。

冻结设置：

- 真实 Pymunk PushObj 接触环境、发布 T-shape checkpoint、发布 `val_T` 片段
- 开发 factor：`-22.5°, -7.5°, +7.5°, +22.5°`
- 每个候选 32 个配对 scenario，每角 8 个
- CEM：200 samples、top 30、10 rounds、5 个 model steps × 5 个 low-level actions
- checkpoint、目标、初态、真实 factor、CEM seed 在 prior/oracle 间严格配对
- 主代价：每步 `block_position_error/20 + wrapped_angle_error/(pi/9)` 的窗口均值

## 冻结门与结果

| 门 | 阈值 | A：发布片段 25-step | B：早期接触 10-step |
|---|---:|---:|---:|
| 完整配对 | 32 | 32，通过 | 32，通过 |
| identity | max abs ≤ 1e-6 | 0，通过 | 0，通过 |
| 规划被 context 改变 | ≥95% | 100%，通过 | 100%，通过 |
| 相对改善 | ≥25% | 13.35%，失败 | 1.64%，失败 |
| delta bootstrap CI 下界 | >0 | 0.3841，通过 | 0.0114，通过 |
| 配对方向正确率 | ≥70% | 65.63%，失败 | 62.50%，失败 |
| 决策 | 全部通过才 GO | **NO-GO** | **NO-GO** |

注意：CI 为正只说明均值上的小改善不是明显的随机正负抵消；它不代表改善足够大或足够稳定。这里预注册的效果量和方向门正是为了避免把“小而统计可见”的变化误报成值得开发的方法。

## factor 分解

候选 A 的结果显示出局部机制，但不构成总体 GO：

| factor | prior | oracle | 相对改善 | 改善配对数/8 |
|---:|---:|---:|---:|---:|
| -22.5° | 6.4530 | 5.4143 | 16.10% | 7 |
| -7.5° | 7.8477 | 7.7514 | 1.23% | 4 |
| +7.5° | 3.3792 | 3.3103 | 2.04% | 3 |
| +22.5° | 6.6641 | 4.6181 | 30.70% | 7 |

也就是说，大角度、完整执行窗口下存在价值，尤其 `+22.5°`；但中等角度几乎没有可利用的行为差异。候选 B 进一步说明，“只挑早期有 block 位移的片段、只看前 10 步”没有把价值放大，反而把总体改善压到 1.64%。这更像是大扰动在较长时间累积后，oracle 偶尔能避免明显走偏，而不是一个在常见 factor 范围内稳定有用的 persistent-context benchmark。

正负角不完全对称也符合接触任务的几何非对称性，但每角仅 8 个开发配对，不能据此另开单边角度 benchmark；那会是在看过结果后改 factor 分布。

## 完整性与独立审计

独立 audit 不导入 runner，而是从 raw state/action 重新计算指标和门：

- A/B 均为每角 8 个、总计 32 个唯一配对
- runner 指标与 raw 重算最大差：`0`
- 记录的有效动作与 `R(theta)u` 最大差：A `6.05e-8`，B `5.39e-8`
- prior/oracle 初态最大差：`0`
- wrapper 在 `theta=0` 时相对 base 的 action 和 rollout 最大差：`0`
- 独立 audit 与 runner 的两个 NO-GO 结论一致

预注册哈希：

- design JSON：`a1c1f077890d2ec591871eab86e0c26cab263557840d2ce8871f69f65a8aa299`
- checkpoint：`6a0a75a94eefa4ca1e261cd1010c2c886e1465ce77035ec8a802eb75db3ead95`
- data：`f08486da48f7a6961e3d0035b48d9cd13515c0322bbacaf9dd9f973bf7bad624`
- repo revision：`a29975964f966f2836a2c7e26f464367c795c333`

## 资源

- GPU：NVIDIA L40
- 全部 64 个 paired scenario（128 次 CEM planning）墙钟时间：`1:43.87`
- 峰值 CUDA allocation：约 `2.66 GiB`
- 峰值 RSS：约 `1.38 GiB`
- 退出状态：0；无 swap

## 对总思路的判断

现有证据应分两层说：

1. **机制层面有一点信号。** 当隐藏旋转很大并经过完整 25 步累积时，true factor 可以改善真实接触行为；所以“persistent context 有时会改变规划价值”不是伪命题。
2. **作为下一阶段研究路线，目前不成立。** 在预先规定的 factor 分布和两个有限候选上，oracle 上限都过不了行为门。此时做 history/RLS 只能再损失估计误差，不可能合理地期待比 oracle 更稳定。因此当前 PushObj rotation 实例应停止，而不是把局部大角度结果包装成成功。

若未来继续，应视为新的研究问题并另开独立预注册，而不是延长本 Stage 0。合理方向是寻找一个在真实 benchmark 中对常见 factor 范围本身就有强 oracle value 的任务（例如明确 action dead zone/delay 且存在不可逆早期 waypoint 代价），仍然先过 true-factor oracle 门，再谈 history。

## 产物

远端目录：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_rotation_stage0/`

- `A_released_raw.jsonl` / `B_early_contact_raw.jsonl`
- `A_released_runner_summary.json` / `B_early_contact_runner_summary.json`
- `A_released_independent_audit.json` / `B_early_contact_independent_audit.json`
- `manifest.json`
- `run.log`
