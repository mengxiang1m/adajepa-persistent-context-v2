# Persistent Context V2 实验结果与价值判断

日期：2026-08-22（Asia/Shanghai）

## 结论先行

机器裁决为：

- Stage 0：`TASK_DYNAMIC_RANGE_ESTABLISHED / GO_STAGE1_CONTRACT`
- Stage 1：`HISTORY_VALUE_SUPPORTED / GO_EXPLICIT_CONTEXT_PILOT`
- Stage 2：`EXPLICIT_CONTEXT_SUPPORTED / GO_CONTEXT_CONDITIONED_MODEL_DESIGN`

因此，目前思路**有继续尝试价值**，但证据范围严格限定为：在一个明确具有跨 episode 持续 actuator gain、且当前 episode 第一次高代价动作前没有辨识数据的合成闭环任务中，正确历史有强 persistence-specific 行为价值；一个只保存三个标量充分统计量的非特权 RLS context 几乎回收全部 oracle gap。

这不是“AdaJEPA context-conditioned world model 已成功”的证据，也不是现实机器人或复杂视觉动力学上的结果。下一步只被授权进入 Stage 3 合同设计与 true-context/model-use 上限验证。

## 机器事实

### Stage 0：独立 development 动态范围

从预先限定的 `[0.20,0.15,0.10]` 三个容差候选中，按“最宽合格者优先”机械选中 `0.20`：

| 指标 | population prior | true factor |
|---|---:|---:|
| mean early task cost | 1.875681 | 0.005588 |
| unsafe fraction | 48.63% | 0.00% |

true-factor 相对 cost 改善 99.70%，paired bootstrap 95% CI 为 `[99.66%, 99.74%]`。512/512 paired scenarios 有效；formal outcomes 在该阶段未生成。

### Stage 1：冻结的 history-value oracle

最终有效目录是 `persistent_context_v2_outputs/stage1_formal_v1_repair1/`。persistent/no-persistence 各 384 sequences，每条 8 episodes，统计单位为 sequence，bootstrap 20,000 次。

| 比较（E2–E8 首动作 cost） | 结果 |
|---|---:|
| persistent current-only | 1.798350 |
| persistent correct history | 0.005589 |
| persistent true factor | 0.005589 |
| correct-history 相对改善 | 99.69% |
| `current-correct` difference 95% CI | [1.652456, 1.934846] |
| history true-oracle gap recovery | 100.00% |
| positive sequence fraction | 100.00% |
| DiD mean | 2.407703 |
| DiD 95% CI | [2.233796, 2.582783] |

击穿替代解释的负对照：

- no-persistence correct-history 相对改善：−34.36%；
- persistent shuffled-history 相对改善：−141.97%；
- persistent wrong-sequence-history 相对改善：−134.21%。

所有 8 条冻结 GO 门均通过。独立审计验证 36,864 raw rows、E1 current/correct 精确一致、persistent factor 生命周期、跨条件 target/noise 精确配对、donor 非 self、history count、文件 hash 和机器裁决。

### Stage 2：非特权显式 RLS context

最终有效目录是 `persistent_context_v2_outputs/stage2_pilot_v1_repair1/`，使用独立于 Stage 1 的新 sequence/noise seeds。RLS 不知道 formal factor support；其跨 episode 状态只有 `sum_u2, sum_uy, transition_count`，prior mean/variance 只由 train factors 得到。

| 比较（E2–E8 首动作 cost） | 结果 |
|---|---:|
| persistent current-only RLS | 1.785755 |
| persistent RLS context | 0.008229 |
| categorical history oracle | 0.005744 |
| true factor oracle | 0.005744 |
| RLS 相对改善 | 99.54% |
| `current-RLS` difference 95% CI | [1.639531, 1.916868] |
| categorical gap recovery | 99.86% |
| true-factor gap recovery | 99.86% |
| positive sequence fraction | 100.00% |
| RLS DiD mean | 2.440218 |
| RLS DiD 95% CI | [2.262611, 2.618492] |

负对照：no-persistence −36.50%，shuffled −143.67%，wrong-sequence −151.96%。所有 8 条 Stage 2 门均通过。独立审计从过去 raw transitions 重建了每一个 RLS `estimate_before`，零失败，并验证 43,008 raw rows、跨条件 nuisance/noise 配对、预算和 hash。

## 有效性修复与偏差记录

第一次 `stage1_formal_v1` / `stage2_pilot_v1` 执行中，每个条件内部策略配对正确，但 persistent 与 no-persistence 同序号 sequences 没有共享 target/noise RNG。证据检查将它们标为 `INVALID_EXECUTION` 并保留，没有覆盖或用于最终结论。

唯一 repair 让两个生成条件共享完全相同的 target/noise arrays。factor treatment、factor split、master seed、样本、策略、预算、指标、bootstrap 和门槛均未改变。repair1 的独立审计显式检查该配对并通过。

## 支持的推断

1. 该任务的历史 transition 确实包含持续 gain 信息。
2. 该信息在新 episode 的第一次不可逆动作前有很大的闭环行为价值。
3. 价值依赖 persistence；没有 persistence 或使用错历史时收益消失并转为伤害。
4. 不需要保存 AdaJEPA 权重或大型 memory；低维、可审计的 sufficient statistics 已足够。
5. 因此，“把跨 episode 稳定动力学放入显式 `q(z_seq)`，而把 episode-local 适应分开”的结构性思路值得进入下一门。

## 不支持的推断与最强替代解释

- 未证明现有 AdaJEPA checkpoint 会使用 context；它没有 context 接口。
- 未证明视觉 observation 中能可靠抽取相同 transition response。
- 未证明 PointMaze、PushObj 或 deformable 环境有同样收益。
- 未证明复杂 learned context 优于 RLS；当前结果反而表明复杂方法必须先超过极强的三标量 baseline 才有价值。
- 任务按“首次错误动作不可挽回”构造，gain 是一维、线性、低噪声，并且一次 transition 几乎可辨识。这解释了接近 100% 的效果，限制外推强度，但不能解释 persistence/no-persistence、shuffled 和 wrong-sequence 的方向分离。
- 错历史造成严重伤害，说明真实部署必须有 sequence identity/change detection；不能无条件复用 context。

## 下一步门禁

下一步是另写 Stage 3 冻结合同，而不是继续调本任务：在 factor-diverse train data 上训练只有一种 context conditioning 机制的 world model，先证明 true-context 会改变 rollout/action ranking 并改善闭环行为，再用 RLS `q(z_seq)` 测试 history benefit。首选在简单 actuator/PointMaze wrapper 上做，不直接进入 deformable，也不加入 LoRA/router/expert/consolidation 或 unconditional weight carry。

## 证据位置与资源

- Stage 0：`/data4/zhaoqing/adajepa/persistent_context_v2_outputs/stage0_dev_v1/`
- Stage 1 有效结果：`/data4/zhaoqing/adajepa/persistent_context_v2_outputs/stage1_formal_v1_repair1/`
- Stage 2 有效结果：`/data4/zhaoqing/adajepa/persistent_context_v2_outputs/stage2_pilot_v1_repair1/`
- 无效但保留的 v1：相应目录内 `INVALIDATION.md`

CPU wall time / peak RSS：Stage 0 0.53 s / 49,864 KB；Stage 1 repair1 28.87 s / 83,252 KB；Stage 2 repair1 3.41 s / 91,824 KB。未使用 GPU。
