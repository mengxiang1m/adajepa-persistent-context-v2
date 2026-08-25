# Persistent Context V2：PushObj early-waypoint rotation history Stage 1 合同

状态：**FROZEN BEFORE FORMAL RESULTS**  
合同 ID：`persistent-context-v2-pushobj-rotation-early-waypoint-history-stage1-v1`  
冻结日期：2026-08-22（Asia/Shanghai）

## 1. 问题

在已经由 true-factor oracle 证明具有早期行为动态范围的 10-action PushObj waypoint 任务上，过去 episode 的可观察 proprio transition 能否估计持续 rotation，并改善 later episode 的 deadline 行为？

跨 episode 只保留上一阶段已经冻结的 Procrustes/MLE 充分统计量；AdaJEPA checkpoint 全程冻结。

## 2. Formal sequences

- formal segment 池为 `500..999` 中 nominal step-10 block displacement `>=10` 的片段，共 346 个；不读取其 prior/history 行为来选片段。
- `numpy.random.default_rng(850000).permutation(pool)[:128]` 选出 32 sequences × 4 episodes，sequence-major 分配，全部唯一。
- formal factors：`[-25,-10,10,25]°`；population prior `0°`。
- persistent E1 factor 按 `sequence_id mod 4` 分配，每个 factor 8 条 sequence，E1–E4 固定。
- no-persistence E1 与 paired persistent 相同；E2–E4 分别用 seed `850100` 生成独立的 balanced factor permutation，拒绝任何与该 sequence 上一 episode 相同的 factor。每个 episode 各 factor 8 次且每条 sequence episode 间必定改变。

每个 condition 32 条 sequence，每条 4 episodes。统计单位为 sequence，主评价只使用 E2–E4。

## 3. History evidence 与 estimator

每个 sequence/episode 先以 population prior `0°` 在对应真实 factor 下生成一条共同 10-action evidence rollout。所有策略读取同一 evidence bank；评价策略的反事实动作不写回 history，避免 policy-induced history confound。

Procrustes/MLE 与 rotation Stage 1 完全相同：只读取 command 与 observation 中的 agent position/velocity；利用已知 PD 方程反演二维有效目标增量；不读取 factor、effective action 或 contact 标签。dynamics-consistent residual、命令范数下限和 angle clip 分别固定为 `0.002`、`1e-4`、`35°`。

## 4. 策略与独立 donor

1. `population_prior`；
2. `current_only`，episode entry 应与 population 完全相同；
3. `correct_history`，读取本 sequence 的过去 evidence；
4. `shuffled_history`，每个历史 episode 从独立错误 donor 读取；
5. `wrong_sequence_history`，所有历史 episode 从一个固定错误 donor 读取；
6. `true_factor_oracle`。

为修正上一 rotation Stage 1 中 factor schedule 与 donor offset 的偶然对齐，本合同用独立 seed `850200` 生成 donor：

- wrong donor 是 32-sequence 的随机 permutation，要求不自指且 donor 的 persistent base factor 与 target 不同；
- shuffled 的 h=0/1/2 分别生成独立 permutation，要求同样不自指、base factor 不同，并要求同一 target 的三个 donor 不重复；
- donor 生成不读取 no-persistence 当期 factor 或任何行为结果。

这不能禁止随机 donor factor 偶然等于 no-persistence 当前 factor，但使该匹配由独立随机化产生，而不是固定 offset 的结构性对齐；必须报告实际匹配比例。

## 5. 任务、预算与指标

- 每个 episode 的 goal 是 nominal 发布轨迹第 10 步 observation/state；model horizon 2×5，deadline 10 actions；
- CEM：200 samples、top30、10 rounds、staged objective；无重规划、无 TTT、无当前 episode estimator update；
- 主指标：每条 sequence E2–E4 mean `pose_auc10_to_waypoint`；
- 主效应：persistent `current_only-correct_history`、no-persistence 同字段及 DiD；
- 报告 20,000 次 sequence bootstrap 95% CI，seed `850301`；
- 报告 positive/tie/negative sequence fraction、true gap recovery、shuffled/wrong、factor/episode 异质性、angle error、deadline success 和 donor factor match。

不设置固定效果量、同向比例或 CI 自动门。由完整连续证据决定是否进入 PushObj dead zone。

## 6. 有效性

- 32+32 sequences、每条 4 episodes 完整；128 个 segment 唯一且 waypoint displacement 合格；
- E1 非 true policies identity；所有 episode current/population identity；
- factor 边际/lifetime、donor 独立性、不自指、不读未来、history count 正确；
- estimator 只读允许字段；raw 可独立复算；theta=0 wrapper identity；
- checkpoint/design/data hash、seed、预算和资源完整。

工程失败时停止并保留；科学结果为负时不换 seed、donor、factor、waypoint 或指标。

## 7. 输出

`repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage1/` 下保存 manifest、两个 condition raw JSONL、summary、independent audit、report、log 和 resource CSV。

