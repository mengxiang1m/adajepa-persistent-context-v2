# Persistent Context V2：PushObj dead-zone history Stage 1 合同

状态：**FROZEN BEFORE FORMAL RESULTS**  
合同 ID：`persistent-context-v2-pushobj-radial-deadzone-history-stage1-v1`  
日期：2026-08-22（Asia/Shanghai）

## 1. 问题

在 10-action waypoint 任务中，过去 episode 的 command/proprio transition 能否估计 sequence 内持续的 radial dead-zone 半径，并改善 later episode deadline 行为？

Stage 0 已观察到明显 factor 异质性：true factor 对高于 prior 的 dead zone 有益，对低于 prior 的 dead zone 有害。因此本 Stage 1 不只报告总体均值，必须把高/低 factor 的正负迁移完整报告。

## 2. Formal 数据与 factor

- formal waypoint pool：`500..999` 且 nominal step-10 block displacement `>=10`；
- 排除 early-waypoint rotation Stage 1 已观察行为的 128 个 formal segments；
- `numpy.random.default_rng(930000).permutation(remaining_pool)[:128]`，32 sequences × 4 episodes，全部唯一；
- formal dead zones：`[0.04,0.08,0.12,0.16]`；population prior `0.10`；
- persistent base factor 按 `sequence_id mod 4`，E1–E4 固定；
- no-persistence E1 与 persistent 相同，E2–E4 用 seed `930100` 生成 balanced permutation，并要求每条 sequence 相邻 episode factor 改变。

## 3. Estimator

只读取 command `u` 和 observation 中的 agent position/velocity。用已知 PD 方程反演有效动作 `y`。

- 若 `||y||>1e-4`，active transition 给出 `d_i=||u||-||y||`；
- 若 `||y||<=1e-4`，它给出 censored lower bound `d>=||u||`；
- estimate 为 active `d_i` 的 median 与最大 lower bound 的较大者；没有 active 时为 population prior 与 lower bound 的较大者；
- estimate clip 到 `[0,0.25]`。

开发 Stage 0 raw 上该固定 estimator 的各 factor MAE 为 `3.5e-8–8.6e-8`，检查发生在本 formal 合同冻结前。正式 estimator 不读取 factor、effective action 或 contact 标签。

## 4. 策略、history 与 donor

沿用共同 population-prior evidence bank：所有策略从同一真实过去轨迹估计，评价分支不写回 history。

策略：population、current-only、correct history、shuffled、wrong-sequence、true-factor oracle。

donor seed `930200`，独立于 factor schedule：wrong donor 是跨 persistent base-factor 的随机 permutation；shuffled 的三个历史位置使用三个不同 base-factor 的独立 permutation；不自指、不重复、不读取行为结果。实际 no-persistence donor/current factor 偶然匹配比例必须报告。

## 5. 任务与统计

- goal：nominal step-10 waypoint；horizon 2×5；CEM 200/top30/10 rounds；
- 新 episode 获取当前 transition 前评价；无 TTT/重规划；
- 主指标：sequence 内 E2–E4 mean pose AUC10；
- 报告 persistent、no-persistence、DiD、true gap recovery、shuffled/wrong、deadline success；
- 按 `0.04/0.08/0.12/0.16`、低于/高于 prior、episode 分层；
- sequence bootstrap 20,000 次，seed `930301`。

不设固定效果门；总体正负不得覆盖 factor 异质性。

## 6. 有效性与输出

32+32 sequences、4 episodes、factor/donor 生命周期、E1 identity、current/pop identity、estimator 重算、metric/hash、design/checkpoint/data/资源全部审计。工程失败停止，科学负结果保留且不换 factor/prior/片段。

输出：`repro_outputs/persistent_context_v2_pushobj_deadzone_stage1/`。

