# PushObj 离散 Action Delay True-Factor Oracle Stage 0 冻结合同

合同 ID：`persistent-context-v2-pushobj-discrete-delay-stage0-v1`

冻结日期：2026-08-22。本文在任何本合同 prior/oracle 行为结果产生前冻结。

## 1. 命题与处理

命题：在 10 个低层动作内必须到达早期 waypoint 的真实 PushObj-T 任务中，若执行器存在 episode 内固定的离散 action delay，直接知道 delay 的 world-model planner 是否比只使用 population prior 的 planner 获得更低的早期闭环位姿代价。

唯一处理变量是 planner world model 使用的 delay context：prior 固定使用 2 步，true-factor oracle 使用环境的真实 delay。环境、checkpoint、初态、目标、CEM seed、采样和控制预算严格配对。

零假设是配对 `pose_auc10(prior) - pose_auc10(oracle)` 的均值不为正。本实验只报告连续证据，不设置效果量、同向比例或置信区间自动裁决门。

## 2. Delay 语义与状态边界

delay `d` 以低层 action 为单位，使用 episode-local FIFO：

```text
effective[t] = 0                 , t < d
effective[t] = commanded[t - d]  , t >= d
```

- 队列在每次 `env.prepare` 和每次独立 world-model rollout 开始时以物理零动作填充；不得跨 episode、pair 或 CEM sample 共享。
- `d=0` 必须是原动作和原 world-model rollout 的精确 identity。
- world-model wrapper 先把 normalized action 还原为物理动作，在完整 10-action 候选序列上移动，再重新归一化；不能在两个 5-action model step 的边界错误重置队列。
- 环境执行相同变换。deadline 后尚未释放的最后 `d` 个命令不再额外执行，因此两种策略的环境动作预算都固定为 10。

该任务模拟命令传输、执行器管线或控制接口中的固定整步时延；oracle 只能提前规划，不能获得额外动作、观测或 deadline。

## 3. Split 与场景冻结

- checkpoint：现有 PushObj shape-shift `model_latest.pth`；数据：`val_T/plan_targets.pkl`。
- 只用 development indices `[0, 500)`；formal `[500,1000)` 保留不看。
- 候选需满足 nominal 发布动作重放到第 10 步时 block displacement `>=10 px`。
- 排除此前 rotation Stage 0 A/B、rotation early-waypoint Stage 0 和 dead-zone Stage 0 已观察过行为结果的 development segment。
- 剩余 237 个候选按 `numpy.random.default_rng(960000).permutation(...)[:32]` 冻结为 design JSON 中的 32 个 index；选择时没有运行或读取这些场景上的 delay prior/oracle 行为。

## 4. Factor、prior 与预算

- 预先声明 train support：`[0,1,2,3,4]` 步，均匀总体中心对应 population prior `2` 步。
- development true factors：`[0,1,3,4]`，每档连续分配 8 个场景；不把 prior 本身作为 true factor。
- 每 pair 共享 segment、initial state、waypoint goal、env seed 和 CEM seed。
- waypoint 是 nominal 无 delay 发布轨迹的第 10 步 observation/state。
- open-loop latent CEM：model horizon 2，每 model step 5 个二维低层动作，共 10 个；200 samples、top 30、10 rounds、staged objective。
- 不做 MPC 重规划、TTT、history estimator、当前 episode 在线辨识或任何跨 episode状态保留。

## 5. 端点与统计

主指标为执行状态 1–10 相对 waypoint 的平均位姿代价：

```text
pose_cost_t = block_position_error_t / 20
            + wrapped_block_angle_error_t / (pi/9)
pose_auc10 = mean_t(pose_cost_t)
```

越低越好；pair delta 定义为 `prior - oracle`。报告均值、相对改善、20,000 次 paired bootstrap 95% CI（seed `960101`）、positive/tie/negative fraction，以及四个 delay 分组。辅助报告 deadline success、step-10 error、计划变化比例和 FIFO prefix/shift 审计。

## 6. 工程审计、停止和禁止动作

- design/checkpoint/data hash 必须记录并匹配；32 个 segment 唯一且与排除集不重合。
- `d=0` action transform 和 wrapper rollout identity 最大误差 `<=1e-6`。
- raw 中每个 policy 的 effective action 必须可由 command 与真实 delay 独立复算；前 `d` 步为物理零，后续精确移位。
- prior/oracle 的场景、目标、seed、CEM iteration 数和环境动作数必须配对；raw JSONL append-only，独立脚本复算汇总。
- GPU 运行前检查 GPU 0，占用异常、显存增长异常、身份/配对/raw 审计失败时停止并保留产物；工程 bug 只允许不改变科学处理的有限修复。
- 不得依据 smoke 或正式结果改 factor、prior、segment、deadline、主指标、bootstrap seed 或 baseline；不得挑选正向 seed，也不得把科学负结果改称执行无效。

## 7. 预期产物

```text
repro_outputs/persistent_context_v2_pushobj_delay_stage0/
  manifest.json
  raw.jsonl
  runner_summary.json
  independent_audit.json
  report.md
  run.log
  resource.csv
```
