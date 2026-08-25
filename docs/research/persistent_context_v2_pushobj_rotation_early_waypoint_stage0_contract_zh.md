# Persistent Context V2：PushObj rotation early-waypoint Stage 0 合同

状态：**FROZEN BEFORE FORMAL RESULTS**  
合同 ID：`persistent-context-v2-pushobj-rotation-early-waypoint-stage0-v1`  
冻结日期：2026-08-22（Asia/Shanghai）

## 1. 唯一问题

把 PushObj 任务改为“必须在第 10 个 low-level action 前达到一个真实中间物体姿态 waypoint”后，向冻结 AdaJEPA planner 提供真实动作旋转角，能否相对 `0°` population prior 改善 deadline 之前的闭环行为？

本阶段只比较 population prior 与 true-factor oracle，不使用 history estimator，不更新模型权重。

## 2. 为什么这是 early-waypoint 而非事后截短

原 Stage 0 的目标是发布轨迹第 25 步终点，`pose_auc10` 仍在追逐远期最终目标，前 10 步不是任务 deadline。本实验把发布 nominal 轨迹的第 10 步 observation/state 本身定义为目标，并把 model/planner horizon 固定为 2 model steps × 5 low-level actions；第 10 步后没有补救预算。

任务语义是限时到达中间接触姿态，例如必须在传送节拍、狭窄通道切换或下一阶段抓取前把物体推到指定 waypoint。它改变的是目标和决策期限，不是从原 25-step 结果中事后挑一个正向窗口。

## 3. 冻结资产与片段选择

- checkpoint、数据和 environment 与 rotation Stage 0 相同；只用 T shape。
- 开发池仍为 `0..499`；本 Stage 0 不读取 `500..999` formal 行为。
- 候选片段必须满足 nominal state 在 step 10 的 block position 相对 step 0 位移 `>=10` 像素，确保 waypoint 要求真实早期接触/推动。
- 排除旧 rotation Stage 0 A/B 已选择的 segment，避免复用已经观察过行为结果的 scenario。
- 从剩余池按 `numpy.random.default_rng(810000).permutation(pool)[:32]` 固定选择。
- 冻结选择为：`[139,249,245,124,258,288,173,156,295,98,476,419,34,372,313,88,30,172,444,293,362,105,0,274,425,237,492,119,296,355,190,345]`。

## 4. Factor、配对和预算

- factors：`[-22.5,-7.5,7.5,22.5]°`，每个 factor 连续分配 8 个片段；population prior `0°`。
- 每个 prior/oracle pair 共享 segment、initial state、waypoint goal、env seed 和 CEM seed。
- planner：open-loop latent CEM，model horizon 2；每 model step 5 个二维动作；总 action/deadline 10。
- 200 samples、top 30、10 CEM rounds，`mu=0`、`sigma=1`，发布 staged objective。
- prior world-model context 为 `0°`；oracle context 和真实环境 action rotation 都使用真实 factor。
- 不做 MPC 重规划、TTT、history 更新或任何 episode-state carry。

## 5. Waypoint 构造与指标

从 segment 初态在 nominal `0°` 环境重放发布动作 1–10；第 10 步 observation 是 planner goal，第 10 步 block pose 是评价 waypoint。

主指标：执行状态 1–10 相对 waypoint 的 mean `pose_auc10`：

```text
pose_cost_t = block_position_error_t / 20
            + wrapped_block_angle_error_t / (pi/9)
```

越低越好。配对 delta 为 `prior - oracle`。

必须报告：mean delta、相对改善、32-pair bootstrap 95% CI、positive/tie/negative fraction、按 factor 异质性。辅助报告 step-10 position/angle error、deadline success、计划命令改变比例和 nominal waypoint 位移。

bootstrap 20,000 次，`numpy.random.default_rng(820001)`；统计单位为 paired scenario。

不设置固定效果量、同向比例或 CI 自动裁决门。结果只作连续证据，由用户决定是否接入 history estimator。

## 6. 工程有效性与停止规则

- design/checkpoint/data hash 必须匹配；
- 32 个 segment 唯一、均来自冻结选择、与旧 A/B 行为 segment 不重合；
- nominal step-10 block displacement 全部 `>=10`；
- theta=0 wrapper/base rollout identity `<=1e-6`；
- prior/oracle scenario、seed、目标和预算严格配对；
- 非零 context 到达 planner，command hash 可审计；
- raw JSONL append-only，可由独立脚本重算。

identity、配对、资源或 raw 审计失败时停止；科学结果为负时保留，不换 seed、factor、deadline、目标或指标。

## 7. 输出

```text
repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage0/
  manifest.json
  raw.jsonl
  runner_summary.json
  independent_audit.json
  report.md
  run.log
  resource.csv
```

