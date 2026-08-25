# Persistent Context V2：PushObj radial action dead zone Stage 0 合同

状态：**FROZEN BEFORE FORMAL RESULTS**  
合同 ID：`persistent-context-v2-pushobj-radial-deadzone-stage0-v1`  
日期：2026-08-22（Asia/Shanghai）

## 1. 问题

在 10-action PushObj waypoint deadline 任务中，若同一 episode 的二维相对动作经过未知但固定的径向 dead zone，向冻结 AdaJEPA planner 提供真实 dead-zone 半径能否相对 population prior 改善行为？

本阶段只测 true-factor oracle，不使用 history、不更新模型。

## 2. Factor

对命令 `u`：

```text
r = ||u||
u_eff = 0                         if r <= d
u_eff = ((r-d)/r) * u             if r > d
```

dead zone 只改变动作执行，不改变图像、初态或目标。world-model wrapper 与真实环境使用完全相同变换。

- train-support calibration：`[0,0.05,0.10,0.15,0.20]`；
- population prior：`d=0.10`；
- development factors：`[0.025,0.075,0.125,0.175]`，每个 8 pairs；
- formal factors 保留为 `[0.04,0.08,0.12,0.16]`。

## 3. 片段和 waypoint

- 开发池 `0..499` 中 nominal step-10 block displacement `>=10` 的片段；
- 排除旧 rotation Stage 0 A/B 以及 early-waypoint Stage 0 的全部已观察片段；
- `numpy.random.default_rng(900000).permutation(remaining_pool)[:32]`；
- 冻结选择：`[142,343,32,428,199,380,148,429,292,407,385,376,35,149,413,321,360,347,398,357,179,18,152,163,330,193,294,322,359,138,336,469]`。

goal 是 nominal 无 dead-zone 发布轨迹的第 10 步 observation/state。model horizon 2×5，执行 deadline 10 actions。

## 4. Planner、配对和指标

- latent CEM：200 samples、top30、10 rounds、staged objective；
- prior/oracle 共享 checkpoint、segment、initial/goal、env seed、CEM seed 和预算；
- prior context `d=0.10`；oracle context与真实环境均用 true d；
- 无重规划、TTT、history 或跨 episode state。

主指标是 step1–10 相对 waypoint 的 mean `pose_auc10`；越低越好。报告 paired delta=`prior-oracle`、相对改善、20,000 次 bootstrap 95% CI（seed `900101`）、positive/tie/negative fraction、factor 异质性、deadline success、无效命令比例和计划变化。

不设置固定效果量或 CI 自动裁决门。

## 5. 有效性

- 32 个唯一、新片段且 waypoint displacement 合格；
- `d=0` wrapper 与 base exact identity；
- dead-zone transform 保方向、不放大、阈值两侧正确；
- prior/oracle factor、目标、seed、预算配对；
- raw append-only、独立重算、checkpoint/design/data hash 与资源记录完整。

工程失败时停止；科学结果为负时不换 factor、prior、片段、deadline 或指标。

## 6. 输出

`repro_outputs/persistent_context_v2_pushobj_deadzone_stage0/`。

