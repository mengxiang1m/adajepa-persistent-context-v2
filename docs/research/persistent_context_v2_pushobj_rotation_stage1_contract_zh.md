# Persistent Context V2：PushObj rotation 跨 episode Procrustes/MLE Stage 1 合同

状态：**FROZEN BEFORE FORMAL RESULTS**  
合同 ID：`persistent-context-v2-pushobj-rotation-history-stage1-v1`  
冻结日期：2026-08-22（Asia/Shanghai）

## 1. 唯一问题

在真实 Pymunk PushObj、发布的 T-shape AdaJEPA checkpoint 和保留的 formal 片段上，过去 episode 中可观察的工具运动 transition 能否估计同一 sequence 持续的隐藏动作旋转，并改善后续 episode 尚未获得当前数据时的闭环规划？

本实验不更新 AdaJEPA 权重，不加入 LoRA、replay、router 或神经 context encoder。跨 episode 唯一持久状态是二维 Procrustes/MLE 的充分统计量。

## 2. 冻结资产与 formal split

- 仓库：`/data4/zhaoqing/adajepa`
- checkpoint：`/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth`
- 数据：`/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl`
- 只使用此前未用于 Stage 0 行为结果的 formal 索引池 `500..999`。
- formal factors：`[-25, -10, 10, 25]` 度；population prior 为 `0` 度。
- `numpy.random.default_rng(610000).permutation(arange(500,1000))[:128]` 生成 32 条 sequence × 4 episodes 的唯一 segment，按 sequence-major 顺序分配。
- 每条 sequence 的 segment、初态、目标、env seed 和 CEM seed 在所有策略及 persistent/no-persistence 条件间配对一致。

## 3. 两个 factor 生命周期

每个条件 32 条独立 sequence，每条 4 episodes。

- `persistent`：sequence `j` 使用 factor `formal_factors[j mod 4]`，4 个 episode 中 factor 不变；32 条 sequence 中每个 factor 各 8 条。
- `no_persistence`：E1 与 paired persistent factor 相同；E2、E3、E4 按冻结 factor 顺序循环到下一个、下两个、下三个 factor。这样每个 episode 和全体样本的 factor 边际分布均衡，但一条 sequence 内 factor 改变。

sequence 是唯一独立统计单位。E1 只用于 identity audit 和产生第一批历史；主统计只使用 E2–E4。

## 4. 共同 history evidence 与可观察量

每个 sequence/episode 先生成一条共同 evidence rollout：使用 `0°` population context、冻结 checkpoint、固定 CEM seed 规划 25 个动作，并在该 episode 的真实 factor 下执行。所有非特权策略读取同一份 append-only evidence，避免策略行为差异污染历史信息对照。

估计器只允许读取：

- agent 发出的二维相对命令 `u_t`；
- transition 前后的 agent position 和 velocity，它们属于 PushObj observation/proprio；
- 已知的发布环境控制频率与 PD 运动方程。

禁止读取：真实 factor、环境内部旋转后的 `effective_action`、`n_contacts`、block factor 标签或未来 episode transition。`n_contacts` 只允许在 raw artifact 中作事后异质性解释。

## 5. Procrustes/MLE estimator

PushObj 的无接触 agent PD 动力学是线性的。由前后 position/velocity 和已知 `k_p=100`、`k_v=20`、`dt=0.01`、每动作 10 个 simulator steps，解析反推出该 transition 对应的二维有效目标增量 `y_t`。

真实关系为：

```text
y_t = R(theta) u_t
```

每条 transition 计算旋转保持范数残差：

```text
r_t = abs(||y_t|| - ||u_t||)
```

只有 `r_t <= 0.002` 且 `||u_t|| >= 1e-4` 的 dynamics-consistent transition 进入充分统计量。该筛选不使用接触标签；它只是排除不满足已知自由运动方程的 transition。统计量为：

```text
C = sum dot(u_t, y_t)
S = sum cross(u_t, y_t)
theta_hat = atan2(S, C)
```

若没有有效 transition，则返回 population prior `0°`。估计角限制在 `[-35°,35°]`，该范围由 Stage 0 冻结的训练支持 `[-30°,30°]` 外加固定 5° 数值余量确定，不读取 formal 结果。

## 6. 冻结策略

每个 episode 的规划都发生在读取当前 episode evidence 之前，因此 E2–E4 全部是 cold-start evaluation。

1. `population_prior`：始终用 `0°`。
2. `current_only`：episode entry 尚无当前 transition，因此从相同 population prior 开始；应与 `population_prior` 逐 action/state/cost 相同。
3. `correct_history`：只累计本 sequence 的 E1 至 E(i-1) evidence。
4. `shuffled_history`：历史 episode 数和 transition 数相同；第 h 个历史 episode从 donor sequence `(sequence_id + 1 + h) mod 32` 读取，混合多个错误 sequence，且永不读取自己。
5. `wrong_sequence_history`：E1 至 E(i-1) 全部从固定 donor `(sequence_id + 1) mod 32` 读取。
6. `true_factor_oracle`：直接使用当前 episode 的真实 factor，只给出 ceiling。

同一个 context 数值使用同一个 CEM seed 时只规划一次，再把确定性结果复制给 context 相同的策略；复制前后必须做 command/state hash identity audit。

## 7. 冻结规划和执行预算

与 PushObj rotation Stage 0 完全相同：

- open-loop latent CEM；5 model steps × 5 low-level actions，共 25 actions；
- 200 samples、top 30、10 CEM rounds；
- `mu=0`、`sigma=1`；发布的 staged objective；
- 无 MPC 重规划、无当前 episode estimator update、无 TTT；
- checkpoint、model weights 和 estimator donor bank 只读。

## 8. 主指标与统计

主指标是 later episodes E2–E4 的 `pose_auc25`，越低越好。先在每条 sequence 内对 E2–E4 求均值，再以 32 条 sequence 为统计样本。

必须报告：

- persistent：`current_only - correct_history` 的均值、相对改善、20,000 次 paired sequence bootstrap 95% CI、positive/tie/negative fractions；
- no-persistence 的同字段效应；
- persistence-specific DiD；
- `true_factor_oracle` 相对 population/current-only 的行为差距；
- correct history 对 true-oracle gap 的回收比例；
- shuffled 与 wrong-sequence 的效应；
- 按 factor、episode、有效 transition 数和接触比例的异质性；
- angle absolute error、无有效 transition 比例，只作机制解释。

bootstrap 使用 `numpy.random.default_rng(7601)`，20,000 resamples。不得按 episode 或 timestep 扩大样本量。

本合同不设置固定效果量、同向比例或 CI 自动裁决门。最终只陈述连续证据和不确定性，由用户决定是否进入 early-waypoint。

## 9. 工程有效性与停止规则

以下属于执行有效性，不是效果门：

- design、checkpoint、data SHA 与 manifest 一致；
- 32+32 条 sequence、每条 4 episodes 完整，segment 不重复；
- E1 的 population/current/correct/shuffled/wrong 逐 command/state/cost 相同；
- current-only 与 population 在所有 episode 精确相同；
- persistent/no-persistence 的 factor 生命周期和边际分布正确；
- donor 不自指、不读未来、历史 transition 数匹配；
- estimator 不读取 factor/effective-action/contact 标签；
- paired seed、规划预算、raw rows、资源记录完整；
- theta=0 wrapper identity audit 通过。

出现 identity/state-lifetime 违反、formal segment 泄漏到 estimator 开发、raw artifact 无法复算、CUDA/OOM、设备争用或连续两次同类失败时停止并保留产物。科学结果为负不触发修复性换 seed、换指标或换 factor。

## 10. 必需产物

```text
repro_outputs/persistent_context_v2_pushobj_rotation_stage1/
  manifest.json
  persistent_raw.jsonl
  no_persistence_raw.jsonl
  runner_summary.json
  independent_audit.json
  report.md
  run.log
  resource.csv
```

代码、测试、合同和机器可读 design 必须在 formal run 前同步到远端并记录 SHA。Stage 0 原始产物不得修改。
