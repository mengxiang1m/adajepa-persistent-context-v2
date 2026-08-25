# PushObj Bayesian Rotation×Gain Matrix History Stage 1 冻结合同

合同 ID：`persistent-context-v2-pushobj-bayesian-matrix-history-stage1-v1`  
冻结日期：2026-08-22；在本合同任何新 formal matrix-history 行为结果产生前冻结。

## 1. 命题与信息边界

命题：过去 episode 的真实 command/proprio transitions 能否估计同一 sequence 中持续的 `gain×R(theta)` actuator matrix，并改善下一个 episode 在任何当前 transition 产生前的 early-waypoint 闭环行为。

非特权 estimator 只读取自己发出的二维物理 command，以及 observation/state 中可见的 agent position、velocity；它不读取 factor、true/effective action、contact count、coverage、goal label或 oracle 输出。AdaJEPA checkpoint 不更新；跨 episode 只保留 Gaussian posterior sufficient statistics。

主零假设为 persistent 条件 E2 的 `pose_auc10(current_only)-pose_auc10(correct_history)` 均值不为正。另报告 no-persistence 和 DiD，不设置效果量、同向比例或 CI 自动裁决门。

## 2. 结构化 Bayesian matrix context

真实矩阵属于：

```text
A(c,s) = [[ c, -s],
          [ s,  c]]
c = gain cos(theta), s = gain sin(theta)
effective = A command
```

使用已知 PushObj agent PD 方程，从相邻可观察 position/velocity 反演本步 effective target displacement，再由 command/effective 得到一条 `z=[c,s]` 观测。只保留 command norm `>=1e-4` 且 inferred gain ratio 位于 `[0.45,1.55]` 的 finite transition；这些范围由 train/formal factor 物理支持在结果前确定。

Gaussian prior 的 mean/covariance由 rotation `[-30,-15,0,15,30]°` × gain `[0.75,1,1.25]` 的 15 个 train matrices计算，covariance 加固定 ridge `1e-4 I`；观测噪声固定 `0.01`。posterior 以 precision 与 information vector 更新，planner 使用 posterior mean 对应的 2×2 matrix。不得投影到 formal factor 列表或读取 factor ID。

上一轮已有 rotation evidence 的只读设计校准显示 128 episodes 均有 10/10 条范围内观测；这只用于确认反演实现，不属于本合同 formal matrix behavior。

## 3. Formal split、sequence 与 factors

- 使用 formal indices `[500,1000)` 中 nominal step-10 block displacement `>=10 px` 的场景。
- 排除前面 rotation/dead-zone Stage 1 已产生行为结果的 formal segments后剩余 73 个；以 `default_rng(1040000)` 冻结选择其中前 64 个，均从未观察过行为结果。
- 每 condition 32 sequences×2 episodes；同一 sequence 的 E1/E2 使用两个不同 frozen segments。
- formal factor 为 rotation `[-25,-10,+10,+25]°` × gain `[0.82,1.18]` 的 8 个组合；每个 persistent factor 有 4 条 sequences。
- persistent：E1/E2 factor 相同。no-persistence：E1 与 persistent 完全相同，E2 factor index 固定循环偏移 `+3 mod 8`；每个 episode 的边际 factor 仍平衡。
- 两 condition 共享 segment、初态、goal、env seed 和 CEM seed；差异只能来自 E2 factor lifetime。

## 4. Policies 与 history controls

1. `population_prior`：始终使用 train Gaussian mean matrix。
2. `current_only`：E1/E2 入口都从同一 prior 开始；评价前没有当前 transition，因此行为与 population prior 必须一致。
3. `correct_history`：E2 使用本 sequence E1 evidence posterior。
4. `shuffled_history`：E2 使用固定 donor seed 从其他 sequences 的 E1 transition pool 抽取与 correct history 完全相同数量的观测。
5. `wrong_sequence_history`：E2 使用 `(sequence_id+1) mod 32` 的单一错误 sequence E1 观测，并循环/截断到相同数量。
6. `true_factor_oracle`：直接使用当前真实 matrix，提供行为 ceiling。

E1 中除 true oracle 外的五种 policy 必须逐 action/state/cost exact identity。所有 estimator policy 的 E2 history observation count 必须相同；donor 不得为自己。

## 5. 执行与主统计

- evidence 使用 population-prior planner 在真实 factor 环境中执行 10 actions；evaluation 使用完全相同的 frozen场景和预算，但独立重置 simulator。
- open-loop latent CEM：2 model steps×5 actions、200 samples、top30、10 rounds；不做 MPC、TTT 或 episode-local update。
- 主端点为每条 sequence E2 `pose_auc10`；sequence 是最小统计单位。
- 报告 persistent correct-history effect、no-persistence effect、DiD、true gap recovery、positive/tie/negative、20,000 paired bootstrap CI（seed `1040301`）、shuffled/wrong、按 factor/gain/rotation 异质性、matrix/gain/angle estimation error 和 deadline success。

## 6. 工程审计与停止纪律

- design/checkpoint/data hash 匹配；64 segments 唯一、fresh、waypoint displacement 合格；factor schedule 与边际平衡。
- identity matrix wrapper误差 `<=1e-6`；raw effective action从 command×true matrix独立复算。
- E1 non-oracle identity、current/population identity、history count、donor isolation、state/command hashes、CEM预算全部审计。
- raw JSONL append-only；runner summary必须由独立脚本 exact复算。
- GPU 0 运行并记录 wall、周期显存、PyTorch peak、命令、退出码；异常/身份/配对/raw失败时停止并保留产物。
- smoke/formal 后不得改 factor、场景、posterior hyperparameter、donor、指标、seed或baseline；科学负结果不得改称 INVALID。
