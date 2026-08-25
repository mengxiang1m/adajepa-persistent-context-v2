# PushObj 离散 Delay 非特权 History Stage 1 开发合同

状态：**FROZEN FOR DEVELOPMENT SMOKE；不是 formal**  
合同 ID：`persistent-context-v2-pushobj-discrete-delay-history-stage1-dev-v1`  
日期：2026-08-24（Asia/Shanghai）

## 1. 本阶段只回答什么

检验一个只读取过去 episode 的 planner command 和 agent position/velocity transition 的离散 delay estimator，能否在 E2 第一个 transition 发生前给出可审计 posterior/MAP，并通过最小闭环 smoke。Smoke 只验证实现、状态边界和因果对照可运行，不产生新的正式行为结论。

旧 Stage 0 的 true-factor 结果已显示强异质性：`d=3/4` 有益，`d=0/1` 反而有害。因此必须把“delay 可辨识”和“使用 exact delay 有行为价值”分开；accuracy 不得替代 `pose_auc10`。

## 2. Estimator 与禁止输入

候选支持固定为 `[0,1,2,3,4]`，先验固定均匀。利用已知 agent PD 离散动力学，从相邻 proprio 的 position/velocity 反演实际生效控制量；对每个候选 delay，把过去 command 做 episode-reset、zero-filled FIFO 移位，按二维各向同性 Gaussian residual 累加 log likelihood。开发噪声固定为 `0.1`，MAP 并列时取较小 delay。

估计器输出完整 posterior、MAP、entropy、evidence transition count、episode count 和 change-detector 状态。它不得读取 true delay、effective action、contact、block state、goal、planner loss、行为 cost 或当前 E2 transition；不得更新 base model、optimizer、replay、running statistics 或全局 memory。

开发参数只依据既有 Stage 0 raw 的 32 条 outcome-exposed 轨迹冻结。该 raw 仅用于可辨识性开发，不能再充当 formal。开发重算中 32/32 MAP 正确；这只是开发事实。

## 3. Smoke 数据与序列

使用作者发布的 `val_T/plan_targets.pkl`。从所有既有 `repro_outputs/**/*.jsonl` 中排除出现过 `segment_index` 的片段，再要求 nominal step-10 block displacement `>=10`，以 seed `1080000` 选出 8 个 smoke 和 16 个 reserve。4 个 sequence 各含 E1/E2，所有 8 个 smoke 片段唯一；reserve 不在本 smoke 读取行为结果。

E1 delay 按 sequence 为 `[0,1,3,4]`。Persistent 的 E2 保持不变；no-persistence 的 E2 在四档 factor index 上循环平移 `+1`，因此 E2 边际分布相同且每条都改变。两个 condition 共享完全相同的 E1 evidence；E2 的场景、初态、目标、CEM seed 和预算按 sequence 匹配，唯一条件差异是真实 delay 是否持续。

## 4. 策略与处理变量

- `population_prior/current_only`：E2 都使用 2 步；两者必须逐 action/state 相同；
- `correct_history_map`：只用本 sequence E1 的非特权 posterior MAP；
- `correct_history_high_delay_gate`：仅当 posterior `P(d>=3)>=0.95` 时使用 MAP，否则回退 2 步；这是基于旧 Stage 0 异质性的预定义二级策略，不是 learned gate；
- `shuffled_history`：固定 seed 打乱本 sequence 的 E1 command 时间顺序，proprio 不变；
- `wrong_sequence_history`：使用下一 factor sequence 的整段 E1 command/proprio；
- `true_factor_oracle`：只作上限，不属于非特权策略。

E1 evidence 一律由 population-prior planner 生成。E2 所有分支只读同一冻结 E1 bank，不把任何评价分支写回 history。主开发对比为 persistent `current_only - correct_history_map`；同时输出 no-persistence 同对比与 DiD。Gate、shuffled、wrong 和 oracle 都是辅助诊断。

## 5. Smoke 有效性

必须满足：4×E1 evidence 与 4×2 condition E2 完整；E1 estimator 可独立重算；E2 前零当前证据；persistent/no-persistence 的场景配对；population/current identity；FIFO 可复算；world-model 参数 hash 前后不变；所有数值有限；单 GPU 资源和 wall time 已记录。独立审计不得调用 runner 的 estimator 或 summary 实现。

任何实现、identity、预算、hash 或审计失败均保留原目录并另开 repair；不因 smoke 行为方向不佳而标 `INVALID`，也不得据此调 threshold、noise 或片段。正式合同、formal split 和样本量只能在 smoke 审计后另行冻结。

