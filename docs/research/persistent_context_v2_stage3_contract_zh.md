# Persistent Context V2 Stage 3：FiLM Context-Conditioned World Model 合同

## 0. 冻结状态与授权

- contract ID：`persistent-context-v2-film-world-model-v1`
- 状态：`FROZEN BEFORE TRAINING/DEVELOPMENT/FORMAL RESULTS`
- 冻结日期：2026-08-22（Asia/Shanghai）
- 授权来源：Stage 2 repair1 为 `EXPLICIT_CONTEXT_SUPPORTED / GO_CONTEXT_CONDITIONED_MODEL_DESIGN`，独立 raw audit 通过。
- 本阶段不修改 AdaJEPA 权重，不加入 TTT、LoRA、router、expert、replay 或 consolidation。

## 1. 核心命题

在 factor-diverse train data 上训练、且只有一个显式 FiLM context 接口的动力学模型，是否会真正使用 `z_seq` 改变 rollout prediction 和 action ranking，并把 true/RLS context 转化为 held-out sequence 的冷启动闭环收益？

零假设：context 只改变 latent/prediction proxy，却不能产生预注册的闭环行为改善；或者 RLS context 无法回收 true-context 的行为收益。

## 2. 模型、训练和 split

动力学仍为 `y=g*u+epsilon`，cost、action limit、noise 和关键首动作与 Stage 0–2 相同。训练 factor labels 可见，测试策略除 `true_context`/审计外不可见。

- train factors：`[0.50,0.65,0.80,0.95,1.05,1.20,1.35,1.50]`；population mean `1.0`。
- development factors：`[0.575,0.725,0.875,1.125,1.275,1.425]`。
- formal factors：`[0.6125,0.7625,0.9125,1.0875,1.2375,1.3875]`；development gate 前禁止生成 formal outcome。
- train transitions 32,768，seed `2026082231`；dev prediction transitions 8,192，seed `2026082232`。
- 模型：`action Linear(1,16)`；`context Linear(1,32)` 产生一组 scale/shift；唯一条件化为 `h = action_features*(1+scale)+shift`；`Linear(16,1)` 输出 response。不得加入第二种 context 机制。
- optimizer：AdamW，lr `3e-3`，weight decay `1e-6`，batch 512，2,000 steps；torch seed `2026082233`。
- 每 100 steps 在固定 dev prediction set 上评估，以最低 true-context dev MSE 选择 checkpoint。训练和 checkpoint 选择不得读取 history closed-loop 方向或 formal data。

所有策略共享同一冻结 checkpoint。planner 在 `[-1.5,1.5]` 上固定评估 401 个等距候选动作，选择 predicted response 与 target 平方误差最小者。

## 3. Development model-use 门

只比较冻结模型的 `population_context(z=1)` 与 `true_context(z=g)`，不使用 history 选模型或调任务。256 paired dev scenarios，seed `2026082234`，bootstrap 5,000 次，seed `2026082235`。

只有全部满足才生成 formal outcomes：

1. true-context prediction MSE 不超过 population-context MSE 的 10%；
2. true-context closed-loop cost 相对 population 至少改善 50%，paired relative bootstrap 95% CI 下界至少 40%；
3. population unsafe fraction 位于 `[0.25,0.75]`；
4. true-context 与 population 的 planner action 至少在 80% scenarios 中不同；
5. checkpoint、split、预算、finite prediction 和 paired nuisance/noise 审计通过。

失败：`MODEL_CONTEXT_USE_NOT_ESTABLISHED / STOP_BEFORE_FORMAL`。

## 4. Formal 生成、策略与统计

dev GO 后，persistent/no-persistence 各 384 条新的 sequences，8 episodes，master seed `2026082241`；同 sequence index 跨生成条件共享完全相同 target/noise。bootstrap 20,000 次，PCG64 seed `2026082242`。

策略：

1. `population_context`：冻结模型始终使用 train prior mean `z=1`；
2. `current_only_context`：RLS 每 episode 重置，第一次动作前仍为 prior；
3. `persistent_rls_context`：跨 episode 保留 `sum_u2,sum_uy,count`；
4. `shuffled_rls_context`：使用随机无固定点 donor 的等量 RLS history；
5. `wrong_sequence_rls_context`：使用固定下一 sequence 的等量 history；
6. `true_context`：冻结模型使用真实 gain，只提供模型使用上限；
7. `analytic_true_factor_oracle`：直接按真实 gain 规划，给出 simulator ceiling。

primary endpoint 是 E2–E8 第一次动作的 mean task cost，统计单位为 sequence。Episode 1 current-only/persistent 必须逐 context、prediction、action、response、cost 一致。每策略每 episode 401 candidate predictions、一次环境动作、一次 observation；无额外探索预算。

## 5. Formal GO 门

全部满足才判 `CONTEXT_CONDITIONED_WORLD_MODEL_SUPPORTED / GO_REAL_BENCHMARK_TRANSFER_CONTRACT`：

1. true-context 相对 population cost 改善至少 50%，paired difference CI 下界高于 population mean 的 40%；
2. true-context 回收至少 90% analytic true-factor gap；
3. persistent RLS context 相对 current-only 改善至少 50%，difference CI 下界高于 current mean 的 40%；
4. RLS context 回收至少 90% true-context improvement；
5. RLS persistence DiD CI 下界高于 persistent current mean 的 30%；
6. 至少 80% persistent sequences 同向改善；
7. no-persistence 中 RLS 相对改善不超过 5%；shuffled/wrong 在 persistent 中各不超过 5%且小于正确 RLS 改善一半；
8. true-context 与 population 的 later action ranking 至少 80% 不同，并记录 `context→prediction→action→outcome` 字段；
9. 384+384 sequences、跨条件 nuisance/noise、factor lifetime、donor、RLS reconstruction、checkpoint hash、预算、raw hash 和资源审计通过。

任一科学门失败：`CONTEXT_CONDITIONED_WORLD_MODEL_NOT_ESTABLISHED / NO_GO_TRANSFER`。工程审计失败：`INVALID_EXECUTION / REPAIR_ONLY`。

## 6. 推断边界与下一步

GO 只说明该合成 actuator world model 会使用 context，且非特权 RLS context 能在冻结模型中转化为行为收益。它不证明视觉 AdaJEPA、PointMaze、PushObj、deformable 或现实机器人有效。

GO 只授权新写一个真实 benchmark transfer 合同：优先 PointMaze/action-calibration wrapper，训练 factor-diverse checkpoint，并沿用 true-context → RLS-context → negative controls 的顺序。禁止直接加入 episode-local TTT 或把当前合成结果包装成 AdaJEPA 成功。
