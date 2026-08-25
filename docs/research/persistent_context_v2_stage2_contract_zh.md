# Persistent Context V2 Stage 2：显式 RLS Sequence Context Pilot 合同

## 0. 冻结状态

- contract ID：`persistent-context-v2-explicit-rls-v1`
- 状态：`FROZEN BEFORE PILOT RESULTS`
- 冻结日期：2026-08-22（Asia/Shanghai）
- 授权：Stage 1 `HISTORY_VALUE_SUPPORTED / GO_EXPLICIT_CONTEXT_PILOT`，且独立 raw audit 通过。
- 目的：检验不知 formal factor support 的最小低维 sufficient-statistics 方法能否回收 categorical history oracle 的闭环收益。

## 1. 方法与唯一主处理

方法只假设 transition 形式 `y=g*u+epsilon`，不知道 formal test gain 列表。它从 train factors 校准高斯 prior：mean 为 train mean，variance 为 train population variance；以已知训练噪声方差进行标量 Bayesian linear regression（等价于带 population prior 的 RLS）。

持久状态严格为：`sum_u2, sum_uy, transition_count`。base model 不变；无 optimizer、replay、neural embedding、factor label 或 formal support。唯一主处理是该 RLS sufficient statistics 是否跨 episode 保留。

## 2. 数据与策略

- 任务、`tolerance=0.20`、train/formal factors、8 episodes 和 primary E2–E8 first-action cost 与 Stage 1 相同。
- 使用新的独立 sequence/noise master seed `2026082221`，不复用 Stage 1 formal sequences；persistent 与 no-persistence 各 384 sequences。
- bootstrap 20,000 次，PCG64 seed `2026082222`。
- 策略：`population_prior`、`current_only_rls`、`persistent_rls`、`shuffled_rls`、`wrong_sequence_rls`、`categorical_history_oracle`、`true_factor_oracle`。
- shuffled donor 每 episode 为无固定点 permutation；wrong donor 固定为下一 sequence；donor history 数量匹配。
- categorical oracle 与 Stage 1 相同，只作已建立 history ceiling 的复验；true factor 是 task ceiling。
- Episode 1 的 current-only RLS 与 persistent RLS 必须 action/response/cost 完全一致。所有策略每 episode 一次动作、一次 observation；非 frozen/oracle estimator 一次 update。

## 3. 指标与 GO 门

统计单位仍为 sequence，主指标仍为 later-episode mean early task cost。定义 persistent/no-persistence RLS effect 和 DiD 与 Stage 1 相同。

只有全部满足才判 `EXPLICIT_CONTEXT_SUPPORTED / GO_CONTEXT_CONDITIONED_MODEL_DESIGN`：

1. persistent RLS 相对 current-only RLS 改善至少 30%，paired difference CI 下界高于 current mean 的 20%；
2. RLS DiD CI 下界高于 persistent current mean 的 15%；
3. persistent RLS 回收至少 80% categorical-history improvement，且至少 50% true-factor gap；
4. 至少 75% persistent sequences 同向改善；
5. no-persistence 中 persistent RLS 相对改善不超过 5%；
6. shuffled/wrong RLS 在 persistent 中相对改善各不超过 5%，且各小于 persistent RLS 改善的一半；
7. categorical history oracle 在新 sequences 上相对 current 至少改善 30%；
8. 384+384 sequences、E1 identity、factor/donor/state-count、预算、finite estimate、hash 与 raw audit 全通过。

失败则 `EXPLICIT_CONTEXT_NOT_ESTABLISHED / NO_GO_CONTEXT_MODEL`；无效执行只允许不改变合同的有限修复。

## 4. 推断边界

GO 只证明：在该标量执行器、一次不可逆冷启动动作的合成任务中，一个可解释三标量 sequence state 足以利用历史。它不证明 AdaJEPA、视觉 latent、现实机器人或 context-conditioned world model 已有效。GO 只授权另写 Stage 3 训练合同；不得直接把 RLS 结果包装成 neural method 成功。
