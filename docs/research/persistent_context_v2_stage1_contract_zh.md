# Persistent Context V2 Stage 1：History-Value Oracle 正式合同

## 0. 冻结状态与授权来源

- contract ID：`persistent-context-v2-history-oracle-v1`
- 状态：`FROZEN BEFORE FORMAL RESULTS`
- 冻结日期：2026-08-22（Asia/Shanghai）
- Stage 0 依据：`persistent_context_v2_outputs/stage0_dev_v1/summary.json` 与 `independent_audit.json` 均通过；机械选择 `tolerance=0.20`。
- 旧 `persistent-actuator-factor-oracle-v1` 的 NO-GO 保持不变；本合同不修改其 factor、horizon、指标或阈值。

## 1. 命题、唯一处理与零假设

命题：在所有策略共享 train split 校准的 population prior 时，过去 episode 的真实 actuator transition 能否识别同一 sequence 中持续的 gain，并在下一 episode 第一次不可逆插接动作前降低 task cost。

唯一主处理变量是：正确 sequence 的 categorical sufficient statistics 是否跨 episode 保留。零假设是 persistent 条件下 `current_only - correct_history` 的 sequence-level early cost 改善不具实际意义，或这种改善不比 no-persistence 更强。

## 2. 冻结生成过程与 split

- 动力学、目标、动作限幅、noise 和 cost 完全继承 Stage 0 选中的 `tolerance=0.20` 任务。
- 每条 sequence 8 个 episode；primary window 是每个 episode 的第一个动作，发生在任何 current-episode update 前。
- train factors：`[0.55,0.70,0.85,1.15,1.30,1.45]`，population prior mean 为 `1.0`。
- formal held-out factors：`[0.575,0.725,0.90,1.10,1.275,1.425]`，development 中从未运行。
- persistent：sequence 内 factor 固定；no-persistence：每 episode 独立重采样，但边际 factor table 相同。
- formal 样本：每个生成条件 384 条 sequence；master seed `2026082211`。每条 paired sequence 的所有策略共享 factor、target、noise、动作/观测预算。

## 3. 冻结策略

1. `population_prior`：始终使用 train prior mean `g_hat=1.0`。
2. `current_only`：每 episode 从相同 uniform categorical prior 开始；第一次动作后用当前真实 `(u,y)` 更新，但 episode 边界丢弃。
3. `correct_history`：使用同一 categorical Bayesian estimator 和 likelihood；唯一差异是保留本 sequence 过去 episode 的 posterior sufficient statistics。
4. `shuffled_history`：历史数量与 estimator 相同，但每个 episode entry 使用随机无固定点 donor sequence 的等量历史。
5. `wrong_sequence_history`：使用固定循环 donor `(sequence_id+1) mod N` 的等量历史。
6. `true_factor_oracle`：直接使用本 episode 的真实 gain。

categorical oracle 知道 formal factor support 和 noise likelihood，但看不到真实 factor ID；它只以过去实际执行的 `(u,y)` 更新 posterior。因此它是 history-value upper bound，不是最终部署方法。shuffled/wrong 的 donor history 来自 donor 自己实际执行的 transition；各策略的 estimator state 分开。

Episode 1 中 `current_only` 与 `correct_history` 必须逐 action、response、cost、unsafe 完全一致。每个 episode 每策略恰好一次动作、一次 observation、一次 posterior update（不更新的 privileged/frozen 策略记录为零）；无额外 planning 或探索预算。

## 4. 主指标与统计

独立统计单位为 sequence。主指标是 later episodes E2–E8 的 mean early task cost：

`((y-d)/0.20)^2 + 1[abs(y-d)>0.20]`，越低越好。

每 sequence 定义：

- `Delta_persistent = cost(current_only)-cost(correct_history)`；
- `Delta_no_persistence = cost(current_only)-cost(correct_history)`；
- `DiD = Delta_persistent-Delta_no_persistence`，按相同 sequence index 配对。

bootstrap：paired sequence percentile bootstrap，20,000 resamples，PCG64 seed `2026082212`，双侧 95% CI。辅助指标为 unsafe fraction、posterior estimate error、E1 identity 和随 episode 的 cost；这些不能覆盖主门失败。

## 5. 冻结 GO 门

只有以下全部成立才判 `HISTORY_VALUE_SUPPORTED / GO_EXPLICIT_CONTEXT_PILOT`：

1. formal true factor 相对 population prior 的 later mean cost 改善至少 25%，其 sequence bootstrap 95% CI 下界至少 20%；
2. persistent 中 correct history 相对 current-only 的 cost 改善至少 30%，且 `Delta_persistent` CI 下界严格高于 current-only mean cost 的 20%；
3. `DiD` CI 下界严格高于 persistent current-only mean cost 的 15%；
4. correct history 回收至少 50% true-oracle gap；
5. 至少 75% persistent sequences 的 later mean cost 同向改善，ties 不计；
6. no-persistence 中 correct history 的相对改善不超过 5%；
7. persistent 中 shuffled 与 wrong-sequence 的相对改善各不超过 5%，且均小于 correct-history 改善的一半；
8. 384/384 + 384/384 sequences 有效，E1 identity、factor lifetime、donor 非同源、预算、hash 和 raw-artifact 审计全部通过。

任一科学门失败：`HISTORY_VALUE_NOT_ESTABLISHED / NO_GO_CONTEXT_METHOD`。工程或身份审计失败：`INVALID_EXECUTION / REPAIR_ONLY`。

## 6. 有限修复、停止与后续动作

只允许修复不改变生成过程、factor split、seed、样本、策略、指标、bootstrap 或门槛的实现错误，并保留失败产物。禁止看结果后更改 tolerance、factor、episode 数、主指标或阈值。

若 GO，只授权一个不知 formal factor support 的标量 Bayesian/RLS sequence-context pilot；不直接授权 context-conditioned neural world model。若 NO-GO，停止该任务实例上的 context、memory、LoRA、router、expert 与 consolidation 开发。

## 7. 设计检查：混淆与负对照

- 总体先验不正确：由对称 train factors 固定 mean=1.0，所有非 true-factor 策略共享。
- 一般平滑/更多样本而非 persistence：no-persistence 必须击穿。
- 历史数量或 donor 自身因素泄漏：shuffled/wrong 使用等量 donor transition，donor 不得为 self。
- 当前 episode 额外数据：primary action 在任何 current update 前，所有策略动作/观测预算相同。
- task 过易或全面失败：Stage 0 已在独立 dev split 得到 prior unsafe 48.63%。
- estimator 直接读取 factor：raw artifact 只允许 true-factor policy 使用 factor；其余只由 prior/posterior mean 产生 action。
