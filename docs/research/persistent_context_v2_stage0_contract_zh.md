# Persistent Context V2 Stage 0：冷启动执行器校准任务合同

## 冻结状态

- contract ID：`persistent-context-v2-stage0-docking-v1`
- 状态：`FROZEN BEFORE DEVELOPMENT RESULTS`
- 冻结日期：2026-08-22（Asia/Shanghai）
- 关系：这是独立于 `persistent-actuator-factor-oracle-v1` 的新任务开发合同；旧 oracle 的 factor table、horizon、指标和裁决均不改变。
- 本阶段只做 benchmark qualification，不产生 history-value 正式结论。

## 核心命题与任务

任务模拟一次性精密插接/脉冲定位：每个 episode 的第一个执行器脉冲发生在当前 episode 尚无 transition 时，脉冲结果决定是否偏离容差带；偏离造成不可逆的碰撞/报废代价。随后才读到执行结果，因此同一 episode 的系统辨识不能挽回第一次动作，但过去 episode 可用于下一次冷启动。

隐藏因素是标量 actuator gain `g_seq`。策略观察目标位移 `d`，动作 `u`，随后观察响应

`y = g_seq * u + epsilon`, `epsilon ~ N(0, 0.015^2)`。

`g_seq` 不出现在 observation 或 task ID 中。知道 gain 时的最优首动作是 `clip(d/g_seq, -1.5, 1.5)`；population policy 使用由 train factors 校准的共同 prior mean。目标符号等概率，绝对位移均匀分布于 `[0.65, 0.85]`，属于 episode nuisance。

主行为 cost 是首动作后的状态误差和不可逆容差违反：

`cost = ((y-d)/tolerance)^2 + 1[abs(y-d) > tolerance]`。

它不含 action-energy 或 prediction-loss 项。辅助 outcome 是 `safe = abs(y-d) <= tolerance`。

## 冻结 split、候选与选择规则

- train factors：`[0.55, 0.70, 0.85, 1.15, 1.30, 1.45]`；只用于得到共享 population prior mean `1.0`。
- development factors：`[0.60, 0.80, 1.20, 1.40]`。
- formal factors（本阶段禁止运行）：`[0.575, 0.725, 0.90, 1.10, 1.275, 1.425]`。
- development sequences：512；master seed `2026082201`；paired bootstrap 20,000 次，seed `2026082202`。
- 只比较 `population_prior` 与 `true_factor_oracle`，共享 factor、target 和 noise。
- 预先限定三个任务几何候选，按最宽容差优先：`tolerance = [0.20, 0.15, 0.10]`。不得依据 history 表现选择候选。

选择第一个同时满足下列条件的候选：

1. population-prior unsafe fraction 位于 `[0.25, 0.75]`，避免过易或全面失败；
2. true oracle 相对 population prior 的 mean early cost 改善至少 25%；
3. paired bootstrap 的相对改善 95% CI 下界至少 20%；
4. true-oracle mean cost 严格低于 population-prior mean cost；
5. 512/512 paired scenarios 有效且共享 nuisance/noise 的审计通过。

若无候选通过，裁决 `TASK_DYNAMIC_RANGE_NOT_ESTABLISHED / STOP_TASK_FAMILY`。若有候选通过，裁决 `TASK_DYNAMIC_RANGE_ESTABLISHED / GO_STAGE1_CONTRACT`，只授权为选中候选另写 Stage 1 冻结合同；不授权 neural memory、LoRA、router、expert 或 consolidation。

## 执行和证据

必须保留 resolved config、population prior 及 hash、逐 scenario raw rows、bootstrap、源码/合同 hash、命令、环境、wall time 和独立复算结果。Stage 0 不得读取或生成 formal outcomes，不得用 history oracle 调整任务。

最强替代解释与击穿条件：若 true oracle 只改善动作正则而非状态误差、population 已饱和/全面失败、或 gap 仅由不公平的 target/noise 产生，则本任务不合格。上述主 cost 明确只由下一状态误差和安全违反构成，paired identity 必须机器检查。
