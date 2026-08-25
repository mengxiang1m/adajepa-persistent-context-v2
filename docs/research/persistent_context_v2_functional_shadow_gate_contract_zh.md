# Persistent Context V2 Functional Shadow Gate V1 合同

合同 ID：`persistent-context-v2-functional-shadow-gate-v1`

## 研究问题

Dead zone、delay 和 rotation×gain matrix 都已保存 population plan 与 context plan 的严格配对真实执行。第一版 functional shadow gate 只读取规划前可获得的低维 context 估计和固定 population prior，在两条已执行计划中做离线选择：它能否减少“无条件使用精确 context”造成的负向尾部，同时保留已经存在的正向收益？

本阶段不训练 gate，不重跑 planner，不生成新动作，也不修改任何既有 raw。它是 retrospective paired shadow evaluation，不包装成新的前瞻随机实验。

## 信息边界

Gate 决策禁止读取：真实执行 cost、success、goal、segment ID、environment/CEM seed、planner `best_loss`、factor 分组结果或 best-of-two 标签。

Gate 只读取：

- dead zone：correct-history censored-MLE posterior mean；
- delay：真实 delay，仅作为 functional-ceiling，尚不可部署；
- matrix：correct-history Bayesian posterior mean matrix，由此计算 estimated gain/rotation；
- 各任务预先固定的 population prior。

Planner `best_loss` 不进入 gate。Schema 审计已经发现它在 dead zone 与 matrix 示例中会给出相反的真实行为方向，因此不能把 latent objective 差直接当可靠 gate。

## 冻结规则及来源

规则只表达早期 Stage 0 已观察到的结构，不按单个 Stage 1 sequence outcome 调整：

1. Dead zone：`estimated d > 0.10` 时使用 history-context plan，否则 population plan。
2. Delay：`delay > 2` 时使用 true-context plan，否则 population plan。该规则只测行为功能上限，不证明 delay 可被非特权估计。
3. Matrix：`estimated gain < 0.932780492` 或 `abs(estimated rotation)>=15°` 时使用 history-context plan，否则 population plan。`15°` 是 Stage 0 小角 `7.5°` 与大角 `22.5°` 的固定中点。

## 输入与统计单位

- Dead zone：Stage 1 persistent raw，32 sequences；每个 sequence 先对 E2–E4 cost 取均值。
- Delay：Stage 0 raw，32 pairs。
- Matrix：Stage 1 persistent raw，32 sequences；使用 E2 cold-start evaluation。
- 三个 raw SHA256 冻结在 design JSON。

对每个单位构造五个 policy outcome：

- `population`；
- `always_context`；
- `functional_gate`；
- `inverted_gate`：采用完全相反的选择，作为结构负对照；
- `best_of_two_behavior_ceiling`：逐单位取 population/context 中真实 cost 较低者，只表示可回收上限，绝不作为 gate 输入。

## 报告

每个任务分别报告 policy mean、population-policy delta、相对改善、paired bootstrap 95% CI、正/平/负比例、context selection rate、相对 population 的 harm fraction、gate 相对 always-context 的变化、best-of-two opportunity recovery 和 factor 分组。另报告三个任务的宏平均相对改善，但不把不同量纲 raw cost 直接合并。

不设置固定效果百分比门槛。结果无论正负都保留。

## 审计与下一步边界

独立 audit 必须重新验证 raw hash、单位抽取、posterior 到 gate decision 的映射、policy outcome 选择、bootstrap 和 summary。

如果 functional gate 在多个任务上同时减少 harm 并保留大部分 best-of-two opportunity，才进入 learned surrogate gate 数据合同；learned gate 必须使用独立 train/dev/formal split，不能直接在本批 raw 上训练和报告同批效果。
