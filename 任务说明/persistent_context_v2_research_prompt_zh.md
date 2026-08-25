# 跨 Episode 持续动力学上下文学习 V2：研究执行 Prompt

> **2026-08-22 用户修订：撤销固定数值效果门。** 后续实验不得再用预设的 `25%`、`20/30/40/50%`、`70/75/80%` 同向比例或“置信区间必须跨/不跨 0”等固定阈值自动裁决 `GO/NO-GO`。必须完整报告效应方向、效应大小、置信区间、同向比例、分组异质性和负对照，由用户结合研究成本决定是否继续。实现正确性、身份配对、预算一致性和 raw artifact 完整性仍是有效性审计，不属于被撤销的效果门。历史冻结合同与原始机器输出保留为当时规则下的审计记录，但其阈值裁决不再约束后续研究。

> **2026-08-24 现状校准：本 Prompt 是治理规范，不是从 Stage 0 重新执行的任务清单。** Rotation 与 rotation×gain matrix 已建立 persistent-history 因果证据；T 数据 P1-V2 显示 F0 相对 population 改善 `13.21%`。96 条跨-shape formal 显示 correct fixed `.75` 改善 `9.97%`，factor 改变时 history 平均无收益，因此跨任务物理 context 父命题成立。只读 D3 未找到稳定 harm selector。P2 delay formal 已完成：command/proprio 可 `32/32` 识别 delay，但 persistence-specific 行为值未建立。P3a/P3b CoG 表示审计均已完成：contact/impulse 有局部预测信息，10 Hz 拼接与 100 Hz event ridge 都未形成可部署 predictor；P3b 主比较 `C10-S100=-0.00728`，S100 也未超过 zero response。CoG V3/history 继续暂停，下一候选回到 matrix scene-level harm 的 rollout-disagreement feasibility。优先级以 [`Persistent_Context_V2_后续实验计划_2026-08-23.md`](./Persistent_Context_V2_后续实验计划_2026-08-23.md) 为准。

## 0. 角色与总目标

你是一名研究世界模型、在线系统辨识、测试时学习和模型预测控制的高级研究工程师。

本研究不继续旧的“无条件继承 AdaJEPA TTT 权重”路线，也不通过修改旧 oracle 的 factor、horizon、阈值或指标来寻求正结果。旧结论与 V2 已完成结果必须完整保留：

- Phase A–H 只证明了状态继承、safe consolidation、rollback 和因果 probe 等工程机制可运行，没有证明长期行为正迁移；
- P0A 在冻结的 PushObj-T 条件下得到 `PREMISE_NOT_ESTABLISHED`，说明当前 AdaJEPA episode 内权重更新不构成稳定、低伤害的跨 episode 知识；
- `persistent-actuator-factor-oracle-v1` 得到 `HISTORY_VALUE_NOT_ESTABLISHED / NO_GO_METHOD`，其弱 cost 收益不足、success 饱和，并且大部分收益可由非持续条件下的总体先验学习或平滑解释。
- PushObj rotation 与 rotation×gain matrix 在各自冻结条件下支持显式 history context 的闭环价值及 persistence-specific 解释；不得外推为所有 factor、shape、planner 或视觉模型都有效；
- dead zone、delay 与 matrix 的负向尾部表明 factor 估计准确不等于行为最优；factor-only learned gate 相对 population 正向，但相对 always-context 的额外均值优势仍不确定；task-interaction D0 提高了 outcome 预测相关性，却在硬决策后退化为 always-context，说明预测精度、context 强度和闭环行为必须分开验证；
- CoG physics oracle 上限稳定存在，但 v1 learned predictor 只回收少量 gap，temporal-GRU v2 未改善；P3a 控制边界拼接和 P3b 100 Hz event ridge 均未形成可部署 predictor。不得直接开发 CoG V3、CoG history encoder 或新 CoG closed-loop formal。

本 V2 是科学上独立的新问题：

> 在所有策略共享一个由训练分布校准的 population prior 时，过去 episode 的 transition 能否识别当前 sequence 中持续但未直接观测的动力学因素，并在新 episode 尚未获得足够本地数据前改善闭环控制？

对任何**尚未验证的新 factor/新环境**，研究顺序必须是：

`构造任务并测量行为动态范围` → `oracle 估计 persistence-specific history value` → `显式上下文估计方法` → `条件化世界模型` → `必要时才加入 episode-local TTT`。

不得用预测损失或参数变化替代行为结果；是否继续下一级由已报告的连续证据和用户决定，不再由固定效果阈值自动停止。

---

## 绑定的科研思路与证据规范

本节将 `docs/research/ai_research_reasoning_standard_zh.md` 的核心要求直接纳入本 Prompt。执行者不得把它当作背景建议；它是本研究的绑定工作规范。

### 规范优先级与单线程责任

- 研究由一个 AI 线程连续负责协议设计、实现、执行、复算和证据审计，不使用 executor/reviewer 双 Agent 角色划分来替代实验可信度；
- 可信度来自预注册、可辨识对照、机器可复算证据、反例检查和停止规则，而不是角色数量；
- 发生冲突时，优先级为：用户最新明确要求 > 当前冻结研究合同 > 本 Prompt/思路规范 > 历史 roadmap、prompt 和任务记录；
- 历史文档和探索性结果只能作为证据，不能自动授权继续旧路线。

### 不可改写的历史证据边界

- P0A 是有效负结果：60/60 正式 runs 有效，主效应 `weight_only - episodic` 的 T2 success-AUC 均值为 `-0.0708333333`，paired bootstrap 95% CI 为 `[-0.2041666667, 0.0125]`，仅 3/12 blocks 正向，10/12 harmed；
- 允许推断：在 P0A 冻结的 checkpoint、PushObj-T blocks/seeds、预算、更新层和无条件权重继承条件下，当前 AdaJEPA 在线更新没有形成可重复、低伤害、值得保存的稳定知识；
- 禁止推断：长期记忆普遍不可能、所有 continual TTT 均无效，或其他任务、模型和持续因素也不可能利用历史；
- 旧 persistent-factor oracle 的 `NO_GO_METHOD` 只约束旧任务实例，不能被改名、调参或换阈值后当作同一实验重开。

### 必须遵循的研究依赖链

任何方法开发前必须逐级验证：

```text
任务确有跨 episode 共享因素
→ 历史观测能够识别该因素
→ 知道该因素能够改善后续闭环行为
→ 非特权方法能够从历史提取该收益
→ 复杂神经方法确实优于简单充分统计量或重新拟合
```

- 父命题为 `UNTESTED`、`INCONCLUSIVE` 或总体负向时必须明确披露；是否继续下游方法由用户决定，不由自动门禁决定；
- prediction loss、参数变化、accept rate、无遗忘、单 seed 改善或漂亮的 latent 可视化不能替代闭环行为正迁移；
- 如果 transition 中没有可辨识的稳定因素，正确结论是“不建立长期变量”，而不是强行学习一个 `z_seq`。

### 合同必须先于正式结果

在任何 formal run 之前，必须冻结：

- 核心命题、唯一主处理变量和可证伪零假设；
- 任务生成机制、持续因素、episode nuisance 和状态边界；
- train/development/formal split、样本量、统计单位和 seeds；
- paired controls、控制与更新预算、主行为端点和评价窗口；
- bootstrap/test 算法、随机种子，以及不带自动裁决阈值的完整报告字段；
- 工程有效性、停止条件、有限修复范围和禁止的后续动作。

合同冻结后不得因 smoke、subgroup 或正式结果改变 factor、seed、样本、阈值、主指标、baseline 或解释范围。工程 bug 只允许不改变科学处理的有限修复，并必须保留失败产物。

### Oracle-first 与可辨识性原则

新方向的第一个科学实验必须是 history-value upper-bound assessment，而不是 memory 网络。最低限度包含：

1. `population_prior/current_only`；
2. 只使用过去真实信息的 `correct_history` oracle；
3. 直接知道隐藏因素的 `true_factor_oracle`；
4. factor 每 episode 重采样的 `no_persistence`；
5. shuffled history 和 wrong-sequence history。

是否进入方法阶段必须依据 held-out later episodes 的完整闭环证据：correct history 相对 current-only 的方向和区间、true-oracle gap 回收、同向比例，以及 no-persistence、shuffled、wrong-sequence 负对照。不得用“显著/不显著”或任一固定效果阈值自动授权或阻断下一阶段；若负对照同样获益，则 persistence-specific 解释不成立。

所有比较必须满足：

- sequence 是最小独立统计单位，不得把 episode、step 或 transition 伪装成独立样本；
- paired 策略共享 factor、start、goal、nuisance、噪声、控制预算和更新预算；
- Episode 1 的 current-only/history 必须精确一致，处理差异只能从跨 episode 边界产生；
- model、optimizer、replay、显式 context/statistics 与 RNG 必须分开记录生命周期；
- 若关键负对照同样获益，不得声称收益来自持续因素。

### 单线程三次检查

同一执行线程必须按顺序完成三次检查，并在每次检查结束时留下机器或书面产物：

1. **设计检查**：结果未知时冻结合同；列出最可能的混淆、替代解释、失败模式和击穿首选解释的负对照；
2. **执行检查**：只按合同实现和运行；记录命令、代码与资产 hash、环境、资源、失败、重试和偏差；不得依据中途结果改变科学处理；
3. **证据检查**：从 raw artifacts 独立复算主结果，主动寻找反例，并分别报告效应方向、大小、不确定性、同向比例、异质性和工程有效性；只有实现或审计错误可标为 `INVALID`，不得依据固定效果阈值自动给出 `GO/NO-GO`。

事实、推断和猜测必须分开书写。执行成功不等于科学假设成立，工程有效性也不能覆盖不利或不确定的主行为结果。

### 结果处置与停止纪律

- 后续处置不再由固定数值门自动决定；报告必须把“均值正向但较小”“均值负向或不确定”“大部分配对正向”“收益集中于子组”等情况分开陈述，由用户决定是否投入下一级实验；
- `INVALID`：仅用于可定位的实现、身份或审计错误；不得把科学失败重命名为无效执行；
- 阶段间不得围绕单个正向 seed 连续发明新解释、指标或模块；探索性异常只能生成新的、独立冻结且跨 seed 复验的假设，不得回写当前 formal 结论；
- 每个方向判断必须回答：已经证明什么、没有证明什么、最强替代解释是什么、下一步为何是被证据授权的最小动作。

---

## 1. 核心概念与状态分解

每条 sequence 记为 `S_j`，含多个 episode。生成过程必须显式分解为：

```text
z_seq,j                  跨 episode 持续的隐藏动力学因素
xi_ep,j,i                第 i 个 episode 独有的起点、目标、接触状态等因素
tau_j,i ~ P(tau | z_seq,j, xi_ep,j,i)
```

策略不得直接观察 `z_seq`。它只能通过真实执行 transition：

```text
(o_t, s_t, a_t, o_{t+1}, s_{t+1})
```

形成对 `z_seq` 的估计或 belief。

状态生命周期必须显式分开：

| 状态 | 生命周期 | 例子 |
|---|---|---|
| base model | 全部实验固定 | 视觉 encoder、基础 dynamics predictor |
| population prior | 全部测试 sequence 共享，只由 train split 得到 | `p(z)` 的均值/协方差 |
| `q(z_seq)` | 同一 sequence 跨 episode 保留 | actuator bias、摩擦、stiffness 的 belief |
| `z_ep` | 每个 episode 重置 | 当前接触模式、局部形变、当前轨迹残差 |
| raw/history statistics | 按 sequence 隔离 | 递归最小二乘充分统计量或有限 transition set |
| TTT optimizer/replay | 默认每 episode 重置 | 用户决定开展相应阶段时使用 |

长期知识的首选载体是低维 `q(z_seq)` 或可解释充分统计量，不是全局模型权重。

---

## 2. 研究假设

### H1：任务具有可利用的持续因素

在相同 start/goal/noise 和控制预算下，估计直接知道 `z_seq` 的 true-factor oracle 相对 population prior 的行为效应，并完整报告其方向、大小、不确定性、同向比例和 factor 异质性。不得用单一固定数值阈值把正向小效应改写成“没有上限”。

### H2：历史价值具有持续因素特异性

正确 sequence history 应在 persistent 条件下改善新 episode 前期行为；当 factor 每 episode 重采样、history 被打乱或来自其他 sequence 时，该收益应消失或显著减弱。

### H3：显式上下文优于无条件权重继承

若 H1/H2 成立，保存 `q(z_seq)` 的方法应比无条件保存 AdaJEPA 权重更可辨识、更稳定，并能在 held-out sequence 上复现收益。

### H4：episode-local 与 sequence-level 信息可以分离

`z_seq` 解释跨 episode 稳定动力学，`z_ep` 或 episodic TTT 只处理当前接触和轨迹残差；加入 episodic 部分不得消除已经建立的 episode-entry history benefit。

---

## 3. Stage 0：任务开发与动态范围校准

本节及 Stage 1–4 是新 factor/新环境的复用模板；已经完成的 rotation、matrix、CoG 等实验不得借此改合同重跑。Stage 0 是开发期 benchmark qualification，不是正式科学结果。必须使用独立的 train/development factors 和 seeds，禁止读取 formal held-out factors/seeds。

### 3.1 因素选择原则

首个任务只改变一个低维因素，候选包括：

- actuator gain、rotation、dead zone 或 action delay；
- PointMaze 的质量、阻尼或 action calibration；
- PushObj 的摩擦、质心或工具标定；
- deformable 环境中的 rope stiffness、摩擦或工具偏差。

选择的因素必须同时满足：

1. 在一条 sequence 中严格固定，在不同 sequence 间变化；
2. 不从单帧 observation 或显式 task ID 直接泄漏；
3. 能从多条真实 transition 中辨识；
4. 改变该因素会改变最优动作，而不仅改变无关预测 loss；
5. 在做出关键动作前，单个 episode 的安全数据不足以完全辨识；
6. 错误动作具有真实的时间、能量、安全或不可逆代价，而不是人为删除正常学习机会。

### 3.2 只允许用 true oracle 校准任务

开发期只能依据以下两种策略选择任务配置：

- `population_prior`：使用由 train factors 得到的正确总体先验；
- `true_factor_oracle`：直接获得真实 `z_seq`。

不得在开发期用 history oracle 的方向选择 factor table、episode 长度或 success threshold，以免把任务调成偏爱历史。

Stage 0 必须在 development split 选择一个主行为端点并量化动态范围：

- 若使用 success：报告 population-prior 与 true oracle 的 success、百分点差、配对不确定性和 ceiling/floor 情况；
- 若使用 early cost/regret：报告 true oracle 相对 population prior 的绝对与相对改善、配对不确定性和同向比例，并确认差异来自状态/任务代价，不得只来自微小 action-energy 正则项。

success 与 cost 仍需在结果前选定一个主指标，不能在看到结果后切换；但该指标不再绑定固定通过线。

### 3.3 冷启动信息约束

任务应测量真实的 cold-start 价值：

- **允许并可能需要重构 benchmark，减少单个 episode 在关键决策前可用于试错和在线辨识的 transition/动作次数。** 当前 benchmark 给 AdaJEPA 足够大的 episode 内试错空间时，current-only/episodic 方法可能在本 episode 内重新学会动力学，从而掩盖跨 episode 信息的价值；新 benchmark 应让历史知识在新 episode 的早期决策中具有可观察优势；
- 每个 episode 的关键决策出现在当前数据足以完整辨识 `z_seq` 之前；
- 所有策略拥有相同动作、观测、规划和更新预算；
- 允许额外报告随 episode 内数据增加，history 与 reset 的差距如何收敛；
- 正式合同必须固定关键决策前允许的最大 environment transitions、exploratory actions、TTT/context updates 和规划次数，不能让 history 方法获得额外在线数据；
- benchmark 的目标是匹配“首次错误动作代价较高、不能无限试错”的跨 episode 学习场景，而不是无原则削弱 episodic baseline；不得仅通过随意缩短 horizon 制造正结果，必须说明该信息约束对应的现实控制代价，并由 true-factor oracle 的行为差距证明任务确有可改善空间。

### 3.4 Stage 0 完成条件

完成预先限定的候选任务集合后，汇总每个候选的效应方向、大小、置信区间、同向比例与 factor/任务异质性。不得因未达到某个预设百分比自动停止，也不得因均值略正就隐瞒不确定性；是否继续 history oracle 或方法实验由用户根据完整证据决定。

---

## 4. Stage 1：冻结的 History-Value Oracle 评估

用户根据 Stage 0 连续证据决定继续后，先写一份独立、冻结的 Stage 1 合同，再生成 formal 结果。合同必须固定 factor split、sequence 数、episodes、seeds、噪声、控制器、预算、主指标和 bootstrap，但不得设置固定效果量自动裁决线。

### 4.1 必需策略

1. `population_prior`：不使用当前 sequence 历史，只使用 train split 校准的 `p(z)`。
2. `current_only`：每个 episode 从相同 population prior 开始，仅用当前 episode transition 更新 `q(z)`。
3. `correct_history`：与 current-only 使用完全相同的估计器、控制器和当前数据，唯一差异是跨 episode 保留 `q(z_seq)` 或充分统计量。
4. `shuffled_history`：历史样本数和预算相同，但打乱 episode/transition 归属。
5. `wrong_sequence_history`：使用另一条独立 sequence 的等量历史。
6. `true_factor_oracle`：直接使用真实 `z_seq`，给出行为 ceiling。

所有非特权策略必须共享相同且正确的 population prior。不得再用错误的固定 `I` prior 使跨 episode 混合数据看似有益。

### 4.2 两个生成条件

- `persistent`：一条 sequence 内 `z_seq` 固定；
- `no_persistence`：每个 episode 独立重采样 factor，但边际 factor 分布与 persistent 完全相同。

同一 paired sequence 内必须共享 start、goal、episode nuisance、环境噪声和预算。Episode 1 中 `current_only` 与 `correct_history` 必须逐 action、state、cost 精确一致。

### 4.3 主评价窗口

主指标只评价 later episodes 的 cold-start 窗口，正式合同必须预先固定以下之一：

- 第一个闭环动作的反事实 regret；
- 前 `K` 个闭环 step 的 cumulative task cost/regret；
- 第一次当前 episode 更新之前的 success-AUC；
- 在辨识置信度达到阈值前的 safety violation 或 irreversible failure。

完整 episode success、最终 cost、factor estimation error 和 prediction loss 只作辅助指标，不能替代预注册的 cold-start 主端点。

### 4.4 Persistence-specific 主效应

若 cost 越低越好，定义每条 sequence：

```text
Delta_persistent = cost(current_only) - cost(correct_history)
Delta_no_persistence = cost(current_only) - cost(correct_history)
DiD = Delta_persistent - Delta_no_persistence
```

`DiD > 0` 才表示历史收益特异地依赖 factor 持续性。success 指标使用相同方向的差分定义。

正式统计单位必须是 sequence。对 paired sequence effect 做预注册 bootstrap 或层级模型，不得把 episode、step 或 transition 当作独立样本。

### 4.5 必报证据（无固定 GO 门）

正式合同不再预设自动裁决的效果量或同向比例阈值。最低限度必须报告：

1. true-factor oracle 相对 population prior 的主行为差距及置信区间；
2. persistent 条件下 `correct_history - current_only` 的配对效应及置信区间；
3. `DiD` 的配对效应及置信区间；
4. correct history 回收 true-oracle gap 的比例；
5. 独立 sequences 的同向改善比例，ties 单独报告；
6. shuffled、wrong-sequence 和 no-persistence 的效应及其与正确 history 的差别；
7. factor、任务难度和 episode 的异质性；
8. 所有 identity、预算、state-lifetime、RNG 和 raw-artifact 审计结果。

不得再因任一统计量未达到固定百分比或置信区间阈值而自动输出 `HISTORY_VALUE_NOT_ESTABLISHED / NO_GO_CONTEXT_METHOD`。报告应使用描述性结论，如“总体正向且区间排除 0”“总体正向但不确定”“均值由大 factor 子组主导”或“总体负向”，并把下一步选择交给用户。

---

## 5. Stage 2：显式上下文估计方法（由用户结合 Stage 1 连续证据决定是否开展）

Stage 2 的目标不是长期记忆网络，而是先证明非特权方法能从历史 transition 恢复 Stage 1 的 oracle 收益。

### 5.1 首选最小方法

若 factor 和动力学形式已知，优先使用：

- recursive least squares；
- Kalman/Bayesian parameter filter；
- 低维 maximum-likelihood/MAP system identification。

持久状态为：

```text
SequenceContextState:
    posterior_mean
    posterior_covariance
    sufficient_statistics
    transition_count
    sequence_id
    change_detection_state
```

该状态存放在独立 sequence state 中，可序列化和审计；不得写入 base model 权重。sequence 结束或检测到 factor change 时重置。

### 5.2 未知因素的 learned context

用户根据显式 estimator pilot 的连续证据决定继续后，可使用 transition set encoder、RNN 或 Transformer 推断低维 `z_seq`。训练必须约束：

- 同 factor、不同 start/goal/trajectory 的 `z_seq` 一致；
- 不同 factor 的 context 可区分；
- `z_seq` 能改善 held-out later-episode transition 和闭环行为；
- context 维度受限，不能编码整条轨迹；
- wrong/shuffled history 不得产生相同收益。

如果 learned `z_seq` 只改善 prediction loss 而不改善 Stage 1 的主行为指标，只能结论为“当前方法尚无闭环行为证据”；不得用辅助指标将其包装成成功，也不得外推为所有 learned context 都无效。

---

## 6. Stage 3：Context-Conditioned World Model（由用户结合 Stage 2 连续证据决定是否开展）

现有 checkpoint 没有显式 context 接口，不能仅在测试时估出 `z` 后期望 predictor 自动使用。必须在 factor-diverse training data 上重新训练：

```text
h_(t+1) = F(h_t, a_t, z_seq, z_ep)
```

其中：

- `z_seq` 表示跨 episode 稳定物理因素；
- `z_ep` 表示当前 episode 的接触、局部形变或轨迹残差，每 episode 重置；
- base encoder/predictor 不通过无条件跨 episode TTT 漂移。

首个实现只选择一种条件化机制：

- predictor context token；或
- FiLM conditioning；或
- 由 `z_seq` 生成的单个小 residual adapter。

不得在第一版同时加入多种 adapter、router、expert、replay gate 和 consolidation。

### 6.1 训练数据要求

- train/dev/formal test 使用隔离的 sequence 和 seeds；
- 至少 formal test factor 组合或连续区间未在训练中逐点出现；
- 每个 factor 覆盖多种 start、goal 和 trajectory，避免 context 与轨迹绑定；
- factor labels 可用于第一版监督训练，但测试时不得提供；
- 必须报告 true-`z` conditioned model 的行为上限，确认模型确实会使用 context。

### 6.2 规划使用方式

MPC 必须在 rollout 时显式条件于 `q(z_seq)`：

- 最小基线：使用 posterior mean 做 certainty-equivalent planning；
- 不确定性扩展：从 `q(z_seq)` 采样多个 context，优化 expected cost 或风险敏感 cost；
- 是否研究 dual control/主动辨识动作由用户结合已报告的行为价值决定。

主因果链必须可记录：

```text
history transition
→ q(z_seq) 变化
→ predicted rollout 变化
→ action ranking/action 变化
→ simulator outcome 变化
```

---

## 7. Stage 4：与 AdaJEPA episode-local TTT 组合（最后阶段）

用户结合 context-conditioned model 的闭环连续证据决定继续后，可加入 episodic AdaJEPA：

```text
z_seq / context estimator：跨 episode 保留稳定动力学
z_ep 或 episodic TTT：只处理当前轨迹局部残差
```

必须比较：

- context only；
- episodic AdaJEPA only；
- context + episodic AdaJEPA；
- unconditional weight carry；
- true-context upper bound。

组合方法必须证明：

1. 保留 context 在 episode entry 的既有收益；
2. episodic TTT 能改善后期局部适应；
3. fast/episodic 更新不会系统性掩盖或破坏 `z_seq`；
4. 不依赖全局权重跨 episode 累积。

---

## 8. Deformable 环境的前置重构门

deformable 环境科学上适合研究持续 stiffness、friction 或 tool calibration，但当前实现不得直接进入正式实验。至少先完成：

1. dataset 每条 rollout 返回并持久化物理 factor、object identity、seed 和生成配置，不能继续返回空 `info`；
2. `FlexEnvWrapper.update_env()` 能在 sequence 开始时设置 factor，并证明 episode reset 不改变该 factor；
3. factor change 必须安全重建 scene，不能留下上一 sequence 的模拟器状态；
4. 修正 `Chamfer distance < 0` 的不可能 success 定义，在 development split 校准有动态范围的 success/AUC；
5. 建立 ordered sequence manifest 和 train/dev/formal factor split；
6. 生成 factor-diverse 数据并训练能够使用 context 的 checkpoint；
7. 验证 deterministic paired replay、状态恢复、factor hash 和资源可审计；
8. 先完成 Stage 0/1 非神经 oracle 证据与负对照报告，再由用户依据连续证据决定是否投入神经 context 方法。

首个 deformable 实验只选一个对象和一个 factor。例如固定 rope、只改变 stiffness；不得同时改变 stiffness、长度、摩擦、颜色和工具。

---

## 9. 实验完整性与报告要求

每个正式阶段必须在结果生成前冻结：

- 核心假设和唯一主处理变量；
- train/dev/formal split；
- factor 生命周期和 change boundary；
- sequence/episode 数、seeds、start/goal/noise；
- population prior 的拟合数据与 hash；
- 控制器、规划预算、观测和更新预算；
- 主行为指标、窗口 `K`、统计方法与连续效应报告字段；
- 无效执行和有限修复规则，以及连续证据的必报字段；
- 允许与禁止的下一阶段动作。

每条 formal run 至少保留：

- resolved task/sequence manifest；
- factor hash（对策略隐藏，对审计可见）；
- observation/action/noise budget；
- 每步 state、action、cost、success 和 belief；
- `q(z_seq)` 更新前后状态及 hash；
- model、optimizer、replay 和 RNG 生命周期审计；
- 原始结果、聚合脚本、bootstrap 配置、资源记录和偏差。

报告必须分开：

1. 机器事实；
2. 由主对照支持的范围内推断；
3. 替代解释；
4. 未测试问题；
5. 用户是否决定开展下一阶段。

---

## 10. 全局禁止事项

- 不得修改旧 P0A 或旧 oracle 的合同后重跑以寻求正结果；
- 不得把 task calibration 使用的 seeds/factors 再作为 formal test；
- 不得用错误 population prior 制造 history 优势；
- 不得把 prediction loss、factor estimation error、accept rate 或参数变化提升为主要行为证据；
- 不得把 step/transition 当作独立统计单位；
- 不得用固定效果门自动阻断或自动授权 neural memory、LoRA、router、expert 或 consolidation；是否继续必须基于完整行为证据和用户明确决定；
- 不得把显式 `z_seq` 偷换成未受约束、可记忆整条轨迹的大型 embedding；
- 不得把一个 task instance 的失败外推为长期学习普遍不可能；
- 不得为了得到正结果同时改变 factor、episode budget、success threshold、主指标和模型结构。

---

## 11. 当前推荐执行顺序

1. 保留并归档 Phase A–H、P0A、旧 oracle、所有冻结合同、负结果和 raw artifacts，不再扩展无条件权重持续路线；
2. matrix D1/D2、作者 T 池 P1-V2、跨-shape formal 与只读 D3 已完成；当前默认简单基线为 fixed `α=0.75`，现有特征不支持新的 harm-aware formal；
3. delay 非特权 32-sequence formal 已完成：可辨识但 persistence-specific 行为值未建立，暂不在相同数据上继续调 gate；
4. CoG P3a/P3b 表示审计已完成；100 Hz S100 full 未超过 C10 或 zero response，因此 CoG V3/history 暂停。下一候选是 matrix rollout-disagreement harm feasibility，只允许先补只读日志并用未暴露开发数据检验；
5. 只有显式 context 与行为选择器都稳定后，才比较 `context only / episode-local TTT only / context + TTT / true-context`；
6. 最后开展 `z_shape + z_physics` 的 T/L/Z 因子化实验，再迁移到完整视觉 AdaJEPA 或 deformable 环境。

具体假设、拆分、端点、对照、资源估计和停止纪律见当前后续实验计划；本 Prompt 不替代每项实验在结果产生前冻结的独立合同。

最终研究目标不是证明“模型能够保存历史”，而是建立完整因果链：

> 历史包含持续因素信息；该因素可从历史辨识；知道它能改善新 episode 的早期闭环行为；显式 context 方法能回收 oracle gap；复杂神经方法确实优于简单充分统计量。
