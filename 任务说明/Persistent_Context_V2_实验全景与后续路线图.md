# AdaJEPA Persistent Context：实验全景、当前结论与后续路线图

更新日期：2026-08-23

文档定位：本文件是证据全景，不是 formal 合同或可直接执行的 prompt。当前规范见 [`persistent_context_v2_research_prompt_zh.md`](./persistent_context_v2_research_prompt_zh.md)，后续实验的具体拆分、端点和资源计划见 [`Persistent_Context_V2_后续实验计划_2026-08-23.md`](./Persistent_Context_V2_后续实验计划_2026-08-23.md)。

## 1. 这项研究到底在解决什么

核心问题不是“模型能不能在单个 episode 内继续训练”，而是：

> 如果某个隐藏物理因素会在多个 episode 之间持续存在，能否把过去 episode 的经验压缩成一个独立、低维、可审计的 `context`，并在新 episode 一开始、还没有机会重新试错时，用它做出更好的动作？

目前研究的主要因果链是：

```text
过去 episode 的 transition
→ 估计持续因素 q(z_seq)
→ world model 的未来预测改变
→ planner 的动作排序和命令改变
→ 新 episode 的真实行为改善
```

这里刻意把两类适应分开：

- `z_seq`：跨 episode 持续的因素，例如工具坐标旋转、动作增益、物体重心；
- `z_ep`：只属于当前 episode 的接触状态、局部误差或短期残差，每次 reset 后清空。

当前路线不再依赖“无条件保存 AdaJEPA 在线更新后的全部权重”。稳定知识优先放进显式 context 或充分统计量，base model 不在 episode 之间无条件漂移。

## 2. 当前总体判断

目前已经得到四层不同强度的证据。

### 2.1 已经建立完整因果链的方向

PushObj 工具旋转和 rotation×gain matrix 已经完成：

```text
真实 factor 有行为价值
→ 非特权历史能估准 factor
→ correct history 改善新 episode
→ factor 不持续时旧历史有害
→ shuffled / wrong history 不能复制收益
```

这说明“跨 episode 显式 context”在真实 PushObj 接触环境中不是只改善预测指标，而是确实可以改善动作和结果。

### 2.2 有明确上限、但 learned model 还不够好的方向

PushObj 水平 CoG 的 physics oracle 很强，第一批改善约 `26.87%`；但是第一版 CoG-conditioned neural predictor 只改善约 `4.68%`，只回收了约 `23.06%` 的 oracle gap。随后在全新场景上测试的 temporal-GRU v2 只改善 `0.37%`，只回收 `2.06%` oracle gap，相对 v1 反而退化 `2.78%`。

这意味着 CoG 方向有价值，但“仅把 flattened MLP 换成 GRU”并没有解决问题。P3a 随后确认旧 7 维输入确实缺少 block velocity、angular velocity 和 contact；contact 摘要能改善低容量 ridge，却不能改善冻结 v1。当前瓶颈进一步收敛为接触冲量响应及时间分辨率，而不是简单缺字段或任务本身没有信号。

### 2.3 只有条件性价值的方向

Dead zone 和 discrete delay 都出现相同现象：

- 当真实 dead zone/delay 大于 population prior 时，精确 context 明显有益；
- 当真实值小于 prior 时，更准确的 context 反而可能使当前 frozen predictor/CEM 行为变差。

因此“物理参数估得更准”不等于“有限预算 planner 一定做得更好”。这类 factor 以后需要行为 gate、robust planning 或不确定性处理，不能无条件使用 posterior mean。

### 2.4 当前没有建立行为价值的方向

在发布的 PointMaze AdaJEPA checkpoint 上，scalar gain 和连续 actuator lag 都会改变 CEM 动作，但 true factor 没有改善真实 MuJoCo 结果。这里的问题不是 context 没有进入 planner，而是更准确的动力学标定没有转化为更好的有限预算动作排序。

因此当前 PointMaze checkpoint/planner 组合不是继续开发 persistent history 的优先平台。

### 2.5 前置 T/L/Z 权重持续路线：安全性成立，长期行为优势未成立

在显式低维 context 路线之前，远端已经完整执行过 Phase A–H，直接研究 AdaJEPA 权重、optimizer、replay、fast/slow LoRA 和长期 memory 在 `T→T`、`T→L→T`、`T→Z→T`、`T→L→Z(red)` 长序列中的持续适应。

这条路线得到的客观结论是：

- naive full-weight carry 会产生可重复的 recurring-T 干扰；
- planning-aware safe consolidation 能把 T2–T4 的 pre-loss 干扰从约 `10^-2` 压到 `10^-6`，并消除 T4 success-AUC 插入损失；
- 但当前 shared slow LoRA 的 long-TLZ success/AUC 只有 `0.500/0.348`，低于 episodic 的 `0.733/0.478`；
- task/shape 梯度没有形成稳定、强烈的 T/L/Z 对立结构；
- 增加 slow update、短期 functional probe、episode-entry probe 和统一 fast scale 都没有建立跨 replicate 的长期行为优势。

因此 Phase A–H 不是“没做过”，而是已经完成的一条重要负结果链。它解释了为什么后续路线转向显式、低维、可审计的 persistent context，而不是继续无条件保存模型权重。

## 3. 已完成实验总表

| 实验 | 环境/因素 | 做了什么 | 主要结果 | 当前解释 |
|---|---|---|---|---|
| 合成 Stage 0 | 一维持续 gain | 比较 population 与 true factor | early cost 改善 `99.70%` | 合成任务有充分动态范围 |
| 合成 Stage 1 | 一维持续 gain | correct history 与多种负对照 | persistent 改善 `99.69%`；no-persistence 退化 | 历史价值依赖 factor 持续 |
| 合成 Stage 2 | 一维持续 gain | 三标量 RLS context | 改善 `99.54%`，回收 `99.86%` oracle gap | 简单充分统计量足够 |
| 合成 Stage 3 | FiLM world model | factor-diverse 训练 + RLS context | persistent RLS 改善 `99.36%` | 合成环境完整链路成立 |
| Persistent actuator oracle | 合成二维 rotation×双轴 gain | episode reset、history oracle、true factor | dense cost 改善 `5.96%`；252/256 正向；persistence DiD `+0.001783` | 小幅稳定收益存在，但大部分不是 persistence-specific |
| PointMaze gain | scalar action gain | 冻结 AdaJEPA + true-context action adapter | 主指标退化 `0.80%` | context 到达动作，但没有行为价值 |
| PointMaze lag | 连续 actuator lag | hard goal 与 local waypoint 两个候选 | 分别退化 `2.45%`、`0.38%` | 当前 PointMaze 分支不适合继续 history |
| Phase A | PushObj T/TLT | ordered manifest、W/O/B 生命周期、只读 probe | 10/10 tests、2/2 checkpoint smoke；hash/cleanup 一致 | 工程与因果审计基础设施建立，不是行为结论 |
| Phase B | PushObj TT/TLT | weights/optimizer/replay 2×2×2 matched 因子实验 | 48/48 runs；replay 改善 latent post-loss，但行为 success `0/8`；W carry 有插入干扰 | latent loss 与真实规划行为不一致，naive carry 不可靠 |
| Phase C | PushObj T/L/Z(red) | fast/slow LoRA、memory、planning-aware accept/rollback | recurring-T 干扰约减少 `99.41%`；long-TLZ `success/AUC=0.500/0.348`，低于 episodic `0.733/0.478` | 安全写入成立，长期行为增益未成立 |
| Phase D | T/L/Z slow-gradient | 58 个只读 gradient signatures | within-cross cosine margin仅 `0.001773/0.002217`；shape 对立不稳定 | 不支持直接增加 task-routed experts |
| Phase E | objective alignment + slow steps | gradient alignment 与 steps=2 paired formal | accepts `6→11`，但 AUC `10.45→10.35`；episodic `14.35` | 更多被接受更新不等于更好行为 |
| Phase F | simulator functional probe | committed/candidate slow 的确定性反事实规划 | 10/10 可辨识；accepted `4/4` 局部改善，但未触发可验证 veto | probe 工具成立，不能解释 delayed 跨 episode 退化 |
| Phase G | episode-entry causal transfer | source slow 对 persistent slow | long-TLZ `6/25` 改善、`19/25` 持平、`0/25` 受损；仅 1/3 replicate 支持整体正向 | 存在局部信号，但不可重复为总体优势 |
| Phase H | matched-fast replay | fast scale `{0,.25,.5,1}` 反事实 | directional recovery 仅 `1/3` replicates | fast masking 依赖 context，不存在统一全局 scale |
| PushObj rotation oracle | 工具坐标旋转 | prior `0°` 对 true rotation | 完整窗口改善 `13.35%`；早期接触旧定义改善 `1.64%` | rotation 有正向行为信号，强度依任务定义而变 |
| PushObj rotation history | Procrustes/MLE | 32 persistent + 32 no-persistence sequences | persistent 改善 `13.88%`；31/32 正向；no-persistence 退化 `6.20%` | 完整因果链成立 |
| Rotation early-waypoint oracle | 第 10 步真实 waypoint | 10-action deadline | 改善 `6.37%`；success `71.875%→93.75%` | 早期不可补救代价更能体现 context 价值 |
| Rotation early-waypoint history | Procrustes/MLE | E2–E4 cold-start | 改善 `12.06%`；success `59.375%→93.75%`；no-persistence 退化 `5.12%` | early-waypoint history 明确有效 |
| PushObj dead zone | radial soft threshold | true factor + censored MLE history | 总体约 `1.58%`；高 dead zone 改善 `9.95%`，低 dead zone 退化 `9.47%` | 只能条件性启用 |
| PushObj discrete delay | 0/1/3/4 步 FIFO delay | true-factor oracle | 总体改善 `1.48%`；3/4 步改善 `5.03%/12.55%`，0/1 步负向 | 需要 one-sided/风险感知策略 |
| Rotation×gain matrix oracle | 2×2 actuator matrix | true matrix 对 population matrix | 改善 `10.23%`；success `62.5%→100%` | 多因素联合 context 有明确价值 |
| Bayesian matrix history | Bayesian `c,s` posterior | E1 evidence → E2 cold-start | 改善 `9.53%`；回收约 `100.43%` oracle gap；no-persistence 退化 `20.02%` | 当前最完整、最有扩展价值的真实证据之一 |
| CoG physics oracle | T 物体水平质心 | ground-truth simulator CEM | 改善 `26.87%`；26/32 正向；success `81.25%→96.875%` | 当前最强的新物理因素行为上限 |
| CoG FiLM predictor v1 | CoG-conditioned residual model | 7,680 train，768 dev，32 formal pairs | prediction MSE 降低 `35.91%`；闭环改善 `4.68%`；23/32 正向；CI 跨 0 | 模型学到信号，但只回收 `23.06%` oracle gap |
| CoG temporal predictor v2 | 单向 GRU + 单一 CoG FiLM | 复用冻结 train/dev，32 个全新 formal pairs | 闭环改善 `0.37%`；相对 v1 退化 `2.78%`；只回收 `2.06%` oracle gap | 单纯增加时序骨干无效，CoG history 暂缓 |
| Functional shadow gate | dead zone、delay、matrix | 96 个冻结配对单位上按低维 factor estimate 选择 population/context | 三任务分别改善 `5.66%/6.11%/11.15%`，宏平均 `7.64%`；always-context 为 `4.20%` | 条件选择能显著减少 context 的负向尾部；delay 部分仍是 true-factor ceiling |
| Matrix learned surrogate gate | 96 条独立 train/dev/formal sequences | 二次 ridge 从 posterior gain/rotation 预测 paired 行为差 | formal 改善 `12.29%`；always-context `11.29%`；harm `31.25%→21.875%` | learned gate 首次独立 split 正向；factor-only 仍无法解释组内场景差异 |

本文不再使用人为固定的效果百分比作为统一裁决门。所有方向都报告连续效应、配对区间、同向比例、factor 异质性和负对照，再结合实现成本与剩余 oracle gap 判断优先级。

## 4. 关键实验的详细含义

### 4.1 合成 Stage 0–3：先验证逻辑是否成立

合成任务使用跨 episode 固定的一维 actuator gain。新 episode 的第一次关键动作前没有当前 episode 的辨识数据，因此历史信息有明确冷启动价值。

结果显示：

- correct history 几乎等价于 true factor；
- 只保存 `sum_u2、sum_uy、count` 三个充分统计量的 RLS 已经够用；
- FiLM-conditioned world model 会真正使用 context 改变预测和动作；
- factor 不持续、history 被打乱或来自错误 sequence 时，收益消失并转为伤害。

它证明了研究逻辑，但任务是一维、线性、低噪声，不能直接代表视觉接触动力学。

详细结果：[`persistent_context_v2_results_zh.md`](../docs/research/persistent_context_v2_results_zh.md)、[`persistent_context_v2_stage3_results_zh.md`](../docs/research/persistent_context_v2_stage3_results_zh.md)。

### 4.2 PointMaze：为什么模型换了动作却没有变好

PointMaze 的 gain 和 lag 实验都证明了 context 已经进入模型：所有 paired scenarios 的 CEM plan 都改变了。但真实 MuJoCo 结果没有改善。

这说明当前发布 predictor、latent goal distance 和有限预算 CEM 之间存在代理误差。更准确的 action transform 可能改变动作，却不保证改变后的动作在真实环境里更好。继续做 RLS 只会回答“能不能估准参数”，不能回答“估准后有没有用”。

详细结果：[`persistent_context_v2_pointmaze_transfer_results_zh.md`](../docs/research/persistent_context_v2_pointmaze_transfer_results_zh.md)、[`persistent_context_v2_pointmaze_lag_stage0_results_zh.md`](../docs/research/persistent_context_v2_pointmaze_lag_stage0_results_zh.md)。

### 4.3 Rotation：最简单、最清楚的真实 PushObj 成功案例

环境把 planner command 变换为 `R(theta)u`。Procrustes/MLE 只读取过去 episode 的 command 和 agent proprio transition，就能把持续角度估到接近数值精度。

在完整 25-step 任务中：

- current-only `pose_auc25=5.376122`；
- correct history `4.629919`；
- 改善 `13.8799%`，CI `[+0.480323,+1.035029]`；
- 31/32 sequences 正向；
- factor 每 episode 改变时，旧历史使行为退化 `6.1967%`。

在更严格的 10-action early-waypoint 中：

- current-only `2.758192`；correct history `2.425518`；
- 改善 `12.0613%`，CI `[+0.204502,+0.460335]`；
- deadline success 提高 `34.375` 个百分点；
- no-persistence 退化 `5.1157%`。

这证明跨 episode context 的价值不只是完整轨迹末端的累计差异，也能出现在新 episode 的早期关键动作中。

详细结果：[`persistent_context_v2_pushobj_rotation_stage1_results_zh.md`](../docs/research/persistent_context_v2_pushobj_rotation_stage1_results_zh.md)、[`persistent_context_v2_pushobj_rotation_early_waypoint_results_zh.md`](../docs/research/persistent_context_v2_pushobj_rotation_early_waypoint_results_zh.md)。

### 4.4 Dead zone 和 delay：为什么“真参数”也可能伤害行为

两组实验都发现，真实 factor 位于 population prior 一侧时，精确 context 有益；位于另一侧时，精确 context 可能有害。

原因不是 estimator 不准。Dead-zone MLE 的 factor MAE 约为 `1.46e-6`；问题在于 frozen world model 和 CEM 不是完美真实控制器。错误 prior 有时恰好产生更大的动作、保守动作或有利的命令整形。

因此这两项实验给出一个重要警告：

> context 系统最终需要判断“这个信息是否会改善行为”，不能只判断“这个参数是否估得准确”。

详细结果：[`persistent_context_v2_pushobj_deadzone_results_zh.md`](../docs/research/persistent_context_v2_pushobj_deadzone_results_zh.md)、[`persistent_context_v2_pushobj_delay_stage0_results_zh.md`](../docs/research/persistent_context_v2_pushobj_delay_stage0_results_zh.md)。

### 4.5 Rotation×gain matrix：从单一角度扩展到联合动作标定

这里隐藏因素不再是一个角度，而是完整的 `gain × R(theta)` 2×2 matrix。Bayesian estimator 在 E1 只看 command 和 proprio transition，估计参数 `c=g cos(theta)`、`s=g sin(theta)`，E2 在没有当前 transition 时直接使用 posterior planning。

主要结果：

- true-matrix oracle 改善 `10.2271%`；
- Bayesian correct history 改善 `9.5308%`，CI `[+0.029035,+0.404384]`；
- deadline success `56.25%→96.875%`；
- no-persistence 退化 `20.0171%`；
- shuffled history 退化 `3.27%`，wrong-sequence 退化 `6.22%`；
- matrix、gain、rotation estimator 均接近 true factor。

这说明显式 context 不只适用于单个标量，结构化低维 matrix posterior 同样可行。

详细结果：[`persistent_context_v2_pushobj_matrix_stage0_results_zh.md`](../docs/research/persistent_context_v2_pushobj_matrix_stage0_results_zh.md)、[`persistent_context_v2_pushobj_matrix_stage1_results_zh.md`](../docs/research/persistent_context_v2_pushobj_matrix_stage1_results_zh.md)。

### 4.6 CoG：任务很有价值，但第一版学习模型只吃到一小部分

CoG 实验改变 T 物体的水平质心，不改变初始图像、proprio 或 raw state。physics oracle 直接知道真实 CoG，并使用真实 Pymunk 做候选 rollout。

第一批 oracle：

- `pose_auc10：1.339719→0.979750`；
- 改善 `26.8690%`，CI `[+0.231569,+0.501601]`；
- 26/32 pairs 正向；
- success `81.25%→96.875%`。

第二批全新场景再次确认 physics oracle 改善 `20.2716%`。

随后训练了第一版 CoG-conditioned FiLM residual predictor：

- 96 train、24 dev、32 formal segments，三组互斥；
- train 与 formal 使用不同的 CoG 数值；
- 7,680 train samples，168,222 个参数；
- zero context residual 构造上严格为 0；
- true CoG 将 held-out prediction MSE 降低 `35.9126%`；
- 正式闭环 `1.383167→1.318500`，改善 `4.6753%`；
- 23/32 pairs 正向，但 CI `[-0.011011,+0.141373]` 跨 0；
- deadline success 不变；
- 只回收 `23.0633%` physics-oracle gap。

当前判断是：模型确实在使用 context，但 flattened 10-step residual 模型没有充分表达接触发生前后的时序和状态依赖。现在直接训练 history encoder，会被 predictor 本身的上限卡住。

Temporal v2 随后用单向 GRU 显式处理 10 步顺序，并在 32 个全新场景上同时比较 v1、v2 和 physics oracle。结果是：v2 相对 population 仅改善 `0.3747%`，CI 跨 0，相对 v1 退化 `2.7814%`；v1 在该新批次也只改善 `3.0707%`。同批 physics oracle 仍改善 `18.1714%`。逐步分析显示 v2 从早期开始就弱于 v1，后五步 prediction error 也更高。因此已排除“只换 causal GRU 就能修复 CoG predictor”的简单方案。

V2详细结果：[`persistent_context_v2_pushobj_cog_temporal_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_temporal_results_zh.md)。

详细结果：[`persistent_context_v2_pushobj_cog_stage0_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_stage0_results_zh.md)、[`persistent_context_v2_pushobj_cog_predictor_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_predictor_results_zh.md)。

### 4.7 Persistent actuator oracle：小幅收益与 persistence-specific 成分

这项前置合成实验使用 8 种 rotation×双轴 gain actuator factors，每条 persistent sequence 有 8 episodes，共 256 条 sequences。History oracle 从过去 episode 获取 factor，true-factor oracle 直接知道当前 factor。

- episode-reset dense cost `0.103650`；history `0.097475`，改善 `5.958%`；
- paired cost delta CI `[+0.005708,+0.006648]`，252/256 sequences 正向；
- true-factor cost `0.097298`，history 回收 `97.213%` true-oracle gap；
- no-persistence 条件也改善 `4.208%`；
- persistent/no-persistence 配对 DiD 为 `+0.001783`，CI `[+0.001245,+0.002322]`，相当于 persistent reset cost 的 `1.720%`。

因此这里有稳定的 history dense-cost 收益，但只有一部分能归因于 factor 跨 episode 持续。该实验是二维线性合成系统，不是 AdaJEPA 视觉 benchmark。

代码：`research/persistent_factor/benchmark.py`；远端产物：`/data4/zhaoqing/adajepa/persistent_factor_outputs/oracle_v1/`。

### 4.8 Phase A–B：状态生命周期成立，但 naive carry/replay 不能由 latent loss 直接裁决

Phase A 建立了 ordered sequence manifest、weights/optimizer/replay 独立 scope、只读 probe、状态 hash、资源日志和 sequence cleanup。10/10 CPU tests 与两条真实 checkpoint GPU smoke 通过；episodic T2 恢复 base hash，weight-carry T2 保留 T1 后 hash，两者在 sequence 结束都恢复 base。

Phase B 随后在 3 组 matched segments 上完成 TT/TLT 的 W/O/B `2×2×2` 因子实验，共 48/48 runs：

- TLT replay carry 的 T2 pre/post loss 分别降低 `2.57648e-5/2.02171e-4`；
- 只要 W=carry，插入 L 后 T2 pre-loss penalty 在四种 O/B policy 下均为 3/3 replicates 正向；
- 修复 replay 分组混杂后的中预算行为批次 8/8 完成，但 T2 success 为 `0/8`；
- replay 可降低 latent post-loss，却可能让 final state distance 变差；W×B 又存在明显交互。

这一步证明状态继承效应可以被严格归因，同时也证明 prediction loss 不能单独作为长期写入 gate。

远端产物：`phase_a_outputs/`、`phase_b_outputs/grouped_aggregate/`、`phase_b_outputs/behavior_r0_m5_o20_grouped/`。旧的 replay 分组混杂批次保留在 `behavior_r0_m5_o20/`，有 `PROTOCOL_INVALID.md`，不用于结论。

### 4.9 Phase C：Safe Consolidated Memory 的 retention–plasticity 结果

Phase C 实现 frozen base、episode-local fast LoRA、persistent slow LoRA、fixed-capacity memory 与 planning-aware accept/rollback。正式 short/long 共 132 runs、567 episodes。

安全性结果很强：

- naive-no-reset 在 T2/T3/T4 的 pre-loss 插入惩罚分别为 `+0.004776/+0.016475/+0.009267`；
- 到 T4，success-AUC 平均下降 `0.35`，final state distance 增加 `41.78`；
- C2.1 将 T2–T4 pre-loss 插入量压到 `10^-6`，T4 success-AUC 插入差为 0，state-distance 插入量仅 `+0.245`，相对 naive 减少约 `99.41%`。

但长期行为收益没有成立：

| policy | long-TLZ success | success AUC |
|---|---:|---:|
| frozen | 0.433 | 0.310 |
| episodic full AdaJEPA | 0.733 | 0.478 |
| naive full-weight carry | 0.800 | 0.527 |
| C2.1 safe consolidation | 0.500 | 0.348 |
| periodic-C2.1 | 0.500 | 0.343 |

Naive carry 在 L/Z 上可塑性强，但会破坏 recurring T；C2.1 守住 T，却因高回滚率和小 adapter 容量没有获得足够行为可塑性。这是明确的 retention–plasticity trade-off。

核心实现：`planning/dual_lora.py`、`planning/long_term_memory.py`、`planning/adajepa_mpc.py`；最终报告：`/data4/zhaoqing/adajepa/phase_c_outputs/phase_c_final/report.md`。

### 4.10 Phase D–H：逐层排查为什么 safe slow memory 没转化为稳定行为收益

| Phase | 检查问题 | 客观结果 | 结论边界 |
|---|---|---|---|
| D | 是否存在稳定 T/L/Z 梯度冲突，需要 routed experts | 58 个 39,232 维 signatures；anchor/online within-cross cosine margin仅 `0.001773/0.002217`，逐 target 方向不一致 | shape identity 不足以支持直接路由；轨迹/接触异质性更重要 |
| E | JEPA gradient 是否与行为目标冲突；增加 slow steps 是否改善 | one-step/contact 95.65% 非负对齐；steps=2 accepts `11` vs C2.1 `6`，但 long-TLZ AUC `10.35` vs `10.45`，episodic `14.35` | objective 不是明显反向；更多 update 未改善行为 |
| F | 当前 episode 的 functional planning probe 能否识别坏 candidate | 10/10 candidate 可辨识；原 gate accepted 4 个在当前 probe 上 4/4 改善；functional regression 0/4 | 当前局部 probe 无法解释 delayed interference，未启用 veto |
| G | persistent slow 在下一 episode 入口是否直接改善行为 | long-TLZ 6/25 改善、19/25 持平、0/25 受损；recurring-T mean distance改善 `0.1987%`，但只有 1/3 replicate 整体支持正向 | 有稀疏局部因果信号，没有可重复总体优势 |
| H | fast adaptation 是否稳定遮蔽 slow signal | scale 0.25 仅在 r1 恢复 1/1 masked；r0 有 harm，r2 无 masking；directional recovery 1/3 | 不支持统一 fast scale，交互依赖 episode/context |

Phase D–H 的 probe、状态恢复和反事实工具都通过审计，但科学结果总体是否定或不确定的。它们排除了“加 task experts”“多做几步 slow update”“加一个局部 veto”“统一调低 fast scale”等简单修复。

最终报告位于远端：`phase_d_outputs/phase_d_final/report.md`、`phase_e_outputs/phase_e_final/report.md`、`phase_f_outputs/phase_f_final/report.md`、`phase_g_outputs/phase_g_final/report.md`、`phase_h_outputs/phase_h_final/report.md`。

## 5. 关于 T、L、Z、T-shape 内部因素的覆盖情况

需要区分两条已经执行过的路线。

第一条是较早的“权重/adapter 持续”路线。Phase A–H 已经正式覆盖：

- `T→T`、`T→L→T`、`T→Z→T` 与 `T→Z(red)→T` 短序列；
- 三周期 `T→L→Z(red)` 后回到 T4 的 long-TLZ；
- frozen、episodic、naive full-weight carry、fast-only、always-write、C1/C2.1 safe consolidation 与 periodic-C2.1；
- T/L/Z gradient、functional planning、episode-entry 和 matched-fast 反事实诊断。

所以不能再写成“T/L/Z 从未做正式 persistent sequence”。准确结论是：跨 shape 序列已经做过，但 current fast/slow weight-memory 方法没有建立超过 episodic/periodic baseline 的可重复长期行为优势。

第二条是当前“显式低维 persistent context”路线。为了固定几何外观、只改变动作标定或物理参数，其正式实验主要在 PushObj `T` shape 上完成。

已经覆盖的是：

- T 内部的工具旋转；
- T 内部的 rotation×gain matrix；
- T 内部的 radial dead zone；
- T 内部的 discrete delay；
- T 内部的水平 CoG；
- 这些因素的 oracle、部分显式 history estimator，以及 CoG 的两版 learned predictor。

显式 context 路线尚未完成的是：

- T/L/Z 等不同 shape 之间的正式 persistent sequence 对比；
- object identity 作为 `z_seq` 时，history 是否能改善新 episode；
- 同一 context 方法在 T、L、Z 上分别训练和 held-out transfer 的矩阵实验；
- shape context 与 rotation/gain/CoG context 的解耦实验。

所以目前不能声称已经验证“显式、因子化 context 在 T/L/Z 之间正向迁移”。已有 Phase A–H 回答的是持续权重/adapter/memory；当前 PushObj factor 实验回答的是 T shape 内隐藏物理因素。两者都做过，但还没有在同一个实验中闭合 `z_shape + z_physics → held-out shape behavior` 的因果链。

## 6. 到目前为止真正学到了什么

### 6.1 持续 context 是否值得做，首先取决于任务有没有行为上限

PointMaze 中 true factor 会改变动作但不改善结果；PushObj rotation、matrix 和 CoG 中 true factor 能改善结果。两者差异说明不能只看参数能否注入或预测误差，必须先测真实行为。

### 6.2 简单充分统计量是非常强的 baseline

对 rotation 和 rotation×gain，Procrustes/MLE/Bayesian posterior 已经几乎回收全部 oracle gap。复杂 learned memory 必须解决它们解决不了的问题，例如 CoG、摩擦、接触模式或不可直接线性辨识的因素，而不是为了复杂而复杂。

### 6.3 Persistence 的因果证据已经存在

在 rotation 和 matrix 中：factor 持续时 correct history 改善；factor 每 episode 改变时同样的旧历史显著伤害；错误或打乱 history 不能复制 correct history。收益不是“多给模型一些数据”这种泛化解释。

### 6.4 准确 context 与行为最优不是同一件事

Dead zone、delay 和少数 matrix factor 组合都出现 true context 负向。这意味着以后需要显式建模 context 的行为价值、planner surrogate error 和不确定性。

### 6.5 learned predictor 是当前主要瓶颈

CoG physics oracle 很强，但 v1 只回收一部分，temporal v2 更差。P3a 中 contact ridge 有增量信息，冻结 v1 correction 仍退化。P3b 随后完成 100 Hz event-response 审计：nominal impulse 相对 event geometry 有增量 `+0.02913`，但完整 S100 error `0.33020`，没有超过 C10 的 `0.32292` 或 zero response 的 `0.30496`；privileged true-contact ridge 也退化。当前 CoG 瓶颈不能通过提高日志频率和固定 ridge 解决，V3/history 继续暂停。

### 6.6 直接保留权重的旧路线给出了清楚的反例和工程资产

Phase A–H 证明 persistent slow/fast 状态、可回滚 memory 和只读 causal probe 都能严谨实现；同时也证明“安全不遗忘”与“得到更好的总体行为”不是一回事。Naive carry 有可塑性但会干扰 recurring T，safe consolidation 能隔离干扰却不如 episodic。当前显式 context 路线应复用这些审计工具，但不应回到无条件权重持续。

## 7. 当前成熟度判断

| 组件 | 当前状态 | 依据 |
|---|---|---|
| persistent context 核心命题 | 已有真实 benchmark 支持 | rotation 与 matrix 的 persistent/no-persistence/负对照 |
| 简单显式 estimator | 已建立 | Procrustes、censored MLE、Bayesian matrix posterior |
| 早期 cold-start 行为价值 | 已建立 | rotation early-waypoint 与 matrix E2 |
| context-conditioned learned world model | 合成环境建立；PushObj 尚不稳定 | 合成 FiLM 很强；CoG v1 小幅正向，temporal v2 未改善 |
| CoG Markov/contact 表示 | 字段采集成立，粗粒度充分性未成立 | P3a R2 优于 R1，但 v1 correction 退化；独立审计有效 |
| CoG 100 Hz event response | impulse 有局部增量，event sufficiency 未成立 | P3b 主比较 `C10-S100=-0.00728`；S100 未超过 zero；独立审计有效 |
| 非线性隐藏物理的 history 推断 | 未建立 | CoG history 尚未开发 |
| 行为 gate | factor-only learned gate 已建立初步证据 | 独立 train/dev/formal matrix 实验改善 `12.29%`，harm 从 always-context 的 `31.25%` 降到 `21.875%`；额外收益 CI 仍跨 0 |
| visual AdaJEPA 原生 context 接口 | 未建立 | 当前 CoG 是 physics-residual predictor，不是完整视觉 AdaJEPA 重训练 |
| 跨 shape T/L/Z 持续权重/adapter | 已正式测试，正向优势未建立 | Phase C–H long-TLZ；safe retention 成立，但低于 episodic/periodic，后续因果 probes 未跨 replicate 通过 |
| 显式 rotation×gain context 的跨 shape 泛化 | 已建立（作者 seed-42 发布池范围） | 96 条跨-shape formal：correct `.75` 改善 9.97%，persistence-specific CI 排除 0；尚未覆盖新生成 seed/视觉分布 |
| persistent slow + episode-local fast TTT | 已实现并正式测试，优势未建立 | Phase C dual LoRA 到 Phase H matched-fast；统一 fast-scale recovery 仅 1/3 replicate |
| 状态生命周期与因果 probe 基础设施 | 已建立 | Phase A–H 的 manifest/hash/rollback/test-retest/cleanup 审计 |

## 8. 接下来怎么做

### 已完成：CoG temporal predictor v2

该实验已按冻结计划完成。V2只改善 `0.37%`，没有超过v1，并且在新场景只回收 `2.06%` oracle gap。结论是停止在相同输入上增加复杂 history encoder，转而检查 state representation 与接触信息缺失。

### 已完成：P3a CoG Markov/contact 表示审计

Simulator 字段采集、legacy 状态同一性和重复 hash 均通过。24-train/16-eval segment 中，R0/R1/R2 ridge error 为 `0.56422/0.54495/0.44623`，说明 contact 聚合对低容量映射有信息；但冻结 v1 为 `0.28831`，v1+Markov 和 v1+Markov/contact correction 退化到 `0.30213/0.32110`。所以“直接补字段”不是经验证的修复。

### 已完成：P3b CoG event-level contact-response audit

P3b 在 24-train/16-eval segment 上记录了 100 Hz substep。C10 与完整 S100 均为 29 维，模型在 train CV 锁定后才生成 eval。主比较 `C10-S100=-0.00728`、CI `[-0.02137,+0.00603]`、8/16 正向。Nominal impulse 相对 geometry-only 有增量 `+0.02913`、14/16 正向，但 S100 full 仍差于 zero response；privileged true-contact ridge 也更差。该结果有效并经独立复算，因此不进入 V3、CoG history 或新 CoG formal。

### 暂缓：CoG 的非特权 history estimator

只有 learned true-CoG model 具有稳定行为价值后才进入；当前 v1/v2 证据不足，因此本项暂缓。

优先从简单方法开始：

- 从过去接触 transition 对不同候选 CoG 做 simulator likelihood/MLE；
- 或维护低维 Bayesian posterior，而不是一开始训练大 history Transformer；
- E1 产生 evidence，E2 在读取当前 transition 前评价；
- 同时做 persistent、no-persistence、shuffled、wrong-sequence controls；
- 记录完整链条：`history → posterior → prediction → action → outcome`。

### 已完成：functional shadow gate

已在 dead zone、delay 和 matrix 的 96 个冻结配对单位上完成可审计 shadow evaluation。规则只读取规划前的低维 factor/context 信息，不读取真实 cost、planner loss 或 best-of-two 标签。

- 相对 population，三任务分别改善 `5.6629%/6.1108%/11.1518%`；
- 宏平均改善 `7.6418%`，高于 always-context 的 `4.1963%`；
- harm fraction 从 always-context 的 `53.125%/43.75%/25.00%` 降至 `6.25%/0%/12.50%`；
- inverted gate 在三个任务上全部退化；
- 独立审计复算 96 个单位，decision mismatch 与数值 replay error 均为 0。

这证明“什么时候相信 context”有明确行为价值。边界是：它是 retrospective shadow evaluation；delay gate 使用真实 factor；规则尚未学习，也没有在新 formal split 上验证泛化。

### 已完成：factor-only learned surrogate gate

已建立 train/dev/formal 各 32 条、共 96 条互斥 matrix sequences。每条包含 E1 history evidence 和 E2 population/context 配对执行；三个 split 共 192 个片段完全不重叠，并排除旧 matrix/shadow 的直接输入。

- 二次 ridge 只读取 posterior gain/rotation；train 拟合、dev 选择 `alpha=1`、train+dev refit；
- formal learned gate 相对 population 改善 `12.2900%`，paired delta CI `[+0.180032,+0.482775]`；
- always-context 改善 `11.2931%`，harm `31.25%`；learned gate harm 降到 `21.875%`；
- learned 相对 always-context 额外降低 cost `1.1238%`，但该差的 CI 跨 0；
- functional gate 改善 `11.9783%`、harm `15.625%`；learned 均值略好但更激进；
- 独立 audit 的 posterior/model/decision/summary replay error 均为 0。

因此 learned gate 方向得到首个独立 split 正向结果，但不能声称已解决负向尾部。

### 已完成：factor + task-interaction D0 feasibility

已在旧 96 条 V1 sequences 上完成只读重放，加入 scene geometry、两套候选 action 差异及 population/posterior 交叉 rollout。旧 formal 在 D0 中降级为 exploration，不构成新的 formal 证据。

- F0/F1/F2 的旧 formal prediction correlation 为 `0.5327/0.6371/0.6674`，说明决策前特征确实含有组内信号；
- 但预定义的 `predicted delta > 0` 硬 gate 使 F1/F2 都选择 `100%` context，harm 从 F0 的 `21.875%` 回到 `31.25%`；
- F1/F2 mean delta 相对 F0 约 `-0.0270`，paired CI `[-0.0912,+0.0212]`；
- 三次 split rotation 中，F1 均未超过 F0；F2 只有一次约 `+0.01`，另外两次为负；
- 独立 audit 对 96 条特征与 outcome 重放误差均为 0，model/RNG identity 通过。

所以“特征有信息”与“当前硬 gate 有行为价值”必须分开。现有结果不支持直接冻结 task-interaction hard-gate V2 formal。

### 已完成：fixed soft-context D1

已固定比较 `α∈{0,0.25,0.5,0.75,1}`，其中 `M_α=(1-α)M_prior+αM_posterior`。旧 96 条全部作为探索，480/480 treatments 完成；`α=0/1` 严格复现原 population/context 分支，独立 audit 全部通过。

- `α=0.75` 相对 population 改善 `11.4495%`，与 full context 的 `11.3645%` 相当，harm 从 `30.208%` 降到 `18.75%`；
- `α=0.5` 改善 `9.4119%`，保留约 `82.8%` full-context mean delta，harm 降到 `14.583%`；
- 三个旧 split 上中间 α 的 harm 都低于 full context；但 `α=0.75` 相对 full 的聚合 mean cost 只低 `0.0021`，不能声称通用最优；
- `44/96` 条的事后最佳 α 位于中间，`54/96` 条剂量曲线非单调，说明有逐场景强度机会，也有显著异质性。

### 已完成：低容量 soft-policy D2

已在旧 96 条上冻结剂量 ridge 并完成三次 split rotation。F0/F2 相对 fixed `α=0.75` 的 mean cost 三轮都正向，但 F2 相对 F0 为 `+0.0194/-0.0292/-0.0199`，没有可重复增量；F2 的 harm 也没有优于 fixed `α=0.75`。因此 soft policy 值得独立验证，但 task-interaction 复杂度不进入下一轮主方法。

### 已完成：作者 T 数据科学收缩后的 F0 soft-policy formal

发布物没有原始目标生成源，不能生成真正的新轨迹分布。审计在 1000 个 T segments 中找到 413 个 eligible 且从未进入 matrix outcome 的片段；因此冻结同一发布池内 matrix-unexposed 的 `64/32/96` train/dev/formal，共使用 384 个互异 E1/E2 segments并保留 29 个备用。主方法只保留 6 维 factor-only soft policy；主对照 fixed `α=0.75`，另保留 population、fixed `α=0.5`、full context 和不可部署 best-α ceiling。

正式 96 条中，F0 相对 population 改善 `13.2141%`，delta CI `[0.2124,0.3963]`；相对 fixed `.75` 的唯一主差为 `+0.01686`，CI `[-0.00865,+0.04122]`。F0 与 fixed `.75` harm 都是 `21.875%`。F0 只选择 `.5/.75/1=12/24/60` 次，同 factor 内选择完全相同，所以它没有成为 scene-aware safety gate。独立端到端审计所有重算误差为 0。结论是 context 在未暴露 T scenes 上稳定有效，但 learned F0 相对简单 fixed `.75` 的额外价值仍不确定。

### 已完成：作者其他 shape 的跨任务复验

作者还发布了 `val_I/val_L/val_+/val_small_tee/val_square/val_Z`，现有 matrix manifests 只使用 `val_T`。只读审计找到 4474 个 eligible 非 T segments，跨池只有 1 个 exact state+action duplicate；正式选择会排除所有 exact duplicate，并要求 segment hash 与 `ep_idx:offset` 全局唯一。

Formal 无需 train/dev，收缩为 96 条：六个 shape pair 各 16 条，8 个 factor 在每 pair 各 2 条。Correct fixed `.75` 相对 population 改善 `9.97%`，主 delta CI `[0.1393,0.3478]`；factor 不持续时同样 history 相对 population 为 `-0.22%`，persistence-specific delta CI `[0.1236,0.3696]`。六个 shape pair 主效应均为正，独立审计所有重算误差为 0。这证明低维物理 context 在当前作者数据上可跨任务迁移，并且收益依赖物理因素持续。

External T-F0 无需 refit 即改善 `11.10%`，相对 fixed `.75` 的均值优势 CI `[0.0017,0.0538]`，但 harm 为 `33.33%`，高于 fixed `.75` 的 `28.13%`。所以均值迁移与安全选择仍是两个问题。

### 已完成：跨 shape 失败归因 D3

当前 96 条只作为 outcome-exposed exploration。只读 D3 用 leave-one-shape-pair-out/leave-one-factor-out 检查 F0/F1/F2：现有特征几乎总是继续使用 context。Fixed `.75` 的最佳 veto 只减少 1 条 harm 且降低 mean；external F0 的最佳 veto 也只减少 1 条 harm，mean 损失 CI 排除 0。因此不授权新的 harm-aware formal，也不在同一批 outcome 上调阈值或扩大模型。

### 已完成：完整预测轨迹分歧 D4

D4 使用 96 条未暴露 development sequence，在任何 E2 执行前保存默认模型/上下文模型对 population/context 两套命令的四条完整 latent 轨迹。Fixed `.75` 平均改善 `11.90%`，区间 `[0.2191,0.4110]`，但仍有 24/96 条受损。预注册风险分数的 harm AUC 为 `0.3906`、区间 `[0.2664,0.5223]`；分歧越大时平均收益反而越高。低容量 ridge 在 96 条中选择 context 95 次，受损率仍为 `25%`。原始轨迹、状态、指标和统计独立审计误差均为 0。因此 rollout disagreement 不能按冻结方向作为风险 veto，也不在同一数据上翻转方向或选择次要分数后启动 formal。

### 已完成：Delay 非特权 estimator development smoke

Estimator 只用过去 E1 command 和 agent position/velocity，通过 PD 反演与五档 FIFO likelihood 得到 posterior。既有 Stage 0 32 条开发轨迹与 4 条全新作者片段上的 MAP 分别为 `32/32`、`4/4`；E2 零当前 evidence、model identity 和独立重算均通过。

4 条 smoke 中 persistent correct-MAP 改善 `7.69%`，但 no-persistence old-history MAP 也改善 `7.83%`，DiD cost 仅 `+0.01497`。这不是 formal；它说明 delay 可辨识，但错误 context 也可能作为命令整形改善当前 frozen planner。开发 wrong donor 在 no-persistence 中与 current factor 相撞，正式设计必须改成无 collision 映射。作者数据排除既有 raw 后剩余池只支持科学收缩的 32 sequences × 2 episodes，而不是原建议的每条件 64 条。

### 已完成：Delay 非特权 history formal

32-sequence formal 中 estimator 对 E1 delay `32/32` MAP 正确，但 persistent correct-MAP 相对 prior 仅改善 `0.28%`，no-persistence old-MAP 改善 `0.32%`，DiD `-0.00083`、CI `[-0.02355,+0.02095]`。Persistent true-factor oracle 与 estimator 逐条相同，也只有 `0.28%` 总体改善；失败点不是辨识，而是 exact delay 在当前 frozen world-model/CEM 上缺乏总体行为价值。

异质性复现：`d=0/1` 分别退化 `13.79%/7.37%`，`d=3/4` 改善 `3.32%/10.91%`。预定义 high-delay gate 避开低-delay伤害，在 persistent 改善 `4.40%`；但 no-persistence 也改善 `4.33%`，gate DiD 近 0。Wrong donor 已保证两个 condition 都不碰 current factor，独立审计全部通过。因此 gate 有平均 command-shaping 价值，却不是 persistence-specific history 证据；当前不直接进入 learned delay gate。

### 已完成：扩大 rotation×gain matrix 的独立复验

本次已新增三个不重叠批次，每批 32 sequences。无条件 context 在 train/dev/formal 分别改善 `11.29%/11.52%/11.29%`；factor 异质性总体存在，但旧批次负向的 `(-10°,1.18)` 在新 formal 变为小幅正向且组内 2 正 2 负，进一步证明 gate 需要场景交互信息，而不能把一次 factor 分组符号当永久规则。

### 后续候选：E2 在线贝叶斯修正

如果用户决定继续，先检验简单充分统计量：E2 开始时把 E1 后验作为物理参数先验，再根据前 1–2 步 command、位置和速度更新。实验必须同时包含 E1/E2 factor 持续和 factor 改变，所有方法共享总动作预算。主比较是“旧 context 全程固定”与“旧 context 初始化后在线修正”；population、current-only 和 true-current-factor 作为对照。该实验尚未冻结或执行。

### 后续步骤：显式 context + episode-local adaptation

Phase C–H 已测试“persistent slow 权重 + episode-local fast LoRA”，但没有建立行为优势。这里计划的不是重复该实验，而是在显式低维 context 已可靠时，测试它与 episode-local adaptation 的组合：

```text
z_seq：跨 episode 的慢变量，例如 rotation、gain、CoG
z_ep：当前 episode 的快变量，例如接触残差或局部模型误差
```

对照至少包括：base、context only、episode-local adaptation only、context + episode-local adaptation、true-context upper bound。必须检查快适应是否保留 episode entry 的 context 收益，以及是否在错误 context/change point 后帮助恢复。

### 后续步骤：从已建立的跨 shape context 进入 factorization 与视觉模型

Rotation×gain 物理 context 的 state-based 跨 shape 迁移已经建立。进一步进入 factorization/视觉前仍需：

1. 先把 shape identity 与物理 factor 分开标注；
2. 每个 shape 覆盖多种 start、goal、trajectory 和 factor；
3. 比较 shared context、shape-specific context 和 factorized `z_shape + z_physics`；
4. formal 中同时保留 held-out trajectory、held-out factor 和 held-out shape 组合；
5. 在这个阶段才把成熟的 context 接口迁移进视觉 AdaJEPA predictor。

## 9. 推荐执行优先级

按当前证据、科学风险和实现成本排序：

1. **证据与源码快照治理（已完成首版）**：后续每次运行继续绑定 patch、untracked source hash 和环境，不能只记录 base commit；
2. **Delay 非特权 estimator formal（已完成）**：可辨识性成立，persistence-specific 行为值未建立；high-delay gate 的收益不依赖持续性，暂不训练 learned gate；
3. **CoG P3a 表示审计（已完成）**：字段可得，但控制边界 Markov/contact 拼接未修正 v1；
4. **CoG P3b event-level contact-response audit（已完成）**：100 Hz impulse 有局部信息，完整表示未超过 10 Hz/zero；CoG 路线暂停；
5. **Matrix rollout-disagreement harm feasibility（已完成）**：fixed `.75` 平均价值再次成立，但预注册 harm AUC 为 `0.391`，ridge 退化为 95/96 条使用 context；不启动当前风险 veto formal；
6. **E2 在线贝叶斯修正（待用户决定）**：以 E1 后验初始化，再用 E2 前 1–2 步反馈更新，同时检验 factor 持续和改变；
7. **T/L/Z 跨 shape factorization**；
8. **迁移到完整视觉 AdaJEPA context-conditioned predictor**。

详细合同草案和样本拆分见当前后续实验计划。这里的排序表示依赖与信息价值，不是固定效果门。

## 10. 主要代码、报告与远端产物

### 核心报告

- Rotation history：[`persistent_context_v2_pushobj_rotation_stage1_results_zh.md`](../docs/research/persistent_context_v2_pushobj_rotation_stage1_results_zh.md)
- Rotation early-waypoint：[`persistent_context_v2_pushobj_rotation_early_waypoint_results_zh.md`](../docs/research/persistent_context_v2_pushobj_rotation_early_waypoint_results_zh.md)
- Dead zone：[`persistent_context_v2_pushobj_deadzone_results_zh.md`](../docs/research/persistent_context_v2_pushobj_deadzone_results_zh.md)
- Delay：[`persistent_context_v2_pushobj_delay_stage0_results_zh.md`](../docs/research/persistent_context_v2_pushobj_delay_stage0_results_zh.md)
- Delay 非特权 history development smoke：[`persistent_context_v2_pushobj_delay_history_stage1_dev_results_zh.md`](../docs/research/persistent_context_v2_pushobj_delay_history_stage1_dev_results_zh.md)
- Delay 非特权 history formal：[`persistent_context_v2_pushobj_delay_history_stage1_formal_results_zh.md`](../docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_results_zh.md)
- Matrix history：[`persistent_context_v2_pushobj_matrix_stage1_results_zh.md`](../docs/research/persistent_context_v2_pushobj_matrix_stage1_results_zh.md)
- CoG oracle：[`persistent_context_v2_pushobj_cog_stage0_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_stage0_results_zh.md)
- CoG predictor：[`persistent_context_v2_pushobj_cog_predictor_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_predictor_results_zh.md)
- CoG temporal v2：[`persistent_context_v2_pushobj_cog_temporal_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_temporal_results_zh.md)
- CoG Markov/contact P3a：[`persistent_context_v2_pushobj_cog_markov_contact_audit_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_markov_contact_audit_results_zh.md)
- Functional shadow gate：[`persistent_context_v2_functional_shadow_gate_results_zh.md`](../docs/research/persistent_context_v2_functional_shadow_gate_results_zh.md)
- Matrix learned gate：[`persistent_context_v2_matrix_learned_gate_results_zh.md`](../docs/research/persistent_context_v2_matrix_learned_gate_results_zh.md)
- Matrix task-interaction D0：[`persistent_context_v2_matrix_task_interaction_d0_results_zh.md`](../docs/research/persistent_context_v2_matrix_task_interaction_d0_results_zh.md)
- Matrix soft-context D1 设计：[`persistent_context_v2_matrix_soft_context_d1_design_zh.md`](../docs/research/persistent_context_v2_matrix_soft_context_d1_design_zh.md)
- Matrix soft-context D1 结果：[`persistent_context_v2_matrix_soft_context_d1_results_zh.md`](../docs/research/persistent_context_v2_matrix_soft_context_d1_results_zh.md)
- Matrix soft-policy D2 结果：[`persistent_context_v2_matrix_soft_policy_d2_results_zh.md`](../docs/research/persistent_context_v2_matrix_soft_policy_d2_results_zh.md)
- Matrix F0 soft-policy 前瞻正式结果：[`persistent_context_v2_matrix_f0_soft_policy_formal_results_zh.md`](../docs/research/persistent_context_v2_matrix_f0_soft_policy_formal_results_zh.md)
- 跨 Shape Matrix History 前瞻正式结果：[`persistent_context_v2_cross_shape_matrix_history_formal_results_zh.md`](../docs/research/persistent_context_v2_cross_shape_matrix_history_formal_results_zh.md)
- 跨 Shape Harm Attribution D3：[`persistent_context_v2_cross_shape_harm_d3_results_zh.md`](../docs/research/persistent_context_v2_cross_shape_harm_d3_results_zh.md)
- 完整预测轨迹分歧 D4：[`persistent_context_v2_matrix_rollout_disagreement_d4_results_zh.md`](../docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_results_zh.md)

### 前置跨任务 Phase A–H 最终报告（远端）

```text
/data4/zhaoqing/adajepa/任务记录/phase A.md
/data4/zhaoqing/adajepa/phase_b_outputs/grouped_aggregate/report.md
/data4/zhaoqing/adajepa/phase_b_outputs/behavior_r0_m5_o20_grouped/behavior_summary/report.md
/data4/zhaoqing/adajepa/phase_c_outputs/phase_c_final/report.md
/data4/zhaoqing/adajepa/phase_d_outputs/phase_d_final/report.md
/data4/zhaoqing/adajepa/phase_e_outputs/phase_e_final/report.md
/data4/zhaoqing/adajepa/phase_f_outputs/phase_f_final/report.md
/data4/zhaoqing/adajepa/phase_g_outputs/phase_g_final/report.md
/data4/zhaoqing/adajepa/phase_h_outputs/phase_h_final/report.md
```

### 核心实现

以下 Persistent Context V2 文件在本地与远端仓库均存在；Phase A–H 的早期基础设施当前以远端 `/data4/zhaoqing/adajepa/` 下的同名路径为权威，本地精简工作区并未包含其全部文件。

```text
research/persistent_context_v2/pushobj_rotation_stage1.py
research/persistent_context_v2/pushobj_rotation_early_waypoint_stage1.py
research/persistent_context_v2/pushobj_deadzone_stage1.py
research/persistent_context_v2/pushobj_delay_stage0.py
research/persistent_context_v2/pushobj_matrix_stage1.py
research/persistent_context_v2/pushobj_cog_stage0.py
research/persistent_context_v2/pushobj_cog_predictor.py
research/persistent_context_v2/functional_shadow_gate.py
research/persistent_context_v2/pushobj_matrix_gate_data.py
research/persistent_context_v2/matrix_learned_gate.py
research/persistent_context_v2/matrix_rollout_disagreement_d4.py
scripts/audit_persistent_context_v2_matrix_rollout_disagreement_d4.py
research/persistent_factor/benchmark.py
planning/sequence_manifest.py
planning/continual_state.py
planning/dual_lora.py
planning/long_term_memory.py
planning/functional_probe.py
planning/matched_fast_probe.py
planning/adajepa_mpc.py
```

### 远端正式输出

```text
/data4/zhaoqing/adajepa/persistent_context_v2_outputs/
/data4/zhaoqing/adajepa/persistent_factor_outputs/oracle_v1/
/data4/zhaoqing/adajepa/phase_a_outputs/
/data4/zhaoqing/adajepa/phase_b_outputs/
/data4/zhaoqing/adajepa/phase_c_outputs/c3_formal_short/
/data4/zhaoqing/adajepa/phase_c_outputs/c3_formal_long/
/data4/zhaoqing/adajepa/phase_d_outputs/d0_formal_gradients/
/data4/zhaoqing/adajepa/phase_e_outputs/e0_formal_alignment/
/data4/zhaoqing/adajepa/phase_e_outputs/e1_steps_2_formal/
/data4/zhaoqing/adajepa/phase_f_outputs/f1_diagnostic_r0_long_tlz/
/data4/zhaoqing/adajepa/phase_g_outputs/g2_positive_transfer_formal/
/data4/zhaoqing/adajepa/phase_h_outputs/h2_fast_scale_formal/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_rotation_stage1/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage1/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_deadzone_stage1/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_delay_stage0/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_matrix_stage1/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_stage0/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_predictor/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_temporal/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_functional_shadow_gate/
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_matrix_learned_gate/
```

## 11. 远端产物对账与归并说明

本次按远端实际目录重新对账。下表说明哪些目录代表独立实验族，哪些只是同一实验的 smoke、inspect、repair 或校准过程。

| 远端实验族 | 全景位置 | 采用的正式/权威产物 | 归并说明 |
|---|---|---|---|
| 合成 Persistent Context Stage 0–3 | 3、4.1 | `stage0_dev_v1`、`stage1_formal_v1_repair1`、`stage2_pilot_v1_repair1`、`stage3_film_v1` | Stage 1/2 的原始与 repair1 均保留；repair1 是修复审计/汇总后的权威版本，不另算科学实验 |
| Persistent actuator factor | 3、4.7 | `persistent_factor_outputs/oracle_v1` | 独立合成实验，已补入 |
| PointMaze gain/lag | 3、4.2 | `pointmaze_transfer/v1`、`pointmaze_lag_stage0_v1/A_hard_goal`、`B_local_waypoint` | dev_smoke/dev_standard 不是额外正式结论 |
| 跨任务 Phase A–B | 3、4.8 | Phase A 两条成功 smoke；Phase B `grouped_aggregate` 与修复后 behavior grouped | Phase A 是工程验收；旧 `behavior_r0_m5_o20` 有协议混杂，只保留不用作结论 |
| Safe continual Phase C | 3、4.9 | `c3_formal_short`、`c3_formal_long` | smoke、calibration、retry 是同一方法开发轨迹，不重复计数 |
| 诊断 Phase D–H | 3、4.10 | D0 formal、E0/E1 formal、F1 diagnostic、G2 formal、H2 formal | 各 phase 的 smoke/diagnostic 分支保留；主结论使用 final report/decision |
| PushObj rotation/dead-zone/delay/matrix/CoG | 3、4.3–4.6 | 各无 `_smoke`/`_inspect` 后缀目录 | smoke/inspect/repair 仅验证管线；正式 raw/summary/audit 已在对应报告中 |
| Functional 与 learned gate | 3、8 | `persistent_context_v2_functional_shadow_gate`、`persistent_context_v2_matrix_learned_gate/evaluation` | 两个是独立 gate 实验；matrix learned gate 的 train/dev/formal 子目录属于同一个 split 实验 |

对账后的处理原则：

- 正式、pilot、diagnostic、工程 smoke 分开标注证据等级；
- 同一科学设计的 repair/retry 不重复计算实验数量；
- 协议无效目录不删除，但不进入结论；
- 负结果与未跨 replicate 的局部正向结果都写入全景；
- 截至本次对账，远端主要实验族都已经在本文件的总表、详细说明或归并表中出现。为避免正向选择偏差，不再维护只收录“正向实验”的独立汇总；局部正向信号必须和同一实验的总体结果、负向配对及结论边界一起读取。

## 12. 最终概括

目前最可靠的结论不是“所有 persistent context 都有用”，而是：

> 当隐藏因素确实跨 episode 持续、历史中可辨识、而且该因素能改变新 episode 的早期行为时，显式低维 context 可以产生真实、可重复、具有 persistence-specific 对照的收益。

Rotation 和 rotation×gain matrix 已经证明这条链。CoG 证明更复杂物理因素存在很大的行为上限，但也暴露出 learned predictor 的表达瓶颈。Dead zone 和 delay 进一步说明不能把更准确的参数无条件送给 planner；functional shadow gate 随后在三个任务上把宏平均改善从 always-context 的 `4.20%` 提高到 `7.64%`。独立 split 的 matrix learned gate 又把 formal 改善从 always-context 的 `11.29%` 提高到 `12.29%`，并将 harm 从 `31.25%` 降到 `21.875%`，但额外收益 CI 仍跨 0。D0 发现 task-interaction 特征能提高 benefit prediction correlation，却不能通过零阈值 hard gate 改善行为；D1/D2 支持 soft context。P1-V2 在未暴露 T scenes 上确认 F0 相对 population 改善 `13.21%`。P1-XShape 又确认 correct `.75` 跨任务改善 `9.97%`，而 factor 不持续时收益消失，建立了 persistence-specific 跨 shape 证据。P2 delay formal 则给出重要边界：delay 虽可 `32/32` 精确辨识，正确 history 的 persistent/no-persistence 改善都只有约 `0.3%`，high-delay gate 的约 `4.4%` 收益也不依赖持续性。P3a 再给出表示边界：接触字段对简单模型有信息，但不能修正冻结 v1；误差更像来自 contact impulse response 和时间聚合，而非简单缺少几个控制边界变量。当前瓶颈包括跨-shape 约 28% 的负向尾部、planner 对 exact factor 的非单调响应，以及复杂接触因素的事件级建模。

前置 Phase A–H 同样是当前判断的一部分：它们证明 naive weight carry 会干扰、safe consolidation 可以保护 recurring task，但尚未获得超过 episodic/periodic 的稳定总体行为优势。这些负结果支持当前转向显式 context，而不是被视为未发生过的实验。

所以当前路线有继续价值，但下一阶段的重点已经从“证明历史有信息”转向两个更难的问题：

1. 如何让 context-conditioned temporal world model 真正回收复杂接触动力学的 oracle gap；
2. 如何在 context 有益与有害的区域之间做可靠、可审计的强度控制与行为选择。
