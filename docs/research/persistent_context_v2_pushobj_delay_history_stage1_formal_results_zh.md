# PushObj Delay 非特权 History Stage 1 正式结果

日期：2026-08-24  
合同 ID：`persistent-context-v2-pushobj-discrete-delay-history-stage1-formal-v1`  
状态：**32-sequence formal 完成，独立审计有效**

## 1. 人话结论

Delay 可以从历史里非常准确地认出来，但“认出来”在这个任务上几乎没有转化为跨 episode 的额外行为收益。

E1 的非特权 estimator 对 32 条 sequence 全部识别正确。Persistent E2 使用正确历史 MAP，相对固定 2-step prior 只改善 `0.28%`；no-persistence 中继续使用已经过时的 E1 MAP 也改善 `0.32%`。两者 DiD 几乎为 0。这意味着当前平均差异不能归因于物理 delay 跨 episode 持续。

预定义 high-delay gate 在 persistent 中改善 `4.40%`，但 no-persistence 中同样改善 `4.33%`，gate DiD 也几乎为 0。它更像一种“在某些 sequence 上使用高-delay command shaping”的策略，而不是有效的跨任务记忆机制。

## 2. 三个主 estimand

`delta = current-only cost - history-policy cost`，越大越好；统计单位均为 32 条 sequence。

| 主对比 | current | history MAP | mean delta | 相对改善 | 95% bootstrap CI | 正/平/负 |
|---|---:|---:|---:|---:|---:|---:|
| Persistent | 2.20476 | 2.19853 | +0.00623 | +0.283% | [-0.10295,+0.11226] | 14/0/18 |
| No-persistence | 2.24266 | 2.23560 | +0.00706 | +0.315% | [-0.10980,+0.12217] | 15/0/17 |
| DiD | — | — | -0.00083 | — | [-0.02355,+0.02095] | 15/0/17 |

这里不能只说“CI 跨 0”。更重要的是：persistent 和 no-persistence 的点估计几乎相同，逐 sequence 配对后的 DiD 区间也很窄地围绕 0。这直接削弱了“收益来自 factor 持续”的解释。

## 3. Estimator 与 oracle

- E1 MAP accuracy：`32/32`；mean posterior entropy：`0.00101`；
- persistent correct-history MAP 与 true-factor oracle 的 context 逐条相同，因此行为结果完全相同，oracle-gap recovery 为 100%；
- 这不是值得庆祝的 100% recovery：因为该 formal scene 集上 true-delay oracle 本身总体只改善 `0.28%`；
- no-persistence 的旧 history 与 E2 current factor match 为 0，wrong donor/current match 也为 0，证明负对照没有再发生 development donor collision。

因此失败环节不是 system identification。Estimator 已经到达该离散确定性环境的近似 ceiling；瓶颈是 frozen world-model/CEM 对 exact delay 的闭环行为价值。

## 4. Factor 异质性

Persistent correct-history MAP：

| E2 delay | mean delta | 相对改善 | 95% CI | 正向数 |
|---:|---:|---:|---:|---:|
| 0 | -0.24046 | -13.79% | [-0.47219,-0.01597] | 2/8 |
| 1 | -0.12256 | -7.37% | [-0.20631,-0.06021] | 0/8 |
| 3 | +0.08845 | +3.32% | [-0.05443,+0.23208] | 5/8 |
| 4 | +0.29950 | +10.91% | [+0.11895,+0.46242] | 7/8 |

低 delay 的负向和高 delay 的正向相互抵消，复现了 Stage 0 的基本异质性。说明 exact factor 对 planner 不是单调有益：低 delay 下错误的 2-step prior 反而是一种有利的保守化或命令整形。

## 5. Gate 与负对照

| 条件/策略 | mean delta | 相对改善 | 95% CI | 正/平/负 |
|---|---:|---:|---:|---:|
| persistent high-delay gate | +0.09699 | +4.40% | [+0.02978,+0.17081] | 12/16/4 |
| no-persistence old-history gate | +0.09716 | +4.33% | [+0.02195,+0.18180] | 12/16/4 |
| persistent shuffled | +0.05178 | +2.35% | [-0.05322,+0.14891] | 17/4/11 |
| persistent wrong-sequence | +0.03062 | +1.39% | [-0.07990,+0.13750] | 17/0/15 |

High-delay gate 的 persistent/no-persistence 描述性 DiD 为 `-0.00017`，区间 `[-0.02062,+0.01843]`。所以 gate 平均收益是真实的预注册辅助结果，但不是 persistence-specific history value。Persistent 中 gate 相对 always-MAP 的 mean delta 增加 `0.09076`；它主要来自避免 `d=0/1` exact context 的系统性伤害。

Shuffled 和 wrong-sequence 也出现小幅正均值且区间跨 0，再次说明“某个非 prior context 改善 planner”不能自动解释为正确物理记忆被利用。

## 6. 有效性与资源

- 32 条 E1、64 条 matched-condition E2 全部完成；64 个 formal segments 唯一；
- E2 决策先于任何 E2 执行，current evidence count 全为 0；population/current 逐 state identity；condition scene/初态/目标完全配对；
- wrong donor factor 与两个 condition 的 current factor 均不相同；world-model 参数 hash 前后不变；
- formal wall time `172.09 s`，仅用物理 GPU 0（NVIDIA L40）；PyTorch peak allocated `2.849 GiB`、reserved `3.305 GiB`；
- 15 项本地相关测试通过；远端无 pytest，`py_compile` 通过；
- 独立审计：posterior replay 最大误差 `1.39e-17`，FIFO、metric、summary replay 最大误差均为 0。

证据 hash：source snapshot `5252c217dd10629919358e43492a10ac1ee7624ec66ddb5f8b36b98ed935c6ea`；pool audit `557d95f910668b1217da04b74a6b7821f4a6eab0fe9dccd6518940d40f7e8fe6`；E1 raw `a051312aaf3b98acbcf41ebe0b8dc211c56569766d6e27676b1af9e92b1c2b53`；E2 raw `419b5adba34c7470bc1d2460f23c5f0e45b18ef695ed5e1af2f43ac16affa17d`；summary `3051a128918567d235b0227467e0fe2882ea3d91a45abf9b3217d4fc5d5bf940`；audit `3a126836c80d0e1a602b48a622874e52d7be0010f27431e09f19c1e85e8a6b7e`。

## 7. 结论边界与下一步

本实验建立：离散 delay 可由过去 command/proprio 近乎确定地辨识；预定义 high-delay usage policy 在该分布上有平均行为价值。

本实验没有建立：delay history 的 persistence-specific 闭环价值，或 learned delay gate 的必要性。主 MAP、gate、shuffled 和 wrong 的结果共同支持更保守的解释：当前收益主要来自 planner regularization/command shaping，而非精确保存持续物理状态。

因此现有证据不直接支持继续训练 delay learned gate。是否继续投入仍由用户基于完整证据决定；按既定依赖顺序，信息价值更高的下一项是 P3 CoG Markov/contact-state 表示审计，而不是在相同 delay 数据上继续调 noise、threshold 或模型。

