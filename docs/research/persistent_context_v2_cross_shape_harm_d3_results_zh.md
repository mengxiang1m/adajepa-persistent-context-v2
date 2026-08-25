# 跨 Shape Harm Attribution D3 探索结果

日期：2026-08-23  
分析 ID：`persistent-context-v2-cross-shape-harm-d3-exploratory-v1`  
证据级别：**outcome-exposed、只读探索，不是新 formal**

## 1. 问题

跨-shape formal 已证明 correct physical context 平均有效，但 fixed `.75` 仍有 `28.125%` harm，external T-F0 harm 为 `33.333%`。D3 只用已有 96 条结果，检查 E2 outcome 前可计算的信号能否跨 shape/factor 识别负向尾部。

固定特征集：

- F0：6 维 posterior factor 二次基；
- F1：F0 + 8 个 scene geometry/candidate-action 差异；
- F2：F1 + 4 个 population/context 交叉 rollout disagreement。

模型只用 ridge continuous-benefit regression，决策为 predicted benefit `>0` 才使用目标 context。评价固定为 leave-one-shape-pair-out 和 leave-one-factor-out；ridge 只在每个 outer-train 内做分组 inner CV。没有搜索树、MLP、Transformer 或事后阈值。

## 2. 有效性

- 96/96 行只读特征完成，未生成任何新 environment outcome；
- 世界模型参数与全局 RNG 前后 hash 不变；scene replay error 为 0；
- 独立审计重算 feature、outcome 和嵌套分组分析，最大误差均为 0；
- features SHA256：`1a447a543eac2ae00336b6bcbd118489cdfd11c53568fa916dd832ae669994ea`；analysis SHA256：`071ebd1f4e76d8795b40afc6109b0bfb2dbd798c828d5318f6fcfcbd2c77db99`；独立 audit SHA256：`9da11ba9d31a55e8a12e575e82c41d52fdda19c62c6c080a4e95b7e09ca6c59c`；源码快照 SHA256：`3116a5057f8717d84a671a3956fc944c78cfb74376a3cc7478301ba5f1991204`。

## 3. 结果

### Fixed `.75` 的 safety veto

| 外组评价 | 特征 | prediction corr | 使用率 | mean delta | harm | 相对原 fixed `.75` mean 改变 |
|---|---|---:|---:|---:|---:|---:|
| shape-pair | F0 | -0.090 | 100% | 0.2400 | 28.125% | 0 |
| shape-pair | F1 | -0.061 | 96.875% | 0.2309 | 27.083% | -0.0091 |
| shape-pair | F2 | +0.209 | 100% | 0.2400 | 28.125% | 0 |
| factor | F0 | -0.357 | 100% | 0.2400 | 28.125% | 0 |
| factor | F1 | -0.211 | 98.958% | 0.2393 | 28.125% | -0.0008 |
| factor | F2 | +0.081 | 96.875% | 0.2246 | 28.125% | -0.0154 |

唯一看到的 harm 下降是 shape-pair/F1 从 `28.125%` 到 `27.083%`，即少 1 条 harm；同时 mean delta 损失 `0.0091`，相对原策略改善区间 `[-0.0247,0.0001]`。这不是稳定的安全增益。

### External T-F0 的 safety veto

Factor-outer 的 F0/F1/F2 全部 100% 使用 F0，没有任何 veto。Shape-pair/F2 使用率 `92.708%`，harm 从 `33.333%` 降到 `32.292%`，但 mean delta 从 `0.2673` 降到 `0.2449`；相对原 F0 的 mean 改变为 `-0.0224`，CI `[-0.0451,-0.0049]`。即少 1 条 harm 的同时，平均行为明确变差。

F2 continuous prediction correlation 在 shape-pair/factor 外组上分别约 `0.063/0.263`，说明仍有弱连续信号；但预定义零阈值无法把它转成有用的风险选择。相关性提高不等于闭环策略改善。

## 4. 结论

现有 18 维以内的 factor、geometry/action、rollout disagreement 不能稳定识别跨-shape负向尾部。结果不授权：

- 直接在剩余作者 segments 上启动 harm-aware formal；
- 根据当前 96 条反复调阈值；
- 换树、MLP 或 Transformer 后从同一批 outcome 中择优。

当前稳健默认继续是 fixed `.75`。External F0 适合作为“均值优先、可容忍更高 harm”的诊断候选，不是安全 gate。

下一项转向独立科学问题 P2：从真实 command/proprio history 非特权估计 action delay，并用 persistent/no-persistence/wrong-history 对照检验闭环价值。显式 context + episode-local adaptation 仍排在 delay 后；不恢复无条件权重 carry。
