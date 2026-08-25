# 任务说明：文档入口与治理规则

更新日期：2026-08-24

本目录只保留三类当前入口，避免历史 prompt、阶段报告和聊天记录同时充当“现行路线图”。

| 文档 | 作用 | 是否可覆盖历史合同 |
|---|---|---|
| [`persistent_context_v2_research_prompt_zh.md`](./persistent_context_v2_research_prompt_zh.md) | 研究治理、证据边界和新方向的通用模板 | 否 |
| [`Persistent_Context_V2_实验全景与后续路线图.md`](./Persistent_Context_V2_实验全景与后续路线图.md) | 已完成实验、证据强度和结论边界的全景索引 | 否 |
| [`Persistent_Context_V2_后续实验计划_2026-08-23.md`](./Persistent_Context_V2_后续实验计划_2026-08-23.md) | 从当前状态出发的实验优先级、设计和资源计划 | 否；每个 formal 实验仍需单独冻结合同 |

## 优先级

发生冲突时使用以下顺序：

1. 用户最新明确要求；
2. 尚未产生 formal 结果的当前冻结合同；
3. 本目录的研究 Prompt 与当前实验计划；
4. 实验全景；
5. 历史 prompt、roadmap、phase 记录和 closeout 文档。

历史合同只约束它原先定义的实验。后来的治理规则不能改写已经生成的 raw artifact、有效性判定或当时的机器裁决，但可以限制其结论外推，并决定是否投入新的独立实验。

## 保留与删除

- 必须保留：冻结合同、design/manifest、raw artifact、聚合与独立审计、负结果、无效执行说明、修复记录和最终报告。
- 可以删除：不承担证据责任的重复 Prompt、恢复出的聊天全文、只挑选正向结果的汇总，以及已被当前入口完整取代且没有历史引用的执行说明。
- 不把 smoke、repair、retry 当成独立科学实验；也不删除它们的原始产物。
- “过时”不等于应删除。若文档是历史合同或被历史 closeout 引用，应保留原文，并按历史证据而非现行指令读取。

## 当前一句话路线

作者 T 与六组跨-shape formal 已建立 rotation×gain context 的跨任务价值，D3 未找到稳定 harm selector。Delay factor 可精确辨识，但 persistence-specific 行为值未建立。CoG P3a/P3b 已完成，10 Hz 拼接和 100 Hz event ridge 都未形成可部署 predictor，CoG V3/history 暂停。下一候选是 matrix rollout-disagreement harm feasibility；episode-local TTT 和视觉迁移继续后置。
