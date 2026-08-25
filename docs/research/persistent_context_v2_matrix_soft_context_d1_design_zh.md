# Persistent Context V2：Matrix soft-context D1 探索设计

日期：2026-08-23  
设计 ID：`persistent-context-v2-matrix-soft-context-d1-exploratory-v1`  
状态：运行前冻结的探索设计；不是 formal 合同

## 1. 问题

D0 表明任务交互特征能提高 `context benefit` 的连续预测相关性，但“预测为正就全量使用 context”的硬开关没有降低 harm。D1 只回答一个更小的问题：

> 在已经得到 posterior context 的情况下，部分向 posterior 收缩是否比 full context 更稳健？

D1 不训练 gate，不选择新特征，不声称泛化到新数据或新 factor。

## 2. 固定处理

对每条旧 Matrix learned-gate sequence 的 E1 posterior matrix `M_posterior` 和 population prior matrix `M_prior`，固定：

```text
M_α = (1 - α) M_prior + α M_posterior
α ∈ {0, 0.25, 0.5, 0.75, 1}
```

- `α=0` 是 population；
- `α=1` 是 always-context；
- 中间三个值表示 context 强度，不是随机抽样 posterior；
- α 网格在任何 D1 outcome 产生前冻结，不得看结果后加密有利区间；
- 所有 α 使用同一 checkpoint、初态、目标、环境 seed、CEM seed、规划预算和执行 horizon。

## 3. 数据与证据等级

使用旧 train/dev/formal 共 96 条 sequence 的冻结 E1 history、factor、初态和目标。三个 split 在 D1 中统一视为已有数据上的探索，不保留 formal 语义。

不根据旧 population/context outcome、D0 prediction、factor subgroup 或 scene geometry 筛选 sequence。所有 96 条都运行完整 α 网格。

## 4. 必需 identity 与无效规则

在汇总任何中间 α 前，必须满足：

1. `α=0` 的 command、state 与 `pose_auc10` 在数值容差内复现原 population 分支；
2. `α=1` 同样复现原 context 分支；
3. 每个 α 的各策略从相同初态和 RNG 状态开始；
4. 策略运行之间 model 参数、running stats、optimizer、replay 不变；
5. 96×5 个处理组合无静默缺失。

任一端点 identity 系统性失败则整批无效，先查协议，不解释剂量曲线。工程修复必须写入新 `repairN/retryN` 目录并保留失败产物。

## 5. 预先固定的报告

主描述量按 α 报告：

- sequence-level `population_pose_auc10 - alpha_pose_auc10` 的均值与 paired bootstrap 95% CI；
- 相对 population 的 harm fraction、positive/tie fraction；
- 相对 full context 的 paired delta；
- 每条 sequence 的五点剂量曲线及非单调比例；
- 按原 factor 平衡分组的均值和 harm，仅作异质性描述。

额外报告两个探索性 ceiling：

- 全局最佳固定 α：在 96 条整体均值上最优，只描述，不视为无偏估计；
- 每条 sequence 事后最佳 α：只表示 soft-context 的机会空间，不是可部署策略。

不得因为某个中间 α 最优而把同一 96 条数据重新称为验证集或 formal，也不设置 GO/NO-GO 百分比阈值。

## 6. 解释规则

- 若中间 α 在多个原 split 上方向一致地降低 harm，并保留大部分 full-context 均值收益：支持后续单独冻结“预测 α”的新数据实验；
- 若中间 α 只在一个 split 或少数 factor 上有利：结论是异质性线索，不能冻结通用 α；
- 若曲线基本单调且 α=1 最好：问题主要仍是场景选择，而不是强度；
- 若所有 α>0 都有明显负尾：优先研究保守 fallback、可校准 abstention 或 benchmark/model mismatch；
- 任何结论都只适用于旧 Matrix benchmark。

## 7. 审计与资源

运行前生成新的 source snapshot，并记录 D1 设计文件 hash、checkpoint/data/raw hash。保存逐 sequence/α 的 command、state、planner trace、posterior/prior matrix、seed 和资源日志。独立 audit 不 import runner 的汇总实现，至少复算 α matrix、端点 identity、`pose_auc10`、harm 和 bootstrap 输入。

先单 GPU smoke 3 条 sequence×5 α；通过后最多使用 1–2 张空闲 L40。不得覆盖任何 V1 或 D0 目录。
