# Persistent Context V2 Matrix F0 Soft Policy Formal V1 冻结合同

合同 ID：`persistent-context-v2-matrix-f0-soft-policy-formal-v1`

## 研究问题

D1 表明固定中间 context 强度能减少 full-context 的负向尾部，D2 表明简单 factor-only 剂量策略在三次旧 split rotation 中相对 fixed `α=0.75` 的 mean cost 方向一致，而 18 维 task-interaction 没有稳定增量。本实验前瞻检验：

> 只根据 E1 posterior gain/rotation 的 6 维二次基选择 context 强度，能否在从未产生过 matrix outcome 的作者发布场景上，优于固定 `α=0.75`？

## 数据边界与科学收缩

作者发布物只有 seed-42 的 1000 个 T-shape segments，没有原始2705条轨迹和生成脚本。701 个片段满足 nominal step-10 block displacement≥10；结构化审计排除288个已进入 matrix outcome 的片段后剩413个。

使用冻结 seed `1130000` 对排序后的413个 index 做 `random.Random(seed).shuffle`，取前384个形成：

- train：64 sequences、128 segments；
- dev：32 sequences、64 segments；
- formal：96 sequences、192 segments；
- reserve：29 segments。

所有片段跨 split、跨 E1/E2 完全互异；每个 split 按 sequence ID modulo 8 平衡 factor。选择时不读取任何新 matrix outcome。该 formal 检验同一作者发布池中 matrix-unexposed scenes，不外推为新轨迹生成分布。

## 处理与信息边界

每条 sequence 的 E1 只用 population matrix 规划和执行，Bayesian estimator 只读取 command 与可观察 agent transition，形成 E2 入口 posterior。E2 固定五个强度：

```text
α ∈ {0,0.25,0.5,0.75,1}
M_α=(1-α)M_population+αM_posterior
```

五个分支共享 checkpoint、factor、初态、目标、env seed、CEM seed、预算和 horizon。Checkpoint、world model、optimizer 和 estimator 均冻结，不做 TTT。

主方法输入只有 posterior 导出的 gain/rotation 二次基 `[1,g,r,g²,gr,r²]`。禁止输入 scene geometry、candidate actions、task-interaction feature、segment/sequence ID、seed、真实 factor、E2 state/contact/success/cost 或 best-α 标签。

## 模型锁定顺序

剂量模型固定为：

```text
predicted benefit(x,α)
=α·xβ_linear+α(1-α)·xβ_curvature
```

1. 完整收集 train 五分支；
2. 对合同中的每个 ridge λ 只在 train 拟合；
3. 完整收集 dev 五分支，按 realized dev policy mean delta 选 λ，完全并列取更大 λ；
4. train+dev refit，写入 locked model、training raw hash、design/contract/source hash；
5. locked model hash 存在且独立审计通过后，才允许启动 formal；
6. formal 不调 λ、阈值、特征、α 网格或模型。

策略在升序 α 网格中选择 predicted benefit 最大者，完全并列选择更小 α，不增加 abstention threshold、risk penalty 或 harm classifier。

## Prospective formal 执行

每条 formal sequence 在 E1 posterior 产生后、任何 E2 outcome 前计算 α decision。先规划并执行被选 α 分支，再按升序执行剩余四个配对分支。raw 必须保存 decision time、predictions、selected α 和 execution order。

因此主 learned outcome 是真实前瞻选择后首先执行的分支；其余分支只用于严格配对对照和不可部署 ceiling。若 selected-first 分支与同 α 复算/审计不一致，该 sequence 无效。

## 端点与统计

唯一主 estimand：formal sequence-level

```text
pose_auc10(fixed α=0.75) - pose_auc10(F0 soft policy)
```

报告 mean unit delta、paired bootstrap 95% CI 和全部96个 unit values。关键次要端点：F0 相对 population/full context、harm fraction 差、positive/tie/negative fraction、α 选择率、按 factor 异质性和 per-sequence best-α ceiling。统计单位只能是 sequence，不设置自动 GO/NO-GO 效果阈值。

## 有效性、修复与审计

- 新 formal 前生成包含 dirty patch 和全部参与源码 hash 的 source snapshot；
- 保存 selection/design/contract/checkpoint/data/model/raw hash、命令、退出码和资源日志；
- smoke 只验证3条 sequence，不读取为科学结论；
- 失败目录不覆盖，有限工程修复写入 `repairN/retryN`；
- 独立 audit 不 import collector/model/evaluator，实现上重新验证 selection eligibility/exposure、posterior、五个 matrix、metrics、train/dev λ、locked model、formal decision、execution order和summary；
- 任何 formal outcome 生成后不得改变处理、样本或主 estimand。
