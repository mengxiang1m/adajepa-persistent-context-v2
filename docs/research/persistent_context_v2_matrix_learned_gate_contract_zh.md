# Persistent Context V2 Matrix Learned Surrogate Gate V1 冻结合同

合同 ID：`persistent-context-v2-matrix-learned-surrogate-gate-v1`

## 研究问题

Rotation×gain 的 context 在不同 factor/场景中既可能改善也可能伤害行为。Functional shadow gate 已证明低维 factor estimate 中存在可用于选择的信息。本实验进一步检验：只用互斥 train/dev 配对行为学习的低维 surrogate，能否在未参与训练或调参的 formal sequences 上，选择 population plan 或 history-context plan，并减少无条件 context 的负向尾部。

## 数据与拆分

- Train、dev、formal 各 32 条 persistent sequences；每条包含 E1 evidence 与 E2 cold-start evaluation。
- 每个 split 使用 64 个互不重复片段，三个 split 间完全不重叠，并排除上一轮 matrix Stage 1/shadow gate 使用的 64 个片段。
- 片段仅按 index 范围和 nominal step-10 block displacement 选择；选择时不读取 matrix 行为结果。
- 由于可用数据限制，不要求这些片段从未在其他 PushObj 任务中出现；本合同只保证本实验 split 间及相对直接 matrix/shadow 输入的隔离。
- 每个 split 都平衡覆盖 8 个 `rotation×gain` 组合，每个组合 4 条 sequences。

## 配对执行

E1 用 population matrix 规划并在真实持续 factor 下执行，Bayesian estimator 只读取 command 与可观察 agent position/velocity，产生 E2 入口 posterior。E2 在相同 initial state、goal、真实 factor、env seed、CEM seed 和预算下分别执行 population plan 与 posterior-context plan，保存两个潜在行为结果。

Checkpoint、predictor、CEM 与 estimator 均冻结，不做 TTT 或参数更新。主指标为 E2 `pose_auc10`，sequence 是最小统计单位。

## Learned surrogate 的信息边界

模型输入只有 E1 posterior 推导出的：

- estimated gain；
- estimated rotation。

固定二次基为 `[1,g,r,g²,gr,r²]`，其中 `g=(gain-population_prior_gain)/0.25`，`r=rotation/30°`。训练目标是 E2 `population_pose_auc10-context_pose_auc10` 的真实配对差。

禁止输入 segment/sequence ID、goal、initial state、factor truth、env/CEM seed、planner `best_loss`、E2 state、cost、success 或 best-of-two 标签。Formal outcome 在决策全部生成之后才允许用于评价。

候选 ridge alpha 固定为 `[0,0.01,0.1,1,10]`。只在 train 拟合候选；在 dev 上按 learned-gate 的真实 paired delta 选择，完全并列时选更大的 alpha。随后用 train+dev 重新拟合该 alpha，formal 只评价一次。决策不设效果量门槛：预测 delta 大于 0 使用 context，否则使用 population。

## 对照与报告

Formal 同时报告：population、always-context、learned gate、上一阶段的 functional rule、inverted learned gate、与 learned gate 精确匹配选择数量的 fixed-seed random gate、best-of-two behavior ceiling。

每项报告 mean、相对 population 改善、paired bootstrap 95% CI、正/平/负与 harm fraction；另报告 learned gate 相对 always-context、选择率、confusion/precision/recall、regression error、按 factor 结果和 best-of-two opportunity recovery。不设置任意固定改善比例裁决线，所有结果均保留。

## 审计

独立审计重新验证 design、checkpoint、data 与 raw hash；片段和 seed 隔离；E1 posterior；feature extraction；alpha 选择；train+dev refit；formal 决策；每项 outcome；bootstrap 和 summary。独立审计不得 import runner 或 learned-gate 实现。
