# Persistent Context V2：Matrix Learned Surrogate Gate 正式结果

## 1. 研究问题与实验性质

上一阶段的 functional shadow gate 使用手写低维规则，证明“什么时候使用 context”有行为价值。本实验进一步训练一个真正的数据驱动 surrogate：它只根据 E1 history posterior 的 estimated gain/rotation，预测 E2 中 context plan 相对 population plan 的行为差，并在未参与训练或调参的 formal sequences 上决定使用哪条 plan。

这是 prospective split paired shadow evaluation：数据合同、split、特征、模型族和决策规则先冻结；随后生成 train/dev/formal 三批严格配对真实执行。Formal 原始行为与 train/dev 同时采集，但只有在 train 拟合、dev 选 alpha、train+dev refit 全部锁定后才被读取评价。

## 2. 数据与算法

- train/dev/formal 各 32 条 persistent sequences，共 96 条；
- 每条 sequence 包含 E1 evidence 和 E2 cold-start evaluation，共 192 个互不重复片段；
- 三个 split 无片段重叠，并排除旧 matrix Stage 1/shadow gate 的 64 个直接输入；
- 每个 split 平衡覆盖 8 个 rotation×gain 组合，各 4 条；
- E2 population/context 共享 initial state、goal、factor、env seed、CEM seed、checkpoint 和预算；
- E1 estimator 只读取 command 与可观察 agent position/velocity。

Gate 特征只有 estimated gain 和 estimated rotation，固定二次基为 `[1,g,r,g²,gr,r²]`。标签为 E2 `population pose_auc10 - context pose_auc10`。候选 ridge alpha 为 `[0,0.01,0.1,1,10]`；train 拟合、dev 按 gate 的 paired behavior delta 选择，最终选中 `alpha=1.0`，再用 train+dev 共 64 条 refit。预测 delta 大于 0 使用 context，否则使用 population，不设额外效果量门槛。

禁止输入真实 factor、segment/sequence ID、goal、initial state、seed、planner `best_loss`、formal outcome 或 best-of-two 标签。

## 3. 数据批次本身

无条件 context 在三个新 split 上均总体正向，但都有负向 sequences：

| split | population mean | context mean | 改善 | context 正向 | context 负向 |
|---|---:|---:|---:|---:|---:|
| train | 2.412964 | 2.140439 | 11.2942% | 62.50% | 37.50% |
| dev | 2.278219 | 2.015687 | 11.5236% | 78.125% | 21.875% |
| formal | 2.703538 | 2.398226 | 11.2931% | 68.75% | 31.25% |

这说明 formal 不是一个“context 几乎总是正确”的简单批次，确实包含 gate 需要处理的负向尾部。

## 4. Formal 主结果

Cost 越低越好；CI 为 sequence-level paired bootstrap 95% CI，区间单位是 raw pose AUC10 delta。

| policy | treatment mean | 相对 population 改善 | delta CI | context 选择率 | harm fraction |
|---|---:|---:|---:|---:|---:|
| always context | 2.398226 | 11.2931% | `[+0.137531,+0.471757]` | 100% | 31.25% |
| **learned gate** | **2.371274** | **12.2900%** | **`[+0.180032,+0.482775]`** | 87.50% | **21.875%** |
| functional gate | 2.379701 | 11.9783% | `[+0.176250,+0.475641]` | 75.00% | **15.625%** |
| selection-matched random | 2.451268 | 9.3311% | `[+0.085577,+0.420290]` | 87.50% | 28.125% |
| inverted learned gate | 2.730490 | -0.9969% | `[-0.091201,+0.021231]` | 12.50% | 9.375% |
| best-of-two ceiling | 2.309728 | 14.5665% | `[+0.269623,+0.523000]` | 逐条最优 | 0% |

Learned gate 相对 population 的 mean delta 为 `+0.332264`，回收 best-of-two opportunity 的 `84.37%`。它比无条件 context 的 cost 再降低 `1.1238%`，但 learned-vs-always 的 paired delta CI 为 `[-0.021231,+0.088537]`，仍跨 0。因此可以客观陈述“均值更好且 harm 下降”，不能声称额外 1.12% 已被本批样本稳定确定。

Functional gate 的均值略弱于 learned gate，但 harm 更低。两者代表不同取舍：learned gate 更积极地保留可能收益，functional rule 更保守。

## 5. Learned gate 学到了什么

Formal 回归/分类诊断：

- prediction 与真实 paired delta 的相关系数：`0.5327`；
- MSE：`0.171714`，MAE：`0.352293`；
- beneficial-context precision：`75.00%`；
- recall：`95.45%`；
- 排除 ties 后 accuracy：`75.00%`；
- confusion：TP `21`、FP `7`、FN `1`、TN `3`。

模型在 `theta=+10°, gain=1.18` 的 4 条 formal sequences 上全部保留 population；该组 context mean delta 为 `-0.215615`，因此成功避开整组总体伤害。其余 7 个 factor 组均选择 context。

值得注意的是，旧批次中明显有害的 `theta=-10°, gain=1.18` 在本批 formal 上均值变为小幅正向 `+0.067417`，组内仍是 2 正、2 负。这说明“某个离散 factor 永远有害”的结论不稳定，行为价值还取决于场景/goal/contact geometry。Learned gate 比固定 factor rule 多选择该组，提升了总体均值，但也保留了一部分组内伤害。

## 6. 审计与复现

独立审计不 import 数据 runner 或 learned-gate 实现，重新完成：split/segment/seed/factor 检查、Bayesian posterior、特征、全部 ridge 候选、dev alpha 选择、train+dev refit、formal 决策、policy outcome、bootstrap 与 summary。

审计结果：

- `valid=true`，`failures=[]`；
- train/dev/formal 均为 32 条；192 个片段全部唯一；
- pairing、seed、factor failure 均为 0；
- segment eligibility failure 为 0，checkpoint/data hash 与三个 collection manifest 一致；
- E2 `pose_auc10` 从 raw states/goal 独立重算最大误差：`0.0`；
- posterior mean/covariance replay 最大误差：`0.0/0.0`；
- selected alpha：`1.0`；beta replay 最大误差：`0.0`；
- formal decision mismatch：`0`；summary replay 最大误差：`0.0`。

关键 SHA256：

```text
design  607d5e943635e34c10883ac16b37162c212e1b0e30fd075bcb1f7e6136f3d756
train   cee0c3cab8ce4e5fd328b789b81f5efe4cf77de87b197bffc9c8a286c3a35acd
dev     d9a33dec9ce8d62e0555d7e7d5a31cc7de637463f850ea98f4b69df7a9ad8f80
formal  f95465d3bd7401f36469bc4c101b57508803b3a3e0f7dcb779861466c37aa0d8
model   17efaa418ba99f872fab7c8f80acae8ca1aef370a53d292f2b0a490d6392a313
```

代码与远端产物：

```text
research/persistent_context_v2/pushobj_matrix_gate_data.py
research/persistent_context_v2/matrix_learned_gate.py
scripts/run_persistent_context_v2_matrix_gate_data.py
scripts/run_persistent_context_v2_matrix_learned_gate.py
scripts/audit_persistent_context_v2_matrix_learned_gate.py
tests/test_persistent_context_v2_matrix_learned_gate.py

/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_matrix_learned_gate/
  train/raw.jsonl
  dev/raw.jsonl
  formal/raw.jsonl
  evaluation/gate_model.json
  evaluation/formal_decisions.jsonl
  evaluation/runner_summary.json
  evaluation/independent_audit.json
```

## 7. 结论与下一步

本实验给出了第一个使用独立 train/dev/formal 数据的 learned behavior gate 正向结果：它在全新 formal sequences 上保留了 context 的主要收益，将无条件 context 的 harm 从 `31.25%` 降至 `21.875%`，总体改善从 `11.29%` 提高到 `12.29%`。方向有继续价值。

同时，factor-only gate 的能力已接近边界。同一 factor 组内仍会同时出现明显正向和负向场景；只看 gain/rotation 无法区分。下一版不应继续堆叠 factor 多项式，而应增加规划前可见、又不使用真实 outcome 的 task-context interaction 特征，例如：

- population/context 候选动作序列的差异；
- 两种 context 下预测轨迹的分歧与不确定性；
- initial-to-waypoint 几何、预期接触时刻与接触方向；
- posterior uncertainty 与计划敏感度的交互。

Planner `best_loss` 仍不应单独作为标签或 gate，因为既有审计已显示它不能可靠代表真实行为方向。下一实验应继续使用独立 split，比较 factor-only 与 factor+task-interaction surrogate，正式端点仍是减害与保留收益。
