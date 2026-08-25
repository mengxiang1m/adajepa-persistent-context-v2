# Persistent Context V2：Functional Shadow Gate 正式结果

## 1. 实验要回答什么

已有 PushObj 实验表明，context plan 在某些 factor 区域明显优于 population plan，在另一些区域却会造成伤害。本实验不再问“context 总体有没有用”，而是问：

> 只读取规划前已经存在的低维 factor 估计，能否选择何时使用 context plan，从而保留主要收益并减少负向尾部？

这是一次对冻结原始轨迹的 retrospective paired shadow evaluation。它没有重新训练策略、重跑 planner 或生成新动作，而是在每个严格配对单位中，根据冻结规则选择已经真实执行过的 population/context 结果。它验证选择逻辑的行为价值，但不是新的前瞻随机实验。

## 2. 输入、信息边界与规则

共使用三个任务、96 个统计单位：

| 任务 | 原始数据 | 单位 | gate 可读取的信息 | 冻结规则 |
|---|---|---:|---|---|
| dead zone | Stage 1 persistent raw | 32 sequences，结果为 E2–E4 均值 | correct-history censored-MLE 的 `d` 估计 | `estimated d > 0.10` 时用 context |
| delay | Stage 0 raw | 32 pairs | 真实 delay | `delay > 2` 时用 context |
| rotation×gain matrix | Stage 1 persistent raw | 32 sequences，结果为 E2 | correct-history Bayesian posterior matrix | `gain < 0.932780492` 或 `abs(rotation) >= 15°` 时用 context |

Delay 的规则读取真实 factor，因此只代表 functional ceiling；它不能直接部署，也不等价于已经完成 delay estimator。Dead zone 与 matrix 使用历史 estimator 的输出，不读取真实执行结果。

所有 gate 均禁止读取真实 cost、success、goal、segment/seed、planner `best_loss` 或逐单位 best-of-two 标签。详细信息边界见冻结合同与 design JSON。

每个任务比较：

- `population`：始终使用 population plan；
- `always_context`：始终使用 context plan；
- `functional_gate`：按上述规则选择；
- `inverted_gate`：做完全相反的选择，作为结构负对照；
- `best_of_two_behavior_ceiling`：事后逐单位选真实 cost 较低者，只用于表示可回收上限。

## 3. 正式结果

以下改善均相对 population，cost 越低越好。CI 是按对应统计单位做的 paired bootstrap 95% CI，括号内为 raw cost delta 的区间。

| 任务 | population mean | always-context 改善 | functional gate 改善 | gate harm fraction | inverted gate | best-of-two 上限 | 上限回收率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| dead zone | 2.253478 | 1.5814% | **5.6629%**，CI `[+0.066199,+0.195067]` | **6.25%** | -4.0815% | 6.4713% | 87.51% |
| delay | 2.790067 | 1.4767% | **6.1108%**，CI `[+0.094041,+0.254564]` | **0.00%** | -4.6341% | 6.4469% | 94.79% |
| matrix | 2.263778 | 9.5308% | **11.1518%**，CI `[+0.073530,+0.427226]` | **12.50%** | -1.6210% | 13.9196% | 80.12% |

三任务宏平均相对改善：

| policy | 宏平均改善 |
|---|---:|
| always context | 4.1963% |
| functional gate | **7.6418%** |
| inverted gate | -3.4455% |
| best-of-two ceiling | 8.9459% |

Functional gate 在三个任务上都优于 population；三个任务的 paired bootstrap delta CI 均在 0 以上。相反的选择规则在三个任务上都退化，说明收益来自“在哪些区域使用 context”这一结构，而不是简单地减少 context 使用次数。

## 4. Gate 相对无条件 context 的变化

| 任务 | always-context harm | functional-gate harm | harm 降低 | gate 相对 always-context 的 cost 改善 | paired CI |
|---|---:|---:|---:|---:|---:|
| dead zone | 53.125% | **6.25%** | 46.875 pp | **4.1470%** | `[+0.042830,+0.150248]` |
| delay | 43.75% | **0.00%** | 43.75 pp | **4.7036%** | `[+0.040318,+0.250080]` |
| matrix | 25.00% | **12.50%** | 12.50 pp | **1.7918%** | `[-0.008340,+0.092506]` |

Dead zone 和 delay 中，gate 相对 always-context 的改善 CI 在 0 以上。Matrix 的均值更好且 harm 减半，但 gate-vs-always 的 CI 仍跨 0；因此不能把 matrix 上额外的 1.79% 当作已经独立确定的提升。

## 5. 分 factor 发生了什么

### Dead zone

- `d=0.04/0.08` 时保留 population，避免了 always-context 在这两档的总体退化；
- `d=0.12/0.16` 时使用 context，分别改善 `4.1854%/13.9798%`；
- gate 选择 context 的比例为 50%。

### Delay

- `delay=0/1` 时保留 population；
- `delay=3/4` 时使用 context，分别改善 `5.0347%/12.5496%`；
- gate 选择 context 的比例为 50%，该结果仍属于真实 factor functional ceiling。

### Rotation×gain matrix

- gate 在 75% 的 sequences 上选择 context；
- 它避开了总体有害的 `theta=-10°, gain=1.18` 区域，同时也会保守地放弃一部分 `theta=+10°, gain=1.18` 的正收益；
- 最终从 best-of-two 上限中回收 `80.12%`，说明简单低维规则已经有效，但仍有可由 learned surrogate 改善的选择空间。

## 6. 审计与复现

远端独立审计结果：

- `valid=true`，`failures=[]`；
- 96 个单位完整复算；
- gate decision mismatch：`0`；
- 单位 outcome replay 最大绝对误差：`0.0`；
- summary replay 最大绝对误差：`0.0`；
- design SHA256：`384a2ae480abe3987dfd7c227a0c6aaf01a10902e17e4aea3c34cd2aa2f80271`；
- 远端 `shadow_units.jsonl` SHA256：`5c3ce370c17f5ed4fefca0bfeb3992a73170876d6bad194d35561ff4db4f637b`。

Windows 与 Linux 对少数 matrix `hypot` 结果存在约末一位浮点表示差异，但所有选择、行为 outcome 和正式 summary 完全一致；远端 manifest/audit 使用 Linux 正式产物哈希。

实现与产物：

```text
research/persistent_context_v2/functional_shadow_gate.py
scripts/run_persistent_context_v2_functional_shadow_gate.py
scripts/audit_persistent_context_v2_functional_shadow_gate.py
tests/test_persistent_context_v2_functional_shadow_gate.py
docs/research/persistent_context_v2_functional_shadow_gate_design.json
docs/research/persistent_context_v2_functional_shadow_gate_contract_zh.md

/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_functional_shadow_gate/
  runner_summary.json
  shadow_units.jsonl
  manifest.json
  independent_audit.json
```

## 7. 客观结论与边界

本实验支持以下判断：

1. Context 的价值具有明显的条件性；一个简单、可解释的选择器可以比 population 和无条件 context 都更好。
2. Gate 的主要作用不是额外提高最佳区域的收益，而是避开 context 会伤害行为的区域。三个任务的 harm fraction 都明显下降。
3. 低维 factor estimate 已经包含足以支持行为选择的信息；不需要用 planner proxy loss 冒充真实价值信号。
4. 当前最值得继续的是 learned surrogate gate，但必须使用新的、相互独立的 train/dev/formal 数据，不能在这 96 个单位上训练后再报告同批结果。

本实验没有证明：

- 已有一个可部署的 delay gate；
- learned gate 已能跨新 seed、factor 或任务泛化；
- shadow 选择在在线前瞻执行中必然保持同样大小的收益；
- 原始视觉 AdaJEPA 已经学会该选择机制。

因此，这一步给出了明确的继续价值，同时也把下一步问题收窄为：用不读取真实 factor 的 surrogate，在独立新数据上复现这种减害与收益保留能力。
