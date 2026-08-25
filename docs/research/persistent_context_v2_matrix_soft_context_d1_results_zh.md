# Persistent Context V2：Matrix soft-context D1 探索结果

日期：2026-08-23  
证据等级：冻结设计下复用旧 96 条 sequence 的探索；不是新的 formal

## 1. 结论

部分使用 posterior context 能明显缓和 full-context 的负向尾部，并保留大部分平均收益。

- `α=0.75` 的平均改善为 `11.4495%`，与 full context 的 `11.3645%` 基本相当，但 harm 从 `29/96` 降到 `18/96`；
- `α=0.5` 的平均改善为 `9.4119%`，保留 full-context mean delta 的约 `82.8%`，harm 降到 `14/96`；
- 中间 α 在 train/dev/formal 三个旧 split 上都比 `α=1` 伤害更少；
- 但 `α=0.75` 相对 `α=1` 的聚合 mean cost 只低 `0.002095`，差异很小，而且这是旧数据上的探索性选择，不能称为已验证的最优 α。

因此 D1 支持把 context 从二元“用/不用”改成“使用多少”。下一步应先在旧数据上冻结并检验一个低容量的逐场景 soft policy，再用全新数据正式比较 learned α 与固定 `α=0.5/0.75`；不能把本次 post-hoc 最佳值直接包装成 formal 方法。

## 2. 固定设计

对旧 Matrix learned-gate train/dev/formal 各 32 条 sequence，统一降级为 exploration，并完整执行：

```text
M_α = (1 - α) M_prior + α M_posterior
α ∈ {0, 0.25, 0.5, 0.75, 1}
```

共 `96×5=480` 个 treatment。每个 α 共享相同 checkpoint、初态、目标、env seed、CEM seed、规划预算和执行 horizon。网格在 outcome 产生前冻结，没有按中途结果增删 α 或筛 sequence。

## 3. 主结果

| α | mean cost | mean delta vs population | 相对改善 | paired bootstrap 95% CI | harm | positive | mean delta vs full context |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.4649 | 0 | 0 | `[0,0]` | 0% | 0% | -0.2801 |
| 0.25 | 2.3386 | 0.1263 | 5.1250% | `[0.0944,0.1591]` | 15.625% | 80.208% | -0.1538 |
| 0.5 | 2.2329 | 0.2320 | 9.4119% | `[0.1798,0.2857]` | 14.583% | 82.292% | -0.0481 |
| 0.75 | 2.1827 | 0.2822 | 11.4495% | `[0.2141,0.3521]` | 18.750% | 80.208% | +0.0021 |
| 1 | 2.1848 | 0.2801 | 11.3645% | `[0.1976,0.3653]` | 30.208% | 69.792% | 0 |

`mean delta vs full context > 0` 表示该 α 的 cost 低于 full context。这里 `α=0.75` 的 `+0.0021` 很小，不能把 harm 改善和均值优势混成一个强结论。

## 4. 三个旧 split 的稳定性

| split | α=0.5：delta / harm | α=0.75：delta / harm | α=1：delta / harm |
|---|---:|---:|---:|
| train | 0.2066 / 12.50% | 0.2511 / 25.00% | 0.2725 / 37.50% |
| dev | 0.2208 / 9.375% | 0.2713 / 9.375% | 0.2625 / 21.875% |
| formal（在 D1 中为旧探索数据） | 0.2686 / 21.875% | 0.3243 / 21.875% | 0.3053 / 31.25% |

两个中间强度在三份数据上都降低 harm。`α=0.75` 的均值在 dev/formal 略好于 full，在 train 略差；所以“相当均值、更低 harm”比“α=0.75 稳定更优”更符合证据。

## 5. 异质性与机会空间

- 逐 sequence 事后最佳 α 计数：`0/0.25/0.5/0.75/1 = 12/10/13/21/40`；
- `44/96`（45.83%）的事后最佳 α 位于中间；
- `54/96`（56.25%）的五点曲线不是单调的；
- 逐 sequence best-α mean cost 为 `2.0739`，但它是不可部署的行为 ceiling。

这说明强度异质性真实存在，同时也说明不能只学一个全局 α。非单调比例较高，后续 policy 必须直接以闭环 outcome 校准，不能假设 posterior 越强行为就单调越好。

## 6. 有效性与资源

- 端点 `α=0/1` 的 command、state、`pose_auc10` 与旧 population/context 逐项完全一致；
- 480/480 treatment、96/96 sequence，无重复或缺失；
- 独立 audit 不 import D1 runner，matrix、metric、summary 重放最大误差均为 0；
- endpoint hash mismatch 和 stored array hash mismatch 均为 0；
- model-state 与 RNG digest 运行前后不变；
- raw SHA256：`6a496663d60a82dfdc782ef28529cb14437c6625bd4c2d2e009a9f963c6d20fa`；
- design SHA256：`0447fe672d8d1d33ca57a2d51c9284a002b8b199bcbbd82de33936b5676c9efd`；
- source snapshot SHA256：`e6f88cd00d3c52361558d528fdb702e42b049ecd9cedefdf2f6f3a09eddfe5a6`；
- 单张 NVIDIA L40，wall time `4:48.63`，峰值 RSS `1,428,028 KiB`，PyTorch peak allocated `2,848,889,344 bytes`；退出码 0，运行后 GPU 释放。

两次正式 treatment 前的 smoke 失败均保留：一次为 `PYTHONPATH` 启动问题，一次为 `true_matrix` 字段路径假设错误；二者都没有产生 treatment，也没有改变科学设计。第三次 smoke 的 15/15 treatment 与独立 audit 通过后才放行完整运行。

## 7. 下一步

先做 D2 soft-policy feasibility，仍只把旧 96 条当 exploration：

1. 固定低容量模型和小型特征集，在 train/dev/test split rotation 中预测每个 α 的闭环 cost；
2. 逐场景选择 `α∈{0,0.25,0.5,0.75,1}`，与固定 `α=0.5`、固定 `α=0.75`、full context 和 factor-only hard gate 比较；
3. 主问题不是能否超过不可部署 per-sequence oracle，而是能否在三次 rotation 中稳定超过固定 `α=0.75` 或在相近 mean 下进一步降 harm；
4. 若 D2 仍不稳定，停止增加 gate 维度，直接把固定 `α=0.5/0.75` 作为新 formal 的安全/效能基线；
5. 若 D2 稳定，再生成全新 target 资产，在独立 train/dev/formal 上训练和一次性验证 learned α，并增加先决策、只执行所选 plan 的单路径 replay。

远端权威产物：

```text
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_matrix_soft_context_d1_exploration_v1/
```
