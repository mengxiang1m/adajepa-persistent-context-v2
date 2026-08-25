# Persistent Context V2：Matrix soft-policy D2 探索结果

日期：2026-08-23  
证据等级：旧 D0/D1 数据的冻结 split-rotation 探索；不是新 formal

## 1. 结论

低容量 soft policy 在三次 split rotation 中都比固定 `α=0.75` 获得更低的 test mean cost，但新增 task-interaction 特征没有稳定增量，而且没有进一步降低 harm。

- F2 task-interaction policy 相对 fixed `α=0.75` 的 mean cost 改善为 `+0.0363/+0.0259/+0.0153`，三轮同向，但三个 paired CI 都跨 0；
- F0 factor-only soft policy 的对应改善为 `+0.0169/+0.0552/+0.0351`，同样三轮同向，其中一轮 CI 排除 0；
- F2 相对 F0 为 `+0.0194/-0.0292/-0.0199`，只有一轮正向，两轮负向；
- F2 相对 fixed `α=0.75` 的 harm fraction 在三轮都增加 `3.125` 个百分点。

因此 D2 支持继续验证“简单 factor-conditioned soft strength”，不支持把 18 维 task-interaction F2 带入下一轮主 formal。下一轮应以 F0 soft policy 对 fixed `α=0.75` 为主比较，并保留 fixed `α=0.5`、full context 和 population。

## 2. 冻结方法

使用 D0 的决策前 feature 和 D1 的五点剂量 outcome，拟合：

```text
predicted benefit(x, α)
= α · xβ_linear + α(1-α) · xβ_curvature
```

F0 使用 6 个 factor 多项式特征，F2 使用全部 18 个 factor+task-interaction 特征。候选 ridge λ、dev 选择、tie-break、三次 train/dev/test rotation 均在分析前冻结。策略只在 `{0,0.25,0.5,0.75,1}` 中选 α。

## 3. 三轮 test 结果

### 3.1 learned policy 相对 fixed α=0.75

| train→dev→test | policy | selected λ | fixed0.75−policy mean cost | paired 95% CI | policy harm | 相对 fixed0.75 harm |
|---|---|---:|---:|---:|---:|---:|
| train→dev→formal | F0 | 10 | +0.0169 | `[-0.0325,+0.0650]` | 28.125% | +6.25 pp |
| train→dev→formal | F2 | 100 | +0.0363 | `[-0.0077,+0.0805]` | 25.00% | +3.125 pp |
| dev→formal→train | F0 | 10 | +0.0552 | `[+0.0209,+0.0909]` | 21.875% | -3.125 pp |
| dev→formal→train | F2 | 1000 | +0.0259 | `[-0.0110,+0.0644]` | 28.125% | +3.125 pp |
| formal→train→dev | F0 | 1 | +0.0351 | `[-0.0144,+0.0906]` | 15.625% | +6.25 pp |
| formal→train→dev | F2 | 1000 | +0.0153 | `[-0.0346,+0.0729]` | 12.50% | +3.125 pp |

均值方向一致是值得独立验证的信号，但区间、harm 与 F0/F2 对照不支持“复杂特征已解决选择问题”。

### 3.2 F2 相对 F0

| test split | F0−F2 mean cost | paired 95% CI |
|---|---:|---:|
| formal | +0.0194 | `[-0.0044,+0.0508]` |
| train | -0.0292 | `[-0.0585,-0.0037]` |
| dev | -0.0199 | `[-0.0575,+0.0154]` |

F2 没有可重复的增量。它在两个 rotation 中因强 ridge 实际只在 `α=0.75/1` 之间选择，表明 18 维输入在 32-sequence fit split 下主要被收缩，而不是学到稳定的场景级剂量结构。

### 3.3 α 选择分布

F0 三轮选择计数分别为：

- `0/0/8/4/20`；
- `0/0/8/4/20`；
- `0/4/8/4/16`。

F2 三轮为：

- `0/2/4/8/18`；
- `0/0/0/17/15`；
- `0/0/0/15/17`。

顺序均为 `α=0/0.25/0.5/0.75/1`。两种策略都没有选择 population，说明当前训练数据以 context 平均有益为主，尚未学出可靠 abstention；这也是下一次 formal 必须保留 harm 和单路径执行检查的原因。

## 4. 有效性

- D0 96 条 feature 与 D1 480 条 treatment 一一完整 join；
- 独立 audit 不 import D2 分析实现；
- ridge 选择 mismatch、decision mismatch 均为 0；
- prediction、beta、summary 最大重放误差均为 0；
- D0 feature SHA256：`a36d6e31ede28699cfd22b663c364a473345ea751ad597b19a94ac040accb706`；
- D1 raw SHA256：`6a496663d60a82dfdc782ef28529cb14437c6625bd4c2d2e009a9f963c6d20fa`；
- D2 analysis SHA256：`59ede16e5ccf2597dff5f8f9b08d311a212210c9f7d39bb4e82f2e9e2e6c8b35`；
- D2 design SHA256：`0ae9d15db2cf406c8908d428c90b9a13c5bbe47c55805c2de5a0ca5c75d70097`；
- source snapshot SHA256：`1ca19b9d19cf7428529e877f5c80c67520168b8a36c98518077ba41f858a1821`；
- CPU wall time `5.55 s`，峰值 RSS `397,124 KiB`，退出码 0。

## 5. 下一步 formal 的收缩方案

新 formal 不再以 F2 为主方法。建议冻结：

- population `α=0`；
- safety baseline：fixed `α=0.5`；
- efficacy baseline：fixed `α=0.75`；
- full context `α=1`；
- 主 learned 方法：F0 factor-only soft policy；
- per-sequence best α 只作 ceiling；
- F2 task features只记录为诊断，不参与 formal 决策。

主 estimand 为新 formal sequence 上 `cost(fixed α=0.75)-cost(F0 soft policy)`；关键次要量为 harm 差、相对 population/full 的 mean delta、α 选择率和 prospective single-path replay identity。Train/dev 必须完整收集五个 α 的 outcome；formal 可以保留配对五分支用于审计，同时增加先预测 α、只执行该分支的单路径子集。

数据可用性审计进一步把规模收缩为 train/dev/formal=`64/32/96`。发布包没有原始轨迹生成源；1000 个 T segments 中 701 个满足 eligibility，排除 288 个 matrix-exposed 后剩 413 个，足够冻结 384 个互异片段并保留 29 个备用。这个 split 对 matrix outcome 未暴露，但仍来自同一个 seed-42 发布池，不能解释为新轨迹分布。

远端权威产物：

```text
/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_matrix_soft_policy_d2_exploration_v1/
```
