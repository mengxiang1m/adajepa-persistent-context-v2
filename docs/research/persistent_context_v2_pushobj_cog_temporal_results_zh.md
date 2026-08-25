# PushObj CoG Temporal Predictor V2 结果

日期：2026-08-23  
合同：`persistent-context-v2-pushobj-cog-temporal-film-predictor-v2`  
正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_temporal/`

## 一句话结论

单向 GRU temporal residual predictor 没有改善 CoG learned-model 上限。在 32 个从未进入 CoG Stage 0 或 v1 train/dev/formal 的新场景上，v2 true-CoG 相对 population 只改善 `0.3747%`，mean delta `+0.005331`，95% CI `[-0.075191,+0.076605]`，19/32 pairs 正向；相对 v1 反而退化 `2.7814%`。v2 只回收 `2.06%` physics-oracle gap，而同一批场景的 physics oracle 改善 `18.1714%`。

所以失败的是当前 causal-GRU predictor，不是 CoG 任务上限。现在不应继续在它上面开发 CoG history estimator。

## 1. 实验做了什么

- 逐字节复用 v1 的 `7680` train、`768` dev samples，data hash 全部匹配。
- 模型把 v1 的 flattened trajectory MLP 替换为：每步 `nominal state_t + state_t+1 + action_t`，`Linear(18,64)`，单层单向 GRU hidden 128，单一 CoG FiLM，逐步 residual head。
- 参数量 `93,123`；没有第二种 conditioning、contact label、history、gate、TTT 或真实 CoG simulator rollout。
- zero-context branch 与共享 head 相减，因此 `context=0` residual 严格为 0。
- AdamW 训练 3000 steps；每 100 step 在冻结 dev 上评价，最低 dev MSE checkpoint 为 step `200`。
- formal 使用 32 个全新 early-contact segments；四档 CoG 各 8 个。population、冻结 v1、冻结 v2、physics oracle 共享 factor、初态、waypoint、环境/CEM seed 和预算。

## 2. 预测结果

### Held-out-factor dev

| 模型/context | residual MSE |
|---|---:|
| population zero context | 0.121486 |
| v1 true CoG | 0.077857 |
| v2 temporal true CoG | 0.085767 |

v2 相对 population 降低 `29.4021%`，说明它确实使用了 CoG；但相对 v1 MSE 增加 `10.1588%`。

### 新 formal 场景的 prediction-execution pose error

| 模型/context | error |
|---|---:|
| population | 0.655674 |
| v1 true CoG | 0.540001 |
| v2 true CoG | 0.610069 |

v1 相对 population 降低 `17.6419%`；v2 只降低 `6.9555%`，并比 v1 高 `12.9755%`。

事后逐步分析显示，v2 并非只在最后一步突然失败：从早期开始就普遍弱于 v1，后五步平均 prediction error 为 population `0.871964`、v1 `0.702772`、v2 `0.794774`。每个 pair 的 prediction-error reduction 与真实行为 delta 相关系数为 v1 `0.862`、v2 `0.789`。这说明当前 CoG 任务里，预测是否变准与行为是否改善高度相关，主要问题是 v2 的预测本身没有超过 v1。

## 3. 正式闭环结果

| policy | pose_auc10 | deadline success | oracle-gap recovery |
|---|---:|---:|---:|
| population prior | 1.422730 | 75.000% | 0% |
| v1 true CoG | 1.379042 | 71.875% | 16.898% |
| v2 temporal true CoG | 1.417398 | 75.000% | 2.062% |
| physics oracle | 1.164199 | 87.500% | 100% |

### V2 相对 population

- mean delta `+0.005331`；相对改善 `0.3747%`；
- paired bootstrap 95% CI `[-0.075191,+0.076605]`；
- positive/tie/negative：`19/0/13`；
- deadline success 不变；
- 32/32 action plans 改变。

### V2 相对 V1

- mean delta `-0.038357`；相对变化 `-2.7814%`；
- CI `[-0.149678,+0.059426]`；
- 虽然 19/32 pairs 的 v2 cost 小于 v1，但少数较大退化使总体均值更差。

### Physics oracle

- 相对 population 改善 `18.1714%`；mean delta `+0.258530`；
- CI `[+0.144342,+0.386995]`；28/32 pairs 正向；
- success `75.0%→87.5%`。

这在第三批全新 CoG 场景上再次建立了行为上限。

## 4. Factor 异质性

| CoG x | population | v1 | v2 | oracle | v2相对population | v2正向 |
|---:|---:|---:|---:|---:|---:|---:|
| -22.5 | 1.811513 | 1.751362 | 1.699538 | 1.402647 | +6.1813% | 6/8 |
| -7.5 | 0.878274 | 0.881956 | 0.898297 | 0.860828 | -2.2798% | 3/8 |
| +7.5 | 1.032779 | 0.977237 | 0.994740 | 0.888040 | +3.6831% | 7/8 |
| +22.5 | 1.968353 | 1.905612 | 2.077018 | 1.505281 | -5.5206% | 3/8 |

v2 的收益明显不对称；`+22.5` 的较大退化抵消了 `-22.5/+7.5` 的改善。

## 5. 审计与工程记录

- 正式 runner exit code `0`；32 unique pairs、factor balance、waypoint 与 plan-change checks 全部通过。
- 第一次独立 CPU 审计只在 CUDA/cuDNN GRU prediction replay 上失败：10 步累计 state max difference `0.001846`，约为 512 像素 workspace 的 `3.6e-6`；环境执行、行为 metric、summary、zero-context、data 与 checkpoint 全部通过。原 `independent_audit.json` 保留。
- 记录逐策略误差并将跨设备 GRU float32 容差修为 `0.005` 后，`independent_audit_repair1.json` 通过。population/oracle replay error为0，v1为 `3.0518e-5`，v2为 `0.001846`；execution与metric replay error均为0。
- design SHA256：`12b06a368fb7c25b9ecf716db6a65d71fccf4b664ed009d611b60d5165617413`。
- v1 checkpoint SHA256：`39bb54d90a863c012dbd05e9ea448d8e213966a1d47c5f116faebc6f0a06c403`。
- v2 checkpoint SHA256：`fbc5b3e01490e13d198b7a82f7f4f13eefde112ecc013ef8f2b180d52f19b9f0`。
- raw SHA256：`ed38b897ee2c9958c8bdad2423aab042105ac3aff0cff0c3e9334de84da3e478`。
- 正式全流程 external wall `3:42.16`，max RSS `1,291,324 KB`；Torch peak allocated `101,184,000 bytes`，GPU 为 NVIDIA L40。

## 6. 结论与下一步

已经排除的简单假设是：“只要把 flattened MLP 换成 causal GRU，CoG residual prediction 和行为就会自然改善。”它没有发生；v2 在 dev prediction、新 formal prediction、闭环行为和 oracle-gap recovery 四项上都没有超过 v1。

当前不开发 CoG history estimator，因为 true-CoG learned model 还没有稳定行为上限。CoG 若继续，应先重构数据表示，显式保存 block linear/angular velocity、contact impulse/contact geometry 等 Markov/contact state，再冻结一个新的 predictor 实验；不能继续在同一输入上只换网络宽度或层数。

按既定优先顺序，下一项转向 `functional shadow gate → learned surrogate gate`：利用 dead zone、delay 和 matrix 中已存在的正负 factor 区域，先离线记录“使用 context plan”相对 population plan 的预测优势与真实行为差异，验证一个可审计的 gate 是否能减少负向尾部，同时保留已建立的正向收益。

## 7. 文件位置

- 合同/设计：`docs/research/persistent_context_v2_pushobj_cog_temporal_contract_zh.md`、`persistent_context_v2_pushobj_cog_temporal_design.json`
- core：`research/persistent_context_v2/pushobj_cog_temporal_predictor.py`
- runner/audit/analysis：`scripts/run_persistent_context_v2_pushobj_cog_temporal.py`、`scripts/audit_persistent_context_v2_pushobj_cog_temporal.py`、`scripts/analyze_persistent_context_v2_pushobj_cog_temporal.py`
- tests：`tests/test_persistent_context_v2_pushobj_cog_temporal.py`
- 远端输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_temporal/`
