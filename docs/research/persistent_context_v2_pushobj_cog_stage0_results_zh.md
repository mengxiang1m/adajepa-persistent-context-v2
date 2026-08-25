# PushObj 水平质心 CoG Simulator-Oracle Stage 0 结果

报告日期：2026-08-23  
合同：`persistent-context-v2-pushobj-horizontal-cog-simulator-oracle-stage0-v1`  
正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_stage0/`

## 一句话结论

在32个全新 early-waypoint pairs 上，知道真实水平 CoG 的 ground-truth physics oracle 将 `pose_auc10` 从 `1.339719` 降至 `0.979750`，改善 `26.8690%`；mean delta `+0.359969`，paired bootstrap 95% CI `[+0.231569,+0.501601]`，26/32 pairs 正向。四档 CoG 均值与分组 CI 全部正向，deadline success 从 `81.25%` 提高到 `96.875%`。

这证明 CoG 任务具有清晰的 true-factor 行为上限，值得进入 factor-conditioned neural predictor 训练。它尚未证明现有 AdaJEPA predictor 能表达 CoG，也尚未证明历史能够估计 CoG。

## 1. 为什么选择 CoG

环境审计发现，当前发布 PushObj 中真正参与 Pymunk 碰撞的 block/agent shapes friction 都是 `0.0`；源码中的 `body.friction=1` 不作用于 shape碰撞。直接修改 friction 会先改变 nominal benchmark接触模型，因此本轮没有把它当成干净的单因素实验。

CoG 是环境原生支持的 body physics。factor只改变 T物体的水平质心：

```text
block center_of_gravity = (cog_x, 45)
```

四档 factor 的初始 visual、agent proprio 和完整 raw state均逐元素相同，最大差异 `0.0`，所以 factor没有从单帧泄漏。

## 2. 实验做了什么

- development factors：`[-22.5,-7.5,+7.5,+22.5] px`，每档8个场景；population prior为train support均值 `0 px`。
- 32个场景来自此前未观察行为的 development early-contact segments。
- nominal CoG下重放发布前10个动作，以step-10 block pose作为waypoint。
- prior与oracle都使用相同ground-truth Pymunk CEM：128 samples、top16、5 rounds、初始sigma `0.2`，并共享CEM seed与初始发布动作mean。
- prior规划模拟器使用 `cog_x=0`；oracle使用环境真实 `cog_x`；两者最终都在真实factor环境执行10步。
- 没有神经predictor、history、TTT、MPC或额外动作。该实验测的是任务ceiling。

## 3. 主结果

| 指标 | prior | oracle | 差异 |
|---|---:|---:|---:|
| pose AUC10 | 1.339719 | 0.979750 | 改善26.8690% |
| mean paired delta |  |  | +0.359969 |
| delta 95% CI |  |  | `[+0.231569,+0.501601]` |
| positive/tie/negative |  |  | 26/0/6 |
| deadline success | 81.25% | 96.875% | +15.625 pp |

deadline success配对区间为 `[+3.125,+28.125]` pp。

终点行为：

- step-10 position error：`10.8124→5.7412 px`，减少 `5.0711 px`，CI `[+2.7150,+7.7890]`；
- step-10 angle error：`0.16177→0.04555 rad`，减少 `0.11623 rad`，CI `[+0.06841,+0.17141]`。

## 4. CoG 分组

| cog_x | prior | oracle | 相对改善 | delta CI | 正向 |
|---:|---:|---:|---:|---:|---:|
| -22.5 | 1.604035 | 1.098829 | 31.4959% | `[+0.229101,+0.761043]` | 6/8 |
| -7.5 | 1.267886 | 1.156953 | 8.7495% | `[+0.032953,+0.196893]` | 5/8 |
| +7.5 | 0.775426 | 0.671924 | 13.3478% | `[+0.042874,+0.171762]` | 7/8 |
| +22.5 | 1.711529 | 0.991294 | 42.0814% | `[+0.486593,+1.032705]` | 8/8 |

收益随CoG偏移幅度总体增大，但两档较小偏移也保持正向均值与正向区间。

## 5. 工程审计

- formal runner exit code `0`；32/32 pairs完整，segments唯一，四factor各8个。
- 无渲染物理rollout与标准`env.step` max state error `0.0`。
- oracle simulator prediction与真实执行max error `0.0`；32/32 prior/oracle计划发生变化。
- 第一次smoke因无渲染循环采用NumPy float64算术，和标准Pymunk `Vec2d`路径产生 `1.53e-5`舍入差；原失败保留。修复为逐行相同算术后，`smoke_repair1`全部identity为0，未改科学配置。
- 第一次independent audit把JSON中的nominal command list按float64重放，产生32个nominal physics failure；正式policy states、metrics、hash和summary均通过。原失败`independent_audit.json`保留；恢复raw运行时float32后，`independent_audit_repair1.json`通过，所有failure count为0。
- 外部wall time `109.75 s`，manifest正式循环 `95.84 s`；CPU max RSS约 `676184 KB`；未使用GPU。

## 6. 证据边界与下一步

已证明：隐藏水平CoG不从episode初始信息泄漏；知道CoG能显著改变最佳早期推动动作，并在真实Pymunk闭环中改善position、angle与success；该收益覆盖四档development factors。

没有证明：神经world model能条件化预测CoG动力学；非特权方法能从历史接触transition估计CoG；history收益具有persistence-specific性。

下一最小动作不是直接做history network，而是冻结factor-diverse数据与模型合同，训练最小CoG-conditioned predictor，并先比较：

```text
population-prior context
vs true-CoG context
vs ground-truth simulator oracle
```

训练必须保留frozen base对照，并在未用于任务开发的formal factors/scenarios上报告latent prediction与10-action闭环行为；只有预测改善而无行为改善不能算成功。

## 7. 文件

- 合同/设计：`docs/research/persistent_context_v2_pushobj_cog_stage0_contract_zh.md`、`persistent_context_v2_pushobj_cog_stage0_design.json`
- core：`research/persistent_context_v2/pushobj_cog_stage0.py`
- runner/audit：`scripts/run_persistent_context_v2_pushobj_cog_stage0.py`、`scripts/audit_persistent_context_v2_pushobj_cog_stage0.py`
- tests：`tests/test_persistent_context_v2_pushobj_cog_stage0.py`
- valid audit：正式输出内 `independent_audit_repair1.json`
