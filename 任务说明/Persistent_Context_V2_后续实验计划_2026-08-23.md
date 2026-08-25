# Persistent Context V2 后续实验计划

日期：2026-08-25
状态：P0、P1-D0/D1/D2/V2/XShape/D3/D4、P2 delay formal、P3a 与 P3b CoG 表示审计已完成；D4 预注册风险方向未成立，是否进入 E2 在线修正由用户决定

## 1. 计划依据

本计划基于本地文档、`adajepa-server:/data4/zhaoqing/adajepa` 的代码、正式报告、raw manifest 和独立审计重新对账。事实与解释分开如下。

### 1.1 已确认事实

- 远端当前没有 Persistent Context/Phase 实验进程；审计时 6 张 NVIDIA L40 均空闲。
- 远端仓库基于 commit `a29975964f966f2836a2c7e26f464367c795c333`，但实验代码和配置是大规模未提交工作树；10 个 tracked 文件有约 2,627 行新增，`research/`、`scripts/`、`tests/`、`docs/` 和多数实验产物未被 Git 跟踪。后续 formal 合同不能只记录 commit，必须记录工作树补丁、全部未跟踪源码清单和逐文件 hash，或先形成可复现快照。
- Rotation history 与 Bayesian rotation×gain matrix 在冻结 PushObj-T early-waypoint 任务中完成了 persistent/no-persistence、shuffled 和 wrong-sequence 对照；matrix correct history 的 E2 `pose_auc10` 改善 `9.5308%`，no-persistence 退化 `20.0171%`，独立审计有效。
- Matrix learned gate V1 使用互斥 train/dev/formal，各 32 sequences。Formal 中 always-context 改善 `11.2931%`、harm `31.25%`；factor-only learned gate 改善 `12.2900%`、harm `21.875%`。Learned-vs-always 的 paired delta CI 为 `[-0.021231,+0.088537]`，所以额外均值优势方向正向但仍不确定。
- P1-D0 已对旧 96 条 V1 sequences 完成只读特征重放和独立审计。F0/F1/F2 在旧 formal 上的 prediction correlation 为 `0.5327/0.6371/0.6674`，说明 task-interaction 特征含有组内信号；但零阈值硬 gate 使 F1/F2 选择率均变成 `100%`、harm 回到 `31.25%`，mean delta 相对 F0 低约 `0.0270`，split rotation 也不稳定。因此不冻结当前 F1/F2 hard-gate V2 formal。
- P1-D1 已完成 `α∈{0,0.25,0.5,0.75,1}` 的 96×5 冻结剂量探索。`α=0.75` 与 full context 平均收益基本相当（`11.4495%` vs `11.3645%`），harm 从 `30.208%` 降到 `18.75%`；`α=0.5` 保留约 `82.8%` full-context mean delta，harm 降到 `14.583%`。这支持 soft context，但不把旧数据上的 post-hoc 最佳 α 当成正式最优值。
- P1-D2 已完成三次 split rotation。F2 task-interaction soft policy 相对 fixed `α=0.75` 的 test mean cost 三轮均正向 `+0.0363/+0.0259/+0.0153`，但 CI 均跨 0、harm 均增加 3.125 pp；F2 相对 F0 为 `+0.0194/-0.0292/-0.0199`。因此继续验证简单 F0 soft policy，但不把 18 维 F2 带入下一轮主 formal。
- P1-V2 已用作者 seed-42 T-shape 发布池完成科学收缩后的 `64/32/96` 前瞻 formal。F0 相对 population 改善 `13.2141%`，95% CI `[0.2124,0.3963]`；相对 fixed `α=.75` 的主差为 `+0.01686`，95% CI `[-0.00865,+0.04122]`，两者 harm 均为 `21.875%`。所以 context 的独立 scene 价值成立，但 learned F0 相对简单 fixed `.75` 的额外价值未建立。
- P1-XShape 已在六个作者非 T shape 池完成 96 条前瞻 formal。Correct-history fixed `.75` 相对 population 改善 `9.9691%`，主 delta `+0.24001`、CI `[0.13935,0.34784]`；factor 不持续时同样 history 相对 population 为 `-0.2249%`、CI 跨 0，harm `59.375%`。Persistence-specific paired delta 为 `+0.24542`、CI `[0.12355,0.36956]`。这支持物理 context 的跨任务、持续因素特异性。
- 同一 rotation×gain 组内仍同时出现获益与受损 sequence。只看 factor 无法识别 scene/goal/contact geometry 引起的行为差异。
- D4 之前的 matrix raw 只保存初态、目标、posterior、commands、真实 states 和 CEM loss trace。D4 已在新 development 数据中补齐默认/上下文模型对两套命令的四条完整 latent rollout，并通过逐数组独立重放。
- Delay 非特权 formal 已完成：E1 estimator `32/32` MAP 正确，但 persistent/no-persistence 的 correct-MAP 改善仅 `0.28%/0.32%`，DiD `-0.00083`、CI `[-0.02355,+0.02095]`。High-delay gate 在两个条件都约改善 `4.4%`，因此有平均行为价值但不是 persistence-specific history value。
- CoG physics oracle 在三批场景中持续显示闭环上限，但 FiLM v1 只回收部分 gap，temporal-GRU v2 相对 population 仅改善 `0.3747%`。P3a 已确认旧 7 维输入缺少 block velocity/angular velocity/contact；R2 contact ridge 相对 R1 改善 `0.09873`、14/16 segment 正向，但仍远差于 v1。更直接的 v1+Markov/contact correction 分别退化 `0.01382/0.03279`，所以“直接拼接控制步聚合字段”未获支持。
- P3b 已完成 24-train/16-eval segment 的 100 Hz event-response audit。唯一主比较 `C10 aggregate - S100 full=-0.00728`、CI `[-0.02137,+0.00603]`、8/16 segment 正向；S100 full error `0.33020`，高于 C10 的 `0.32292` 和 zero response 的 `0.30496`。Nominal event impulse 相对 geometry-only 有增量 `+0.02913`、14/16 正向，但 privileged true-contact ridge 也退化。独立审计有效，因此不训练 CoG V3，不启动 CoG history estimator。
- Phase A–H 已表明无条件权重 carry 会干扰 recurring T；safe slow memory 可减少干扰，但尚未超过 episodic/periodic 行为基线。继续调 slow step、统一 fast scale 或 shape-routed expert 没有现成证据授权。

### 1.2 当前最重要的结论边界

已建立的是：在特定冻结 PushObj benchmark、factor、checkpoint 和 CEM 预算下，显式低维 history context 可产生 persistence-specific 的早期闭环收益，并能从一种作者 shape 迁移到另一种 shape。

尚未建立的是：

- learned gate 相对 always-context 的可重复额外优势；
- gate 对新 factor 连续区间、不同 planner 预算或新 target-generation seed 的泛化；
- delay 的非特权 history persistence-specific value（formal 结果接近 0，未建立）；
- learned CoG predictor 能稳定利用 true CoG 改善行为；
- 显式 context 与 episode-local TTT 的互补性；
- `z_shape + z_physics` 在 T/L/Z 或视觉 AdaJEPA 中的因果链。

因此下一步不直接训练更大的 memory/Transformer，也不回到无条件权重持续。P2 delay、P3a/P3b CoG 和 D4 rollout risk 均已给出边界。D4 不是 formal，预注册风险方向未成立，也没有验证在线 gate。若继续，先用简单的 E2 在线贝叶斯更新检验真实反馈能否修正旧 context；是否投入由用户决定。

## 2. 总体依赖顺序

```text
P0 证据与代码快照治理（已完成）
  ↓
P1-D0 task-interaction feasibility（已完成；hard gate 未获支持）
  ↓
P1-D1 fixed soft-context 剂量探索（已完成）
  ↓
P1-D2 低容量 soft-policy feasibility（已完成）
  ↓
P1-V2 作者 T 池未暴露 scene 的 F0 soft-policy formal（已完成）
  ↓
P1-XShape 作者其他 shape 池的跨任务 history formal（已完成）
  ↓
P1-D3 跨 shape 失败归因与 harm-aware selector feasibility（已完成；未获支持）
  ↓
P2 Delay 非特权 history estimator 与 gate
  （formal 已完成；persistence-specific value 未建立）
  ↓
P3a CoG Markov/contact-state 表示审计（已完成；粗粒度拼接未修正 v1）
  ↓
P3b event-level contact-response audit（已完成；100 Hz 表示未超过 10 Hz/zero）
  ↓
P1-D4 matrix rollout-disagreement harm feasibility（已完成；预注册风险方向未成立）
  ↓
P4 显式 context + episode-local adaptation
  ↓
P5 z_shape + z_physics 与视觉 AdaJEPA
```

P1、P2、P3 回答彼此不同的问题，不以单一固定效果阈值自动停止。但在父命题总体负向或不确定时，必须把继续下游的科学成本和替代解释明确交给用户，不能把“代码可运行”当成授权。

## 3. P0：在新 formal 前固定证据基线

目标不是整理代码美观，而是让任何新结果可复现。

1. 生成 `source_snapshot.json`：base commit、`git diff --binary` hash、所有参与运行的 untracked `.py/.yaml/.json/.md` hash、checkpoint/data hash、Python/CUDA/PyTorch 和 GPU 信息。
2. 给正式 runner 增加 `dirty_worktree=true/false`、patch hash、源码 hash 清单；manifest 不再只记 base commit。
3. 校验 runner 的幂等 resume：成功 run 跳过，失败 run 保留原目录并使用显式 `repairN/retryN`。
4. 为 paired planner 增加只读 identity test：同 context、同 seed 必须逐 command 一致；不同 policy 的环境执行必须从相同初态和 RNG state 开始。
5. 旧合同、负结果和 raw 只归档、不改写；删除的只应是不承担证据责任的误导性文档。

完成产物：`scripts/create_source_snapshot.py`、独立 audit、最小测试和单 GPU smoke。P1-XShape 有效快照记录 base commit、dirty patch、331 个参与源码文件及运行环境，SHA256 为 `73cf3bb41c8546403a6dfa3d4e2aad6a0bb48f8d91513016ce84b179a09d0001`。P0 已完成，后续每次运行仍需生成对应快照；它是有效性要求，不是效果门。

## 4. P1：Matrix context 使用策略

### 4.1 核心问题

在规划前可获得的信息中，加入 scene geometry、两套候选 action 的差异及模型反事实分歧，能否比 factor-only gate 更好地预测 `population cost - context cost`，并在全新 formal sequences 上降低实际 `pose_auc10`？

唯一主处理变量是 gate 的特征集合；factor、history estimator、checkpoint、planner、动作预算和主端点全部保持不变。

### 4.2 已完成：D0 task-interaction 可行性

现有 96 条 learned-gate V1 sequences 只用于 D0 read-only feasibility：

- 重放并验证能在 outcome 不可见时计算候选特征；
- 检查缺失、常量、尺度、共线性和 feature/label 泄漏；
- 用 leave-one-factor-group-out 和 split-rotation 检查方向是否完全由某个 factor 组驱动；
- 现有 formal 在 D0 中降级为 exploration，之后不得再次称为新 formal；
- D0 可以删除无信息特征，但不得根据某条 sequence 的好坏反复发明模型族。

D0 的只读重放、模型/RNG identity 和独立 audit 均通过。任务特征提高了连续预测相关性，但没有改善预定义硬 gate：F1/F2 在主 split 上均退化为 always-context；三次 split rotation 中 F1 均未超过 F0，F2 只出现一次约 `+0.01` 的微小正差。因此 D0 不授权冻结下述原 V2 formal。完整结果见 [`../docs/research/persistent_context_v2_matrix_task_interaction_d0_results_zh.md`](../docs/research/persistent_context_v2_matrix_task_interaction_d0_results_zh.md)。

### 4.3 已完成：D1 fixed soft-context 剂量探索

固定比较 `α∈{0,0.25,0.5,0.75,1}`，使用 `M_α=(1-α)M_prior+αM_posterior`。旧 96 条全部降级为 exploration；不得根据旧 outcome 或 D0 prediction 选场景。`α=0/1` 必须复现原 population/context command、state 和 `pose_auc10`，否则整批无效。

主报告是各 α 的 sequence-level mean delta、paired CI、harm、非单调比例和三个原 split 的方向一致性。全局最佳固定 α 和逐场景最佳 α 只作为乐观 ceiling。详细冻结设计见 [`../docs/research/persistent_context_v2_matrix_soft_context_d1_design_zh.md`](../docs/research/persistent_context_v2_matrix_soft_context_d1_design_zh.md)。

结果满足继续研究 soft policy 的最低科学理由：中间 α 在三个旧 split 上都降低 harm；`α=0.75` 保留并略高于聚合 mean benefit，`α=0.5` 以约 17.2% 的 mean-benefit 损失换来 harm 减半。端点 identity、480 treatment 完整性和独立 audit 全部通过。完整结果见 [`../docs/research/persistent_context_v2_matrix_soft_context_d1_results_zh.md`](../docs/research/persistent_context_v2_matrix_soft_context_d1_results_zh.md)。

### 4.4 已完成：D2 低容量 soft-policy feasibility

继续复用旧 96 条但保持 exploration 语义。固定一个低容量模型，预测五个 α 的闭环 cost 并在网格内选择强度；使用三次 train/dev/test rotation，比较固定 `α=0.5`、固定 `α=0.75`、full context 和 factor-only hard gate。不得同时尝试多个模型族后择优。

D2 的 learned soft policy 相对 fixed `α=0.75` 三轮 mean 方向一致，但没有降低 harm。更关键的是 F2 相对 F0 不稳定，所以 task-interaction 复杂度没有得到授权。完整结果见 [`../docs/research/persistent_context_v2_matrix_soft_policy_d2_results_zh.md`](../docs/research/persistent_context_v2_matrix_soft_policy_d2_results_zh.md)。

### 4.5 已完成：作者 T 数据科学收缩后的 F0 soft-policy formal

主 learned 方法收缩为 6 维 factor-only 剂量模型，主对照为 fixed `α=0.75`，并保留 population、fixed `α=0.5` 和 full context。F2 task features 未进入 formal decision。

远端发布物只包含 seed-42 的 1000 个 T-shape segments，没有原始 2705 条轨迹或生成脚本，不能声称可生成新分布资产。结构化审计发现 701 个 segment 满足 nominal step-10 block displacement≥10；排除所有已经进入 matrix outcome 的 288 个后，只剩 413 个合格未暴露 segment，原建议 `128/64/128` 所需的 640 个互异片段不可行。

实际设计收缩为 train/dev/formal=`64/32/96` sequences，共使用 384 个互异 E1/E2 segments，8 个 factor 分别每 split `8/4/12` 条，保留 29 个 eligible reserve。选择只依据 eligibility、matrix exposure 和冻结随机 seed，不读取 rotation/dead-zone/CoG 或任何 matrix outcome。该资产只能称“同一发布池中对 matrix treatment 未暴露的新 split”，不能称新轨迹分布。Train/dev/formal 均完整执行五个 α；formal 在读取 E2 outcome 前决策并先执行所选分支。

正式结果：F0 相对 population 的改善为 `13.2141%`，sequence bootstrap CI `[0.2124,0.3963]`；主 estimand `fixed .75-F0=+0.01686`，CI `[-0.00865,+0.04122]`。F0 与 fixed `.75` 的 harm 均为 `21.875%`。F0 选择 `.5/.75/1` 分别 `12/24/60` 次，没有选择 `0/.25`，且同 factor 内选择相同，说明它仍是 factor-level 剂量表。端到端独立审计所有数值重放误差为 0。完整结果见 [`../docs/research/persistent_context_v2_matrix_f0_soft_policy_formal_results_zh.md`](../docs/research/persistent_context_v2_matrix_f0_soft_policy_formal_results_zh.md)。

### 4.6 已完成：作者其他 shape 的跨任务物理 context 复验

只读审计已完成。六个非 T 作者池各 1000 segments，step-10 displacement eligibility 分别为 `I 818 / L 694 / + 804 / small_tee 662 / square 762 / Z 734`，合计 4474。现有 28 个记录了 data path 的 manifest 全部只绑定 `val_T`。跨池只有 1 个完全相同的 state+action segment hash（`I:578` 与 `small_tee:519`）；各池还有 4–8 个内部重复，正式抽样全部排除。虽然 1678 个 `ep_idx:offset` 在不同 shape 间重名，但这不等于轨迹内容相同；为保守起见，正式选择仍要求 segment hash 与 `ep_idx:offset` 全局唯一。审计产物 SHA256 为 `5702af0277ffa772dc25fbcc224d8af976dff26a0bfc77bd375d85d8199298ab`。

正式合同无需重新训练，科学收缩为 96 条 formal sequences，另用 6 条互异 smoke sequences且未回流 formal。六个有向 shape pair 固定为 `I→L、L→+、+→small_tee、small_tee→square、square→Z、Z→I`；每 pair 16 条、每个 rotation×gain factor 在每 pair 2 条，因此每 factor 共 12 条。E1 与 E2 shape 不同，所有 formal E1/E2 共 192 个 author segments 全局互异。

每条 formal sequence 在同一个 E1 scene 上生成两份等预算 history：一份 E1/E2 rotation×gain 持续，另一份 E1 factor 按冻结 derangement 改变而 E2 truth 不变。主 estimand `population-correct .75=+0.24001`，CI `[0.13935,0.34784]`；persistence-specific `no-persistence .75-correct .75=+0.24542`，CI `[0.12355,0.36956]`。No-persistence history 相对 population 平均为 `-0.2249%`，harm `59.375%`，所以收益明确依赖 factor 持续。External T-F0 相对 fixed `.75` 均值优势 `+0.02733`、CI `[0.00167,0.05384]`，但 harm 从 `28.125%` 增至 `33.333%`。独立审计所有重算误差为 0。完整结果见 [`../docs/research/persistent_context_v2_cross_shape_matrix_history_formal_results_zh.md`](../docs/research/persistent_context_v2_cross_shape_matrix_history_formal_results_zh.md)。

### 4.7 已完成：跨 shape 失败归因 D3

当前 96 条降级为 outcome-exposed exploration。D3 只读补算了 E2 前可获得的 scene geometry、population/context candidate action 差异和交叉 rollout disagreement，分别预测 fixed `.75` 与 external F0 的连续 benefit；固定使用 leave-one-shape-pair-out、leave-one-factor-out 和低容量 ridge。

D3 未找到能转化为行为优势的稳定风险信号。Fixed `.75` 的最佳 veto 仅少 1 条 harm，却损失 mean delta `0.0091`；external F0 的最佳 veto 同样少 1 条 harm，但 mean 损失 `0.0224`，CI `[-0.0451,-0.0049]`。所以不启动 harm-aware formal、不在当前 96 条调阈值，保留 fixed `.75` 为默认并转向 P2 delay。完整结果见 [`../docs/research/persistent_context_v2_cross_shape_harm_d3_results_zh.md`](../docs/research/persistent_context_v2_cross_shape_harm_d3_results_zh.md)。

### 4.8 已完成：D4 完整预测轨迹分歧

D3 只保存了四个预测目标损失，没有保存模型逐步预测的完整 latent。D4 在任何 E2 真实执行前固定生成两套命令，并保存四条完整预测轨迹：默认模型和上下文模型分别评价 population、context 命令。

主风险分数固定为“context 命令在默认模型与上下文模型之间的全轨迹 RMS 分歧”，方向固定为分数越大、harm 风险越高。真实标签仍是 sequence-level `pose_auc10(population)-pose_auc10(context .75)`。主报告为 harm ROC AUC、sequence bootstrap 区间、与负收益的相关性和风险四分位；低容量 ridge 只作次要机制分析。

数据继续使用作者 seed-42 的六个非 T shape 池。新 selection 排除既有跨 shape smoke、formal、reserve 的全部 segment hash 和 `ep_idx:offset`，使用 6 条 smoke 和 `6 shape pairs × 8 factors × 2 replicates=96` 条 development sequence。抽样不读取新 E2 outcome。

结果中 fixed `.75` 平均改善 `11.90%`，24/96 条受损。预注册主风险分数 AUC 为 `0.3906`，区间 `[0.2664,0.5223]`，且与负收益相关为负；最高分四分位的平均收益反而最大。低容量 ridge 在 96 条中选择 context 95 次，没有降低受损率。原始数据和统计独立审计均通过，重算最大误差为 0。因此 D4 没有支持 rollout-disagreement risk veto，也不授权在同一数据上翻转方向或调阈值。完整结果见 [`../docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_results_zh.md`](../docs/research/persistent_context_v2_matrix_rollout_disagreement_d4_results_zh.md)。

### 4.9 放弃当前版本：task-interaction hard-gate V2

#### 原拟特征

所有特征必须在 E2 执行动作前计算。建议只保留一个小型、预定义的层级特征集：

1. `F0 factor-only`：完整复用 V1 的 `[1,g,r,g²,gr,r²]`；
2. `F1 geometry/action`：在 F0 上加入 agent→block 距离、block→goal 距离、两向量夹角、block-goal 角度误差，以及两套 command 的 RMS 差、首动作夹角、动作幅值比、时间变化量差；
3. `F2 model-interaction`：在 F1 上加入固定 action sequence 在 population/posterior context 下交叉 rollout 得到的 latent goal-cost 差和 context sensitivity。

不能读取 true factor、真实 E2 cost/state/contact/success、segment/sequence ID、seed 或 best-of-two 标签。Planner `best_loss` 不单独作为 gate；如 D0 发现它有增量信息，只能作为预先冻结的辅助 ablation，不能替换主特征集。

Rollout 日志至少保存四个反事实量：`J(prior,a_prior)`、`J(prior,a_context)`、`J(context,a_prior)`、`J(context,a_context)`，以及计算代码/hash。只读重放不得改变模型参数、running stats、optimizer、replay 或 RNG。

#### 原拟模型和选择规则

- 主模型固定为标准化 ridge value regression，标签仍是连续的 `population_pose_auc10 - context_pose_auc10`；不并行尝试树、MLP、Transformer 和多个 loss 后择优。
- 候选正则强度在合同中预先固定；train 拟合、dev 选 alpha，最终用 train+dev refit，formal 只评价一次。
- 决策规则固定为 predicted delta `> 0` 时选择 context；不在 formal 上调阈值。
- `F0` 与 `F2` 使用相同模型族、alpha 选择程序和数据，确保比较只归因于 task-interaction 特征。`F1` 作为机制消融。

#### 新数据与拆分（暂不启动）

现有 `plan_targets.pkl[500:1000]` 已被多轮 matrix 设计大量占用，不应继续从剩余少量片段拼接新的“独立” formal。优先生成并冻结一份新的 PushObj-T target 数据资产。

建议规模：

| split | sequences | 每个 factor | 用途 |
|---|---:|---:|---|
| train | 128 | 16 | 拟合 ridge |
| dev | 64 | 8 | 只选 alpha、检查预注册数值稳定性 |
| formal | 128 | 16 | 一次性评价 |

每条 sequence 含 E1 evidence 和 E2 cold-start；8 个 rotation×gain factor 平衡。共需 640 个互不重复 target segments。数据生成先固定 eligibility（例如 nominal step-10 block displacement）、再抽样；不能看 matrix paired outcome 后筛 segment。Train 完成后才能读取 train 标签，dev 选择完成并写入锁定模型 hash 后，才启动 formal collection。

额外报告一个 held-out-factor 插值/轻度外推 split（例如训练不逐点覆盖的 rotation/gain），但它是独立泛化端点，不能替换主 formal，也不能在看到主 formal 后才决定 factor。

#### Formal 对照与端点（暂不启动）

每条 formal sequence 在完全相同 factor、初态、目标、env seed、CEM seed、checkpoint 和预算下比较：

- population；
- always-context；
- 原 V1 frozen factor-only gate（外部迁移诊断）；
- 在新 train/dev 上 refit 的 `F0`；
- `F1`；
- 主方法 `F2`；
- selection-rate matched random 与 inverted-F2；
- best-of-two 仅作不可部署 ceiling。

主 estimand：sequence-level `pose_auc10(F0-refit) - pose_auc10(F2)`。
关键次要 estimand：F2 相对 population/always-context 的均值差、harm fraction 差、context 选择率、best-of-two opportunity recovery、precision/recall、factor 与 geometry 子组异质性。Deadline success、prediction error 和 CEM proxy 只作辅助。

采用按 factor 分层的 sequence bootstrap，同时报告不分层 bootstrap 作为敏感性分析。报告完整区间和 unit deltas，不设置自动 GO/NO-GO 效果门。

#### Prospective 语义检查

主统计可以继续保存 population/context 两个潜在 outcome 以得到严格配对，但必须增加一批单路径 replay：gate 先作决定，只执行被选 plan；其结果应与 paired artifact 中对应分支在允许误差内一致。若不一致，只能称 shadow evaluation，不能声称在线 gate 已验证。

#### 资源估计

V1 的 96 sequences 在 3 张 L40 上约 58 秒完成，单 sequence 约 `1.68 s`，峰值 PyTorch allocated 约 `2.8 GiB` 量级。V2 增加交叉 rollout 和 320 sequences 后，预计纯收集仍是分钟级；正式预算以 smoke 的实测值线性外推，并只用 1–2 张空闲 GPU。新 target 生成、审计和代码工作预计比 GPU 运行本身更耗时。启动前记录 `nvidia-smi`，输出目录使用新 contract ID，绝不覆盖 V1。

## 5. P2：Delay 非特权 estimator 与 persistence-specific 行为

P1 后优先补齐 delay 的证据链，因为当前 delay gate 读取 true delay。

### 5.1 最小 estimator

- 候选 delay 固定为合同中的离散支持；
- 只用过去 episode 的 command 与 agent proprio transition，按候选 lag 对齐计算 likelihood/MAP posterior；
- population prior 只由 train split 拟合；
- 输出 posterior probability、entropy、evidence count 和 change detector 状态，不写入 base model 权重。

### 5.2 必需条件与策略

- persistent 与边际匹配的 no-persistence；
- population/current-only、E1 true-factor history ceiling、非特权 correct-history estimator、shuffled、wrong-sequence、true-current-factor oracle；
- E1 所有非特权策略逐 action/state 一致；E2 在读取当前 transition 前评价；
- 统计单位为 sequence，主效应同时报告 persistent correct-history、no-persistence 和 DiD。

Dev 已用于固定可辨识 transition、likelihood noise 和 posterior 使用方式。原先“每个生成条件至少 64 sequences”的建议因作者池审计不可执行，修订为两个条件共享场景的 32 paired sequences、4 个 delay 各 8 条；统计精度、配对方差和有限池约束必须在 formal 合同中书面报告，不根据 dev 均值方向筛选。

### 5.3 与 gate 的关系

先单独报告 estimator 的 factor accuracy和闭环行为；随后才把 estimated delay/uncertainty 接入与 P1 同构的 gate。Estimator 变准但行为不改善时，结论停在“可辨识但无闭环收益”，不能用 accuracy 掩盖行为结果。

### 5.4 已完成：development smoke

离散 posterior 只用 E1 command/proprio，在旧 Stage 0 32 条和 4 条新作者片段上分别 `32/32`、`4/4` MAP 正确。4-sequence smoke 中 persistent correct-MAP 改善 `7.69%`，no-persistence old-history MAP 也改善 `7.83%`，DiD cost 仅 `+0.01497`；这不是 formal，说明下一步必须用足够 sequence 区分 persistence 与错误 context 的命令整形。

数据审计排除所有既有 raw `segment_index` 后只剩 114 个合格作者 T 片段，smoke 使用 8 个后剩 106 个。因此 formal 建议科学收缩为 32 sequences × 2 episodes、两个 condition 共享 64 个场景片段，并保留 16 个 reserve。该收缩由有限作者池决定，不依据 smoke 方向。正式 wrong donor 必须避开 E1 与两种 E2 current factor；开发合同的 `(s+1)` donor 在 no-persistence 中与 current factor 相撞，只保留为 smoke 记录。

完整结果见 [`../docs/research/persistent_context_v2_pushobj_delay_history_stage1_dev_results_zh.md`](../docs/research/persistent_context_v2_pushobj_delay_history_stage1_dev_results_zh.md)。

### 5.5 已完成：32-sequence formal

E1 estimator `32/32` MAP 正确。Persistent correct-MAP 相对 prior 改善 `0.283%`，CI `[-0.10295,+0.11226]`；no-persistence old-MAP 改善 `0.315%`，CI `[-0.10980,+0.12217]`；DiD 为 `-0.00083`，CI `[-0.02355,+0.02095]`。所以 system identification 成立，但 persistence-specific 闭环价值未建立。

预定义 high-delay gate 在 persistent/no-persistence 分别改善 `4.40%/4.33%`，两者 DiD 近 0。它避免了 `d=0/1` exact context 的系统性伤害，但更像 planner regularization/command shaping，不是跨 episode 物理记忆证据。Wrong donor 已修复为两个 condition 都无 current-factor collision；独立审计所有行为与统计重算误差为 0。完整结果见 [`../docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_results_zh.md`](../docs/research/persistent_context_v2_pushobj_delay_history_stage1_formal_results_zh.md)。

## 6. P3：CoG Markov/contact-state predictor V3

CoG 的父命题“真实 CoG 有行为价值”已成立；失败点是 learned predictor。V3 不再只换时序骨干，而先检验状态充分性。

### 6.1 数据表示审计

新 rollout 必须持久化并审计：agent position/velocity、block position、angle、linear velocity、angular velocity、contact count、contact point/normal、impulse（若引擎可得）、动作和 CoG。若某项无法由 simulator 稳定读取，要在合同中明确缺失，不得用名称暗示已记录。

先做 Markov sufficiency 诊断：给定 `(state_t, action_t, CoG)` 预测一步变化，比较 contact/non-contact、pre/post-contact 与 CoG 分组残差。只有确认新增字段能解释现有系统性误差后才进入 closed-loop formal。

### 6.2 受控模型比较

- V1 flattened FiLM 作为冻结 baseline；
- 一个固定的 enriched-state Markov residual model；
- 如需时序，仅增加一个与 Markov 模型参数量近似的 causal model；
- 所有模型保留 zero-context identity，训练数据、步数、checkpoint 选择和 formal 场景配对一致；
- true-CoG、population 和 physics oracle 同场比较。

主端点仍是 held-out formal `pose_auc10` 的闭环 paired delta；one-step/trajectory MSE 按 contact strata 报告但不替代行为。V3 若仍只改善 prediction loss，继续 history encoder 没有证据基础。

### 6.3 P3a 已完成：控制边界 Markov/contact 表示审计

24 个 train segment、16 个 eval segment 的开发诊断通过独立复算。旧状态补 block velocity/angular velocity 后，ridge error 从 `0.56422` 降到 `0.54495`，逐 segment delta CI 跨 0；再补 nominal contact 摘要降到 `0.44623`，R1-R2 delta `+0.09873`、CI `[+0.04487,+0.15216]`、14/16 正向。但冻结 v1 本身为 `0.28831`；v1+Markov 与 v1+Markov/contact correction 分别变成 `0.30213/0.32110`，多数 segment 退化。因此新增字段能描述难 transition，但当前粗粒度表示不足以解释或修正 v1，不进入 enriched neural formal。

事后诊断显示 nominal/true 首次 contact step 完全相同，只有 `2.23%` 控制步的 contact/no-contact 事件不一致；主要差异是 impulse，轨迹级 impulse 差与 v1 error Spearman `0.387`。完整结果见 [`../docs/research/persistent_context_v2_pushobj_cog_markov_contact_audit_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_markov_contact_audit_results_zh.md)。

### 6.4 P3b 已完成：100 Hz event-response 审计

P3b 使用相同 24/16 开发拆分，记录 100 Hz pre/post state 与 post-solve contact。Train/eval 分别包含 6,234/3,558 个 nominal event，所有统计仍以 segment 为单位。C10 与完整 S100 均为 29 维，固定 ridge 在 train CV 锁定后才生成 eval。

主比较 `C10-S100=-0.00728`、CI `[-0.02137,+0.00603]`，说明 100 Hz 表示没有超过 10 Hz aggregate。事件 impulse 相对 geometry-only 改善 `+0.02913`、14/16 segment 正向，但 S100 full 仍差于 zero response `0.02524`；privileged true-contact ridge 也更差。Runner 与独立审计均有效。P3b 不支持 V3、CoG history 或新 CoG closed-loop formal。完整结果见 [`../docs/research/persistent_context_v2_pushobj_cog_event_response_audit_results_zh.md`](../docs/research/persistent_context_v2_pushobj_cog_event_response_audit_results_zh.md)。

## 7. P4：显式 context 与 episode-local adaptation

仅在 P1/P2 给出可审计选择策略后，在 rotation×gain 上做最小组合实验：

- base/population；
- context only；
- episode-local AdaJEPA only（每 episode reset）；
- context + episode-local AdaJEPA；
- true-context upper bound；
- unconditional carry 只作历史/同协议负对照，不再作为主方法。

预先分开 episode-entry、adaptation 前 K 步和后期窗口。主问题是组合是否保留 entry context 收益，而不是完整 episode 均值是否略升。需要逐状态审计 `q(z_seq)` 跨 episode 保留、TTT optimizer/replay 每 episode 重置，并测试错误 context/change point 后的恢复。

## 8. P5：跨 shape 与视觉模型

最后才进入 T/L/Z：每个 shape 覆盖多个 factor、start、goal 和 trajectory；比较 shared physics context、shape-specific context 和 factorized `z_shape + z_physics`。Formal 至少分 held-out trajectory、held-out factor 组合和 held-out shape-factor 组合。只有低维接口和 gate 在 state-based PushObj 中成立后，才迁移到完整视觉 AdaJEPA predictor。

Deformable 环境仍受 Prompt 第 8 节的环境重构要求约束，不与 T/L/Z 或视觉迁移并行扩张。

## 9. 每项实验的统一交付

每项 formal 前：冻结合同、design JSON、数据/checkpoint/source hash、唯一主 estimand、对照、bootstrap、无效执行与有限修复规则。
执行中：保存 command、resolved manifest、每步 raw、资源日志、heartbeat、退出码和失败产物。
执行后：由不 import runner 核心实现的独立脚本从 raw 复算；报告机器事实、范围内推断、最强替代解释、异质性、负对照和未测试问题。

任何阶段都不以固定百分比、同向比例或 CI 是否跨 0 自动裁决后续投资；但也不得把总体负向、区间宽或负对照同样获益写成“已验证”。

## 10. 当前建议

P0、P1-D0/D1/D2/V2/XShape/D3/D4、P2 delay formal 与 P3a/P3b 已完成。跨 shape matrix correct history 的父命题成立；delay persistence-specific 闭环价值未建立；CoG 的 10 Hz 拼接和 100 Hz event ridge 均未形成可部署 predictor；D4 完整轨迹分歧不能作为风险 veto。若用户决定继续，最小候选是 E2 在线贝叶斯修正：把 E1 后验作为初始先验，再用前 1–2 步真实反馈更新，并同时测试 factor 持续和改变。Episode-local TTT 和视觉扩张继续暂停。
