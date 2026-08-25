# PushObj 离散 Delay 非特权 History Stage 1 正式合同

状态：**FROZEN BEFORE FORMAL RESULTS**  
合同 ID：`persistent-context-v2-pushobj-discrete-delay-history-stage1-formal-v1`  
日期：2026-08-24（Asia/Shanghai）

## 1. 假设与唯一处理

命题：在真实离散 actuator delay 跨 episode 持续时，只由 E1 command 与 agent proprio transition 得到的非特权 delay posterior/MAP，能否在 E2 第一个 transition 发生前改善 early-waypoint 闭环行为；若 delay 不持续，同一旧 history 的行为效应是否消失或改变。

主处理变量仅为 E2 planner 使用的 delay context。环境真实 delay、checkpoint、作者场景、初态、目标、CEM seed、规划/执行预算和评价顺序在可配对分支间固定。Estimator 不更新 base model 权重、optimizer、replay、running statistics、RNG memory 或任何 TTT 状态。

## 2. 数据科学收缩与独立单位

作者 `val_T/plan_targets.pkl` 共 1000 个 segments。选择前扫描既有 `repro_outputs/**/*.jsonl` 的顶层 `segment_index`，仅用于排除已产生过行为结果的索引；再要求 nominal step-10 block displacement `>=10`。剩余 106 个，seed `1090000` 一次性排列为：8 个 formal-smoke、64 个 formal、16 个 reserve、18 个继续未使用。选择不读取候选上的 context/policy outcome。

Formal 为 32 sequences × E1/E2；64 个 formal segments 全部唯一。Persistent 与 no-persistence 使用相同的 sequence 场景对、E1 evidence、E2 segment、初态、目标和 seed，因此 condition 差异仅为 E2 factor 是否持续。Sequence 是唯一独立统计单位；不得把 episode、action 或 transition 当作独立样本。

原建议每 condition 64 sequences 需要 128 个新片段，作者池不足。32 paired sequences 是有限作者池约束下的科学收缩，不依据 development smoke 的效应方向。Reserve 只用于合同允许的身份/文件损坏修复，不用于补科学负结果。

## 3. Factor、estimator 与生命周期

真实支持为 `[0,1,3,4]`，候选 posterior 支持为 `[0,1,2,3,4]`，population prior 为 2。E1 factor 由 `sequence_id mod 4` 平衡，各 8 条。Persistent E2 保持 E1 factor；no-persistence E2 在四档真实支持上循环平移 `+1`，所以边际分布完全相同且每条改变。

Estimator 与已审计 development 版本冻结一致：从 agent position/velocity 的已知 PD 方程反演生效控制量；对五个 episode-reset、zero-filled FIFO command 候选累计 `σ=0.1` Gaussian likelihood；均匀先验，MAP 并列时取较小 delay。输出 posterior、MAP、entropy、high-delay probability、episode/evidence count 和 change-detector 状态。

Estimator 只能读取 E1 command 和 agent proprio transition。禁止读取 true delay、effective action、contact、block state、goal、planner loss、行为 cost 或 E2 当前 transition。E1 bank 在任何 E2 分支执行前冻结；E2 评价分支不写回 history。

## 4. 策略与负对照

- `population_prior/current_only`：都使用 2 步，必须逐 command/state 相同；
- `correct_history_map`：本 sequence E1 posterior MAP，三个主 estimand 的处理策略；
- `correct_history_high_delay_gate`：仅在 `P(d>=3)>=0.95` 时使用 MAP，否则使用 2；阈值由旧 Stage 0 异质性预定义，只是次要策略；
- `shuffled_history`：固定 seed `1090200` 打乱本 sequence E1 command 的时间顺序，proprio 不变；
- `wrong_sequence_history`：donor 为 `(sequence_id+2) mod N`。因 E1 factor 为 `s mod 4`、no-persistence E2 为 `s+1 mod 4`，该 donor factor 与 persistent/no-persistence 的当前 factor 都不同；
- `true_factor_oracle`：使用 E2 true current delay，只作行为上限。

所有 donor provenance、factor match、shuffle order、posterior 和 E2 current-evidence count 必须落盘。Wrong/shuffled 若因 estimator 偶然输出 current factor，照实报告，不能事后换 donor 或 permutation。

## 5. 端点与统计

唯一行为指标为 E2 `pose_auc10_to_waypoint`，越低越好。三个并列主 estimand 均以 32 条 sequence 配对：

1. persistent `current_only - correct_history_map`；
2. no-persistence `current_only - old correct-history MAP`；
3. 两者逐 sequence delta 的 difference-in-differences。

分别报告 mean、relative improvement（DiD 不作相对比例）、20,000 次 paired sequence bootstrap 95% interval（seed `1090301` 的固定 stream）、positive/tie/negative counts。辅助报告 high-delay gate、shuffled、wrong、true oracle、deadline success、E1 MAP accuracy/entropy、按 E1/E2 delay 分组、donor/current match 和 oracle-gap recovery。不得用 accuracy、局部 factor 正向或固定百分比自动替代主行为证据或裁决后续投资。

## 6. 有效性与复现

Formal 前必须通过 formal 专用 4-sequence 单 GPU smoke，但 smoke outcome 不进入 formal。冻结 design、selection、作者池 audit、contract、checkpoint/data hash 与完整 dirty source snapshot。正式数据在上述对象锁定后读取一次；成功或失败目录不覆盖。

审计至少独立重算：segment hash/选择身份、PD 反演 posterior、donor 与 shuffle、context decision、zero-filled FIFO、`pose_auc10`、三个主 estimand/bootstrap、场景配对、population/current identity、E2 零当前 evidence、world-model 参数不变和资源记录。Runner/audit 所有数值必须有限。

`INVALID` 只用于实现、identity、预算、hash 或审计失败；科学负结果仍有效。唯一自动 repair 仅限可定位工程错误且不得改变 hypothesis、selection、factor、policy、endpoint、seed 或预算。禁止根据 formal 结果调 `σ`、gate threshold、donor、factor、片段，或把当前 formal 重新命名为新 split。

