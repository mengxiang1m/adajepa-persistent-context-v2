# Persistent Context V2：PushObj rotation early-waypoint 实验结果

日期：2026-08-22  
状态：**Stage 0 true-factor oracle 与 Stage 1 history estimator 均完成，独立 raw 审计通过**

## 1. 任务改变

旧 early-contact 候选仍把第 25 步最终姿态当目标，只在前 10 步提前评价。本实验把 nominal 第 10 步物体姿态本身定义为必须按时到达的 waypoint，并把 world-model/CEM horizon 固定为 2×5、执行 deadline 固定为 10 actions。第 10 步后没有补救预算。

所有选中片段的 nominal block 在前 10 步位移均至少 10 像素。Stage 0 使用 32 个未在旧 rotation A/B 中观察行为的开发片段；Stage 1 使用 formal 池中的 128 个唯一片段。

## 2. Stage 0：true-factor oracle

32 个 paired scenario，factors `[-22.5,-7.5,7.5,22.5]°`。

| 指标 | population prior | true factor | 变化 |
|---|---:|---:|---:|
| pose AUC10 | 2.474427 | 2.316920 | 改善 `6.3654%` |
| step-10 waypoint success | 71.875% | 93.750% | `+21.875` 个百分点 |

- mean paired delta：`+0.157507`；
- bootstrap 95% CI：`[+0.048471,+0.268901]`；
- 21/32 pairs 正向，11/32 负向；
- 32/32 计划命令改变；
- nominal waypoint block displacement 均值 `62.73` 像素。

按 factor：

| factor | 相对改善 | positive pairs |
|---:|---:|---:|
| −22.5° | `8.6011%` | 6/8 |
| −7.5° | `7.7225%` | 6/8 |
| +7.5° | `0.4747%` | 4/8 |
| +22.5° | `9.1370%` | 5/8 |

四组均值正向，但 `+7.5°` 的行为动态范围很小。

## 3. Stage 1：跨 episode Procrustes/MLE

32 条 persistent 与 32 条 no-persistence sequence，每条 4 episodes。所有策略共享 population-prior evidence；新 episode 在获得当前 transition 前评价。

### Persistent 主结果

| policy | later E2–E4 pose AUC10 | deadline success |
|---|---:|---:|
| current-only / population | 2.758192 | 59.375% |
| correct history | 2.425518 | 93.750% |
| true-factor oracle | 2.425518 | 93.750% |
| shuffled history | 2.947187 | 43.750% |
| wrong-sequence history | 2.982078 | 50.000% |

`current_only-correct_history`：

- mean delta `+0.332675`；
- 相对改善 `12.0613%`；
- bootstrap 95% CI `[+0.204502,+0.460335]`；
- 25/32 sequences 正向，7/32 负向；
- deadline success 提高 `34.375` 个百分点；
- 回收 true-factor oracle gap `100%`。

### Persistence-specific control

No-persistence 中：

- current-only `2.746055`；correct history `2.886534`；
- history 相对变化 `−5.1157%`，即 pose cost 增加；
- mean delta `−0.140479`，CI `[−0.221081,−0.053031]`；
- 8/32 sequences 正向，24/32 负向；
- current-only 与 correct-history 的 binary deadline success 均为 `52.083%`，说明 binary 指标在该负对照中比连续 pose AUC 粗糙。

DiD：

- mean `+0.473154`；
- 95% CI `[+0.335278,+0.615494]`；
- 29/32 paired sequences 为正。

### 负对照

Persistent 中：

- shuffled history 使 pose cost 退化 `6.8521%`；
- wrong-sequence history 退化 `8.1171%`；
- 两者 donor factor 与当前 factor 匹配比例均为 0。

No-persistence 的 factor schedule 与 donor 由独立随机源生成：

- shuffled donor factor match `27.60%`；
- wrong donor match `24.48%`；
- 接近四因子下约 25% 的随机匹配，而不是上一 Stage 1 固定 offset 产生的结构对齐。

### Estimator 与 factor 异质性

- persistent correct-history angle MAE `0.00000846°`；
- no-persistence correct-history angle MAE `22.8384°`；
- correct history 的行为均值在四个 persistent factor 上都正向：
  - `−25°`：`13.9597%`；
  - `−10°`：`5.2198%`；
  - `+10°`：`3.2397%`；
  - `+25°`：`24.4514%`。
- 小角度 `±10°` 的单 factor CI 包含 0；总体改善主要由大角度放大，但不是只由一个 factor 产生。

按 episode 的 persistent 改善：E2 `13.92%`、E3 `10.98%`、E4 `11.38%`，没有随历史累积消失。

## 4. 工程审计与资源

### Stage 0

- 32/32 pairs，exit 0；independent audit passed；所有 failure count 0；
- wall `42.95 s`；GPU0 L40；峰值 `3674 MiB`。

### Stage 1

- 256/256 evidence、256/256 evaluation；exit 0；
- independent audit 的 raw-summary exact match、factor、donor、estimator、metric、hash 与 identity 全部通过；
- wall `809.85 s`；160 个资源 samples；GPU0 L40；峰值 `3674 MiB`。

## 5. 结论

这组实验把原来的“完整 25-step AUC 有益”推进成了更强的行为事实：跨 episode rotation history 能显著改善新 episode 的 10-action waypoint deadline，同时大幅提高按时成功率；factor 每 episode 改变时，旧历史在连续 pose 指标上反而有害。

因此 early-waypoint 方向可行，且简单充分统计量已经回收全部 oracle gap。当前不需要为 rotation 增加神经 context encoder。

局限是 rotation 在已知、无 observation noise 的 PD 动力学中几乎可以由一个 episode 精确辨识；结论是明确的 episode-entry calibration value，不等于复杂未知物理因素已解决。

按用户指定顺序，下一实验进入 PushObj action dead zone：先检验 true-factor oracle 是否改善 early-waypoint 行为，再决定接入 history estimator。

## 6. 产物

- Stage 0 core：`research/persistent_context_v2/pushobj_rotation_early_waypoint_stage0.py`
- Stage 0 runner/audit：`scripts/run_persistent_context_v2_pushobj_rotation_early_waypoint_stage0.py`、`scripts/audit_persistent_context_v2_pushobj_rotation_early_waypoint_stage0.py`
- Stage 0 输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage0/`
- Stage 1 core：`research/persistent_context_v2/pushobj_rotation_early_waypoint_stage1.py`
- Stage 1 runner/audit：`scripts/run_persistent_context_v2_pushobj_rotation_early_waypoint_stage1.py`、`scripts/audit_persistent_context_v2_pushobj_rotation_early_waypoint_stage1.py`
- Stage 1 输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_rotation_early_waypoint_stage1/`

