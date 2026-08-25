# Persistent Context V2：PushObj rotation 跨 episode Procrustes/MLE Stage 1 结果

日期：2026-08-22  
合同：`persistent-context-v2-pushobj-rotation-history-stage1-v1`  
状态：**正式实验完成，独立 raw 审计通过**

## 1. 实验做了什么

在保留的 PushObj formal 片段 `500..999` 上运行 32 条 persistent 与 32 条 no-persistence sequence，每条 4 episodes。formal 旋转为 `[-25,-10,10,25]°`，每个 factor 在每个 episode 边际均衡。

所有策略共享由 `0°` population planner 实际生成的 history evidence。Procrustes/MLE estimator 只读取自己发出的二维命令和 observation 中可见的 agent position/velocity，不读取真实 factor、旋转后的 effective action或接触标签。每个新 episode 在读取当前 transition 前评价，因此 E2–E4 都是 cold-start。

比较：

- `population_prior` / `current_only`；
- `correct_history`；
- `shuffled_history`；
- `wrong_sequence_history`；
- `true_factor_oracle`。

主指标是每条 sequence 的 E2–E4 mean `pose_auc25`，越低越好；32 条 sequence 是统计单位，bootstrap 20,000 次。

## 2. 主结果

### Persistent factor

| policy | later E2–E4 pose AUC25 |
|---|---:|
| current-only / population | 5.376122 |
| correct history | 4.629919 |
| true-factor oracle | 4.629919 |
| shuffled history | 5.659539 |
| wrong-sequence history | 5.619254 |

`current_only - correct_history`：

- mean delta：`+0.746203`；
- 相对改善：`13.8799%`；
- sequence bootstrap 95% CI：`[+0.480323,+1.035029]`；
- 31/32 sequences 改善，1/32 退化，无 ties；
- correct history 回收 true-factor oracle gap：约 `100.0000%`。

true-factor oracle 自身相对 current-only 的改善同为 `13.8799%`，CI `[+0.475983,+1.040628]`，31/32 sequences 正向。

### No-persistence control

factor 每 episode 改变时：

- current-only：`5.499237`；
- correct history：`5.840006`；
- mean delta：`−0.340769`；
- 相对变化：`−6.1967%`，即历史使 cost 增加；
- bootstrap 95% CI：`[−0.520296,−0.161898]`；
- 5/32 sequences 改善，27/32 退化。

### Persistence-specific DiD

```text
DiD = (current-history)_persistent - (current-history)_no_persistence
```

- mean DiD：`+1.086972`；
- bootstrap 95% CI：`[+0.747244,+1.442823]`；
- 28/32 paired sequences 的 DiD 为正。

这说明行为收益不是“多看历史一般都会变好”，而是依赖同一 rotation 在 episode 间持续。

## 3. Factor 与 episode 异质性

### Persistent factor

| factor | current | history | mean delta | 相对改善 | positive sequences |
|---:|---:|---:|---:|---:|---:|
| −25° | 5.340332 | 4.358122 | +0.982210 | `18.3923%` | 8/8 |
| −10° | 5.756138 | 5.436653 | +0.319485 | `5.5503%` | 7/8 |
| +10° | 4.268784 | 4.139133 | +0.129651 | `3.0372%` | 8/8 |
| +25° | 6.139235 | 4.585769 | +1.553465 | `25.3039%` | 8/8 |

所有四个 factor 的均值都正向；收益明显由大旋转角放大，但不只存在于大角度组。

### Later episode

| episode | persistent 相对改善 | persistent delta CI | no-persistence 相对变化 |
|---:|---:|---:|---:|
| E2 | `14.6652%` | `[+0.289100,+1.248631]` | `−4.1219%` |
| E3 | `13.4081%` | `[+0.324491,+1.183492]` | `−7.9811%` |
| E4 | `13.6613%` | `[+0.511181,+1.130795]` | `−6.3423%` |

收益没有随着 episode 推进消失；E4 的逐 episode 同向比例为 29/32。

## 4. Estimator 与因果链

Persistent 条件：

- correct-history angle MAE：`0.00000625°`；
- median absolute error：`0.00000402°`；
- later evaluation 的 zero-accepted-history fraction：`0`；
- 每个 later episode 平均累计 50 条 dynamics-consistent transition；
- history context 在 96/96 later episodes 都改变了 planner command hash。

No-persistence 条件下，过去 factor 与当前 factor 不同：

- correct-history angle MAE：`24.9924°`；
- history 在 96/96 later episodes 改变动作，但总体使行为变差。

因此机器证据闭合了：

```text
过去 episode 的可观察 proprio transition
→ Procrustes 充分统计量
→ rotation estimate
→ world-model rollout context
→ CEM action ranking/命令改变
→ simulator pose AUC 改变
```

## 5. 接触异质性

Persistent evidence 中有接触的 transition 比例均值为 `61.96%`，sequence 范围 `25.33%–96.00%`。以事后中位数 `62.67%` 描述性分组：

- 低/等接触组 18 sequences：mean behavior delta `+0.882637`；
- 高接触组 14 sequences：mean behavior delta `+0.570787`；
- contact fraction 与 sequence delta 的 Pearson correlation：`−0.213`。

两组均正向，但接触越多时平均收益略小。该分组是事后异质性解释，不是裁决门。

## 6. 负对照的结构性限制

Persistent 条件中：

- shuffled history 相对 current-only：`−5.2718%`；
- wrong-sequence history：`−4.5224%`；
- 两者 donor factor 与当前 factor 的匹配比例均为 0，angle MAE 分别为 `25.14°/25.00°`。

但 no-persistence 使用冻结的循环 factor schedule，而 donor 也由固定 sequence offset 生成，二者产生偶然对齐：

- shuffled donor episode 的 factor 与当前 factor 匹配 `1/3`；angle MAE `14.28°`；
- wrong-sequence donor 匹配 `1/2`；angle MAE `10.02°`；
- 因而 no-persistence 的 shuffled/wrong 分别出现约 `3.49%/7.94%` 行为改善。

这两个辅助 control 在 no-persistence 下不是纯无信息 control，不能用来声称“任意错误历史都一定有害”。它不改变两个主处理的含义：correct history 在 persistent 中使用同 factor 历史并改善，在 no-persistence 中使用本 sequence 但已过期的历史并退化；paired DiD 仍直接比较这两个生命周期。不过，若未来需要精确比较多种错误历史，应在独立合同中随机化 factor schedule 与 donor assignment，使二者统计独立。

## 7. 工程与资源审计

- formal evidence：256/256；formal evaluation：256/256；exit code 0；
- E1 population/current/correct/shuffled/wrong identity：通过；
- current-only/population 所有 episode identity：通过；
- factor lifetime、segment 唯一性、donor 不自指/不读未来：通过；
- independent audit：runner summary 与 raw 重算完全一致；所有 failure count 为 0；
- wall time：`839.38 s`（约 13.99 min）；
- GPU：单张 NVIDIA L40（GPU0）；166 个资源 samples；设备总显存峰值 `3676 MiB`；
- 本地相关测试：11/11 通过；远端直接 pure-function tests：4/4 通过；远端环境未安装 pytest，因此没有伪装 pytest 结果。

## 8. 客观结论与下一步

本实验说明：在当前真实 PushObj rotation wrapper 中，一个只保存低维充分统计量的跨 episode estimator 可以从非特权 observation 恢复几乎精确的持续角度，并把 later-episode pose AUC25 改善 `13.88%`；收益在 31/32 sequences 和全部四个 factor 组的均值上为正，并在 factor 不持续时转为负。

因此 direction 1 有明确尝试价值，而且当前简单 Procrustes/MLE 已足够，不需要先开发神经 memory。

同时，这个 factor 在无 observation noise 的已知 PD 动力学中可由很少 transition 几乎精确辨识；当前价值集中在 episode-entry cold start，不能据此声称复杂视觉 context 或长期神经记忆已经解决。

按用户指定顺序，下一项是构造 rotation 的 early-waypoint PushObj 任务，检验同一 history context 是否能改善前几个不可逆关键动作，而不仅是完整 25-step open-loop AUC。

## 9. 产物

- 算法：`research/persistent_context_v2/pushobj_rotation_stage1.py`
- runner：`scripts/run_persistent_context_v2_pushobj_rotation_stage1.py`
- independent audit：`scripts/audit_persistent_context_v2_pushobj_rotation_stage1.py`
- descriptive analysis：`scripts/analyze_persistent_context_v2_pushobj_rotation_stage1.py`
- tests：`tests/test_persistent_context_v2_pushobj_rotation_stage1.py`
- 远端正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_rotation_stage1/`
- runner summary：同目录 `runner_summary.json`
- independent audit：同目录 `independent_audit.json`
- descriptive analysis：同目录 `descriptive_analysis.json`

