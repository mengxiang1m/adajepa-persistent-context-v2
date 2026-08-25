# PushObj Delay 非特权 History Stage 1 开发 Smoke 结果

日期：2026-08-24  
合同 ID：`persistent-context-v2-pushobj-discrete-delay-history-stage1-dev-v1`  
证据等级：**development smoke，不是 formal**

## 1. 结论先行

非特权 delay estimator 的实现链已经打通。它只用过去 E1 的 command 与 agent position/velocity，在旧 Stage 0 的 32 条开发轨迹上 MAP `32/32` 正确，在 4 条全新作者片段 smoke 上也为 `4/4`；E2 前没有读取任何当前 transition，world model 参数不变，独立审计有效。

闭环 smoke 不能证明 persistence-specific value。4 条 persistent sequence 中，always-MAP 相对 2-step prior 平均改善 `7.69%`，但 no-persistence 也改善 `7.83%`，DiD cost 只有 `+0.01497`。这与旧 Stage 0 的观察一致：错误 delay context 有时会成为有利的命令整形，不能用 factor accuracy 代替行为因果证据。

## 2. 固定方法

- 候选 delay：`[0,1,2,3,4]`，均匀先验；
- 从已知 agent PD 方程反演每步实际生效控制量；
- 对 zero-filled FIFO command 的五个候选移位累计 Gaussian residual likelihood，`σ=0.1`；
- 输出 posterior、MAP、entropy、evidence count 和 change-detector 状态；
- 禁止输入 true delay、effective action、contact、block state、goal、planner loss、行为 cost 和 E2 transition。

4 条 sequence 的 E1 factor 为 `0/1/3/4`。Persistent 在 E2 保持，no-persistence 循环平移一档；两个条件共享 E1 evidence 和匹配的 E2 scene/seed/budget。8 个 smoke segment 均来自作者 `val_T`，且未在既有 raw 结果中出现。

## 3. Smoke 结果

| 条件/策略 | current mean | treatment mean | mean delta | 相对改善 | 正/平/负 |
|---|---:|---:|---:|---:|---:|
| persistent / correct MAP | 2.9637 | 2.7358 | +0.2279 | +7.69% | 2/0/2 |
| persistent / high-delay gate | 2.9637 | 2.7025 | +0.2612 | +8.81% | 2/2/0 |
| no-persistence / old-history MAP | 2.7192 | 2.5062 | +0.2129 | +7.83% | 2/0/2 |
| no-persistence / old-history gate | 2.7192 | 2.4729 | +0.2463 | +9.06% | 2/2/0 |

Persistent MAP 的逐 delay delta 为 `[-0.0712,-0.0619,+0.5273,+0.5175]`：低 delay 受损，高 delay 获益。Estimator 与 true-factor oracle 在 persistent 中逐 context 一致，所以两者行为完全相同。这说明 estimator 在 smoke 上回收了 oracle，但不说明历史持续性带来额外收益；no-persistence 的相似改善正是必须保留的反证。

样本数只有 4，不计算或解释置信区间，不据此调 posterior noise、gate threshold、factor 或片段。

## 4. 审计与工程记录

- 11 项本地相关测试通过；远端环境没有安装 `pytest`，远端 `py_compile` 通过；
- 成功 smoke：4 条 E1、8 条 E2，wall time `17.59 s`；仅用物理 GPU 0（NVIDIA L40）；现场约 3.7 GiB，PyTorch peak allocated `2.849 GiB`、reserved `3.305 GiB`；
- population/current action-state identity、E2 零当前 evidence、condition scene pairing、数值有限和 model-state identity 全通过；
- 独立审计重算 posterior 最大误差 `6.94e-18`，FIFO、metric、summary 最大误差均为 0；
- source snapshot SHA256：`628c8f6218e5d0b0b3443eaabee56dbdcf91ad48628783db06d09f2082d105e5`；E1 raw：`6208858ec202db49913c926febbfe5de755631ad31582834e934098413385474`；E2 raw：`248918bb5a9ccbed9088b82386e28495859cd21b5a87dd6bf8bc86013667c608`；audit：`e3dcaeff06dba2c8b0ccf7eaf35da4fce5ff6651af35d782c33070d486a3c08d`。

启动阶段保留两条无行为结果的工程失败：直接执行 runner 时缺少 `PYTHONPATH=.`；第一次 smoke 的后台监控 shell 丢失工作目录。成功结果位于新目录 `repro_outputs/persistent_context_v2_pushobj_delay_history_stage1_dev_smoke_repair1/`，未覆盖失败目录。

## 5. Smoke 暴露的合同修正

`wrong_sequence_history=(s+1) mod 4` 在 persistent 中确实来自错误 factor，但 no-persistence 的 E2 factor 也恰好平移 `+1`，因此该 donor 在 no-persistence 中撞上 true-current factor。数值审计仍然有效，但这个分支不能在 no-persistence 中解释为错误-factor负对照。

正式合同必须：

- 把 wrong donor 改为同时避开 E1 factor 和两种 E2 current factor 的预冻结映射，并显式报告 donor/current factor match；
- 保留 `correct-history MAP` 为主处理、high-delay gate 为预定义次要处理；
- 使用作者数据科学收缩到 32 sequences × 2 episodes。当前排除所有已有 raw 后共有 114 个合格片段，smoke 使用 8 个，剩余 106 个足以冻结 64 个 formal + 16 个 reserve；
- 两个 condition 共享这 64 个场景片段，sequence 是独立统计单位；样本量收缩原因是有限作者池，不是 smoke 均值方向。

后续 32-sequence formal 已完成：可辨识性保持 `32/32`，但 persistent/no-persistence MAP 改善分别仅 `0.28%/0.32%`，DiD 近 0。正式结论见 [`persistent_context_v2_pushobj_delay_history_stage1_formal_results_zh.md`](./persistent_context_v2_pushobj_delay_history_stage1_formal_results_zh.md)；本文件继续只作为 development smoke 记录。
