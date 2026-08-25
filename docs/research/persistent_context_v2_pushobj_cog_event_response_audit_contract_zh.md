# Persistent Context V2：PushObj CoG P3b 事件级接触响应审计合同

日期：2026-08-24  
合同 ID：`persistent-context-v2-pushobj-cog-event-response-audit-v1`  
性质：开发集表示审计，不产生新的闭环 formal 结论

## 1. 问题与假设

P3a 显示 nominal 与 true-CoG rollout 的首次接触控制步一致，主要差异集中在接触冲量。P3b 检查 10 Hz 控制步聚合是否丢失了预测 CoG 接触响应所需的信息。

主假设：在相同 nominal agent-block contact event 上，使用 100 Hz 接触前状态、接触几何和 nominal impulse 的低容量模型，比使用 10 Hz 控制边界状态和整步聚合 contact summary 的同类模型具有更低的 held-out segment 平均响应误差。

唯一主比较：

```text
C10_aggregate error - S100_state_geometry_impulse error
```

正值表示 100 Hz event representation 更好。结果按连续证据报告，不设置固定百分比、方向比例或置信区间过零的自动 GO/NO-GO 门。

## 2. 数据与拆分

- 数据只来自既有 CoG predictor train/dev segment；所有 CoG formal segment 禁止使用。
- train：P3a 的 24 个 train segment。
- eval：P3a 的 16 个 held-out dev segment。它们已经用于 P3a，因此本结果始终标记为 development audit。
- train CoG：`[-30,-15,0,15,30]`。
- eval CoG：`[-25,-10,10,25]`。
- 每个 segment 使用 nominal action 加三档冻结噪声，共 4 个 variant。
- train 和 eval 使用不同 variant RNG stream。
- segment 是独立单位；event、substep、factor 和 variant 不作为独立样本。

## 3. 轨迹与事件定义

- simulator 为 PushT/Pymunk，physics 频率 100 Hz，control 频率 10 Hz。
- 每个 10 步 action rollout 记录 100 个 physics substep。
- 每个 substep 保存 agent/block 的 pre/post state、command、controller target、control/substep index 和四类 post-solve contact 字段。
- nominal 与 true-CoG rollout 按相同 control index 和 substep index 对齐，不使用动态时间规整或 outcome 驱动的重对齐。
- 主事件集合由 nominal rollout 决定：`agent_block point_count > 0` 且同一 substep 的 nominal `block_wall point_count == 0`。
- eval segment 若没有主事件，审计标为 `INVALID`；不得读取 true rollout 后修改 eligibility。

预测目标是同一 substep 的 block generalized-velocity response residual：

```text
[(v_post-v_pre)_true - (v_post-v_pre)_nominal,
 (omega_post-omega_pre)_true - (omega_post-omega_pre)_nominal]
```

三个输出分量使用 train 主事件的非零 CoG target RMS 归一化；每维下限 `1e-6`。归一化参数不得使用 eval target。

## 4. 冻结表示

所有非特权表示都读取真实 E2 执行前可由 nominal rollout 产生的信息，并通过 CoG context 调制。

- `C10_aggregate`：当前控制步起点的 10 维 Markov state、2 维 command、该控制步 17 维 nominal agent-block aggregate contact summary，共 29 维。
- `S100_state`：event pre-state 10 维、command 2 维、controller target 相对 agent 2 维、substep phase 1 维，共 15 维。
- `S100_state_geometry`：`S100_state` 加 8 维 nominal event geometry。
- `S100_state_geometry_impulse`：再加 6 维 nominal event impulse/solver summary，共 29 维。
- `P100_true_contact`：在 `S100_state_geometry_impulse` 后加入同一 substep 的 true contact geometry 和 impulse字段；仅用于 privileged attribution。
- `zero_response`：恒为零的负控制。

低容量模型固定为 ridge，feature map 为 `[c,c²,c*x,c²*x]`，其中 CoG context 在 `c=0` 时使预测严格为零。ridge alpha 只用 train segment 的 4-fold grouped CV 选择；精确平局选择更大的 alpha。每种表示锁定 alpha、标准化参数和系数 hash 后才读取 eval target。

## 5. 指标与统计

每个 event 的误差为三个归一化 response 分量的 RMSE。先在每个 segment 内对 factor、variant 和 event 求均值，再对 16 个 eval segment 求均值。

必须报告：

- 每种表示的 segment 平均 error；
- 主比较的均值、20,000 次 segment bootstrap 95% 区间、正/平/负 segment 数；
- `S100_state → S100_state_geometry → S100_state_geometry_impulse` 的嵌套比较；
- deployable `S100` 与 privileged `P100` 的剩余差距；
- factor 异质性、event 数、nominal/true event mismatch 和输出尺度；
- zero-CoG identity、determinism 和 rollout identity。

## 6. 有效性与修复

以下情况标记 `INVALID`：

- contract/design/source snapshot hash 不一致；
- train/eval 重叠或任何 forbidden formal segment 被读取；
- substep rollout 的 10 Hz boundary state 与原 `rollout_physics` 最大误差超过 `1e-6`；
- 重复 rollout 的 state/contact 不完全一致；
- zero-CoG target 或 model prediction 不是严格零；
- 任一冻结 eval segment 没有 nominal eligible event；
- 模型在读取 eval target 前未锁定；
- 数组非有限、样本数、factor、variant 或 event provenance 不完整；
- 独立审计无法逐项复算。

smoke 若在完整 eval 生成前发现实现问题，可以形成带原因的新 repair 合同和 design。完整 eval target 一旦生成，任何改变 eligibility、特征、target、alpha grid、指标或比较的修改都必须使用新版本，当前批次保留。

不利科学结果仍为 `VALID`，不能以模型容量不足为由改判无效。

## 7. 禁止的后续动作

本合同不授权：

- 训练 V3 neural predictor、Transformer 或 CoG history encoder；
- 运行新的 CEM 闭环 formal；
- 将 true-contact、true state 或 eval outcome 输入可部署模型；
- 将 event 数当作独立样本数；
- 根据 eval 结果修改 event eligibility、输出尺度或模型容量；
- 覆盖 P3a 或本次失败/负结果产物。

只有非特权 100 Hz 表示在 held-out segment/factor 上显示稳定增量，才另行冻结最小 contact-response predictor 的闭环开发合同。
