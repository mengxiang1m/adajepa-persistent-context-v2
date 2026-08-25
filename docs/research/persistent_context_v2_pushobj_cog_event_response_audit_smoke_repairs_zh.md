# PushObj CoG P3b smoke 修复记录

日期：2026-08-24  
主合同：`persistent-context-v2-pushobj-cog-event-response-audit-v1`

## Smoke 1：包装器输出目录冲突

- 目录：`persistent_context_v2_pushobj_cog_event_response_audit_smoke_v1`
- 现象：资源监控文件先写入 runner 输出目录，append-only 防覆盖检查立即拒绝启动。
- 数据状态：未生成 train 或 eval 数据。
- 修复：资源监控和 console 先写入同级文件，runner 完成后复制到产物目录。实验代码、合同和 design 未修改。

## Smoke Repair1：JSON bool 类型

- 目录：`persistent_context_v2_pushobj_cog_event_response_audit_smoke_v1_repair1`
- 范围：2 个 train segment、2 个 eval segment；仅为 smoke 子集。
- 现象：100 Hz 数据生成、ridge 选择和 eval 预测完成；写 `runner_summary.json` 时，结构检查中的 NumPy `bool_` 被 JSON serializer 拒绝。
- 数据状态：完整冻结 24/16 eval 未读取；该 smoke 子集不用于效果解释。
- 修复：将 `finite` 检查的 NumPy 布尔值显式转换为 Python `bool`。事件定义、特征、target、模型、alpha grid、拆分、指标和主比较均未改变。
- 后续要求：创建包含修复源码的新 source snapshot；使用新 append-only smoke 目录重新运行。若通过，才允许完整开发审计。

## Smoke Repair2：command dtype 身份偏移

- 目录：`persistent_context_v2_pushobj_cog_event_response_audit_smoke_v1_repair2`
- 范围：2 个 train segment、2 个 eval segment；仅为 smoke 子集。
- 现象：数据、拟合和 summary 全部完成，其余结构检查通过；`rollout_identity=false`。train/eval 的最大 boundary state 差均为 `3.0517578125e-05`，超过冻结的 `1e-6`。
- 原因：substep logger 入口把作者的 float32 command 显式提升为 float64，原 `rollout_physics` 保留 command dtype。两条控制轨迹因此产生微小数值差。
- 修复：删除 dtype 提升，使用与 `rollout_physics` 相同的 `np.asarray(command) * action_scale`。事件 definition、特征、target、模型、split 和主比较不变。
- 数据状态：完整冻结 24/16 eval 仍未读取；Repair2 smoke 结果不用于效果解释。

## Smoke Repair3：通过

- 目录：`persistent_context_v2_pushobj_cog_event_response_audit_smoke_v1_repair3`
- source snapshot：`persistent_context_v2_p3b_source_snapshot_v1_repair2`。
- 范围：2 个 train segment、2 个 eval segment；仅为 smoke 子集。
- 结果：runner exit `0`，全部结构检查通过，包括 rollout identity、deterministic repeat、zero-CoG identity、event rule、finite、model lock before eval。
- 决定：允许使用同一合同、design 和 Repair2 source snapshot 运行完整 24/16 development audit。
