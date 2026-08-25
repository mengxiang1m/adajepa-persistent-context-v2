# PushObj CoG 条件预测器结果

日期：2026-08-23  
合同：`persistent-context-v2-pushobj-cog-film-residual-predictor-v1`  
正式输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_predictor/`

## 一句话结论

第一版 CoG-conditioned 模型学到了真实但偏弱的行为价值：32 个未见 formal 场景上，true-CoG context 相对 population prior 将 `pose_auc10` 从 `1.383167` 降到 `1.318500`，平均改善 `4.6753%`，23/32 场景正向；但 paired 95% CI `[-0.011011,+0.141373]` 跨 0，deadline success 没有提高，只回收 physics oracle gap 的 `23.06%`。

因此客观判断是“有可学习信号，但当前 predictor 还不够准、也不够稳定”。这个结果支持继续改进条件动力学模型，不支持现在就把它当成成熟方案，也不适合立刻在其上训练复杂 history encoder。

## 1. 做了什么

### 数据

- 从 355 个 early-contact T-shape 片段中按冻结 seed 选择 96 train、24 dev、32 formal segment；三组互斥，并排除了上一 CoG oracle 的 32 个场景。
- train CoG：`[-30,-15,0,15,30]`；dev：`[-25,-10,10,25]`；formal：`[-22.5,-7.5,7.5,22.5]`。formal 数值未在训练中逐点出现。
- 每个 train segment、每个 factor 使用 16 条动作轨迹，共 `7680` 个训练样本；dev 使用 `768` 个样本。动作覆盖发布动作及四档固定噪声扰动。

### 模型

- 参数量 `168,222`。
- base trajectory 是 nominal `CoG=(0,45)` Pymunk 的 10 步 rollout；网络预测真实 CoG 相对 nominal 的 10 步 block `(x,y,angle)` residual。
- 唯一 context 路径是 `cog_x/30` 产生的一组 FiLM scale/shift；没有 history、TTT、gate、router 或第二个 adapter。
- zero-context branch 从同一 head 输出中相减，因此 `cog_x=0` 时 residual 严格为 0；审计实测 max absolute error 为 `0`。
- 固定训练 3000 steps，每 100 step 在 held-out-factor dev 上评估，最低 dev MSE 的 checkpoint 位于 step `900`。

## 2. 预测结果

held-out-factor dev residual MSE：

| context | MSE |
|---|---:|
| population `cog_x=0` | 0.121486 |
| true CoG | 0.077857 |

true context 将 dev prediction MSE 降低 `35.9126%`。

在正式 CEM 最终动作上，预测轨迹相对真实执行的平均归一化 pose error 从 `0.532936` 降至 `0.455379`，降低 `14.5527%`。说明 context 确实改善了预测，但剩余误差仍大。

## 3. 正式闭环行为

| policy | mean pose_auc10 | deadline success |
|---|---:|---:|
| population prior context | 1.383167 | 81.25% |
| learned true-CoG context | 1.318500 | 81.25% |
| true-physics simulator oracle | 1.102777 | 96.875% |

learned true-CoG 相对 population：

- mean delta `+0.064667`；相对改善 `4.6753%`；
- paired bootstrap 95% CI `[-0.011011,+0.141373]`；
- 23/32 正向，0 tie，9/32 负向；
- 32/32 的 CEM action plan 都发生变化；
- 回收 simulator-oracle improvement 的 `23.0633%`；
- deadline success 不变，均为 `81.25%`。

true-physics oracle 相对 population 在这批全新场景上改善 `20.2716%`，mean delta `+0.280390`，CI `[+0.182752,+0.401784]`，29/32 正向。这再次确认可用的 CoG 行为上限存在；主要瓶颈是 learned predictor 尚未回收该上限。

## 4. 按 CoG 分组

| CoG x | population | learned | oracle | learned改善 | 正向数 |
|---:|---:|---:|---:|---:|---:|
| -22.5 | 1.676154 | 1.515125 | 1.225379 | 9.6070% | 7/8 |
| -7.5 | 1.298140 | 1.262699 | 1.190819 | 2.7301% | 6/8 |
| +7.5 | 1.323797 | 1.306793 | 1.168501 | 1.2844% | 5/8 |
| +22.5 | 1.234578 | 1.189383 | 0.826408 | 3.6608% | 5/8 |

四个 factor 分组的均值都正向，但正向强度明显不对称；`-22.5` 最强，其余三组较小。每组只有 8 个样本，分组值只作异质性描述。

## 5. 审计与工程记录

- 正式 runner exit code `0`；32 pairs、factor balance、split、waypoint、action-change 检查均通过。
- 第一次独立审计只在 CPU 重算 CUDA float32 prediction 时失败：max difference `3.0518e-5` 超过原 `1e-5` 容差；环境执行重放、指标与摘要误差均为 0。原始 `independent_audit.json` 保留。
- 容差按跨设备 float32 GEMM 修到 `1e-4` 后，`independent_audit_repair1.json` 与包含正确 trajectory pose diagnostic 的 `independent_audit_repair2.json` 均通过；formal raw、checkpoint 与结果方向没有改动。
- repair2：zero-context error `0`，execution replay error `0`，prediction replay max error `3.0518e-5`，metric replay error `0`，failure list 为空。
- design SHA256：`e09973efeaf0bd291a35cd0f4627888aace591134e05eb0a273cf05ff1947c1f`。
- checkpoint SHA256：`39bb54d90a863c012dbd05e9ea448d8e213966a1d47c5f116faebc6f0a06c403`。
- raw SHA256：`8bd3e9ac6719d0664a048eb8eb27a2058e6ee7db6805fc21c5c0b9dc47a2a2f1`。
- L40 GPU 正式全流程 wall time `3:22.95`，max RSS `1,183,300 KB`，Torch peak allocated `24,363,520 bytes`。

## 6. 证据边界与下一步

已证明：factor-diverse 训练的单一 FiLM residual predictor 会在 held-out CoG 上使用 context；它同时改善 prediction 与平均闭环行为，四档 factor 均值均为正。

尚未证明：该改善在重复批次上稳定；模型能回收大部分 oracle gap；CoG 可从历史非特权估计；视觉 AdaJEPA predictor 已获得相同能力。

下一步应先针对 predictor 的主要误差源做一个冻结改进实验：把当前“整段 flattened trajectory residual”替换为保留接触时序的 recurrent/temporal residual predictor，仍只保留一个 FiLM context 路径，并在同一 formal split 上先比较 true-CoG 行为上限。只有 learned true-context 的闭环收益稳定后，才值得增加 CoG history estimator。

## 7. 文件位置

- 合同/设计：`docs/research/persistent_context_v2_pushobj_cog_predictor_contract_zh.md`、`persistent_context_v2_pushobj_cog_predictor_design.json`
- core：`research/persistent_context_v2/pushobj_cog_predictor.py`
- runner/audit：`scripts/run_persistent_context_v2_pushobj_cog_predictor.py`、`scripts/audit_persistent_context_v2_pushobj_cog_predictor.py`
- tests：`tests/test_persistent_context_v2_pushobj_cog_predictor.py`
- 远端输出：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_predictor/`
