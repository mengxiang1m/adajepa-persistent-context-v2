# Persistent Context V2：PushObj CoG Markov/contact 表示审计结果

日期：2026-08-24  
合同：`persistent-context-v2-pushobj-cog-markov-contact-audit-v1`  
性质：开发集表示诊断，不是新的闭环 formal 结果  
结论：`VALID`；新增字段能描述接触难点，但没有解释到足以改善冻结 v1 predictor，暂不进入 enriched neural predictor 或 CoG history encoder。

## 1. 一句话结论

旧 7 维状态确实漏掉了 block 线速度、block 角速度和接触过程；补 contact 摘要能让低容量 ridge 明显优于只看旧状态的 ridge，但仍远弱于旧 v1。更直接的检验中，用这些新增字段修正冻结 v1，评估误差反而增加。因此“缺字段”是合理的问题定位，但当前这套控制步边界/聚合表示不是已经验证的解决方案。

## 2. 字段审计

旧 `PushTEnv._get_obs()` 的 7 维依次是：

1. agent position `(x,y)`；
2. block position `(x,y)`；
3. block angle；
4. agent velocity `(vx,vy)`。

它不含 block velocity 或 angular velocity。Pymunk 内部可稳定读取：

- `block.velocity`、`block.angular_velocity`、mass、moment、center of gravity；
- post-solve arbiter 的 contact points、normal、distance、total impulse、total kinetic energy 和 first-contact 标记。

本审计把 block 动量保存在每个 10 Hz 控制边界，把碰撞字段在控制步内按 agent-block、block-wall、agent-wall 分开聚合。替换原只计数的 callback 后，legacy 状态与原 `rollout_physics` 最大绝对差为 0；重复 rollout 的状态与 contact array hash 完全相同。

## 3. 冻结设计

- 来源：作者 `plan_targets.pkl` 的既有 CoG predictor train/dev segment；未读取任何 CoG formal segment。
- 拟合：24 个 v1-train segment，5 个 CoG，4 个 action variant，共 480 条轨迹。
- 评估：16 个 v1-dev segment，4 个 held-out CoG，4 个 action variant，共 256 条轨迹。
- 独立单位：segment；每个 segment 内的 factor、variant 和 transition 不作为独立样本。
- R0：旧 7 维 nominal trajectory + action + CoG。
- R1：R0 + nominal block velocity/angular velocity。
- R2：R1 + nominal agent-block contact 几何/冲量摘要。
- 模型：固定 ridge 与零上下文严格恒等映射；alpha 只在 train segment 四折选择。
- 直接归因：冻结 v1，再分别用 Markov 增量和 Markov+contact 增量拟合 v1 residual correction。

初版 2+2 segment smoke 发现空 CV fold 和缺少“直接修正 v1”对照；在完整数据读取前形成 Repair1。初版合同、设计和两个 smoke 目录均保留，partial smoke 的结构检查为 false，不用于效果解释。

## 4. 主要结果

主指标越低越好，是每条轨迹 10 步平均 normalized pose error。

| 模型 | 评估误差 | 相邻比较 | segment 配对证据 |
|---|---:|---:|---|
| 冻结 v1 FiLM | 0.28831 | — | 参考模型 |
| R0 legacy ridge | 0.56422 | — | — |
| R1 Markov ridge | 0.54495 | R0-R1 = +0.01927 | CI `[-0.02157,+0.06293]`；9 正/7 负 |
| R2 nominal-contact ridge | 0.44623 | R1-R2 = +0.09873 | CI `[+0.04487,+0.15216]`；14 正/2 负 |
| v1 + Markov correction | 0.30213 | v1-(v1+C1) = -0.01382 | CI `[-0.03237,+0.00026]`；5 正/11 负 |
| v1 + Markov/contact correction | 0.32110 | v1-(v1+C2) = -0.03279 | CI `[-0.07278,-0.00400]`；3 正/13 负 |

解释：

- 仅补 block velocity/angular velocity，对低容量模型只有小幅、异质的改善。
- 再补 nominal contact 摘要，低容量模型相对 R1 有一致得多的改善，说明接触相位/冲量确有解释信息。
- 但 R2 仍比 v1 差 `0.15792`，16 个 segment 中 15 个更差。
- 最关键的冻结-v1 correction 没有泛化：Markov correction 平均略退化；加入 contact 后退化更明显。这排除了“把这批聚合字段直接拼到 v1 就足够”的简单解释。

## 5. 异质性和接触诊断

冻结 v1 在极端 CoG 上更难：

| CoG x | v1 error |
|---:|---:|
| -25 | 0.39097 |
| -10 | 0.17408 |
| +10 | 0.20095 |
| +25 | 0.38722 |

逐 transition 的描述性 Spearman 相关为：v1 error 与 block speed `0.274`，与绝对 angular speed `0.263`，与真实 contact impulse `0.250`。这些 transition 不独立，相关只用于定位，不是显著性检验。

事后 nominal-vs-true contact 对账进一步显示：

- 只有 `2.23%` 的控制步 contact/no-contact 标记不一致；256 条中 47 条至少一步不一致；
- 首次 contact 控制步 `256/256` 相同；
- 但每轨迹 contact impulse 绝对差总和平均为 `1039.51`；
- 有 contact-event 分叉的轨迹，v1 error 为 `0.40774`，无分叉为 `0.26145`；
- v1 error 与 nominal/true impulse 差的轨迹级 Spearman 为 `0.387`。

这更像“接触发生时间大致对，但 CoG 改变了碰撞内的冲量响应”，而不是“模型完全不知道何时接触”。当前 10 Hz 边界状态和每控制步聚合量可能把 100 Hz 碰撞过程压缩得过粗。

## 6. 已证明、未证明与替代解释

已证明：

- simulator 中所需 Markov/contact 字段真实存在、非平凡、可确定性读取；
- nominal contact 摘要对低容量 CoG residual mapping 有增量信息；
- 现有 v1 误差集中在高速、旋转和大冲量 transition，并在极端 CoG 上更大；
- 当前字段拼接/线性 correction 不能改善冻结 v1。

未证明：

- enriched neural predictor 没有价值；本审计没有训练它；
- contact impulse 可从非特权 history 准确预测；
- prediction error 的任何下降能转化为 CEM 闭环收益；本审计没有产生新 formal 行为结果；
- CoG history transfer 成立；history encoder 仍未启动。

最强替代解释：ridge 容量可能不足；但冻结 v1 correction 同样退化，说明不能仅用“ridge 太弱”解释全部负结果。更可信的下一问题是表示时间尺度和 contact-response 建模：nominal 与 true 的首次接触相同，主要差异出现在接触内冲量，而不是控制边界的粗状态。

## 7. 下一步：P3b event-level contact-response audit

下一项只做开发诊断，不直接训练大网络，也不做行为 formal：

1. 在 100 Hz physics substep 保存接触前后 block velocity/angular velocity、接触点/法向和 canonical block impulse；
2. 明确分开 nominal 可用字段、真实 factor 的事后归因字段和任何 privileged upper bound；
3. 先测 `nominal pre-contact state + CoG → true impulse delta` 能否在 held-out segment/factor 上泛化；
4. 用 privileged true-contact impulse 做只读上限，判断 v1 residual 是否主要可由接触响应解释；
5. 若 event-level、非泄漏输入仍不能解释 v1 残差，不训练 V3 neural predictor；若能解释，再冻结一个最小 contact-response model，并先做 true-CoG 闭环开发验证，最后才考虑 formal 和 history estimator。

## 8. 可复现性

- base commit：`a29975964f966f2836a2c7e26f464367c795c333`；dirty worktree 已完整快照。
- design SHA256：`3c93649f79c979115f2031c84d97b4d0ab4b087c14ad10ca21eead96f61e9a6a`。
- contract SHA256：`8eb8dd8387ea58e7f4a191beede998ceae1eb2539089fa81c4cb2e53901d5d5e`。
- source snapshot SHA256：`0a38db55c9b152a330f15855bd3883ef7933c1b021ee6432d3c7de294d28b27c`；独立 snapshot audit 有效。
- v1 checkpoint SHA256：`39bb54d90a863c012dbd05e9ea448d8e213966a1d47c5f116faebc6f0a06c403`。
- runner summary SHA256：`ae1ffe2aa78644b863ac7222805e078559ab82484edf99437c68dd97e82cce8d`。
- independent audit Repair1 SHA256：`a29ba7a7f3e224785c3aaef415e02455f7ae836957b0245995d2d4e09d83be1d`，`valid=true`、无 failure。
- runner 使用物理 GPU 0（NVIDIA L40），manifest `exit_status=0`；周期 `nvidia-smi` 已保存。Python 完成后，启动包装器因 PowerShell/SSH 的 PID 转义错误未自动停止监控；监控 PID 随后被定向终止。该包装错误发生在 runner summary、manifest 完成和 `exit_status=0` 写入之后，不改变实验数组或统计，原日志保留。

远端主产物：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_markov_contact_audit_v1/`。
