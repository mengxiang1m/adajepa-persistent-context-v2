# Persistent Context V2：PushObj CoG P3b 事件级接触响应审计结果

日期：2026-08-24  
合同：`persistent-context-v2-pushobj-cog-event-response-audit-v1`  
性质：开发集表示审计，不是新的闭环 formal 结果  
结论：`VALID`；100 Hz nominal impulse 有局部增量，但完整事件表示未超过 10 Hz 聚合表示或零响应，不进入 CoG V3 predictor 和 CoG history estimator。

## 1. 研究问题

P3a 发现 nominal 与 true-CoG rollout 的首次接触控制步一致，主要差异集中在接触冲量。P3b 将日志从 10 Hz 控制边界提高到 100 Hz physics substep，检验事件级状态、接触几何和 nominal impulse 能否解释 true-CoG 相对 nominal-CoG 的单步速度响应差。

唯一主比较为：

```text
C10_aggregate error - S100_state_geometry_impulse error
```

正值支持 100 Hz event representation。

## 2. 冻结设计

- train：24 个 P3a train segment，CoG `[-30,-15,0,15,30]`。
- eval：16 个 P3a held-out dev segment，CoG `[-25,-10,10,25]`。
- 所有既有 CoG formal segment 均未使用。
- 每个 segment 有 nominal action 和三档噪声 action，共 4 个 variant。
- 主事件由 nominal rollout 冻结：agent-block contact 且同一 substep 没有 block-wall contact。
- nominal 与 true rollout 按相同 control/substep index 对齐，不做 outcome 驱动的重对齐。
- 目标为 block `dv_x,dv_y,domega` 的 true-minus-nominal response residual。
- 模型为固定 feature map 加 ridge；alpha 只在 train segment 四折选择，模型锁定后才生成 eval。
- 统计单位为 segment，bootstrap 20,000 次。

数据规模：

| split | segment | nominal event | factor-event rows |
|---|---:|---:|---:|
| train | 24 | 6,234 | 31,170 |
| eval | 16 | 3,558 | 14,232 |

train target RMS 归一化尺度为：

```text
dv_x: 14.86161
dv_y: 16.45538
domega: 0.16559
```

## 3. 表示

| 表示 | 信息权限 | 输入维数 |
|---|---|---:|
| zero response | 恒预测 0 | 0 |
| C10 aggregate | 10 Hz 控制边界 state、command、整步 contact summary | 29 |
| S100 state | 100 Hz event pre-state、command、target、substep phase | 15 |
| S100 state+geometry | S100 state 加 nominal contact geometry | 23 |
| S100 state+geometry+impulse | 再加 nominal impulse/solver summary | 29 |
| P100 true contact | 再加同一 substep 的 true contact 字段 | 43；privileged |

C10 和完整 S100 的输入维数同为 29。所有 ridge 都在 train CV 中选择 `alpha=100`。

## 4. 主要结果

误差越低越好：

| 模型 | eval segment 平均 error |
|---|---:|
| zero response | **0.30496** |
| C10 aggregate | **0.32292** |
| S100 state | 0.34402 |
| S100 state+geometry | 0.35932 |
| S100 state+geometry+impulse | 0.33020 |
| P100 true contact | 0.37289 |

唯一主比较：

```text
C10 - S100 full = -0.00728
95% CI = [-0.02137, +0.00603]
positive / tie / negative segments = 8 / 0 / 8
```

100 Hz 完整表示没有超过 10 Hz 聚合表示。

嵌套比较：

| 比较 | mean delta | 95% CI | 正/平/负 segment |
|---|---:|---:|---:|
| S100 state - state+geometry | `-0.01530` | `[-0.02318,-0.00758]` | `4/0/12` |
| state+geometry - state+geometry+impulse | `+0.02913` | `[+0.01781,+0.03958]` | `14/0/2` |
| S100 full - privileged P100 | `-0.04269` | `[-0.05891,-0.02775]` | `1/0/15` |
| zero response - S100 full | `-0.02524` | `[-0.04170,-0.00898]` | `3/0/13` |

事件级 nominal impulse 对 geometry-only 模型有稳定增量，说明冲量字段包含信息。加入 impulse 后的模型仍比 C10 高 `0.00728`，比 zero response 高 `0.02524`。raw true-contact 字段加入固定 ridge 后也没有形成 privileged upper bound。

## 5. Factor 与事件诊断

| CoG x | zero | C10 | S100 full | privileged P100 |
|---:|---:|---:|---:|---:|
| -25 | 0.37946 | 0.40498 | 0.42389 | 0.48464 |
| -10 | 0.24763 | 0.25267 | 0.25687 | 0.27192 |
| +10 | 0.22223 | 0.22892 | 0.23162 | 0.25219 |
| +25 | 0.37051 | 0.40511 | 0.40841 | 0.48281 |

- nominal event 对齐到 true rollout 时，`4.49%` 的 factor-event row 在同一 substep 没有 true agent-block contact。
- true/nominal event impulse norm 的平均绝对差为 `20.34`。
- 极端 CoG 的 target 和模型误差都更大。

这些结果支持“CoG 会改变接触响应”，但当前对齐目标和固定低容量表示无法把这种差异转成可泛化预测。事件时间分辨率本身没有解决 P3a 的表示瓶颈。

## 6. 结论边界

已建立：

- 100 Hz substep 日志可确定性采集，且与原 10 Hz rollout identity 一致；
- nominal event impulse 在低容量 event model 中有增量信息；
- CoG 极值和 true/nominal impulse 分歧仍是主要难点。

未建立：

- 100 Hz 非特权 event representation 优于 10 Hz aggregate；
- event model 优于恒零 response；
- raw true-contact ridge 构成有效 privileged upper bound；
- event prediction 能提高 true-CoG CEM 闭环行为；
- CoG history transfer 有效。

privileged P100 的负结果只约束当前 target、对齐方式、feature map 和 ridge，不能扩展成所有 mechanics-aware contact model 都无效。

## 7. 决定

- 不训练 CoG V3 neural predictor。
- 不启动 CoG history estimator。
- 不运行新的 CoG closed-loop formal。
- P3b 作为有效负结果保留，CoG 路线停在事件表示诊断阶段。
- 若以后重启 CoG，需要新的开发合同，先建立 mechanics-aware contact-response upper bound，再讨论可部署 predictor。

当前研究资源应回到已经建立闭环价值的 rotation-gain Context，重点处理跨 shape 中约 `28%` 的 scene-level harm。下一候选是前瞻记录 population/context 两套 predicted rollout，并只读检验 rollout disagreement 能否解释负向场景；该候选仍需独立冻结合同和未暴露开发数据。

## 8. 有效性与复现

- runner：`valid=true`，全部结构检查通过。
- 独立审计：`valid=true`，failure 为空，所有模型、比较和 bootstrap 复算一致。
- rollout boundary identity 最大误差：`0`。
- deterministic repeat：通过。
- zero-CoG target/model identity：通过。
- 24/16 segment 全部存在 eligible event。
- 模型在 eval 生成前锁定。
- 完整运行 wall time：`15.88 s`，CPU-only；周期 GPU 监控已保存。
- design SHA256：`e9cf733782795997a45196f606656f1955784c2c58002f40c8707c2fa8719de6`。
- contract SHA256：`e326c8838e4f0d9d09008b35d7be17d0adfe5436163c38f75480358bdd3abd03`。
- source snapshot SHA256：`f89ed4f7baa98f6b5db6e25e90180e2d6e92436a3efe6f22be9041277a31eda1`。
- independent audit SHA256：`1b88f8f45a14a95e7ea114b4069216a2e3b2b806e1e38a486b15fc7ae4495f23`。
- 远端产物：`/data4/zhaoqing/adajepa/repro_outputs/persistent_context_v2_pushobj_cog_event_response_audit_v1/`。
