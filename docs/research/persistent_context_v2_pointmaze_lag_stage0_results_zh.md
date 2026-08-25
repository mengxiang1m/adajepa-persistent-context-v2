# Persistent Context V2：PointMaze Actuator-Lag Stage 0 结果

## 结论

机器判定：`NO_GO_POINTMAZE_ACTUATOR_LAG_TASK`。

两个在结果未知时预注册的候选任务都没有建立 true-factor 行为上限。true context 在全部 64 个独立配对场景中都改变了 AdaJEPA/CEM 动作计划，但真实 MuJoCo 结果没有改善。因此不能进入 history/RLS、no-persistence、shuffled 或 wrong-sequence formal 实验。

用人话说：这次不再只改“动力大小”，而是让动作慢慢生效。模型知道真实延迟程度后确实换了动作，但无论在原始远距离迷宫目标还是短距离 waypoint，换后的动作都没有更好。既然直接知道答案都没有收益，历史学习没有可回收的上限。

## 冻结任务

唯一 factor 为连续 actuator lag `rho`：

```text
e_-1 = 0                         # 每个 episode 重置
e_t = rho e_(t-1) + (1-rho) u_t
```

- train factors：`[0.0, 0.2, 0.4, 0.6, 0.8]`；population prior mean 为 `0.4`。
- development factors：`[0.1, 0.3, 0.5, 0.7]`。
- reserved formal factors：`[0.15, 0.35, 0.55, 0.75]`，本次没有读取或运行。
- 发布的 PointMaze AdaJEPA checkpoint 全冻结。
- CEM 固定为 `200` candidates、`30` elites、`10` rounds、5 个 model steps（25 个低层动作）。
- population 与 oracle 严格共享 factor、场景、初始状态、目标、CEM RNG 和动作预算。

冻结设计 SHA256：`ff173c47bf78746b01adce8dea0779e526ca548522823be94f4aafe75c45de2f`。

## 候选 A：原始 hard-goal

4 个 development factors × 每 factor 8 个场景，共 32 个独立 paired scenarios。主指标为前 25 步 position-distance AUC，越低越好。

| 指标 | population prior | true context | prior − true | 相对改善 | 95% CI | 同向比例 |
|---|---:|---:|---:|---:|---|---:|
| `auc_k25`（主） | 2.5600 | 2.6227 | −0.0628 | −2.45% | [−0.1626, 0.0161] | 50.0% |
| `auc_k10` | 3.5562 | 3.5495 | +0.0067 | +0.19% | [−0.0144, 0.0271] | 65.6% |
| `cost_k25` | 1.4619 | 1.6937 | −0.2318 | −15.85% | [−0.5112, 0.0002] | 31.3% |

动作计划 32/32 改变，平均绝对 command 差 `0.1626`。身份和配对审计通过；CI、25% 实际效应和 70% 方向门均失败，候选 A 为 `NO_GO`。

## 候选 B：local waypoint early-time cost

候选 A 失败后才按冻结顺序解锁 B。start/goal 在 free-cell graph 上 BFS 距离严格为 2，位置 jitter 为 ±0.1，初始与目标速度均为 0。共 32 个独立 paired scenarios，主指标为前 10 步 position-distance AUC。

| 指标 | population prior | true context | prior − true | 相对改善 | 95% CI | 同向比例 |
|---|---:|---:|---:|---:|---|---:|
| `auc_k10`（主） | 1.0463 | 1.0503 | −0.0040 | −0.38% | [−0.0154, 0.0080] | 37.5% |
| `auc_k25` | 0.7433 | 0.7554 | −0.0121 | −1.62% | [−0.0391, 0.0139] | 56.3% |
| `cost_k10` | 0.4791 | 0.5100 | −0.0308 | −6.43% | [−0.0733, 0.0112] | 46.9% |

动作计划 32/32 改变，平均绝对 command 差 `0.1792`。身份和配对审计通过；三个行为门均失败，候选 B 为 `NO_GO`。

## 证明了什么

### 机器事实

- lag factor 的 sequence/episode 生命周期和 analytic rollout 可执行；episode-local actuator state 每个 episode 从 0 重置。
- true factor 明确进入 world-model rollout 和 action ranking，因为所有 paired action plans 都发生变化。
- 两个候选共 128 条 raw policy records，数量和配对身份完整。
- 两个 raw artifact 分别通过独立 bootstrap 复算。
- 21 个 Persistent Context V2 测试全部通过。

### 证据支持的推断

- 在当前发布 checkpoint 和 latent CEM objective 下，知道 actuator lag 没有转化成早期控制价值。
- 前一个 scalar gain 失败并非仅因为“只改了动作幅度”：改变动作时序的连续 lag 在两个任务上也失败。
- 当前证据越来越指向 PointMaze 的冻结 world-model/planner 组合对这类低维 actuator context 缺乏可利用的行为敏感性，而不是 context 没被输入模型。

### 没有证明的事

- 不能外推为 persistent context 普遍无效。
- 没有测试离散 pure delay、dead zone、不同 world-model architecture 或重新训练的 temporal context predictor。
- 没有证明 PushObj、deformable 或其他有接触/不可逆代价的任务也会失败。

## 为什么没有继续 RLS/history

本阶段门禁要求 true-factor oracle 至少相对 population prior 改善 25%，paired CI 下界大于 0，并有至少 70% 场景同向。两个候选没有任何一个接近门槛。运行 RLS 只能证明能否估参数，不能证明参数有行为价值；因此 formal history 合同未获授权，`history_rls_executed=false`。

## 资产与资源审计

- checkpoint SHA256：`df2caeb62f6fc8d67b58cab43dcbbbba7d4821226744613f52137b7c27c25528`。
- 执行代码 SHA256：`63cd2489b10dcc9c1036b147b266abb0888248e88426af203d34642e48044939`。
- 候选 A raw SHA256：`3da2629aad3a427576dd61a41480c9f54108179ce2cc22db6f446e6897a66ff8`。
- 候选 B raw SHA256：`2d1dbbfec2fd09592d9c44f88a6bf4a207d2682575b47f9678eec0927cdd402b`。
- GPU：显式 `CUDA_VISIBLE_DEVICES=0`，NVIDIA L40；两个候选外部采样峰值均为 `2,563 MiB`，Torch peak allocation 均为 `1,579,901,952 bytes`。
- 模型加载后的执行时间：A `19.90 s`，B `20.06 s`。

远端产物：

```text
/data4/zhaoqing/adajepa/persistent_context_v2_outputs/pointmaze_lag_stage0_v1/
```

## 下一步边界

PointMaze actuator-lag 分支停止，不增加第三候选，不根据 factor subgroup 换指标重跑。

如果继续真实 benchmark，优先级应从 PointMaze 移到具有明显接触或首次错误代价的 PushObj/deformable 任务，但仍需重新开始独立 Stage 0：先预注册单一摩擦、工具标定或质心 factor，只比较正确 population prior 与 true-factor oracle。不能把当前合成正结果称为真实 PushObj 已成立。

