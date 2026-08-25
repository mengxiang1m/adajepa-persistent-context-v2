# Persistent Context V2：真实 PushObj 工具坐标旋转 Stage 0 预注册

状态：**冻结，尚未查看任何本实验 prior/oracle 行为结果**。

## 1. 唯一问题

在真实 PushObj 接触动力学、发布的 AdaJEPA checkpoint 和发布的 T 形验证片段上，若同一 episode 内存在一个隐藏且恒定的工具坐标系旋转，向 planner 提供真实旋转角（true-factor oracle）是否会比只使用训练总体均值（population prior）显著改善行为？

本 Stage 0 只回答这个父问题。父门不通过时，不训练、不运行 history/RLS，也不查看保留 formal split。

## 2. 与旧实验的边界

- 不是旧 P0A 的跨 episode AdaJEPA 权重持续适应；checkpoint 全程冻结，且不做 TTT。
- 不是旧 `persistent_factor/benchmark.py` 的合成线性 actuator oracle；这里使用真实 Pymunk 接触、真实图像观测、发布的 PushObj AdaJEPA checkpoint 和 latent CEM。
- 不是 PointMaze action gain/lag 的重新调参；环境、动力学、任务和 factor 均不同。

## 3. 冻结资产

- 仓库：`/data4/zhaoqing/adajepa`
- checkpoint：`/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth`
- checkpoint 配置：同目录 `hydra.yaml`
- 片段：`/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl`
- 只用 T 形，避免将 shape shift 混入 factor 检验。
- 开发索引池：`0..499`；保留 formal 池：`500..999`。本 Stage 0 禁止读取 formal 池的行为结果。

## 4. 冻结 factor

actor/planner 输出二维相对动作 `u_t`。真实环境实际收到：

`u_eff,t = R(theta) u_t`

其中 `theta` 在整个 episode 内固定，旋转发生在动作缩放到 Pymunk PD 目标之前。角度不改变图像或初始状态，因此单帧看不到；它会持续改变接触点附近的推力方向。

- 训练支持角：`[-30, -15, 0, 15, 30]` 度
- population prior：训练角均值 `0` 度
- Stage 0 开发角：`[-22.5, -7.5, 7.5, 22.5]` 度
- 保留 formal 角：`[-25, -10, 10, 25]` 度

三组互不重合。Stage 0 不拟合 factor 模型。

true-factor world-model wrapper 的唯一操作：将归一化候选动作反归一化，按真实 `theta` 旋转每个二维 low-level action，再按发布统计量归一化后送入冻结 checkpoint。真实环境用同方向、同角度旋转。这是已知坐标变换，不是用结果反推的校准器。

## 5. 冻结任务候选与顺序停止

每个候选包含 4 个开发角，每角 8 个片段，共 32 个配对 scenario。每个 scenario 的 prior/oracle 使用相同初态、目标、CEM 随机种子和真实 factor。

### 候选 A：发布片段原分布

- 池：开发索引 `0..499`
- 选择：`numpy.random.default_rng(410000).permutation(pool)[:32]`，依 factor 顺序连续分配 8 个
- 目标：从片段初态出发，在无旋转 nominal 环境重放该片段 25 个发布动作，所得最终真实观测/状态
- 主窗口：执行 25 个 low-level action 后的 `pose_auc25`

### 候选 B：发布片段中的早期接触子集

仅当候选 A 为 NO-GO 时运行。

- 池：开发索引中，发布 state 轨迹前 10 步的 T-block 位移 `>=10` 像素者
- 选择：`numpy.random.default_rng(420000).permutation(pool)[:32]`，依 factor 顺序连续分配 8 个
- 目标同 A，仍是发布 25-step nominal replay 的最终目标
- 主窗口：前 10 个 low-level action 的 `pose_auc10`

候选 B 只改变预先定义的接触性筛选和评价窗口，不改变 checkpoint、factor 或 CEM。若 A 为 GO，跳过 B；若 A、B 都 NO-GO，父门关闭并停止。

## 6. 冻结 planner 和执行预算

- planner：paired open-loop latent CEM
- model horizon：5；每个 model step 聚合 5 个二维动作，总执行长度 25
- samples：200；elite：30；CEM rounds：10
- 初始化：`mu=0, sigma=1` 的归一化动作
- objective：发布 PushObj planner 的 `alpha=1, base=2, mode=staged`；episode-entry `step=0`
- prior/oracle 共用每个 scenario 的 CEM seed；环境 deterministic
- 不裁剪 checkpoint 反归一化后的物理命令，因为发布 PushObj 数据和环境允许超出 `[-1,1]`
- 不做 MPC 重规划、在线梯度更新、history 更新或跨 episode 状态携带

## 7. 冻结指标

对每个执行后状态（不计双方相同的初态）：

`pose_cost_t = block_position_error_t / 20 + wrapped_block_angle_error_t / (pi/9)`

20 像素和 `pi/9` 来自环境现有成功容差。主指标为指定窗口内 pose cost 的算术平均：A 用 `pose_auc25`，B 用 `pose_auc10`。越低越好。

配对改善：

- `delta_i = prior_cost_i - oracle_cost_i`
- `relative_improvement = mean(delta) / mean(prior_cost)`
- 方向正确率：`mean(delta_i > 0)`
- 95% CI：对 32 个配对 delta 做 20,000 次有放回 bootstrap，seed `6401`

辅助指标只用于解释：窗口末 position/angle error、命令轨迹 L2 差异、环境 success、按 factor 的均值。辅助指标不得覆盖主门结论。

## 8. 父门（全部满足才 GO）

1. identity audit：`theta=0` wrapper 与 base checkpoint rollout 数值一致（`max_abs <= 1e-6`）。
2. intervention audit：非零 context 的候选有效动作与 prior 不同，且 paired prior/oracle 规划命令 L2 `>1e-6` 的比例 `>=95%`。
3. 主指标 `relative_improvement >=25%`。
4. paired bootstrap 95% CI 下界严格大于 0。
5. 配对方向正确率 `>=70%`。

决策顺序：A 达门即 GO 并停止 Stage 0；A 失败才跑 B；B 达门则 GO；两者均失败则 NO-GO。禁止再加第三个候选。

## 9. GO 之后允许的工作

只有父门 GO 后，才新建并再次冻结独立 formal contract：在 `500..999` 和 formal 角上比较 history/RLS、population prior、true oracle，以及 reset/shuffled-history/zero-context 等负对照。Stage 0 的开发结果不能充当 history 方法结果。

## 10. 证据与可复核性

- 预注册 Markdown 和机器可读 JSON 在远端运行前同步并记录 SHA-256。
- 每个 scenario 写 append-only raw JSONL，保留片段索引、factor、两种命令/有效动作、state 轨迹、planner trace 和资源记录。
- 运行结束后用独立 audit 子命令仅从 raw JSONL 重算统计量和门，不信任 runner summary。
- 记录 git revision、checkpoint/data/design 哈希、命令、GPU/CPU/内存采样和时间。

