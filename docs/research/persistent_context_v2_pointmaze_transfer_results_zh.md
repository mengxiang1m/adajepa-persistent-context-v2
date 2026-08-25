# Persistent Context V2：PointMaze 真实 benchmark 迁移结果

## 结论（人话）

这次把合成任务里成功的思路接到了真实 AdaJEPA PointMaze checkpoint 上，但当前这版 PointMaze action-calibration 任务 **不值得继续做 history/RLS 正式实验**。

原因不是模型没有读到 context。知道真实 actuator gain 后，所有场景的 CEM 动作都发生了变化；原因是这些变化没有让真实 MuJoCo 控制变好。标准 CEM 预算下，true-context 的前 25 步累计位置距离反而比 population prior 略差 `0.80%`，配对置信区间跨过 0，离预设的至少 `25%` 改善很远。

因此当前证据支持的结论是：

> “显式持续 context”在合成 Stage 3 中有机制价值，但把它直接迁移为已发布 PointMaze AdaJEPA 模型前的 scalar action-gain FiLM，在当前 hard-goal/CEM 控制任务上没有建立行为上限。既然直接知道真值都不更好，历史估计得再准也没有可回收的闭环收益。

机器判定：`NO_GO_POINTMAZE_ACTION_CALIBRATION_TASK`。

## 做了什么

### 1. 资产和注入边界审计

- 真实环境：`point_maze_medium` MuJoCo；动作经 5 个 simulator frames 合并成一个模型动作。
- 真实模型：发布的 `mediummaze_dynamics_shift/model_latest.pth`，视觉 encoder、proprio/action encoder 和 ViT predictor 全部冻结。
- 数据：4,000 条 PointMaze rollouts，每条 100 步；前 3,600 条只用于训练/拟合，后 400 条只用于 held-out 检查。
- 唯一隐藏因素：scalar command-to-executed-action gain：

  ```text
  executed_action = clip(gain * clip(command, -1, 1), -1, 1)
  ```

- gain 不进入图像、proprio 或 task ID；只在 raw audit artifact 中保存真值。
- population prior 只由 train factors `(0.50, 0.75, 1.00, 1.25, 1.50)` 校准，均值严格为 `1.0`。
- development factors 为未在训练中逐点出现的 `(0.60, 0.90, 1.10, 1.40)`。

### 2. 训练一个最小 context 接口

在冻结 AdaJEPA 前加入一个 8 参数 FiLM action adapter：scalar context 生成两个动作轴的 affine gain/bias，再经过真实 actuator clipping。训练样本来自 PointMaze train actions 与 factor-diverse relabeling，未训练 encoder 或 predictor。

结果：

- 参数量：`8`；
- held-out development factor action MSE：`5.4e-13` 至 `1.1e-10`；
- held-out formal factor action MSE：`3.0e-12` 至 `8.5e-11`。

这证明接口学会了命令标定关系，不是因为 adapter 没训练好而失败。

### 3. 真实闭环父门

开发集只比较两臂，没有读取 history 结果来选任务：

- `population_prior`：模型 rollout 用 gain `1.0`；
- `true_context`：模型 rollout 直接使用真实 gain。

两臂共享 factor、起点、目标、初始速度、CEM 随机数和动作预算。CEM 使用仓库标准预算：每轮 `200` candidates、`30` elites、`10` rounds、模型 horizon `5`（25 个 MuJoCo 动作）。共 `4 factors × 5 paired scenarios = 20` 个独立配对场景。

主 task-qualification 指标是前 25 步位置距离 AUC，越低越好。所有时间窗都独立复算：

| 指标 | population prior | true context | prior − true | 95% CI / 方向 |
|---|---:|---:|---:|---|
| 5 步距离 AUC | 3.9066 | 3.9220 | −0.0154 | [−0.0503, 0.0077] |
| 25 步距离 AUC（主） | 2.8444 | 2.8672 | −0.0229 | [−0.0700, 0.0197] |
| 第 5 步距离 | 3.6894 | 3.7372 | −0.0477 | [−0.1290, 0.0061] |
| 第 10 步距离 | 3.1055 | 3.1137 | −0.0081 | [−0.0839, 0.0521] |
| 第 25 步距离 | 1.8262 | 1.8397 | −0.0134 | [−0.1933, 0.1399] |

主指标相对改善为 `−0.804%`，同向场景比例 `55%`。门槛要求至少 `+25%` 且配对 CI 下界大于 0；两项都失败。

## 验证了什么

### 已验证的机器事实

- context 接口可训练，并能在 held-out gain 上准确完成 action calibration。
- true-context 会进入 AdaJEPA rollout 并改变规划：20/20 配对场景的 action plan 不同，平均绝对 command 差 `0.1903`。
- PointMaze 离线状态回归能从 nominal transition 识别动作响应，held-out-independent 拟合的 velocity `R²` 为 `0.866/0.859`；RLS 所需的低维 sufficient-statistic 工程路径可实现。
- raw artifact 数量、scenario/plan seed、起点和目标配对完全一致。
- 18 个 Stage 0–3 与 PointMaze transfer 测试通过。

### 由行为对照支持的推断

- 在当前发布 checkpoint、hard-goal protocol、scalar action-gain 和标准 CEM 预算下，知道真实 gain 没有产生可利用的 early-control 行为上限。
- 因此不能声称跨 episode history 在这个任务实例上有价值；即使 RLS 最终估计准确，也没有证据表明该信息能改善控制。
- 合成 Stage 3 的正结果不能直接外推成“真实 AdaJEPA benchmark 已可行”。

### 没有验证、也没有执行的部分

- 没有执行 persistent history、current-only、no-persistence、shuffled 或 wrong-sequence formal 矩阵。
- 没有把 RLS estimation error 当作行为证据。
- 没有加入 episode-local AdaJEPA TTT、权重继承、LoRA、router、expert 或 consolidation。

这些不是遗漏，而是父门失败后的停止纪律：true factor 都无收益时，下游 history 方法没有科学解释空间。

## 最强替代解释

最强替代解释是：factor 映射本身正确，但冻结的 nominal AdaJEPA predictor、latent goal objective 与 CEM 形成的组合并不保证“更物理正确的 action conditioning”会带来更好的有限预算动作排序。population prior 可能反而起到动作幅度正则化作用；墙体接触、动作饱和和有限 CEM 搜索也会削弱 gain 真值的控制价值。

这不等于持续 context 思路普遍无效，只说明当前 PointMaze action-gain 实例没有建立最上游行为前提。

## 执行与审计

- 首次开发运行在任何结果生成前因 Gym `TimeLimit/OrderEnforcing` 外壳与项目自定义 `prepare()` 不兼容而退出。修复只改为使用与项目 vector worker 相同的底层 env；factor、seed、预算和指标未变，失败日志保留。
- 一个 64×4 CEM smoke 先得到相同方向的负结果；随后按标准 200×10 预算复验，未据中途结果改变 factor 或指标。
- 标准复验 GPU：显式 `CUDA_VISIBLE_DEVICES=0`，NVIDIA L40；外部采样峰值 `2,563 MiB`，Torch peak allocation `1,579,901,952 bytes`，模型加载后的实验 wall time `12.65 s`。
- 模型 checkpoint SHA256：`df2caeb62f6fc8d67b58cab43dcbbbba7d4821226744613f52137b7c27c25528`。
- FiLM adapter SHA256：`18d17e934838a65f2af0b6d0a8d502297c29cefb255a41a41499fe15f1ac3708`。
- 标准开发 raw SHA256：`d2485417c7295886cbba0cc584867d58e15f59af40f84125670b5e487d6d5a16`。

远端主要产物位于：

```text
/data4/zhaoqing/adajepa/persistent_context_v2_outputs/pointmaze_transfer/v1/
/data4/zhaoqing/adajepa/persistent_context_v2_outputs/pointmaze_transfer/dev_standard/
```

## 下一步边界

当前 action-gain/PointMaze 实例停止，不围绕已看到的 subgroup 改 factor、换指标或只挑正向 seed 重开 formal。

如果继续真实 benchmark，最小且科学上独立的下一步应是重新做一个 Stage 0 任务资格实验，并在看结果前冻结有限候选表。例如选择会改变路径时序而不仅是动作幅度的单一 factor（action delay/dead zone），或采用有明确早期 waypoint/不可逆代价的任务；仍必须先让 true-factor oracle 达到行为动态范围，之后才允许 history/RLS。

