# Persistent Context V2：PointMaze Actuator-Lag Stage 0 设计冻结

状态：`FROZEN_BEFORE_DEVELOPMENT_RESULTS`

冻结时间：2026-08-22（Asia/Shanghai）

## 独立性与核心命题

本实验是相对于 `NO_GO_POINTMAZE_ACTION_CALIBRATION_TASK` 的独立任务实例。上一实例改变动作幅度：`executed = gain × command`；本实例只改变命令生效的时间响应，禁止用上一实例的 subgroup、factor 或指标改写结果。

核心命题：在发布的 PointMaze AdaJEPA checkpoint、固定 CEM 预算和相同场景下，直接知道一条 sequence 内持续的 actuator-lag factor，是否能显著降低 episode entry 的早期真实任务代价。

零假设：true-factor oracle 相对由 train split 校准的 population-prior context，不能达到预设的早期位置距离改善门。

Stage 0 只允许比较 `population_prior` 和 `true_factor_oracle`，禁止查看 history/RLS 结果选择任务。

## 唯一处理变量

对每个低层 command `u_t`，真实执行动作递推为：

```text
e_-1 = [0, 0]                 # 每个 episode 重置
e_t  = rho * e_(t-1) + (1-rho) * clip(u_t, -1, 1)
```

- `rho` 是 sequence-level persistent factor；`rho=0` 表示无 lag，越大表示响应越慢。
- actuator state `e_t` 是 episode-local state，每个 episode 清零，不跨 episode 保存。
- `rho` 不进入视觉、proprio、目标或 task ID；真值只进入 oracle rollout 和审计文件。
- MuJoCo、maze layout、质量、阻尼、视觉、success threshold 和 AdaJEPA 权重全部固定。

## Factor split 与总体先验

```text
train factors:   [0.00, 0.20, 0.40, 0.60, 0.80]
development:     [0.10, 0.30, 0.50, 0.70]
reserved formal: [0.15, 0.35, 0.55, 0.75]
population prior mean rho = 0.40
```

三个 split 逐点不相交。Stage 0 不读取 reserved formal 场景。population policy 使用 train factor 均值 `0.40` 做 certainty-equivalent planning；true oracle 使用真实 `rho`。两者在真实环境中面对同一个真实 `rho`。

## 固定模型与规划预算

- checkpoint：`/home/zhaoqing/adajepa/checkpoints/mediummaze_dynamics_shift/checkpoints/model_latest.pth`
- encoder、action/proprio encoder、predictor 全部冻结；Stage 0 不训练 lag adapter。
- analytic lag transform 只把 command 序列转换成发布 checkpoint 所理解的 executed-action 序列。
- CEM：`200` candidates、`30` elites、`10` rounds、model horizon `5`，即一次规划覆盖 `25` 个低层动作。
- 两臂共享相同 CEM normal samples、起点、目标、速度、渲染和环境确定性。
- 每个场景只做一次 episode-entry 规划；主窗口前 context updates、exploratory actions 和 TTT steps 均为 `0`。

## 预注册的有限候选任务

只允许按以下顺序最多测试两个候选。候选 A 通过即停止任务校准并进入独立 formal 合同；A 失败才允许运行 B。B 也失败则整个 PointMaze lag 任务族停止。

### 候选 A：原始 hard-goal

- 使用仓库 `goal_source=hard` 的 distant-cell sampling 和原始宽 qvel 分布。
- 4 个 development factors，每个 factor `8` 个独立 paired scenarios，共 `32` 个 sequence-level pairs。
- scenario seed：`120000 + 1000 * factor_index + local_index`。
- 主指标：前 25 个真实低层动作的 position-distance AUC，记为 `auc_k25`，越低越好。
- 辅助：`auc_k5`、`cost_k5/10/25`、action-plan difference。

### 候选 B：local waypoint early-time cost

- 仅在候选 A 失败后运行。
- 在 Medium Maze free-cell graph 中采样 BFS 距离严格为 `2` 的 start/goal cells；位置 jitter 固定为 `[-0.10, 0.10]`，初始和目标速度都为 `0`。
- 4 个 development factors，每个 factor `8` 个独立 paired scenarios，共 `32` 个 sequence-level pairs。
- scenario seed：`220000 + 1000 * factor_index + local_index`。
- 主指标：前 10 个真实低层动作的 position-distance AUC，记为 `auc_k10`，越低越好。
- 现实含义：局部移动命令需要在短控制窗口内生效；lag 引起的等待时间直接增加早期任务成本，而不是删除正常观测或更新机会。
- 辅助：`auc_k5/25`、`cost_k5/10/25`、success_k25、action-plan difference。

## 每个候选的 GO 门

以下必须全部满足：

1. raw artifact 数量、factor、seed、start/goal/velocity、CEM RNG 和动作预算身份审计通过；
2. true context 在 `100%` pairs 上改变 CEM action plan；
3. 主指标 `population - true` 的 paired bootstrap 95% CI 下界严格大于 `0`；
4. 主指标相对改善 `(population - true) / population >= 25%`；
5. 至少 `70%` independent pairs 同向改善，ties 不计。

bootstrap 以 scenario pair 为最小统计单位，`20,000` 次有放回重采样，seed `5401`。

## 停止、修复和下游授权

- A、B 都失败：判定 `NO_GO_POINTMAZE_ACTUATOR_LAG_TASK`，不得执行 history/RLS、shuffled、wrong-sequence、no-persistence 或 neural context formal 矩阵。
- 任一候选通过：只授权另写一份结果未知时冻结的 Stage 1/2 formal 合同；本设计不预先授权 learned memory 或 episodic TTT。
- 只允许修复阻止执行、且不改变 factor、候选、seed、预算、指标和门槛的工程 bug；失败日志必须保留。
- 禁止依据 factor subgroup、辅助指标或单个 seed 改判或产生第三候选。

## 预期审计产物

- frozen JSON manifest 和 SHA256；
- 每 pair 的 command、executed action、state、factor hash、CEM trace、cost；
- checkpoint/code hash、命令、环境、GPU samples 与 Torch peak memory；
- 从 raw JSONL 独立复算的 candidate summary 和最终机器判定。

