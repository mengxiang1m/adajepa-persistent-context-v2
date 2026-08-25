# Persistent Context V2 Stage 3 结果：FiLM Context-Conditioned World Model

日期：2026-08-22（Asia/Shanghai）

## 裁决

`CONTEXT_CONDITIONED_WORLD_MODEL_SUPPORTED / GO_REAL_BENCHMARK_TRANSFER_CONTRACT`

这说明当前思路在合成 actuator world-model 层面可行：一个只含单一 FiLM context 接口的冻结动力学模型会真正使用 context 改变预测和动作排序；测试时不知 formal factor support 的 RLS `q(z_seq)` 能把过去 transition 转化为新 episode 的冷启动闭环收益。

这仍不是视觉 AdaJEPA 或真实 benchmark 的成功结果。它只授权下一步在 PointMaze/action-calibration wrapper 上重新冻结 transfer 合同。

## 做了什么

- train factors：`[0.50,0.65,0.80,0.95,1.05,1.20,1.35,1.50]`；32,768 transitions。
- development factors：`[0.575,0.725,0.875,1.125,1.275,1.425]`；8,192 prediction transitions 和 256 closed-loop scenarios。
- formal factors：`[0.6125,0.7625,0.9125,1.0875,1.2375,1.3875]`；dev 门通过前未生成 formal outcome。
- 模型只有 113 个参数：action projection、一个 FiLM scale/shift context 接口、response output；没有第二种 conditioning、TTT、LoRA、router、replay 或权重继承。
- AdamW 固定训练 2,000 steps；checkpoint 只按固定 dev true-context prediction MSE 选择，best step 为 900。
- 所有闭环策略共享同一冻结 checkpoint，每个动作固定比较 401 个候选。

## Development model-use gate

| 指标 | population context | true context |
|---|---:|---:|
| held-out prediction MSE | 0.067940 | 0.000224 |
| closed-loop mean cost | 1.920976 | 0.004389 |
| unsafe fraction | 56.25% | 0.00% |

- prediction MSE ratio：0.00330；
- behavior relative improvement：99.77%，bootstrap 95% CI `[99.72%,99.82%]`；
- true/population planner action change：100%；
- 五项 development 门全部通过，随后才生成 formal outcomes。

## Formal closed-loop evidence

persistent/no-persistence 各 384 sequences，每条 8 episodes，primary window 为 E2–E8 第一次动作；统计单位为 sequence，bootstrap 20,000 次。

| persistent 策略 | later mean cost |
|---|---:|
| population/current-only context | 1.324355 |
| persistent RLS context | 0.008485 |
| true context（冻结 FiLM 模型） | 0.005946 |
| analytic true-factor oracle | 0.005925 |
| shuffled RLS context | 3.263242 |
| wrong-sequence RLS context | 3.391706 |

主要效应：

- true context 相对 population 改善 99.55%，difference CI `[1.188428,1.448418]`；回收 99.998% analytic gap；
- persistent RLS 相对 current-only 改善 99.36%，difference CI `[1.187508,1.444967]`；384/384 sequences 同向；
- RLS 回收 99.81% true-context improvement；
- persistence DiD 为 1.882022，95% CI `[1.730231,2.038734]`；
- true-context 与 population 的 later action ranking 在 100% decisions 中不同。

负对照：

- no-persistence RLS relative improvement：−40.89%；
- shuffled-history：−146.40%；
- wrong-sequence-history：−156.10%。

formal 九项 GO 门全部通过。

## 独立证据检查

`independent_audit.json` 为 `passed: true`。审计器没有调用实验聚合代码来接受结论，而是独立完成：

- 重新加载 checkpoint 并验证 SHA-256；
- 验证 checkpoint 是 training curve 上最低 dev MSE 的 step 900；
- 重建 dev prediction set 和 development gate；
- 对全部 raw rows 重新评估 401 candidates，planner mismatch 为 0；
- 验证 43,008 条 formal rows、E1 identity、跨条件 target/noise 配对、persistent factor、donor 非 self 和相同预算；
- 从每条 policy 的过去 raw transitions 重建所有 RLS `context_before`；
- 重算 simulator response、task cost、sequence aggregation、bootstrap、DiD、负对照和九项裁决。

## 支持与不支持的结论

支持：

1. 训练后的 world model 确实使用了 context，不只是 latent 发生变化；context 改变了预测、动作排名和 simulator outcome。
2. 真实 factor label 只在训练和 true-context oracle 中使用；非特权 RLS context 几乎回收全部收益。
3. 收益具有 persistence-specific 因果特征，错误/打乱/非持续 context 都不能复制收益。
4. `history → q(z_seq) → prediction → action → outcome` 的完整链条在该任务中建立。

不支持：

- 该模型不是现有视觉 AdaJEPA checkpoint；
- task 是一维、线性、低噪声，并且 formal factors 是 train range 内插值；
- 未测试图像 encoder、接触动力学、非线性、多因素或 distribution extrapolation；
- 未证明 learned context encoder 优于三个标量的 RLS；
- 未证明 episode-local TTT 与 sequence context 可以安全组合。

最强替代解释是：这个 task 与 FiLM 模型都和标量乘法结构高度匹配，因此接近 ceiling 是预期的。该解释限制外推，但不能解释负对照的严格方向分离，也不能否定“context-conditioned model 能把正确历史变成行为收益”的任务内结论。

## 下一步

冻结一个独立的真实 benchmark transfer 合同：优先 PointMaze/action-calibration wrapper，只改变一个跨 episode 固定的 action gain/rotation；生成 factor-diverse train checkpoint；先做 true-context behavior upper bound，再接 RLS context 和 no-persistence/shuffled/wrong controls。该门通过前不加入 AdaJEPA episode-local TTT。

## 产物与资源

- 远端目录：`/data4/zhaoqing/adajepa/persistent_context_v2_outputs/stage3_film_v1/`
- raw formal rows：43,008；development rows：512。
- GPU：物理 GPU 0，NVIDIA L40；framework peak allocated 67,552,256 bytes，peak reserved 88,080,384 bytes。
- 总 wall time 32.29 s（模型内部计时 8.75 s）；最大 host RSS 867,160 KB；退出码 0。
