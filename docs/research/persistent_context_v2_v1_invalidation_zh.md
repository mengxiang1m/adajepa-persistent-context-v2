# Persistent Context V2 v1 条件配对无效声明

证据检查发现 `stage1_formal_v1` 与 `stage2_pilot_v1` 的 persistent/no-persistence 同序号 sequences 没有共享 target/noise RNG。每个条件内部的策略配对、E1 identity 和 raw hash 均有效，但跨条件 DiD 不满足 V2 prompt 的严格 nuisance/noise 配对要求。

因此这两个目录保留为失败产物，裁决为 `INVALID_EXECUTION`，不得作为最终证据。有限修复只改变 RNG stream 的配对方式，使两个生成条件共享完全相同的 target/noise arrays；factor treatment、factor split、master seed、样本、策略、预算、指标、bootstrap 与门槛均保持不变。修复结果写入新的 `stage1_formal_v1_repair1` 与 `stage2_pilot_v1_repair1`，不覆盖 v1。
