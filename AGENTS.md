# AdaJEPA Project Guidance

## Current source of truth

The active research governance is `任务说明/persistent_context_v2_research_prompt_zh.md`. The evidence snapshot is `任务说明/Persistent_Context_V2_实验全景与后续路线图.md`, and the current execution order is `任务说明/Persistent_Context_V2_后续实验计划_2026-08-23.md`.

Historical continual-TTT prompts, phase records, frozen contracts, decisions, negative results, invalidation notes, repairs, and raw artifacts remain evidence. They are not instructions to restart a terminated route or to alter an old contract.

## Scientific scope

- Preserve the original episodic AdaJEPA behavior and P0A/Phase A–H artifacts.
- Do not extend unconditional weight carry merely by changing thresholds, factors, horizons, or metrics.
- For a new factor or environment, establish the chain `behavioral oracle value -> history identifiability -> persistence-specific closed-loop value -> non-privileged estimator -> learned model only if needed`.
- Current priority is matrix factor-plus-task-interaction gating, then a non-privileged delay estimator. CoG history inference remains deferred until an enriched Markov/contact-state predictor demonstrates closed-loop true-CoG value.
- Treat weights, optimizer, episode replay, sequence context, sufficient statistics, and RNG as separate states with explicit lifetimes.

## Evidence and decisions

- Freeze a contract before each new formal result: hypothesis, sole treatment, splits, unit, controls, budgets, primary estimand, statistics, invalidity/repair rules, and forbidden follow-ups.
- A sequence is the independent unit unless the contract justifies another unit. Never count steps or transitions as independent samples.
- Report effect direction, magnitude, interval, positive/tie/negative counts, heterogeneity, negative controls, and raw-artifact validity.
- Do not use a fixed effect percentage, sign fraction, or whether a confidence interval crosses zero as an automatic GO/NO-GO rule. The user decides further investment from the full continuous evidence.
- `INVALID` is only for an implementation, identity, budget, or audit failure. An unfavorable scientific result is not invalid.
- Prediction loss, parameter change, acceptance rate, factor accuracy, or a local positive seed cannot replace the pre-registered closed-loop endpoint.
- Keep facts, scoped inference, alternative explanations, and untested claims separate.

## Reproducibility

- The remote experiment tree is currently a dirty, largely untracked worktree. A commit hash alone is insufficient. Every new formal run must record the base commit, binary diff hash, participating untracked-source hashes, checkpoint/data hashes, resolved command/config, environment, and dirty status.
- Use ordered, versioned manifests. Paired policies must share factor, initial state, goal, nuisance, environment/CEM seeds, observation/action/update budgets, and evaluation order.
- Formal data is read once after train/dev choices and model hashes are locked. Existing formal results may be used for explicitly labeled exploration but never recycled as a new formal split.
- Evaluation and counterfactual probes are read-only: no optimizer step, replay insertion, running-stat update, memory consolidation, or hidden RNG consumption.
- Preserve failed runs, protocol-invalid batches, negative transfer, repairs, and audits. Never overwrite a successful or failed artifact directory.

## GPU policy

- One or two GPUs are pre-approved. Inspect `nvidia-smi`, use only idle devices, set `CUDA_VISIBLE_DEVICES`/device arguments explicitly, and record allocation rationale.
- Ask before using three or more GPUs. Never expand GPU count automatically after OOM or slowdown.
- Run CPU/unit tests and a smallest single-GPU smoke before formal collection.
- Record physical GPU IDs/models, CUDA/PyTorch, wall time, exit status, periodic `nvidia-smi`, and framework peak allocated/reserved memory.

## Editing and verification

- Preserve unrelated user changes; inspect local and remote status before editing.
- Prefer small patches and explicit configuration. Add state-isolation tests when changing reset, optimizer, replay, context, manifest, or gate logic.
- Do not delete frozen contracts, reports, raw results, or historical negative evidence. Remove only redundant or misleading non-evidence documents after verifying they are not required by the audit chain.
