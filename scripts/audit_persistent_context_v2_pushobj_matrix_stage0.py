#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.persistent_context_v2.pushobj_matrix_stage0 import CONTRACT_ID, EXPECTED_DESIGN_SHA256, FACTORS, POPULATION_PRIOR_MATRIX, SEGMENT_INDICES, apply_action_matrix, array_sha256, deadline_success, pose_metrics, read_jsonl, sha256, summarize


def close(a, b):
    return bool(np.isclose(float(a), float(b), rtol=1e-8, atol=1e-9))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",type=Path,default=Path("repro_outputs/persistent_context_v2_pushobj_matrix_stage0")); args=parser.parse_args()
    manifest=json.loads((args.output_dir/"manifest.json").read_text(encoding="utf-8")); runner=json.loads((args.output_dir/"runner_summary.json").read_text(encoding="utf-8")); rows=[row for row in read_jsonl(args.output_dir/"raw.jsonl") if row.get("record_type")=="paired_matrix"]
    failures={key:[] for key in ("manifest","count","scenario","metric","matrix","hash","pairing")}
    if manifest.get("contract_id")!=CONTRACT_ID or manifest.get("design_sha256")!=EXPECTED_DESIGN_SHA256 or sha256(Path(manifest["design_path"]))!=EXPECTED_DESIGN_SHA256: failures["manifest"].append("contract_or_design")
    if max(manifest.get("identity_audit",{}).values(),default=np.inf)>1e-6: failures["manifest"].append("identity")
    if len(rows)!=32 or len({row["ordinal"] for row in rows})!=32: failures["count"].append("rows")
    factor_counts={index:0 for index in range(len(FACTORS))}
    for row in rows:
        ordinal=int(row["ordinal"]); factor_index=int(row["factor_index"]); factor_counts[factor_index]=factor_counts.get(factor_index,0)+1
        if ordinal>=32 or row["segment_index"]!=SEGMENT_INDICES[ordinal] or row["nominal_block_displacement_at_10"]<10: failures["scenario"].append(str(ordinal))
        for policy in ("prior","oracle"):
            payload=row[policy]; metrics=pose_metrics(np.asarray(payload["states"]),np.asarray(row["goal_state"]),10)
            if any(not close(metrics[key],payload["metrics"][key]) for key in metrics) or bool(payload["deadline_success"])!=deadline_success(np.asarray(payload["states"]),np.asarray(row["goal_state"])): failures["metric"].append(f"{ordinal}:{policy}")
            commands=np.asarray(payload["commands"],dtype=np.float32); effective=np.asarray(payload["effective_actions"],dtype=np.float32); expected=apply_action_matrix(commands,np.asarray(row["true_matrix"],dtype=np.float32)).astype(np.float32)
            if not np.array_equal(effective,expected): failures["matrix"].append(f"{ordinal}:{policy}")
            if payload["command_sha256"]!=array_sha256(commands) or payload["effective_action_sha256"]!=array_sha256(effective) or payload["state_sha256"]!=array_sha256(np.asarray(payload["states"],dtype=np.float32)): failures["hash"].append(f"{ordinal}:{policy}")
        if not np.allclose(row["prior"]["context_matrix"],POPULATION_PRIOR_MATRIX,rtol=0,atol=1e-12) or not np.allclose(row["oracle"]["context_matrix"],row["true_matrix"],rtol=0,atol=1e-7): failures["pairing"].append(f"{ordinal}:context")
        if len(row["prior"]["planner"]["trace"])!=10 or len(row["oracle"]["planner"]["trace"])!=10: failures["pairing"].append(f"{ordinal}:budget")
    if any(value!=4 for value in factor_counts.values()): failures["scenario"].append("factor_balance")
    recomputed=summarize(args.output_dir/"raw.jsonl",manifest["identity_audit"]); checks={"raw_recomputes_runner_summary_exactly":recomputed==runner,"all_failure_counts_zero":all(not values for values in failures.values()),"runner_valid":bool(recomputed.get("valid"))}; result={"schema":"persistent-context-v2-pushobj-matrix-stage0-audit-v1","passed":all(checks.values()),"checks":checks,"failure_counts":{key:len(value) for key,value in failures.items()},"failure_examples":{key:value[:20] for key,value in failures.items() if value},"recomputed_summary":recomputed}
    (args.output_dir/"independent_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2,sort_keys=True)); return 0 if result["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
