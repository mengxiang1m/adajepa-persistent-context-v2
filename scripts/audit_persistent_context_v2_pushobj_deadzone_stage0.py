#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.persistent_context_v2.pushobj_deadzone_stage0 import CONTRACT_ID, EXPECTED_DESIGN_SHA256, SEGMENT_INDICES, array_sha256, summarize
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import deadline_success
from research.persistent_context_v2.pushobj_rotation_stage0 import pose_metrics, read_jsonl, sha256


def close(a,b): return bool(np.isclose(float(a),float(b),rtol=1e-8,atol=1e-9))


def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("repro_outputs/persistent_context_v2_pushobj_deadzone_stage0"));a=p.parse_args()
    m=json.loads((a.output_dir/"manifest.json").read_text(encoding="utf-8"));runner=json.loads((a.output_dir/"runner_summary.json").read_text(encoding="utf-8"));rows=[r for r in read_jsonl(a.output_dir/"raw.jsonl") if r.get("record_type")=="paired_deadzone"]
    fail={k:[] for k in ("manifest","count","scenario","metric","hash","pairing")}
    if m.get("contract_id")!=CONTRACT_ID or m.get("design_sha256")!=EXPECTED_DESIGN_SHA256 or sha256(Path(m["design_path"]))!=EXPECTED_DESIGN_SHA256:fail["manifest"].append("contract_or_design")
    if max(m.get("identity_audit",{}).values(),default=np.inf)>1e-6:fail["manifest"].append("identity")
    if len(rows)!=32 or len({r["ordinal"] for r in rows})!=32:fail["count"].append("rows")
    for r in rows:
        o=int(r["ordinal"])
        if r["segment_index"]!=SEGMENT_INDICES[o] or r["nominal_block_displacement_at_10"]<10:fail["scenario"].append(str(o))
        for policy in ("prior","oracle"):
            x=r[policy];metrics=pose_metrics(np.asarray(x["states"]),np.asarray(r["goal_state"]),10)
            if any(not close(metrics[k],x["metrics"][k]) for k in metrics) or bool(x["deadline_success"])!=deadline_success(np.asarray(x["states"]),np.asarray(r["goal_state"])):fail["metric"].append(f"{o}:{policy}")
            if x["command_sha256"]!=array_sha256(np.asarray(x["commands"],dtype=np.float32)) or x["state_sha256"]!=array_sha256(np.asarray(x["states"],dtype=np.float32)):fail["hash"].append(f"{o}:{policy}")
        if len(r["prior"]["planner"]["trace"])!=len(r["oracle"]["planner"]["trace"]):fail["pairing"].append(str(o))
    recomputed=summarize(a.output_dir/"raw.jsonl",m["identity_audit"]);checks={"raw_recomputes_runner_summary_exactly":recomputed==runner,"all_failure_counts_zero":all(not v for v in fail.values()),"runner_valid":bool(recomputed.get("valid"))}
    result={"schema":"persistent-context-v2-pushobj-deadzone-stage0-audit-v1","passed":all(checks.values()),"checks":checks,"failure_counts":{k:len(v) for k,v in fail.items()},"failure_examples":{k:v[:20] for k,v in fail.items() if v},"recomputed_summary":recomputed}
    (a.output_dir/"independent_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0 if result["passed"] else 1


if __name__=="__main__":raise SystemExit(main())
