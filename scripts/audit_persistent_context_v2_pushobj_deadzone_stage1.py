#!/usr/bin/env python3
import argparse,json,pickle,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from research.persistent_context_v2.pushobj_deadzone_stage1 import CONDITIONS,CONTRACT_ID,EXPECTED_DESIGN_SHA256,POLICIES,array_sha256,build_estimator,scenario,summarize
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import deadline_success
from research.persistent_context_v2.pushobj_rotation_stage0 import pose_metrics,read_jsonl,sha256

def close(a,b):return bool(np.isclose(float(a),float(b),rtol=1e-8,atol=1e-8))
def main():
 p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,default=Path("repro_outputs/persistent_context_v2_pushobj_deadzone_stage1"));a=p.parse_args();m=json.loads((a.output_dir/"manifest.json").read_text(encoding="utf-8"));runner=json.loads((a.output_dir/"runner_summary.json").read_text(encoding="utf-8"));n,e=m["sequences"],m["episodes"]
 with Path(m["data"]).open("rb") as h:segments=pickle.load(h)["segments"]
 fail={k:[] for k in ("manifest","count","scenario","estimator","donor","metric","hash","identity")}
 if m.get("contract_id")!=CONTRACT_ID or m.get("design_sha256")!=EXPECTED_DESIGN_SHA256 or sha256(Path(m["design_path"]))!=EXPECTED_DESIGN_SHA256:fail["manifest"].append("contract_or_design")
 if max(m.get("identity_audit",{}).values(),default=np.inf)>1e-6:fail["manifest"].append("identity")
 for c in CONDITIONS:
  raw=read_jsonl(a.output_dir/f"{c}_raw.jsonl");er=[r for r in raw if r.get("record_type")=="evidence_episode"];vr=[r for r in raw if r.get("record_type")=="evaluation_episode"];ev={(r["sequence_id"],r["episode_index"]):r for r in er};vv={(r["sequence_id"],r["episode_index"]):r for r in vr}
  if len(ev)!=n*e or len(vv)!=n*e:fail["count"].append(c)
  for s in range(n):
   for j in range(e):
    if (s,j) not in ev or (s,j) not in vv:continue
    expected=scenario(segments,c,s,j,n,e);x,y=ev[(s,j)],vv[(s,j)]
    if x["segment_index"]!=expected["segment_index"] or y["segment_index"]!=expected["segment_index"] or x["factor"]!=expected["factor"] or y["factor"]!=expected["factor"]:fail["scenario"].append(f"{c}:{s}:{j}")
    if x["evidence_sha256"]!=array_sha256(np.asarray(x["commands"],dtype=np.float32))+":"+array_sha256(np.asarray(x["states"],dtype=np.float32)):fail["hash"].append(f"{c}:{s}:{j}:evidence")
    for policy in ("correct_history","shuffled_history","wrong_sequence_history"):
     est,donors=build_estimator(ev,policy,s,j,n);saved=y["policies"][policy]
     if not close(est.estimate,saved["context"]):fail["estimator"].append(f"{c}:{s}:{j}:{policy}")
     if donors!=saved["donors"]:fail["donor"].append(f"{c}:{s}:{j}:{policy}")
    if y["policies"]["current_only"]["state_sha256"]!=y["policies"]["population_prior"]["state_sha256"]:fail["identity"].append(f"{c}:{s}:{j}:current")
    if j==0 and len({(y["policies"][q]["command_sha256"],y["policies"][q]["state_sha256"]) for q in POLICIES if q!="true_factor_oracle"})!=1:fail["identity"].append(f"{c}:{s}:e1")
    for policy in POLICIES:
     z=y["policies"][policy];metrics=pose_metrics(np.asarray(z["states"]),np.asarray(y["goal_state"]),10)
     if any(not close(metrics[k],z["metrics"][k]) for k in metrics) or bool(z["deadline_success"])!=deadline_success(np.asarray(z["states"]),np.asarray(y["goal_state"])):fail["metric"].append(f"{c}:{s}:{j}:{policy}")
     if z["command_sha256"]!=array_sha256(np.asarray(z["commands"],dtype=np.float32)) or z["state_sha256"]!=array_sha256(np.asarray(z["states"],dtype=np.float32)):fail["hash"].append(f"{c}:{s}:{j}:{policy}")
 recomputed=summarize(a.output_dir,n,e);checks={"raw_recomputes_runner_summary_exactly":recomputed==runner,"all_failure_counts_zero":all(not v for v in fail.values()),"runner_valid":bool(recomputed.get("valid"))};result={"schema":"persistent-context-v2-pushobj-deadzone-stage1-audit-v1","passed":all(checks.values()),"checks":checks,"failure_counts":{k:len(v) for k,v in fail.items()},"failure_examples":{k:v[:20] for k,v in fail.items() if v},"recomputed_summary":recomputed};(a.output_dir/"independent_audit.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(json.dumps(result,indent=2));return 0 if result["passed"] else 1
if __name__=="__main__":raise SystemExit(main())
