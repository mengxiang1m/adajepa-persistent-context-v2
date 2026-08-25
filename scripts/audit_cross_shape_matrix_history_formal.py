#!/usr/bin/env python3
"""Independent replay audit for cross-shape matrix-history formal results."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle: return [json.loads(line) for line in handle if line.strip()]


def segment_hash(segment: dict) -> str:
    digest = hashlib.sha256()
    for key in ("states", "actions"):
        value = np.asarray(segment[key]); digest.update(key.encode("ascii")); digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes()); digest.update(value.tobytes())
    return digest.hexdigest()


def posterior(observations) -> dict:
    samples = np.asarray([[gain * math.cos(math.radians(theta)), gain * math.sin(math.radians(theta))]
                          for theta in (-30., -15., 0., 15., 30.) for gain in (.75, 1., 1.25)])
    mean = samples.mean(0); covariance = np.cov(samples.T, bias=True) + 1e-4 * np.eye(2)
    precision = np.linalg.inv(covariance); information = precision @ mean
    obs = np.asarray(observations, dtype=np.float64).reshape(-1, 2)
    if len(obs): precision += len(obs) * 10000. * np.eye(2); information += 10000. * obs.sum(0)
    covariance = np.linalg.inv(precision); mean_z = np.linalg.solve(precision, information)
    return {"precision": precision, "information": information, "covariance": covariance, "mean_z": mean_z,
            "mean_matrix": np.asarray([[mean_z[0], -mean_z[1]], [mean_z[1], mean_z[0]]]), "observation_count": len(obs)}


def feature(value: dict, design: dict) -> np.ndarray:
    z = np.asarray(value["mean_z"]); gain = np.linalg.norm(z); rotation = math.degrees(math.atan2(z[1], z[0]))
    g = (gain - design["gain_center"]) / design["gain_scale"]; r = rotation / design["rotation_scale_degrees"]
    return np.asarray([1., g, r, g*g, g*r, r*r])


def prediction(model: dict, x: np.ndarray) -> np.ndarray:
    normalized = (x - np.asarray(model["mean"])) / np.asarray(model["scale"]); beta = np.asarray(model["beta"])
    return np.asarray([np.concatenate([a*normalized, a*(1-a)*normalized]) @ beta for a in model["alphas"]])


def metric(treatment: dict) -> float:
    states = np.asarray(treatment["states"], dtype=np.float64); goal = np.asarray(treatment["goal_state"], dtype=np.float64)
    post = states[1:11]; position = np.linalg.norm(post[:,2:4] - goal[None,2:4], axis=1)
    delta = np.abs(post[:,4] - goal[4]) % (2*np.pi); angle = np.minimum(delta, 2*np.pi-delta)
    return float(np.mean(position/20. + angle/(np.pi/9.)))


def bootstrap(values: np.ndarray, design: dict, stream: int) -> list[float]:
    rng=np.random.default_rng(int(design["bootstrap_seed"])+stream); ix=rng.integers(0,len(values),(int(design["bootstrap_resamples"]),len(values)))
    return np.quantile(values[ix].mean(1),[.025,.975]).tolist()


def contrast(values: np.ndarray, design: dict, stream: int) -> dict:
    values=np.asarray(values,dtype=np.float64)
    return {"mean":float(values.mean()),"bootstrap_ci95":bootstrap(values,design,stream),
            "positive_fraction":float(np.mean(values>1e-12)),"tie_fraction":float(np.mean(np.abs(values)<=1e-12)),
            "negative_fraction":float(np.mean(values<-1e-12)),"unit_deltas":values.tolist()}


def max_error(expected, observed) -> tuple[float, list[str]]:
    error, failures = 0., []
    def visit(a, b, path):
        nonlocal error
        if isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b): failures.append(path+" keys"); return
            for key in a: visit(a[key],b[key],path+"."+key)
        elif isinstance(a, list):
            if not isinstance(b,list) or len(a)!=len(b): failures.append(path+" length"); return
            for index,(x,y) in enumerate(zip(a,b)): visit(x,y,f"{path}[{index}]")
        elif isinstance(a,(int,float)) and not isinstance(a,bool):
            try: error=max(error,abs(float(a)-float(b)))
            except Exception: failures.append(path+" numeric")
        elif a != b: failures.append(path+" value")
    visit(expected,observed,"summary"); return error,failures


def audit(args) -> dict:
    design=json.loads(args.design.read_text(encoding="utf-8")); selection=json.loads(args.selection.read_text(encoding="utf-8"))
    model=json.loads(args.external_f0_model.read_text(encoding="utf-8")); observed_summary=json.loads(args.summary.read_text(encoding="utf-8"))
    pools={}
    for shape in design["shapes"]:
        path=Path(design["data_root"])/f"val_{shape}"/"plan_targets.pkl"
        with path.open("rb") as handle: pools[shape]=pickle.load(handle)["segments"]
    global_counts=Counter(segment_hash(segment) for values in pools.values() for segment in values)
    failures=[]; all_hashes=[]; all_provenance=[]
    for split in ("smoke","formal"):
        for expected_id,row in enumerate(selection[split]):
            if int(row["sequence_id"])!=expected_id: failures.append(f"{split} selection sequence")
            for episode in ("e1","e2"):
                item=row[episode]; segment=pools[item["shape"]][int(item["segment_index"])]; digest=segment_hash(segment)
                provenance=f"{int(segment['ep_idx'])}:{int(segment['offset'])}"; all_hashes.append(digest); all_provenance.append(provenance)
                states=np.asarray(segment["states"]); displacement=float(np.linalg.norm(states[10,2:4]-states[0,2:4]))
                if digest!=item["segment_sha256"] or provenance!=item["provenance_key"] or global_counts[digest]!=1 or displacement<10: failures.append(f"{split} selection identity")
    for shape,items in selection["reserve"].items():
        for item in items:
            segment=pools[shape][int(item["segment_index"])]; digest=segment_hash(segment); provenance=f"{int(segment['ep_idx'])}:{int(segment['offset'])}"
            all_hashes.append(digest); all_provenance.append(provenance)
            if digest!=item["segment_sha256"] or provenance!=item["provenance_key"] or global_counts[digest]!=1: failures.append("reserve identity")
    if len(all_hashes)!=228 or len(set(all_hashes))!=228 or len(set(all_provenance))!=228: failures.append("selection uniqueness")
    rows=[row for row in read_jsonl(args.output_dir/"formal/raw.jsonl") if row.get("record_type")=="cross_shape_sequence"]
    if len(rows)!=96: failures.append("formal row count")
    posterior_error=feature_error=prediction_error=context_error=metric_error=scene_error=0.; decision_mismatches=order_mismatches=0
    prior=np.eye(2)*float(design["gain_center"])
    for row in rows:
        selected=selection["formal"][int(row["sequence_id"])]
        if row["selection"]!=selected: failures.append("row selection mismatch")
        replayed={}
        for name in ("correct","no_persistence"):
            value=posterior(row["e1"][name]["matrix_observations"]); replayed[name]=value
            for key in ("precision","information","covariance","mean_z","mean_matrix"):
                posterior_error=max(posterior_error,float(np.max(np.abs(value[key]-np.asarray(row["e1"][name]["posterior"][key])))))
        x=feature(replayed["correct"],design); feature_error=max(feature_error,float(np.max(np.abs(x-np.asarray(row["features"])))))
        pred=prediction(model,x); prediction_error=max(prediction_error,float(np.max(np.abs(pred-np.asarray(row["external_f0_predicted_benefits"])))))
        selected_alpha=float(model["alphas"][int(np.argmax(pred))]); decision_mismatches+=int(selected_alpha!=float(row["external_f0_selected_alpha"]))
        expected_order=[selected_alpha]+[float(a) for a in design["correct_context_alphas"] if float(a)!=selected_alpha]
        order_mismatches+=int([float(a) for a in row["correct_execution_order"]]!=expected_order)
        starts=[]; goals=[]
        for alpha in expected_order:
            treatment=row["correct_treatments"][str(alpha)]; expected_context=(1-alpha)*prior+alpha*replayed["correct"]["mean_matrix"]
            context_error=max(context_error,float(np.max(np.abs(expected_context-np.asarray(treatment["context_matrix"])))))
            metric_error=max(metric_error,abs(metric(treatment)-float(treatment["metrics"]["pose_auc10"])))
            starts.append(np.asarray(treatment["initial_state"])); goals.append(np.asarray(treatment["goal_state"]))
        no=row["no_persistence_treatment"]; alpha=float(design["fixed_alpha"]); expected_context=(1-alpha)*prior+alpha*replayed["no_persistence"]["mean_matrix"]
        context_error=max(context_error,float(np.max(np.abs(expected_context-np.asarray(no["context_matrix"])))))
        metric_error=max(metric_error,abs(metric(no)-float(no["metrics"]["pose_auc10"])))
        starts.append(np.asarray(no["initial_state"])); goals.append(np.asarray(no["goal_state"]))
        scene_error=max(scene_error,max(float(np.max(np.abs(v-starts[0]))) for v in starts),max(float(np.max(np.abs(v-goals[0]))) for v in goals))
        order_mismatches+=int(int(row["correct_treatments"][str(selected_alpha)]["execution_index"])!=0)
        order_mismatches+=int(int(no["execution_index"])!=len(expected_order))
        order_mismatches+=int(float(row["decision_unix"])>min(float(t["execution_started_unix"]) for t in list(row["correct_treatments"].values())+[no]))
    if posterior_error>1e-12: failures.append("posterior replay")
    if feature_error>1e-12: failures.append("feature replay")
    if prediction_error>1e-12: failures.append("prediction replay")
    if context_error>1e-12: failures.append("context replay")
    if metric_error>1e-12: failures.append("metric replay")
    if scene_error>1e-12: failures.append("paired scene mismatch")
    if decision_mismatches: failures.append("decision mismatch")
    if order_mismatches: failures.append("execution order mismatch")
    def correct(row,a): return float(row["correct_treatments"][str(float(a))]["metrics"]["pose_auc10"])
    population=np.asarray([correct(r,0) for r in rows]); c05=np.asarray([correct(r,.5) for r in rows]); c075=np.asarray([correct(r,.75) for r in rows]); full=np.asarray([correct(r,1) for r in rows])
    no=np.asarray([r["no_persistence_treatment"]["metrics"]["pose_auc10"] for r in rows]); f0=np.asarray([correct(r,r["external_f0_selected_alpha"]) for r in rows]); best=np.minimum.reduce([population,c05,c075,full])
    primary,persistence=population-c075,no-c075
    def effect(cost,stream):
        delta=population-cost; value=contrast(delta,design,stream); value.update({"mean_cost":float(cost.mean()),"relative_improvement":float(delta.mean()/population.mean()),"harm_fraction":float(np.mean(delta<-1e-12))}); return value
    by_pair={}; by_factor={}
    for field,target,count in (("shape_pair_index",by_pair,6),("factor_index",by_factor,8)):
        ids=np.asarray([int(r[field]) for r in rows])
        for index in range(count):
            keep=ids==index; target[str(index)]={"n":int(keep.sum()),"primary_mean":float(primary[keep].mean()),"persistence_mean":float(persistence[keep].mean())}
    expected={"contract_id":design["contract_id"],"n_formal":96,"primary":contrast(primary,design,100),"persistence_specific":contrast(persistence,design,101),
              "policies":{"correct_fixed_alpha_0.5":effect(c05,1),"correct_fixed_alpha_0.75":effect(c075,2),"correct_full_context":effect(full,3),"external_T_F0":effect(f0,4),"no_persistence_alpha_0.75":effect(no,5),"per_sequence_best_fixed_grid_ceiling":effect(best,6)},
              "external_F0_vs_fixed_0.75":contrast(c075-f0,design,102),
              "external_f0_selection_counts":{str(float(a)):int(sum(float(r["external_f0_selected_alpha"])==float(a) for r in rows)) for a in rows[0]["external_f0_alphas"]},
              "by_shape_pair":by_pair,"by_factor":by_factor,"selected_branch_first_valid":True}
    summary_error,summary_failures=max_error(expected,observed_summary); failures.extend(summary_failures)
    if summary_error>1e-12: failures.append("summary numeric replay")
    manifest=json.loads((args.output_dir/"formal/manifest.json").read_text(encoding="utf-8"))
    if manifest.get("raw_sha256")!=sha256(args.output_dir/"formal/raw.jsonl") or not manifest.get("model_state_unchanged"): failures.append("formal manifest")
    result={"contract_id":design["contract_id"],"valid":not failures,"failures":failures,"row_count":len(rows),
            "posterior_replay_max_abs":posterior_error,"feature_replay_max_abs":feature_error,"prediction_replay_max_abs":prediction_error,
            "context_replay_max_abs":context_error,"metric_replay_max_abs":metric_error,"paired_scene_max_abs":scene_error,
            "decision_mismatches":decision_mismatches,"execution_order_mismatches":order_mismatches,"summary_replay_max_abs":summary_error,
            "formal_raw_sha256":sha256(args.output_dir/"formal/raw.jsonl"),"formal_summary_sha256":sha256(args.summary),"source_snapshot_sha256":manifest.get("source_snapshot_sha256")}
    args.audit_output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--design",type=Path,required=True); p.add_argument("--selection",type=Path,required=True)
    p.add_argument("--external-f0-model",type=Path,required=True); p.add_argument("--summary",type=Path,required=True); p.add_argument("--audit-output",type=Path,required=True)
    result=audit(p.parse_args()); raise SystemExit(0 if result["valid"] else 1)


if __name__=="__main__": main()
