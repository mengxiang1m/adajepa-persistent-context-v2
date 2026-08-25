#!/usr/bin/env python3
"""Independent CPU audit of cross-shape D3 features and nested-group analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path):
    d=hashlib.sha256()
    with Path(path).open("rb") as h:
        for c in iter(lambda:h.read(1024*1024),b""): d.update(c)
    return d.hexdigest()


def read_jsonl(path): return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def cosine(a,b):
    a=np.asarray(a,dtype=np.float64); b=np.asarray(b,dtype=np.float64); d=float(np.linalg.norm(a)*np.linalg.norm(b)); return float(a@b/d) if d>1e-12 else 0.


def angle(a,b):
    d=abs(float(a)-float(b))%(2*np.pi); return min(d,2*np.pi-d)


def replay_features(original, policy, cross):
    z=np.asarray(original["e1"]["correct"]["posterior"]["mean_z"]); gain=np.linalg.norm(z); rotation=math.degrees(math.atan2(z[1],z[0]))
    g=(gain-cross["gain_center"])/cross["gain_scale"]; r=rotation/cross["rotation_scale_degrees"]
    base=np.asarray([1.,g,r,g*g,g*r,r*r]); alpha=float(policy["target_alpha"])
    pop=original["correct_treatments"]["0.0"]; target=original["correct_treatments"][str(alpha)]
    initial=np.asarray(pop["initial_state"]); goal=np.asarray(pop["goal_state"]); pc=np.asarray(pop["commands"]); tc=np.asarray(target["commands"])
    ab=initial[2:4]-initial[0:2]; bg=goal[2:4]-initial[2:4]; pn=np.linalg.norm(pc,axis=1); tn=np.linalg.norm(tc,axis=1)
    pv=np.linalg.norm(np.diff(pc,axis=0),axis=1).mean(); tv=np.linalg.norm(np.diff(tc,axis=0),axis=1).mean()
    geo=np.asarray([np.linalg.norm(ab),np.linalg.norm(bg),cosine(ab,bg),angle(initial[4],goal[4]),np.sqrt(np.mean((tc-pc)**2)),
                    1-cosine(pc[0],tc[0]),math.log((float(tn.mean())+1e-8)/(float(pn.mean())+1e-8)),math.log((float(tv)+1e-8)/(float(pv)+1e-8))])
    s=policy["model_scores"]; pp,pt,tp,tt=[float(s[k]) for k in ("J_prior_a_prior","J_prior_a_context","J_context_a_prior","J_context_a_context")]
    return np.concatenate([base,geo,np.asarray([pt-pp,tp-tt,tp-pp,tt-pt])])


def fit(rows,count,ridge):
    x=np.asarray([r["x"][:count] for r in rows]); y=np.asarray([r["y"] for r in rows]); mean=x.mean(0); scale=x.std(0); mean[0]=0.; scale[0]=1.; scale[scale<1e-12]=1.
    z=(x-mean)/scale; p=np.eye(count); p[0,0]=0.; return mean,scale,np.linalg.solve(z.T@z+ridge*p,z.T@y)


def predict(model,rows,count):
    mean,scale,beta=model; x=np.asarray([r["x"][:count] for r in rows]); return ((x-mean[:count])/scale[:count])@beta


def nested(rows,count,ridges,field):
    pred=np.zeros(len(rows)); decision=np.zeros(len(rows),dtype=bool); chosen={}
    for outer in sorted({r[field] for r in rows}):
        train=[r for r in rows if r[field]!=outer]; ids=[i for i,r in enumerate(rows) if r[field]==outer]; test=[rows[i] for i in ids]; candidates=[]
        for ridge in ridges:
            values=[]
            for inner in sorted({r[field] for r in train}):
                a=[r for r in train if r[field]!=inner]; b=[r for r in train if r[field]==inner]; q=predict(fit(a,count,float(ridge)),b,count)
                values.extend([r["y"] if v>0 else 0. for r,v in zip(b,q)])
            candidates.append((float(np.mean(values)),float(ridge)))
        ridge=max(candidates,key=lambda x:(x[0],x[1]))[1]; chosen[str(outer)]=ridge; q=predict(fit(train,count,ridge),test,count); pred[ids]=q; decision[ids]=q>0
    return pred,decision,chosen


def bootstrap(values,design,stream):
    values=np.asarray(values); rng=np.random.default_rng(int(design["bootstrap_seed"])+stream); ix=rng.integers(0,len(values),(int(design["bootstrap_resamples"]),len(values)))
    return np.quantile(values[ix].mean(1),[.025,.975]).tolist()


def expected_analysis(feature_rows,design):
    result={"analysis_id":design["analysis_id"],"evidence_level":design["evidence_level"],"targets":{}}; stream=0
    for target in design["targets"]:
        rows=[{"x":np.asarray(r["policies"][target]["features"]),"y":float(r["policies"][target]["benefit"]),"shape_pair":int(r["shape_pair_index"]),"factor":int(r["factor_index"])} for r in feature_rows]
        y=np.asarray([r["y"] for r in rows]); tr={"base_mean_delta":float(y.mean()),"base_harm_fraction":float(np.mean(y<-1e-12)),"evaluations":{}}
        for evaluation,field in (("leave_one_shape_pair_out","shape_pair"),("leave_one_factor_out","factor")):
            models={}
            for name,count in design["feature_sets"].items():
                pred,dec,ridges=nested(rows,int(count),design["ridge_alphas"],field); delta=np.where(dec,y,0.); improvement=delta-y
                corr=float(np.corrcoef(pred,y)[0,1]) if np.std(pred)>1e-12 and np.std(y)>1e-12 else 0.; stream+=1
                models[name]={"prediction_correlation":corr,"selection_rate":float(dec.mean()),"mean_delta":float(delta.mean()),"bootstrap_ci95_delta":bootstrap(delta,design,stream),
                    "harm_fraction":float(np.mean(delta<-1e-12)),"positive_fraction":float(np.mean(delta>1e-12)),"mean_improvement_vs_target":float(improvement.mean()),
                    "bootstrap_ci95_improvement_vs_target":bootstrap(improvement,design,stream+100),"chosen_ridge_by_outer_group":ridges,
                    "out_of_group_predictions":pred.tolist(),"out_of_group_decisions":dec.astype(int).tolist(),"unit_deltas":delta.tolist()}
            tr["evaluations"][evaluation]=models
        result["targets"][target]=tr
    return result


def compare(a,b):
    error=0.; failures=[]
    def visit(x,y,path):
        nonlocal error
        if isinstance(x,dict):
            if not isinstance(y,dict) or set(x)!=set(y): failures.append(path+" keys"); return
            for k in x: visit(x[k],y[k],path+"."+k)
        elif isinstance(x,list):
            if not isinstance(y,list) or len(x)!=len(y): failures.append(path+" length"); return
            for i,(u,v) in enumerate(zip(x,y)): visit(u,v,f"{path}[{i}]")
        elif isinstance(x,(int,float)) and not isinstance(x,bool): error=max(error,abs(float(x)-float(y)))
        elif x!=y: failures.append(path+" value")
    visit(a,b,"analysis"); return error,failures


def main():
    p=argparse.ArgumentParser(); p.add_argument("--d3-dir",type=Path,required=True); p.add_argument("--formal-raw",type=Path,required=True); p.add_argument("--design",type=Path,required=True); p.add_argument("--cross-design",type=Path,required=True); p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    design=json.loads(a.design.read_text(encoding="utf-8")); cross=json.loads(a.cross_design.read_text(encoding="utf-8")); manifest=json.loads((a.d3_dir/"manifest.json").read_text(encoding="utf-8"))
    original=[r for r in read_jsonl(a.formal_raw) if r.get("record_type")=="cross_shape_sequence"]; features=[r for r in read_jsonl(a.d3_dir/"features.jsonl") if r.get("record_type")=="cross_shape_harm_feature"]
    observed=json.loads((a.d3_dir/"analysis.json").read_text(encoding="utf-8")); lookup={int(r["sequence_id"]):r for r in original}; failures=[]; feature_error=outcome_error=0.
    if sha256(a.formal_raw)!=manifest.get("input_raw_sha256") or sha256(a.d3_dir/"features.jsonl")!=manifest.get("features_sha256"): failures.append("hash identity")
    if not manifest.get("model_state_unchanged") or not manifest.get("rng_unchanged") or max(manifest.get("initial_replay_max_abs",1),manifest.get("goal_replay_max_abs",1))>1e-6: failures.append("read-only identity")
    if len(features)!=96 or len(lookup)!=96: failures.append("row count")
    for row in features:
        original_row=lookup[int(row["sequence_id"])]
        for target in design["targets"]:
            policy=row["policies"][target]; replay=replay_features(original_row,policy,cross); feature_error=max(feature_error,float(np.max(np.abs(replay-np.asarray(policy["features"])))))
            pop=float(original_row["correct_treatments"]["0.0"]["metrics"]["pose_auc10"]); cost=float(original_row["correct_treatments"][str(float(policy["target_alpha"]))]["metrics"]["pose_auc10"])
            outcome_error=max(outcome_error,abs((pop-cost)-float(policy["benefit"])),abs(pop-float(policy["population_cost"])),abs(cost-float(policy["target_cost"])))
    if feature_error>1e-12: failures.append("feature replay")
    if outcome_error>1e-12: failures.append("outcome replay")
    expected=expected_analysis(features,design); analysis_error,analysis_failures=compare(expected,observed); failures.extend(analysis_failures)
    if analysis_error>1e-12: failures.append("analysis replay")
    result={"valid":not failures,"failures":failures,"evidence_level":design["evidence_level"],"rows":len(features),"feature_replay_max_abs":feature_error,
            "outcome_replay_max_abs":outcome_error,"analysis_replay_max_abs":analysis_error,"features_sha256":sha256(a.d3_dir/"features.jsonl"),"analysis_sha256":sha256(a.d3_dir/"analysis.json"),"source_snapshot_sha256":manifest.get("source_snapshot_sha256")}
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["valid"] else 1)


if __name__=="__main__": main()
