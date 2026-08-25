#!/usr/bin/env python3
"""Independent pre-formal audit of train/dev model locking."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def features(row, design):
    z = np.asarray(row["posterior"]["mean_z"], dtype=np.float64)
    gain = float(np.linalg.norm(z)); rotation = math.degrees(math.atan2(z[1], z[0]))
    g = (gain - float(design["gain_center"])) / float(design["gain_scale"]); r = rotation / float(design["rotation_scale_degrees"])
    return np.asarray([1.0, g, r, g*g, g*r, r*r])


def rows(path, design):
    result=[]
    for row in read_jsonl(path):
        if row.get("record_type") != "f0_soft_policy_sequence": continue
        costs=np.asarray([row["treatments"][str(a)]["metrics"]["pose_auc10"] for a in design["alphas"]])
        result.append({"x":features(row,design),"costs":costs})
    return result


def fit(data, ridge):
    x=np.stack([r["x"] for r in data]); mean=np.zeros(6); scale=np.ones(6); mean[1:]=x[:,1:].mean(0); scale[1:]=x[:,1:].std(0); scale[1:][scale[1:]<1e-12]=1
    x=(x-mean)/scale; a0=np.asarray([.25,.5,.75,1.]); xx=np.repeat(x,4,0); a=np.tile(a0,len(data)); phi=np.concatenate([a[:,None]*xx,(a*(1-a))[:,None]*xx],1); y=np.concatenate([r["costs"][0]-r["costs"][1:] for r in data])
    penalty=np.eye(12); penalty[0,0]=penalty[6,6]=0; beta=np.linalg.solve(phi.T@phi+ridge*penalty,phi.T@y)
    return mean,scale,beta


def predict(model,data,alphas):
    mean,scale,beta=model; x=(np.stack([r["x"] for r in data])-mean)/scale; out=np.zeros((len(data),len(alphas)))
    for i,a in enumerate(alphas): out[:,i]=np.concatenate([a*x,a*(1-a)*x],1)@beta
    return out


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--design",type=Path,required=True); p.add_argument("--locked-model",type=Path,required=True); p.add_argument("--audit-output",type=Path,required=True); a=p.parse_args()
    design=json.loads(a.design.read_text(encoding="utf-8")); locked=json.loads(a.locked_model.read_text(encoding="utf-8")); train=rows(a.output_dir/"train/raw.jsonl",design); dev=rows(a.output_dir/"dev/raw.jsonl",design)
    failures=[]; candidates=[]
    if len(train)!=64 or len(dev)!=32: failures.append("incomplete train/dev")
    for ridge in design["ridge_alphas"]:
        model=fit(train,float(ridge)); pred=predict(model,dev,design["alphas"]); selected=np.argmax(pred,1); costs=np.stack([r["costs"] for r in dev]); candidates.append((float(np.mean(costs[:,0]-costs[np.arange(32),selected])),float(ridge)))
    chosen=max(candidates,key=lambda x:(x[0],x[1]))[1]; model=fit(train+dev,chosen)
    beta_error=float(np.max(np.abs(model[2]-np.asarray(locked["beta"]))))
    mean_error=float(np.max(np.abs(model[0]-np.asarray(locked["mean"]))))
    scale_error=float(np.max(np.abs(model[1]-np.asarray(locked["scale"]))))
    if chosen!=float(locked["selected_ridge_alpha"]): failures.append("ridge mismatch")
    if max(beta_error,mean_error,scale_error)>1e-12: failures.append("model replay mismatch")
    for key,path in (("train_raw_sha256",a.output_dir/"train/raw.jsonl"),("dev_raw_sha256",a.output_dir/"dev/raw.jsonl"),("design_sha256",a.design)):
        if locked[key]!=sha256(path): failures.append(f"hash mismatch {key}")
    result={"valid":not failures,"failures":failures,"selected_ridge_alpha":chosen,"beta_max_abs":beta_error,"mean_max_abs":mean_error,"scale_max_abs":scale_error,"locked_model_sha256":sha256(a.locked_model),"train_rows":len(train),"dev_rows":len(dev)}
    a.audit_output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8"); print(json.dumps(result,indent=2)); raise SystemExit(0 if result["valid"] else 1)


if __name__=="__main__": main()
