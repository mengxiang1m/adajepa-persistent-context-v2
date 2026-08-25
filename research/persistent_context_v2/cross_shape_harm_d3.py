"""Read-only cross-shape harm attribution and low-capacity veto feasibility."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch

from research.persistent_context_v2.matrix_task_interaction_d0 import (
    fixed_action_goal_loss, model_interaction_features, model_sha256, rng_state_digest,
    safe_cosine, wrapped_angle_error,
)
from research.persistent_context_v2.pushobj_matrix_stage0 import POPULATION_PRIOR_MATRIX, load_matrix_world_model
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import prepare_waypoint
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl, dump_json, make_env, make_preprocessor, read_jsonl, resource_snapshot, sha256,
)


ANALYSIS_ID = "persistent-context-v2-cross-shape-harm-d3-exploratory-v1"
EXPECTED_DESIGN_SHA256 = "4b09aa03030abeece2ec267afbd4d62e7e4a460d81c5038e0cab0493de989e61"


def geometry(initial, goal, population_commands, target_commands) -> np.ndarray:
    initial=np.asarray(initial,dtype=np.float64); goal=np.asarray(goal,dtype=np.float64)
    population=np.asarray(population_commands,dtype=np.float64); target=np.asarray(target_commands,dtype=np.float64)
    agent_to_block=initial[2:4]-initial[0:2]; block_to_goal=goal[2:4]-initial[2:4]
    pn=np.linalg.norm(population,axis=1); tn=np.linalg.norm(target,axis=1)
    pv=np.linalg.norm(np.diff(population,axis=0),axis=1).mean(); tv=np.linalg.norm(np.diff(target,axis=0),axis=1).mean()
    return np.asarray([np.linalg.norm(agent_to_block),np.linalg.norm(block_to_goal),safe_cosine(agent_to_block,block_to_goal),
        wrapped_angle_error(initial[4],goal[4]),np.sqrt(np.mean((target-population)**2)),1-safe_cosine(population[0],target[0]),
        math.log((float(tn.mean())+1e-8)/(float(pn.mean())+1e-8)),math.log((float(tv)+1e-8)/(float(pv)+1e-8))])


def factor_features(row: dict, cross_design: dict) -> np.ndarray:
    z=np.asarray(row["e1"]["correct"]["posterior"]["mean_z"],dtype=np.float64); gain=np.linalg.norm(z); rotation=math.degrees(math.atan2(z[1],z[0]))
    g=(gain-cross_design["gain_center"])/cross_design["gain_scale"]; r=rotation/cross_design["rotation_scale_degrees"]
    return np.asarray([1.,g,r,g*g,g*r,r*r])


def extract(args, design: dict) -> dict:
    if args.output_dir.exists() and any(args.output_dir.iterdir()): raise FileExistsError(f"non-empty output: {args.output_dir}")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if sha256(args.input_raw)!=design["input_raw_sha256"]: raise RuntimeError("input raw hash mismatch")
    cross=json.loads(args.cross_design.read_text(encoding="utf-8")); device=torch.device(args.device)
    if device.type=="cuda": torch.cuda.set_device(device); torch.cuda.reset_peak_memory_stats(device.index or 0)
    pools={}
    for shape in cross["shapes"]:
        with (Path(cross["data_root"])/f"val_{shape}"/"plan_targets.pkl").open("rb") as handle: pools[shape]=pickle.load(handle)["segments"]
    _,wm,_=load_matrix_world_model(args.checkpoint,device); preprocessor,env=make_preprocessor(),make_env()
    before,rng_before=model_sha256(wm),rng_state_digest(); started=time.time(); max_initial=max_goal=0.; out=args.output_dir/"features.jsonl"
    rows=[row for row in read_jsonl(args.input_raw) if row.get("record_type")=="cross_shape_sequence"]
    for row in rows:
        py,npstate,tstate=random.getstate(),np.random.get_state(),torch.get_rng_state(); cuda_index=device.index or 0
        cstate=torch.cuda.get_rng_state(cuda_index) if device.type=="cuda" else None
        selected=row["selection"]["e2"]; segment=pools[selected["shape"]][int(selected["segment_index"])]
        initial,goal_obs,nominal,_=prepare_waypoint(env,segment,int(row["env_seed"])+1); start_obs,_=env.prepare(int(row["env_seed"])+1,initial)
        population=row["correct_treatments"]["0.0"]; max_initial=max(max_initial,float(np.max(np.abs(np.asarray(initial)-np.asarray(population["initial_state"])))))
        max_goal=max(max_goal,float(np.max(np.abs(np.asarray(nominal[-1])-np.asarray(population["goal_state"])))))
        base=factor_features(row,cross); policy_rows={}
        for target,alpha in (("fixed_alpha_0.75",.75),("external_T_F0",float(row["external_f0_selected_alpha"]))):
            treatment=row["correct_treatments"][str(float(alpha))]; context=np.asarray(treatment["context_matrix"],dtype=np.float64)
            scores={"J_prior_a_prior":fixed_action_goal_loss(wm,preprocessor,start_obs,goal_obs,population["commands"],POPULATION_PRIOR_MATRIX),
                    "J_prior_a_context":fixed_action_goal_loss(wm,preprocessor,start_obs,goal_obs,treatment["commands"],POPULATION_PRIOR_MATRIX),
                    "J_context_a_prior":fixed_action_goal_loss(wm,preprocessor,start_obs,goal_obs,population["commands"],context),
                    "J_context_a_context":fixed_action_goal_loss(wm,preprocessor,start_obs,goal_obs,treatment["commands"],context)}
            features=np.concatenate([base,geometry(initial,nominal[-1],population["commands"],treatment["commands"]),model_interaction_features(scores)])
            policy_rows[target]={"features":features,"model_scores":scores,"target_alpha":alpha,
                "population_cost":float(population["metrics"]["pose_auc10"]),"target_cost":float(treatment["metrics"]["pose_auc10"]),
                "benefit":float(population["metrics"]["pose_auc10"])-float(treatment["metrics"]["pose_auc10"])}
        true=row["correct_factor"]; z=np.asarray(row["e1"]["correct"]["posterior"]["mean_z"]); pg=np.linalg.norm(z); pr=math.degrees(math.atan2(z[1],z[0]))
        append_jsonl(out,{"record_type":"cross_shape_harm_feature","analysis_id":ANALYSIS_ID,"sequence_id":int(row["sequence_id"]),
            "shape_pair_index":int(row["shape_pair_index"]),"factor_index":int(row["factor_index"]),"e2_shape":selected["shape"],
            "feature_names":design["feature_names"],"policies":policy_rows,
            "privileged_diagnostics":{"posterior_gain_error":float(pg-float(true["gain"])),"posterior_rotation_error":float(pr-float(true["rotation_degrees"]))}})
        random.setstate(py); np.random.set_state(npstate); torch.set_rng_state(tstate)
        if cstate is not None: torch.cuda.set_rng_state(cstate,cuda_index)
        print(f"D3 feature sequence={row['sequence_id']} complete",flush=True)
    after,rng_after=model_sha256(wm),rng_state_digest(); end=resource_snapshot(device)
    if device.type=="cuda": end["cuda_max_reserved_bytes"]=int(torch.cuda.max_memory_reserved(device.index or 0))
    manifest={"analysis_id":ANALYSIS_ID,"evidence_level":design["evidence_level"],"command":" ".join(__import__("sys").argv),
        "started_unix":started,"finished_unix":time.time(),"rows":len(rows),"design_sha256":sha256(args.design),
        "input_raw_sha256":sha256(args.input_raw),"cross_design_sha256":sha256(args.cross_design),"checkpoint_sha256":sha256(args.checkpoint),
        "source_snapshot_sha256":sha256(args.source_snapshot),"features_sha256":sha256(out),"model_state_unchanged":before==after,
        "model_state_sha256_before":before,"model_state_sha256_after":after,"rng_unchanged":rng_before==rng_after,
        "rng_digest_before":rng_before,"rng_digest_after":rng_after,"initial_replay_max_abs":max_initial,"goal_replay_max_abs":max_goal,
        "resource_end":end}
    if not manifest["model_state_unchanged"] or not manifest["rng_unchanged"] or max(max_initial,max_goal)>1e-6: raise RuntimeError("read-only identity failure")
    dump_json(args.output_dir/"manifest.json",manifest); print(json.dumps(manifest,indent=2)); return manifest


def fit(rows, count, ridge):
    x=np.asarray([row["x"][:count] for row in rows]); y=np.asarray([row["y"] for row in rows]); mean=x.mean(0); scale=x.std(0); mean[0]=0.; scale[0]=1.; scale[scale<1e-12]=1.
    z=(x-mean)/scale; penalty=np.eye(count); penalty[0,0]=0.; beta=np.linalg.solve(z.T@z+ridge*penalty,z.T@y); return mean,scale,beta


def predict(model, rows, count):
    mean,scale,beta=model; x=np.asarray([row["x"][:count] for row in rows]); return ((x-mean[:count])/scale[:count])@beta


def nested_oog(rows, count, ridge_values, group_field):
    predictions=np.zeros(len(rows)); decisions=np.zeros(len(rows),dtype=bool); chosen={}
    groups=sorted({row[group_field] for row in rows})
    for outer in groups:
        train=[row for row in rows if row[group_field]!=outer]; test_indices=[i for i,row in enumerate(rows) if row[group_field]==outer]; test=[rows[i] for i in test_indices]
        candidates=[]
        for ridge in ridge_values:
            deltas=[]
            for inner in sorted({row[group_field] for row in train}):
                inner_train=[row for row in train if row[group_field]!=inner]; inner_test=[row for row in train if row[group_field]==inner]
                pred=predict(fit(inner_train,count,float(ridge)),inner_test,count); deltas.extend([row["y"] if value>0 else 0. for row,value in zip(inner_test,pred)])
            candidates.append((float(np.mean(deltas)),float(ridge)))
        ridge=max(candidates,key=lambda value:(value[0],value[1]))[1]; chosen[str(outer)]=ridge
        pred=predict(fit(train,count,ridge),test,count); predictions[test_indices]=pred; decisions[test_indices]=pred>0
    return predictions,decisions,chosen


def bootstrap(values, design, stream):
    values=np.asarray(values); rng=np.random.default_rng(int(design["bootstrap_seed"])+stream); ix=rng.integers(0,len(values),(int(design["bootstrap_resamples"]),len(values)))
    return np.quantile(values[ix].mean(1),[.025,.975]).tolist()


def analyze(args, design):
    feature_rows=[row for row in read_jsonl(args.output_dir/"features.jsonl") if row.get("record_type")=="cross_shape_harm_feature"]
    if len(feature_rows)!=96: raise RuntimeError("analysis requires 96 rows")
    result={"analysis_id":ANALYSIS_ID,"evidence_level":design["evidence_level"],"targets":{}}
    stream=0
    for target in design["targets"]:
        rows=[{"x":np.asarray(row["policies"][target]["features"]),"y":float(row["policies"][target]["benefit"]),
               "shape_pair":int(row["shape_pair_index"]),"factor":int(row["factor_index"])} for row in feature_rows]
        y=np.asarray([row["y"] for row in rows]); target_result={"base_mean_delta":float(y.mean()),"base_harm_fraction":float(np.mean(y<-1e-12)),"evaluations":{}}
        for evaluation,field in (("leave_one_shape_pair_out","shape_pair"),("leave_one_factor_out","factor")):
            models={}
            for name,count in design["feature_sets"].items():
                pred,decision,ridges=nested_oog(rows,int(count),design["ridge_alphas"],field); delta=np.where(decision,y,0.); improvement=delta-y
                corr=float(np.corrcoef(pred,y)[0,1]) if np.std(pred)>1e-12 and np.std(y)>1e-12 else 0.
                stream+=1; models[name]={"prediction_correlation":corr,"selection_rate":float(decision.mean()),"mean_delta":float(delta.mean()),
                    "bootstrap_ci95_delta":bootstrap(delta,design,stream),"harm_fraction":float(np.mean(delta<-1e-12)),
                    "positive_fraction":float(np.mean(delta>1e-12)),"mean_improvement_vs_target":float(improvement.mean()),
                    "bootstrap_ci95_improvement_vs_target":bootstrap(improvement,design,stream+100),"chosen_ridge_by_outer_group":ridges,
                    "out_of_group_predictions":pred.tolist(),"out_of_group_decisions":decision.astype(int).tolist(),"unit_deltas":delta.tolist()}
            target_result["evaluations"][evaluation]=models
        result["targets"][target]=target_result
    dump_json(args.output_dir/"analysis.json",result); print(json.dumps(result,indent=2)); return result


def main():
    p=argparse.ArgumentParser(); p.add_argument("mode",choices=("extract","analyze")); p.add_argument("--design",type=Path,default=Path("docs/research/persistent_context_v2_cross_shape_harm_d3_design.json"))
    p.add_argument("--cross-design",type=Path,default=Path("docs/research/persistent_context_v2_cross_shape_matrix_history_design.json")); p.add_argument("--input-raw",type=Path,default=Path("repro_outputs/persistent_context_v2_cross_shape_matrix_history_formal_v1/formal/raw.jsonl"))
    p.add_argument("--checkpoint",type=Path,default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth")); p.add_argument("--source-snapshot",type=Path,required=True)
    p.add_argument("--output-dir",type=Path,default=Path("repro_outputs/persistent_context_v2_cross_shape_harm_d3_exploratory_v1")); p.add_argument("--device",default="cuda:0")
    args=p.parse_args()
    if sha256(args.design)!=EXPECTED_DESIGN_SHA256: raise RuntimeError("frozen D3 design hash mismatch")
    design=json.loads(args.design.read_text(encoding="utf-8"))
    if design.get("analysis_id")!=ANALYSIS_ID: raise RuntimeError("analysis id mismatch")
    if args.mode=="extract": extract(args,design)
    else: analyze(args,design)


if __name__=="__main__": main()
