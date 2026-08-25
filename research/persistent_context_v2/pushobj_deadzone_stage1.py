"""Cross-episode radial dead-zone estimation on PushObj early waypoints."""

from __future__ import annotations

import argparse, hashlib, json, pickle, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

from research.persistent_context_v2.pushobj_deadzone_stage0 import (
    POPULATION_PRIOR, DeadZoneWorldModel, execute_deadzone, load_deadzone_world_model,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage0 import (
    deadline_success, nominal_block_displacement_at_10, plan_waypoint_cem, prepare_waypoint,
)
from research.persistent_context_v2.pushobj_rotation_early_waypoint_stage1 import formal_segment_indices as rotation_formal_segments
from research.persistent_context_v2.pushobj_rotation_stage0 import (
    append_jsonl, dump_json, git_revision, identity_audit, make_env, make_preprocessor,
    pose_metrics, read_jsonl, resource_snapshot, seed_all, sha256,
)
from research.persistent_context_v2.pushobj_rotation_stage1 import pd_coefficients

CONTRACT_ID="persistent-context-v2-pushobj-radial-deadzone-history-stage1-v1"
EXPECTED_DESIGN_SHA256="f30e8949a6db04a77c68d9897aefc0661c1be4f7aae38a6157bb441d68884e88"
FACTORS=(0.04,0.08,0.12,0.16)
POLICIES=("population_prior","current_only","correct_history","shuffled_history","wrong_sequence_history","true_factor_oracle")
CONDITIONS=("persistent","no_persistence")
N_SEQUENCES,N_EPISODES=32,4
SELECTION_SEED,FACTOR_SEED,DONOR_SEED,BOOTSTRAP_SEED=930000,930100,930200,930301
BOOTSTRAP_RESAMPLES=20_000

def array_sha256(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

@dataclass
class DeadZoneMLE:
    active_norm_min:float=1e-4
    clip_max:float=.25
    active_values:list=None
    lower_bound:float=0.0
    transition_count:int=0
    def __post_init__(self):
        if self.active_values is None:self.active_values=[]
    def update(self,commands,states):
        commands=np.asarray(commands,dtype=np.float64);states=np.asarray(states,dtype=np.float64)
        pc,vc,tc=pd_coefficients()
        for i,u in enumerate(commands):
            self.transition_count+=1;p0,p1=states[i,:2],states[i+1,:2];v0=states[i,5:7]
            target=(p1-pc*p0-vc*v0)/tc;y=(target-p0)/100.;un,yn=float(np.linalg.norm(u)),float(np.linalg.norm(y))
            if yn>self.active_norm_min:self.active_values.append(max(0.,un-yn))
            else:self.lower_bound=max(self.lower_bound,un)
    @property
    def estimate(self):
        center=float(np.median(self.active_values)) if self.active_values else POPULATION_PRIOR
        return float(np.clip(max(center,self.lower_bound),0.,self.clip_max))
    def as_dict(self):return {"transition_count":self.transition_count,"active_count":len(self.active_values),"lower_bound":self.lower_bound,"estimate":self.estimate}

def formal_segment_indices(segments):
    used=set(rotation_formal_segments(segments).tolist())
    pool=np.asarray([i for i in range(500,min(1000,len(segments))) if i not in used and nominal_block_displacement_at_10(segments[i])>=10],dtype=np.int64)
    if len(pool)<128:raise RuntimeError("dead-zone formal pool too small")
    return np.random.default_rng(SELECTION_SEED).permutation(pool)[:128]

def factor_schedules(n=N_SEQUENCES,e=N_EPISODES):
    if n%4:raise ValueError("n must divide four")
    base=np.arange(n)%4;p=np.repeat(base[:,None],e,axis=1);q=np.empty_like(p);q[:,0]=base;rng=np.random.default_rng(FACTOR_SEED);balanced=np.repeat(np.arange(4),n//4)
    for j in range(1,e):
        while True:
            c=rng.permutation(balanced)
            if np.all(c!=q[:,j-1]):q[:,j]=c;break
    return p,q

def _cross_map(rng,n,shift):
    out=np.empty(n,dtype=int);ids=np.arange(n)
    for f in range(4):
        targets=ids[ids%4==f];donors=ids[ids%4==(f+shift)%4];out[targets]=rng.permutation(donors)
    return out

def donor_maps(n=N_SEQUENCES):
    rng=np.random.default_rng(DONOR_SEED);wrong=_cross_map(rng,n,int(rng.integers(1,4)));shifts=rng.permutation([1,2,3]);shuffled=np.stack([_cross_map(rng,n,int(s)) for s in shifts],axis=1);return wrong,shuffled

def factor_for(condition,s,e,n=N_SEQUENCES,episodes=N_EPISODES):
    p,q=factor_schedules(n,episodes);return FACTORS[int((p if condition=="persistent" else q)[s,e])]

def scenario(segments,condition,s,e,n=N_SEQUENCES,episodes=N_EPISODES):
    idx=formal_segment_indices(segments).reshape(32,4)
    return {"condition":condition,"sequence_id":s,"episode":e+1,"episode_index":e,"segment_index":int(idx[s,e]),"factor":factor_for(condition,s,e,n,episodes),"env_seed":940000+s*100+e,"cem_seed":950000+s*100+e}

def donor_for(policy,s,h,n):
    if policy=="correct_history":return s
    wrong,shuffled=donor_maps(n)
    if policy=="wrong_sequence_history":return int(wrong[s])
    if policy=="shuffled_history":return int(shuffled[s,h])
    raise ValueError(policy)

def build_estimator(evidence,policy,s,e,n):
    est=DeadZoneMLE();donors=[]
    for h in range(e):
        d=donor_for(policy,s,h,n);row=evidence[(d,h)];est.update(row["commands"],row["states"]);donors.append({"history_episode":h+1,"donor_sequence_id":d,"donor_factor":row["factor"],"evidence_sha256":row["evidence_sha256"]})
    return est,donors

def context_payload(evidence,policy,s,e,factor,n):
    if policy in ("population_prior","current_only"):return {"context":POPULATION_PRIOR,"estimator":None,"donors":[]}
    if policy=="true_factor_oracle":return {"context":factor,"estimator":None,"donors":[]}
    est,donors=build_estimator(evidence,policy,s,e,n);return {"context":est.estimate,"estimator":est.as_dict(),"donors":donors}

def _keys(path,kind):return {(r["sequence_id"],r["episode"]) for r in read_jsonl(path) if r.get("record_type")==kind}
def _evidence(path):return {(r["sequence_id"],r["episode_index"]):r for r in read_jsonl(path) if r.get("record_type")=="evidence_episode"}

def generate_evidence(args,condition,wm,prep,segments,env):
    path=args.output_dir/f"{condition}_raw.jsonl";done=_keys(path,"evidence_episode")
    for s in range(args.sequences):
        for e in range(args.episodes):
            meta=scenario(segments,condition,s,e,args.sequences,args.episodes)
            if (s,e+1) in done:continue
            initial,goal_obs,nominal,nominal_actions=prepare_waypoint(env,segments[meta["segment_index"]],meta["env_seed"]);start_obs,_=env.prepare(meta["env_seed"],initial)
            commands,planner=plan_waypoint_cem(wm,prep,start_obs,goal_obs,POPULATION_PRIOR,meta["cem_seed"]);states,effective,contacts,coverage=execute_deadzone(env,initial,meta["env_seed"],commands,meta["factor"])
            row={"record_type":"evidence_episode","contract_id":CONTRACT_ID,**meta,"initial_state":initial,"goal_state":nominal[-1],"nominal_actions":nominal_actions,"commands":commands,"states":states,"effective_actions":effective,"contacts":contacts,"coverages":coverage,"planner":planner,"evidence_sha256":array_sha256(commands)+":"+array_sha256(states),"resource":resource_snapshot(next(wm.parameters()).device)};append_jsonl(path,row);print(f"EVIDENCE {condition} s={s} e={e+1} d={meta['factor']:.2f}",flush=True)
    return _evidence(path)

def run_eval(args,condition,wm,prep,segments,env,evidence):
    path=args.output_dir/f"{condition}_raw.jsonl";done=_keys(path,"evaluation_episode")
    for s in range(args.sequences):
        for e in range(args.episodes):
            meta=scenario(segments,condition,s,e,args.sequences,args.episodes)
            if (s,e+1) in done:continue
            initial,goal_obs,nominal,_=prepare_waypoint(env,segments[meta["segment_index"]],meta["env_seed"]);start_obs,_=env.prepare(meta["env_seed"],initial)
            cps={p:context_payload(evidence,p,s,e,meta["factor"],args.sequences) for p in POLICIES};cache={};policies={}
            for p in POLICIES:
                c=float(cps[p]["context"]);key=round(c,10)
                if key not in cache:
                    commands,planner=plan_waypoint_cem(wm,prep,start_obs,goal_obs,c,meta["cem_seed"]);states,effective,contacts,coverage=execute_deadzone(env,initial,meta["env_seed"],commands,meta["factor"])
                    cache[key]={"commands":commands,"states":states,"effective_actions":effective,"contacts":contacts,"coverages":coverage,"metrics":pose_metrics(states,nominal[-1],10),"deadline_success":deadline_success(states,nominal[-1]),"command_sha256":array_sha256(commands),"state_sha256":array_sha256(states),"planner":planner}
                policies[p]={**cps[p],**cache[key]}
            append_jsonl(path,{"record_type":"evaluation_episode","contract_id":CONTRACT_ID,**meta,"initial_state":initial,"goal_state":nominal[-1],"policies":policies,"resource":resource_snapshot(next(wm.parameters()).device)})
            print(f"EVAL {condition} s={s} e={e+1} d={meta['factor']:.2f} hist={policies['correct_history']['context']:.4f} current={policies['current_only']['metrics']['pose_auc10']:.4f} history={policies['correct_history']['metrics']['pose_auc10']:.4f}",flush=True)

def bootstrap(values,stream):
    v=np.asarray(values,float);rng=np.random.default_rng(BOOTSTRAP_SEED+stream);idx=rng.integers(0,len(v),size=(BOOTSTRAP_RESAMPLES,len(v)));m=v[idx].mean(1);return [float(np.quantile(m,.025)),float(np.quantile(m,.975))]
def effect(a,b,stream):
    a,b=np.asarray(a),np.asarray(b);d=a-b;return {"current_mean":float(a.mean()),"treatment_mean":float(b.mean()),"mean_delta":float(d.mean()),"relative_improvement":float(d.mean()/a.mean()),"bootstrap_ci95_delta":bootstrap(d,stream),"positive_fraction":float(np.mean(d>1e-12)),"tie_fraction":float(np.mean(np.abs(d)<=1e-12)),"negative_fraction":float(np.mean(d< -1e-12)),"sequence_deltas":d.tolist()}

def summarize(root,n=N_SEQUENCES,episodes=N_EPISODES):
    rows={c:[r for r in read_jsonl(root/f"{c}_raw.jsonl") if r.get("record_type")=="evaluation_episode"] for c in CONDITIONS};vals={};succ={}
    for c in CONDITIONS:
        lk={(r["sequence_id"],r["episode_index"]):r for r in rows[c]};vals[c]={};succ[c]={}
        for p in POLICIES:
            vals[c][p]=np.asarray([np.mean([lk[(s,e)]["policies"][p]["metrics"]["pose_auc10"] for e in range(1,episodes)]) for s in range(n)])
            succ[c][p]=np.asarray([np.mean([lk[(s,e)]["policies"][p]["deadline_success"] for e in range(1,episodes)]) for s in range(n)])
    pe=effect(vals["persistent"]["current_only"],vals["persistent"]["correct_history"],100);ne=effect(vals["no_persistence"]["current_only"],vals["no_persistence"]["correct_history"],200);te=effect(vals["persistent"]["current_only"],vals["persistent"]["true_factor_oracle"],300);did=np.asarray(pe["sequence_deltas"])-np.asarray(ne["sequence_deltas"])
    out={"contract_id":CONTRACT_ID,"primary_metric":"later_E2_E4_mean_pose_auc10_to_waypoint","persistent_correct_history":pe,"no_persistence_correct_history":ne,"persistent_true_factor":te,"persistent_shuffled_history":effect(vals["persistent"]["current_only"],vals["persistent"]["shuffled_history"],400),"persistent_wrong_sequence_history":effect(vals["persistent"]["current_only"],vals["persistent"]["wrong_sequence_history"],500),"did":{"mean":float(did.mean()),"bootstrap_ci95":bootstrap(did,600),"positive_fraction":float(np.mean(did>1e-12))},"true_gap_recovery":float(pe["mean_delta"]/te["mean_delta"]),"deadline_success":{c:{p:float(succ[c][p].mean()) for p in POLICIES} for c in CONDITIONS},"policy_means":{c:{p:float(vals[c][p].mean()) for p in POLICIES} for c in CONDITIONS},"by_factor":{},"by_side":{},"estimator":{},"donor_match":{}}
    for f in FACTORS:
        ids=[s for s in range(n) if factor_for("persistent",s,0,n,episodes)==f];out["by_factor"][str(f)]=effect(vals["persistent"]["current_only"][ids],vals["persistent"]["correct_history"][ids],700+int(f*100))
    for name,fs in (("below_prior",(.04,.08)),("above_prior",(.12,.16))):
        ids=[s for s in range(n) if factor_for("persistent",s,0,n,episodes) in fs];out["by_side"][name]=effect(vals["persistent"]["current_only"][ids],vals["persistent"]["correct_history"][ids],800+(name=="above_prior"))
    for c in CONDITIONS:
        later=[r for r in rows[c] if r["episode_index"]>0]
        for p in ("correct_history","shuffled_history","wrong_sequence_history"):
            errs=[abs(r["policies"][p]["context"]-r["factor"]) for r in later];matches=[d["donor_factor"]==r["factor"] for r in later for d in r["policies"][p]["donors"]];out["estimator"].setdefault(c,{})[p]={"mae":float(np.mean(errs)),"median_absolute_error":float(np.median(errs))};out["donor_match"].setdefault(c,{})[p]=float(np.mean(matches))
    checks={}
    for c in CONDITIONS:
        checks[c]={"complete":len(rows[c])==n*episodes,"current_population_identity":all(r["policies"]["current_only"]["state_sha256"]==r["policies"]["population_prior"]["state_sha256"] for r in rows[c]),"episode_one_identity":all(len({(r["policies"][p]["command_sha256"],r["policies"][p]["state_sha256"]) for p in POLICIES if p!="true_factor_oracle"})==1 for r in rows[c] if r["episode_index"]==0)}
    out["structural_checks"]=checks;out["valid"]=all(all(v.values()) for v in checks.values());return out

def main():
    p=argparse.ArgumentParser();p.add_argument("mode",choices=("inspect","run","summarize"));p.add_argument("--smoke",action="store_true");p.add_argument("--checkpoint",type=Path,default=Path("/home/zhaoqing/adajepa/checkpoints/pushobj_shape_shift/checkpoints/model_latest.pth"));p.add_argument("--data",type=Path,default=Path("/home/zhaoqing/adajepa/data/pushobj_eval/val_T/plan_targets.pkl"));p.add_argument("--design",type=Path,default=Path("docs/research/persistent_context_v2_pushobj_deadzone_stage1_design.json"));p.add_argument("--contract",type=Path,default=Path("docs/research/persistent_context_v2_pushobj_deadzone_stage1_contract_zh.md"));p.add_argument("--output-dir",type=Path,default=Path("repro_outputs/persistent_context_v2_pushobj_deadzone_stage1"));p.add_argument("--device",default="cuda:0");a=p.parse_args();a.sequences=4 if a.smoke else 32;a.episodes=2 if a.smoke else 4;a.output_dir.mkdir(parents=True,exist_ok=True)
    with a.data.open("rb") as h:segments=pickle.load(h)["segments"]
    if a.mode=="inspect":
        pp,qq=factor_schedules();w,sh=donor_maps();x={"contract_id":CONTRACT_ID,"design_sha256":sha256(a.design),"contract_sha256":sha256(a.contract),"segments":formal_segment_indices(segments).reshape(32,4).tolist(),"persistent_factor_ids":pp.tolist(),"no_persistence_factor_ids":qq.tolist(),"wrong_donors":w.tolist(),"shuffled_donors":sh.tolist()};dump_json(a.output_dir/"selection_audit.json",x);print(json.dumps(x,indent=2));return
    if a.mode=="summarize":x=summarize(a.output_dir,a.sequences,a.episodes);dump_json(a.output_dir/"runner_summary.json",x);print(json.dumps(x,indent=2));return
    if sha256(a.design)!=EXPECTED_DESIGN_SHA256:raise RuntimeError("design hash mismatch")
    seed_all(9501);device=torch.device(a.device);mp=a.output_dir/"manifest.json"
    if not mp.exists():m={"contract_id":CONTRACT_ID,"mode":"smoke" if a.smoke else "formal","git_revision":git_revision(),"design_path":str(a.design),"design_sha256":sha256(a.design),"contract_path":str(a.contract),"contract_sha256":sha256(a.contract),"checkpoint":str(a.checkpoint),"checkpoint_sha256":sha256(a.checkpoint),"data":str(a.data),"data_sha256":sha256(a.data),"command":" ".join(__import__("sys").argv),"sequences":a.sequences,"episodes":a.episodes,"started_unix":time.time(),"resource_start":resource_snapshot(device)};dump_json(mp,m)
    else:m=json.loads(mp.read_text(encoding="utf-8"))
    base,wm,_=load_deadzone_world_model(a.checkpoint,device);prep=make_preprocessor();env=make_env();meta=scenario(segments,"persistent",0,0,a.sequences,a.episodes);initial,_,_,_=prepare_waypoint(env,segments[meta["segment_index"]],meta["env_seed"]);obs,_=env.prepare(meta["env_seed"],initial);m["identity_audit"]=identity_audit(base,wm,prep,obs);dump_json(mp,m)
    for c in CONDITIONS:e=generate_evidence(a,c,wm,prep,segments,env);run_eval(a,c,wm,prep,segments,env,e)
    x=summarize(a.output_dir,a.sequences,a.episodes);dump_json(a.output_dir/"runner_summary.json",x);print(json.dumps(x,indent=2));m["finished_unix"]=time.time();m["resource_end"]=resource_snapshot(device);dump_json(mp,m)

if __name__=="__main__":main()
