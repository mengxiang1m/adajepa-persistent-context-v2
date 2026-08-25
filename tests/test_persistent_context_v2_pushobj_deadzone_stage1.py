import numpy as np
from research.persistent_context_v2.pushobj_deadzone_stage1 import DeadZoneMLE,donor_maps,factor_schedules
from research.persistent_context_v2.pushobj_rotation_stage1 import pd_coefficients

def _states(commands,d):
    commands=np.asarray(commands,float);effective=[]
    for u in commands:
        n=np.linalg.norm(u);effective.append(np.zeros(2) if n<=d else u*(n-d)/n)
    states=np.zeros((len(commands)+1,7));pc,vc,tc=pd_coefficients();states[0,:2]=[200,250]
    for i,y in enumerate(effective):
        p0=states[i,:2];v0=states[i,5:7];target=p0+100*y;states[i+1,:2]=pc*p0+vc*v0+tc*target
    return states

def test_censored_mle_recovers_deadzone():
    u=np.array([[.03,0],[.2,0],[0,.3]]);e=DeadZoneMLE();e.update(u,_states(u,.12));assert abs(e.estimate-.12)<1e-8

def test_schedules_and_donors_are_isolated():
    p,q=factor_schedules();assert np.all(q[:,1:]!=q[:,:-1]);assert np.array_equal(p[:,0],q[:,0])
    w,s=donor_maps();ids=np.arange(32);assert np.all(w!=ids);assert np.all(w%4!=ids%4);assert np.all(s!=ids[:,None]);assert all(len(set(x))==3 for x in s.tolist())
