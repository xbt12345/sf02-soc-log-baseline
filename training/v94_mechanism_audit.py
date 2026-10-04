"""Frozen checkpoint factorial audit; no parameter estimation or model selection."""
import json,time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT,read,save,sha,data,raw_counts
from v93_regression_audit import PREV,forward,counts_metrics

DEST=ROOT/'artifacts/v94_research_mechanism_20260928'

def main():
    assert not DEST.exists();DEST.mkdir();start=time.monotonic()
    save(DEST/'registration.json',{'scope':'Adaptive frozen audit based on v93. 2x2 old/new encoder and head swaps; no fitted parameters, selected candidate or causal claim from off-path hybrids.',
        'new_classifier_fits':0,'new_calibration_fits':0,'source_sha256':sha(__file__),
        'input_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in [PREV/'U0_model.pt',PREV/'U0R_model.pt',PREV/'ASA_R0.npz',PREV/'ASA_input_ids.npy',ROOT/'docs/V93_500_M_REGRESSION_ROOT_CAUSE_AND_PLAN.md']}})
    r,y,fid,z0,old,_,fit=data();ids=np.load(PREV/'ASA_input_ids.npy');x=sparse.load_npz(PREV/'ASA_R0.npz').astype(float);z=np.asarray(z0[ids]);asa=r.route.eq('asa').to_numpy()
    roles={'fit':fit,'inner':r.fold.eq(1).to_numpy(),'C':r.fold.eq(2).to_numpy(),'H':r.fold.eq(0).to_numpy()}
    counts={role:raw_counts(fid,y,mask,len(old)) for role,mask in roles.items()};states={};hidden={};reports=[]
    for name in ['U0','U0R']:
        states[name]={k:v.numpy().astype(float) for k,v in torch.load(PREV/(name+'_model.pt'),map_location='cpu',weights_only=True)['state_dict'].items()}
        hidden[name]=forward(states[name],x,hidden=True)
    predictions={}
    for enc in ['U0','U0R']:
        for head in ['U0','U0R']:
            name=enc+'_encoder_'+head+'_head';sc=z+hidden[enc]@states[head]['out.weight'].T+states[head]['out.bias'];pred=old.copy();pred[ids]=sc.argmax(1)
            if enc==head:assert np.array_equal(pred,np.load(PREV/(enc+'_prediction.npy')))
            np.save(DEST/(name+'_prediction.npy'),pred);predictions[name]=pred
            reports.append({'encoder':enc,'head':head,'name':name,'roles':{role:counts_metrics(cc,old,pred) for role,cc in counts.items()},
                'scope':'Hybrids are off training path; cannot identify unique causal fault or certify reusable head.'})
    save(DEST/'encoder_head_factorial.json',reports)
    u=np.load(PREV/'U0_prediction.npy');v=np.load(PREV/'U0R_prediction.npy');zs={n:np.load(PREV/(n+'_ASA_scores.npy'))-z for n in ['U0','U0R']}
    dm0=zs['U0'][:,2]-zs['U0'][:,1];dm1=zs['U0R'][:,2]-zs['U0R'][:,1];inner=roles['inner'];target=inner&(y==1)&(old[fid]==1)&(v[fid]!=1)
    regress=np.flatnonzero(target);ii=np.searchsorted(ids,fid[regress]);gain=np.flatnonzero(inner&(y==2)&(old[fid]!=2)&(v[fid]==2));g=np.searchsorted(ids,fid[gain])
    new_m=np.flatnonzero(inner&(y==1)&(u[fid]==1)&(v[fid]!=1));repair_m=np.flatnonzero(inner&(y==1)&(u[fid]!=1)&(v[fid]==1))
    transitions=[]
    for role,mask in roles.items():
        for cls in [0,1,2]:
            iirow=np.flatnonzero(mask&(y==cls));a=u[fid[iirow]];b=v[fid[iirow]]
            transitions.append({'role':role,'class':cls,'U0_correct':int((a==cls).sum()),'U0R_correct':int((b==cls).sum()),
                'newly_wrong':int(((a==cls)&(b!=cls)).sum()),'recovered':int(((a!=cls)&(b==cls)).sum())})
    save(DEST/'continuation_transitions.json',transitions)
    cosine=lambda a,b:float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
    w0=states['U0']['out.weight'][2]-states['U0']['out.weight'][1];w1=states['U0R']['out.weight'][2]-states['U0R']['out.weight'][1]
    # A component describes a grouping proxy; purity does not certify label leakage.
    population=[]
    for role,mask in roles.items():
        rr=r[mask&asa];table=rr.groupby(['component','label_index']).size().unstack(fill_value=0).reindex(columns=[0,1,2],fill_value=0)
        population.append({'role':role,'ASA_rows':len(rr),'components':len(table),'pure_components':int(((table>0).sum(1)==1).sum()),'mixed_components':int(((table>0).sum(1)>1).sum()),
            'rows_in_mixed_components':int(table.loc[(table>0).sum(1)>1].sum().sum()),'rows_same_component_majority_wrong':int((table.sum(1)-table.max(1)).sum())})
    save(DEST/'component_population.json',population)
    summary={'status':'completed_frozen_mechanism_audit','new_classifier_fits':0,'new_calibration_fits':0,'selected':None,'quality_acceptance':False,
        'M500_already_wrong_at_U0':int((u[fid[regress]]!=1).sum()),'M500_previously_correct_at_U0':int((u[fid[regress]]==1).sum()),
        'M500_old_residual_direction_toward_S':int((dm0[ii]>0).sum()),'M500_old_residual_direction_toward_M_or_zero':int((dm0[ii]<=0).sum()),
        'S58_already_correct_U0':int((u[fid[gain]]==2).sum()),'S58_old_residual_toward_S':int((dm0[g]>0).sum()),
        'continuation_new_M_errors':len(new_m),'continuation_repaired_M_errors':len(repair_m),
        'MS_head_direction_cosine_U0_U0R':cosine(w0,w1),'MS_head_norm_ratio_U0R_U0':float(np.linalg.norm(w1)/np.linalg.norm(w0)),
        'source_sha256':sha(__file__),'seconds':time.monotonic()-start}
    save(DEST/'mechanism_summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
