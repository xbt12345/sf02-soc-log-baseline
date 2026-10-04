"""Real zero-accepted stages: unchanged parameters/probabilities, no model calls."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v158_fusion_contract import STAGES

BASE=ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001'
OUT=ROOT/'artifacts/v158_zero_update_stage_identity_audit_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();files=[];paths={Path(__file__).resolve()}
    for path in BASE.glob('outer*_inner*/*/fit.json'):
        r=read(path)
        if r['accepted_updates']==0:
            folder=path.parent;index=STAGES.index(r['stage']);assert index>=1 and r['termination']=='numerical_no_change'
            assert not (folder/'progress.json').exists() and not r['last5_FIT_stats']
            files.append((folder,r,index))
            paths.update(folder/name for name in ['fit.json','endpoint.pt','member_logits.npy','mean_probability.npy','classifier_calls.jsonl','gradients.jsonl'])
            paths.update([folder.parent/STAGES[index-1]/'endpoint.pt',folder.parent/STAGES[index-1]/'mean_probability.npy'])
            if index==3:paths.update([folder.parent/STAGES[0]/'endpoint.pt',folder.parent/STAGES[2]/'endpoint.pt'])
    assert len(files)==4;OUT.mkdir();bindings={q.relative_to(ROOT).as_posix():sha(q) for q in paths}
    save(OUT/'pre_saved_state_bindings.json',dict(source_sha256=bindings,official_model_calls=0))
    cases=[]
    for folder,r,index in files:
        state=torch.load(folder/'endpoint.pt',map_location='cpu',weights_only=True)['state']
        if index==2:expected=torch.load(folder.parent/STAGES[index-1]/'endpoint.pt',map_location='cpu',weights_only=True)['state']
        elif index==3:
            backbone=torch.load(folder.parent/STAGES[0]/'endpoint.pt',map_location='cpu',weights_only=True)['model']
            head=torch.load(folder.parent/STAGES[2]/'endpoint.pt',map_location='cpu',weights_only=True)['state'];expected={}
            for key in ['weight','r','s','bias']:
                expected[key]=backbone['second.'+key].double();expected['reference_'+key]=expected[key]
            for key in ['weight','bias','facts']:expected['head_'+key]=head[key].double()
        else:raise ValueError('Unexpected zero-update stage')
        assert set(state)==set(expected) and all(torch.equal(state[k],v) for k,v in expected.items())
        current=np.load(folder/'mean_probability.npy');previous=np.load(folder.parent/STAGES[index-1]/'mean_probability.npy')
        gap=float(np.abs(current-previous).max());assert gap<=2e-12 and np.array_equal(current.argmax(1),previous.argmax(1))
        calls=[json.loads(z) for z in (folder/'classifier_calls.jsonl').read_text().splitlines()]
        gradients=[json.loads(z) for z in (folder/'gradients.jsonl').read_text().splitlines()]
        assert sum(v['event']=='attempt' for v in calls)==sum(v['event']=='completed' for v in calls)==r['classifier_forward_calls']
        assert sum(v['event']=='full_gradient_attempt' for v in gradients)==sum(v['event']=='full_gradient_completed' for v in gradients)==r['full_gradients']
        cases.append(dict(stage_path=folder.relative_to(ROOT).as_posix(),accepted_updates=0,termination=r['termination'],
            parameters_identical_to_registered_initial_state=True,all_probability_gap=gap,argmax_identical=True,
            actual_full_gradients=r['full_gradients'],actual_classifier_calls=r['classifier_forward_calls'],progress_absence_valid=True))
    check_bindings(bindings);save(OUT/'audit.json',dict(status='all_four_missing_progress_zero_update_stages_actual_identity_verified',cases=cases,
        official_model_calls=0,features_calls=0,new_gradients=0,new_fits=0,new_updates=0,quality_acceptance=False,source_sha256=bindings))
    print(dict(zero_update_stage_identity_verified=True,cases=4,new_model_calls=0,new_fits=0))

if __name__=='__main__':main()
