"""Verify new fixed-S reporting against original gold and actual saved safe candidates."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v169_saved_state_quality import review

OUT=ROOT/'artifacts/v169_saved_state_quality_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();goldfile=ROOT/'data/official/train.parquet';gold=pd.read_parquet(goldfile,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert np.bincount(gold,minlength=3).tolist()==[1899723,111728,45420]
    check_bindings(read(ROOT/'training/review_policy/v169_learnable_prior_pair_draft.json')['source_sha256'])
    files=[Path(__file__).resolve(),ROOT/'training/v169_saved_state_quality.py',goldfile];roles=[]
    for role in [0,1,2]:
        base=ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}';safe=ROOT/f'artifacts/v167_trial_point_restoration_diagnostic_20261002/role{role}/probe0' if role!=1 else ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0'
        refpath=base/'endpoint/OOF_original_rows.parquet';reference=pd.read_parquet(refpath);cohortpath=ROOT/f'artifacts/v169_learnable_prior_pair_plan_20261002/role{role}_all_S_initial_prior_cohort.parquet';cohort=pd.read_parquet(cohortpath)
        assert np.array_equal(reference.truth,gold[reference.row_position]) and np.array_equal(cohort.truth,gold[cohort.row_position])
        opath=ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy';opinion=np.load(opath,mmap_mode='r');prior=np.log(np.maximum(np.asarray(opinion[:,:16],np.float64).mean(1),1e-12))
        state=torch.load(base/'endpoint.pt',map_location='cpu',weights_only=True)['state'];u=np.load(safe/'direction.npy');w=state['output_weight'].numpy()+u[-48:].reshape(16,3)
        q,lp=np.load(safe/'OOF_q.npy'),np.load(safe/'OOF_logq.npy');after=pd.read_parquet(safe/'OOF_original_rows.parquet');assert np.array_equal(after.truth,gold[after.row_position])
        report=review(reference,q,lp,cohort,prior,w,1.)
        assert report['classes']['0']['recall'] is None
        for cls in [1,2]:
            mask=after.truth.eq(cls);wrong=after.pred.ne(after.truth)
            assert report['classes'][str(cls)]['original_rows']==int(mask.sum()) and report['classes'][str(cls)]['errors']==int((mask&wrong).sum())
        assert report['fixed_all_S_slices']['initial_low_prior']['original_rows']==[831,813,319][role]
        assert report['fixed_all_S_slices']['initial_low_prior']['repairs_of_initial_errors']==0
        assert report['fixed_all_S_slices']['initial_other_prior_correct_controls']['regressions_of_initial_correct']==0
        try:review(reference,q,lp,cohort.iloc[:-1],prior,w,1.)
        except ValueError:pass
        else:raise AssertionError('Missing correct/remaining S row must refuse')
        roles.append(dict(role=role,review=report,all_fixed_S_and_controls_retained=True,complete_gold_rows_checked=True))
        files.extend([refpath,cohortpath,opath,base/'endpoint.pt',safe/'direction.npy',safe/'OOF_q.npy',safe/'OOF_logq.npy',safe/'OOF_original_rows.parquet'])
    result=dict(status='V169_saved_actual_original_gold_full_class_and_fixed_initial_S_reporting_qualified',roles=roles,low_prior_initial_rows=[831,813,319],no_safe_candidate_low_prior_S_repairs=True,all_complete_S_cohort_and_correct_controls_required=True,missing_S_row_refused=True,missing_N_support_not_estimated=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,not_new_beta_training_or_gain=True,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(result,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=result['status'],low_prior_S_rows=[831,813,319],official_calls=0)))

if __name__=='__main__':main()
