"""Recompute cached tail original gold, full correction and all costs."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v162_cached_function_restoration_tail import OUT as TRIAL,SOURCE,DIAG
from v162_independent_actual_restoration_review import actual_proposal,rows_review,close,parameter_hash
from v162_multi_function_finite_restoration_v2 import propose
OUT=ROOT/'artifacts/v162_saved_cached_restoration_tail_audit_v2_20261002'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir();goldpath=ROOT/'data/official/train.parquet';cohort=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002/role2/fixed_pure_error_targets.parquet';endpoint=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002/fold2_B/endpoint.pt'
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{p for p in SOURCE.rglob('*') if p.is_file()}|{Path(__file__).resolve(),goldpath,cohort,endpoint,ROOT/'training/v162_independent_actual_restoration_review.py',ROOT/'training/v160_independent_fixed_diagnostic_review.py',ROOT/'training/v161_independent_all_finite_results_review.py',ROOT/'training/v162_multi_function_finite_restoration_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py'}|{DIAG/f'role2/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));folder=TRIAL/'role2';diag=read(folder/'diagnostic.json');assert diag['exception'] is None and diag['new_margin_gradients']==diag['new_fits']==diag['permanent_updates']==diag['new_fixed_error_target_gradients']==diag['new_full_original_class_gradients']==0
    state=torch.load(endpoint,map_location='cpu',weights_only=True)['state'];assert parameter_hash(state)==diag['initial_parameter_sha256']==diag['restored_parameter_sha256'];assert diag['restored_joint_TRAIN_retention']['passed']
    gold=pd.read_parquet(goldpath,columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy();baseline={s:pd.read_parquet(folder/'baseline'/f'{s}_original_rows.parquet') for s in ['OOF','deployment']}
    for scope,f in baseline.items():
        rows_review(folder/'baseline',scope,f,gold);restored,_,_,_=rows_review(folder/'restored',scope,f,gold);assert np.array_equal(restored.pred,f.pred);close(restored[['p0','p1','p2']].to_numpy(),f[['p0','p1','p2']].to_numpy())
    targets=pd.read_parquet(cohort).row_position;gs=[np.load(DIAG/f'role2/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]];u=np.load(SOURCE/'displacement.npy');a=np.load(SOURCE/'active_complete_margin_normals.npy');b=np.load(SOURCE/'base_margins.npy');reports=[]
    for path in sorted(folder.glob('restoration*/original_unit_restoration_review.json')):
        assert np.array_equal(u,np.load(path.parent/'current_displacement.npy'));assert np.array_equal(a,np.load(path.parent/'active_complete_margin_normals.npy'));assert np.array_equal(b,np.load(path.parent/'base_margins.npy'));assert read(path.parent/'active_functions.json')==read(SOURCE/'active_functions.json')
        c=np.load(path.parent/'actual_current_margins.npy');result=propose(u,a,b,c,*gs);assert result['status']==read(path)['status'];assert np.array_equal(result['correction'],np.load(path.parent/'correction.npy')) and np.array_equal(result['displacement'],np.load(path.parent/'displacement.npy'))
        report=actual_proposal(path.parent/'finite_probe',baseline,targets,gs,state,gold);reports.append(dict(restoration=path.parent.name,actual=report,class_reviews=result['class_reviews'],all_local_residuals_passed=all(r['passed'] for r in result['residual_reviews'])));u=result['displacement']
    calls=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];assert not any(c['kind'] not in ['head','feature'] for c in calls)
    for kind in ['head','feature']:
        for event,key in [('attempt','attempts'),('completed','completed')]:
            events=[c for c in calls if c['kind']==kind and c['event']==event];assert [c['ordinal'] for c in events]==list(range(1,len(events)+1));assert len(events)==diag['counts'][kind+'_'+key]
    assert diag['counts']['head_attempts']==diag['counts']['feature_attempts']==(2+len(reports))*22<=88 and diag['finite_proposals']==diag['restoration_solves']==len(reports)==2 and sum(r['actual']['accepted'] for r in reports)==int(diag['actual_finite_restoration_pass'])
    check_bindings(bindings);save(OUT/'audit.json',dict(status='saved_cached_restoration_tail_original_gold_complete_vectors_risks_guards_hashes_and_costs_passed',actual_finite_pass=diag['actual_finite_restoration_pass'],reports=reports,actual_new_heads=88,actual_new_features=88,actual_new_gradients=0,new_fits=0,permanent_updates=0,cumulative_heads=17334,cumulative_all_complete_derivatives=422,all_parameters_restored=True,all_original_joint_retention_passed=True,quality_acceptance=False,official_audit_calls=0,source_sha256=bindings));print(json.dumps(dict(status='V162_saved_cached_restoration_tail_audit_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
