"""New review protocol tests with toy seals/metadata/labels, no head calls."""
import copy,json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review_v159_v2 import ROOT,PROTOCOL,ReviewError,review_plan,seal_run,require_run_seal,require_checkpoint,evaluate_primary,sha,check_bindings

OUT=ROOT/'artifacts/v159_protocol_synthetic_qualification_v2_20261002'

def rejected(fn):
    try:fn()
    except (ReviewError,FileNotFoundError):return True
    raise AssertionError('Expected protocol rejection')

def main():
    assert not OUT.exists();OUT.mkdir()
    sources=[Path(__file__).resolve(),ROOT/'training/experiment_review_v159_v2.py',ROOT/'training/experiment_review.py',ROOT/'training/v159_boundary_runtime_v3.py',ROOT/'training/v159_boundary_train_v3.py',ROOT/'training/v159_boundary_evaluate_v3.py']
    bound={p.relative_to(ROOT).as_posix():sha(p) for p in sources}
    (OUT/'pre_synthetic_bindings.json').write_text(json.dumps(dict(status='before_protocol_synthetic_seals_metadata_labels',source_sha256=bound),indent=2)+'\n',encoding='utf-8')
    p=copy.deepcopy(json.loads((ROOT/'training/review_policy/v159_boundary_execution_contract.json').read_text(encoding='utf-8')))
    p.update(protocol=PROTOCOL,risk_registry='training/review_policy/history_cases.json',activation_entries=['training/v159_boundary_train_v3.py','training/v159_boundary_evaluate_v3.py'])
    p['total_future_caps']['classifier_forward_chunks']=78072
    p['total_future_caps']['opinion_feature_blocks']=78072
    p['total_future_caps']['evaluation_classifier_forward_chunks']=720
    for b in p['role_call_budgets']:b['evaluation_classifier_cap_per_arm']=72+6*b['chunks']
    dummy=OUT/'synthetic_dependency.txt';dummy.write_text('original',encoding='utf-8')
    p['source_sha256']={f.relative_to(ROOT).as_posix():sha(f) for f in sources+[dummy,ROOT/p['risk_registry']]}
    plan=OUT/'synthetic_plan.json';plan.write_text(json.dumps(p,indent=2)+'\n',encoding='utf-8');assert review_plan(p)['plan_review_passed']
    seal=OUT/'synthetic_run_seal.json';trainer=ROOT/p['activation_entries'][0];evaluator=ROOT/p['activation_entries'][1]
    cases={}
    cases['unsealed_rejected']=rejected(lambda:require_run_seal(seal,trainer))
    seal_run(plan,trainer,sources+[dummy,ROOT/p['risk_registry']],seal);assert require_run_seal(seal,trainer)['protocol']==PROTOCOL and require_run_seal(seal,evaluator)['protocol']==PROTOCOL
    cases['both_real_registered_entries_accepted']=True
    cases['wrong_entry_rejected']=rejected(lambda:require_run_seal(seal,ROOT/'training/v159_boundary_train.py'))
    cases['seal_overwrite_rejected']=rejected(lambda:seal_run(plan,trainer,sources,seal))
    dummy.write_text('changed',encoding='utf-8');cases['changed_source_rejected']=rejected(lambda:require_run_seal(seal,trainer));dummy.write_text('original',encoding='utf-8')
    r=dict(status='V159_boundary_fit_executed',selected_by_score=False,fold=0,arm='B',gradient_iterations=2,full_class_gradients=4,proposal_evaluations=3,accepted_updates=2,termination='no_feasible_step',counts=dict(head_attempts=8,head_completed=8,feature_attempts=8,feature_completed=8,gradient_attempts=4,gradient_completed=4),last5=[dict(update=1,parameter_sha256='state1'),dict(update=2,parameter_sha256='state2')],endpoint_parameter_sha256='state2')
    assert require_checkpoint(p,r)['endpoint_review_passed'];cases['finite_stop_accepted_without_claiming_convergence']=True
    for name,key,value in [('score_selected_rejected','selected_by_score',True),('extra_proposal_rejected','proposal_evaluations',601),('wrong_stop_rejected','termination','accepted_update_budget'),('wrong_endpoint_rejected','endpoint_parameter_sha256','state1')]:
        bad=copy.deepcopy(r);bad[key]=value;cases[name]=rejected(lambda:require_checkpoint(p,bad))
    bad=copy.deepcopy(r);bad['last5'][1]['parameter_sha256']='state1';cases['fabricated_window_rejected']=rejected(lambda:require_checkpoint(p,bad))
    y=np.array([0,0,0,0,1,2,1,2,1,2,1,2]);ref=pd.DataFrame(dict(row_position=np.arange(12),truth=y,fold=[0,0,1,1,0,0,0,0,1,1,1,1],root=np.arange(100,112),route=['other']*4+['asa']*8))
    a=y.copy();a[[4,8]]=2;pred=pd.DataFrame(dict(row_position=np.arange(12),pred_A=a,pred_B=y))
    profile=dict(expected_full_rows=12,asa_error_limits=dict(M=1,S=1,total=1),minimum_improved_folds=2,protected_S_roots=[],required_full_classes=[0,1,2])
    assert evaluate_primary(ref,pred,profile)['primary_quality_passed'];cases['legitimate_small_three_class_gain_accepted']=True
    cases['missing_row_rejected']=rejected(lambda:evaluate_primary(ref,pred.iloc[:-1],profile))
    bad=pred.copy();bad.loc[5,'pred_B']=1;failed=evaluate_primary(ref,bad,profile);assert not failed['gates']['paired_M_S_protected'] and not failed['primary_quality_passed'];cases['S_regression_cannot_hide_in_M_gain']=True
    bad=pred.copy();bad['truth']=y;bad.loc[4,'truth']=2;cases['changed_gold_rejected']=rejected(lambda:evaluate_primary(ref,bad,profile))
    check_bindings(bound)
    result=dict(status='new_finite_six_fit_protocol_actual_synthetic_rejection_and_positive_cases_passed',protocol=PROTOCOL,cases=cases,synthetic_classifier_head_calls=0,official_classifier_calls=0,official_features_calls=0,official_gradients=0,official_fits=0,official_updates=0,quality_acceptance=False,source_sha256=bound)
    (OUT/'qualification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
