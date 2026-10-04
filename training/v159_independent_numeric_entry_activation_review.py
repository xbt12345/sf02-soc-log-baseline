"""Review the prospective numeric entry without official model/function calls."""
import ast
import json
from pathlib import Path
from experiment_review_v159_v3 import review_plan, check_bindings, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v159_independent_numeric_entry_activation_review_20261002'


def node(path, name):
    found = [n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
             if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
    assert len(found)==1
    return ast.dump(found[0],include_attributes=False)


def main():
    assert not OUT.exists()
    plan_path=ROOT/'training/review_policy/v159_boundary_execution_contract_v4.json'
    old_path=plan_path.with_name('v159_boundary_execution_contract_v3.json')
    p=json.loads(plan_path.read_text(encoding='utf-8')); old=json.loads(old_path.read_text(encoding='utf-8'))
    review=review_plan(p);assert review['plan_review_passed']
    assert p['task_quality']==old['task_quality'] and p['solver']==old['solver']
    assert p['OOF_retention_refinement']==old['OOF_retention_refinement']
    assert p['numeric_repeat_policy']['operational_empirical_scale_not_universal_theoretical_error_bound']
    assert p['numeric_repeat_policy']['gradient_absolute_floor']==0
    ks=[10,4,10]
    caps=dict(preflight=8*sum(ks)+48*3,fit=2*(1603*sum(ks)+12*3),evaluation=2*(72*3+6*sum(ks)))
    future=p['total_future_caps'];cumulative=p['total_cumulative_caps'];carry=p['historical_technical_cost']
    assert caps==dict(preflight=336,fit=77016,evaluation=720)
    assert future['classifier_forward_chunks']==future['opinion_feature_blocks']==sum(caps.values())==78072
    assert future['fit_full_class_gradients']==2400 and future['full_class_gradients']==2412
    assert cumulative['classifier_forward_chunks']==cumulative['opinion_feature_blocks']==78072+124==78196
    assert cumulative['full_class_gradients']==2412+8==2420
    assert carry['fits']==carry['updates']==0 and carry['full_class_gradients']==8 and carry['classifier_forward_chunks']==124
    assert future['fits']==6 and future['proposal_evaluations']==3600 and future['accepted_updates']==1200
    original=ROOT/'training/v159_boundary_train_v3.py';new=ROOT/'training/v159_boundary_train_v4.py'
    unchanged={n:node(original,n)==node(new,n) for n in ['Counter','context','inputs','probabilities','risk','stats','rows','assign','restore']}
    assert all(unchanged.values())
    text=new.read_text(encoding='utf-8')
    assert 'resolved_class_direction as class_direction' in text and 'finite_step_review(base_risks,trial' in text
    assert "ds['mastered'] and ds['new_errors_vs_initial']==0 and (arm=='A' or os['protected_regressions']==0)" in text
    evaluator=ROOT/'training/v159_boundary_evaluate_v4.py';ev=evaluator.read_text(encoding='utf-8')
    assert "for item in r['last5']:" in ev and "ov,oq,olp,_=risk(model,ctx,'OOF',ctx['ids'])" in ev
    assert "np.array_equal(actual_oo.pred,oo.pred)" in ev and "np.array_equal(actual.pred,stored.pred)" in ev
    assert "repeat_values(q,np.load(folder/'endpoint_deployment_probability.npy'),'probability')" in ev
    evidence=[ROOT/'artifacts/v159_protocol_synthetic_qualification_v3_20261002/qualification.json',
              ROOT/'artifacts/v159_nonzero_real_dimension_numeric_qualification_v2_20261002/qualification.json',
              ROOT/'artifacts/v159_numeric_batch_replay_qualification_v2_20261002/qualification.json',
              ROOT/'artifacts/v159_independent_numeric_policy_saved_state_audit_20261002/audit.json',
              ROOT/'artifacts/v159_independent_combined_guard_audit_20261002/audit.json']
    for ep in evidence:
        q=json.loads(ep.read_text(encoding='utf-8'));check_bindings(q['source_sha256'])
    q=json.loads(evidence[0].read_text(encoding='utf-8'));assert len(q['cases'])==20 and all(q['cases'].values())
    q=json.loads(evidence[2].read_text(encoding='utf-8'));assert q['synthetic_real_dimension_finite_B_probe']['accepted']
    q=json.loads(evidence[4].read_text(encoding='utf-8'));assert all(r['targets_compatible'] for r in q['roles'])
    paths=[Path(__file__).resolve(),plan_path,old_path,original,new,evaluator,
           ROOT/'training/v159_boundary_runtime_v4.py',ROOT/'training/experiment_review_v159_v3.py',
           ROOT/'training/v159_float64_repeat_policy_v2.py']+evidence
    report=dict(status='prospective_numeric_entry_eligible_for_seal_and_real_preflight',protocol=p['protocol'],
                unchanged_official_input_loss_guard_and_counter=unchanged,quality_gates_unchanged=True,
                budgets_independently_recomputed=caps,total_cumulative_caps=cumulative,
                historical_failed_cost_conserved=True,source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in paths},
                official_classifier_calls=0,official_feature_calls=0,official_gradients=0,official_fits=0,official_updates=0,
                permission_scope='Worker may register/seal the numeric entry and run 336 forward/feature chunks and 12 full class preflight gradients. Parameter updates require successful real preflight.',
                quality_acceptance=False,first_training_issue_passed=False,model_promoted=False)
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','unchanged_official_input_loss_guard_and_counter','quality_gates_unchanged','budgets_independently_recomputed','permission_scope','quality_acceptance']},ensure_ascii=False))


if __name__=='__main__':main()
