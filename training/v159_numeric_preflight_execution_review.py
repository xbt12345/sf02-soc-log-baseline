"""Reconcile actually completed preflight and failed side binding, no new calls."""
import json
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_runtime_v4 import OUT as TRIAL,require,save
from v159_float64_repeat_policy_v2 import repeat_gradient,repeat_direction,repeat_values
OUT=ROOT/'artifacts/v159_numeric_preflight_execution_review_20261002'
def main():
    assert not OUT.exists();OUT.mkdir();plan=require(ROOT/'training/v159_boundary_train_v4.py')
    independent=ROOT/'artifacts/v159_independent_numeric_entry_activation_review_20261002/audit.json';audit=read(independent);check_bindings(audit['source_sha256'])
    p=read(TRIAL/'preflight.json');assert p['actual_head_forward_calls']==p['actual_opinion_feature_calls']==336 and p['actual_full_class_gradients']==12 and p['official_fits']==p['official_updates']==0 and p['joint_TRAIN_retention']['passed']
    check_bindings(p['source_sha256'])
    sources=[Path(__file__).resolve(),ROOT/'training/v159_bind_numeric_independent_activation.py',independent,TRIAL/'preflight.json',TRIAL/'preflight_original_console.txt',TRIAL/'registration.json',TRIAL/'run_seal.json',ROOT/'training/v159_float64_repeat_policy_v2.py']+[ROOT/k for k in p['source_sha256']]
    binding={q.relative_to(ROOT).as_posix():sha(q) for q in sources};save(OUT/'pre_saved_preflight_bindings.json',dict(status='before_saved_preflight_replay_no_new_model_calls',source_sha256=binding))
    actual=dict(head_attempts=0,head_completed=0,feature_attempts=0,feature_completed=0,gradient_attempts=0,gradient_completed=0);cases=[]
    for report in p['reports']:
        f,arm=report['fold'],report['arm'];folder=TRIAL/f'preflight{f}_{arm}';events=[json.loads(z) for z in (folder/'calls.jsonl').read_text().splitlines()]
        for kind,key in [('head','head'),('feature','feature'),('full_class_gradient','gradient')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                count=sum(e['kind']==kind and e['event']==event for e in events);assert count==report['counts'][key+'_'+suffix];actual[key+'_'+suffix]+=count
        assert report['parameters_unchanged'] and report['deployment_stats']['mastered'] and report['OOF_stats']['protected_regressions']==0
        if arm=='A':
            gs=[[np.load(folder/f'repetition{r}_class{c}_complete_gradient.npy') for c in [1,2]] for r in [0,1]]
            gradient=[repeat_gradient(gs[0][j],gs[1][j]) for j in [0,1]];mass=plan['role_call_budgets'][f]['original_class_mass'];directions=[repeat_direction(*gs,*mass[1:],a) for a in ['A','B']]
            assert all(z['passed'] for z in gradient+directions) and all(z['descent_qualified'] for z in directions)
            risk=np.load(folder/'repetition0_class1_risk.npy');risk_reports=[]
            for r in [0,1]:
                for c in [1,2]:risk_reports.append(repeat_values(risk,np.load(folder/f'repetition{r}_class{c}_risk.npy'),'risk'))
            assert all(z['passed'] for z in risk_reports)
            cases.append(dict(fold=f,gradient=gradient,directions=directions,risks=risk_reports))
    assert actual==dict(head_attempts=336,head_completed=336,feature_attempts=336,feature_completed=336,gradient_attempts=12,gradient_completed=12)
    assert not (TRIAL/'independent_activation_review_binding.json').exists() and not list(TRIAL.glob('fold*_*'))
    oldest=min((TRIAL/f'preflight{f}_A/calls.jsonl').stat().st_ctime_ns for f in range(3));review_time=independent.stat().st_ctime_ns
    # Filesystem chronology is reported as such, never a successful binding.
    result=dict(status='actual_three_role_preflight_saved_vectors_costs_and_unchanged_sources_reconciled',preflight_head_calls=336,preflight_opinion_feature_calls=336,preflight_full_class_gradients=12,cumulative_actual=dict(head_calls=460,opinion_feature_calls=460,full_class_gradients=20,fits=0,updates=0),cases=cases,joint_TRAIN_retention=p['joint_TRAIN_retention'],remaining_cumulative_caps=dict(head_calls=77736,opinion_feature_calls=77736,full_class_gradients=2400,fits=6),
      side_binding_failure=dict(status='failed_no_binding_receipt_created',error_type='AssertionError',source='training/v159_bind_numeric_independent_activation.py',reason='preflight_original_console created while binder still checking full physical sources; launcher did not await binder completion',source_sha256=sha(ROOT/'training/v159_bind_numeric_independent_activation.py'),independent_review_itself_preexisted=review_time<oldest,filesystem_independent_review_creation_ns=review_time,filesystem_first_preflight_journal_creation_ns=oldest,not_relabelled_as_prospective_binding_success=True),
      official_new_heads=0,official_new_features=0,official_new_gradients=0,official_new_fits=0,official_new_updates=0,first_training_issue_passed=False,quality_acceptance=False,goal_status='active',source_sha256=binding)
    check_bindings(binding);save(OUT/'review.json',result);print(json.dumps(dict(status=result['status'],cumulative_actual=result['cumulative_actual'],old_retention=p['joint_TRAIN_retention']['passed'],side_binding_receipt_success=False),ensure_ascii=False))
if __name__=='__main__':main()
