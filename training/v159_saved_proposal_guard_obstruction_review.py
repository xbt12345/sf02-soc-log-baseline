"""Disentangle actually saved rejected proposal guards; no model/feature calls."""
import json
from pathlib import Path
from collections import Counter
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import finite_step_review
TRIAL=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
OUT=ROOT/'artifacts/v159_saved_proposal_guard_obstruction_review_20261002'
def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve(),ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_boundary_train_v4.py',TRIAL/'run_seal.json',ROOT/'training/review_policy/v159_boundary_execution_contract_v4.json']
    paths += [TRIAL/f'fold{f}_{arm}'/name for f in range(3) for arm in ['A','B'] for name in ['fit.json','proposals.jsonl','directions.jsonl','started.json','calls.jsonl']]
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in paths};(OUT/'pre_saved_journal_bindings.json').write_text(json.dumps(dict(status='bound_before_saved_journal_decode_no_model_calls',source_sha256=binding),indent=2)+'\n',encoding='utf-8')
    reports=[]
    for f in range(3):
        for arm in ['A','B']:
            folder=TRIAL/f'fold{f}_{arm}';r=read(folder/'fit.json');mass=read(folder/'started.json')['original_class_mass'][1:];prop=[json.loads(z) for z in (folder/'proposals.jsonl').read_text().splitlines()]
            assert len(prop)==r['proposal_evaluations'];totals=Counter();terminal=[]
            for e in prop:
                oo,dd=e['OOF_stats'],e['deployment_stats']
                guards=dict(deployment_mastered=dd['mastered'],deployment_no_new_initial_errors=dd['new_errors_vs_initial']==0,OOF_pure_initial_correct_retained=arm=='A' or oo['protected_regressions']==0)
                guard=all(guards.values());assert guard==e['classification_guard']
                review=finite_step_review(e['base_risks'],e['trial_risks'],e['class_slopes'],*mass,arm,e['step'],guard)
                assert review==e['finite_numeric_review'] and review['accepted']==e['accepted']
                risk_only=finite_step_review(e['base_risks'],e['trial_risks'],e['class_slopes'],*mass,arm,e['step'],True)
                totals['accepted' if e['accepted'] else 'rejected']+=1
                if not e['accepted']:
                    for k,okay in guards.items():
                        if not okay:totals['failed_'+k]+=1
                    totals['risk_only_'+risk_only['reason']]+=1
                    if not guard and risk_only['accepted']:totals['classification_guard_only_obstruction']+=1
                if e['iteration']==r['gradient_iterations']:
                    terminal.append(dict(proposal=e['proposal'],step=e['step'],guard_components=guards,OOF_protected_regressions=oo['protected_regressions'],deployment_new_errors=dd['new_errors_vs_initial'],deployment_pure_errors=dd['pure_errors'],deployment_total_errors=dd['total_errors'],recorded_finite_reason=review['reason'],risk_only_finite_review=risk_only))
            reports.append(dict(fold=f,arm=arm,accepted_updates=r['accepted_updates'],iterations=r['gradient_iterations'],full_class_gradients=r['full_class_gradients'],proposals=r['proposal_evaluations'],head_calls=r['counts']['head_attempts'],termination=r['termination'],endpoint_OOF_stats=r['OOF_stats'],initial_class_risks=r['initial_stable_class_risks'],final_class_risks=r['final_stable_class_risks'],proposal_rejection_totals=dict(totals),terminal_iteration_proposals=terminal,
              endpoint_selection='last actually accepted state; no label-selected best',conclusion_scope='Only this fixed direction and registered step domain; not proof of empty feasible set, no learned solution, or convergence'))
    check_bindings(binding)
    result=dict(status='six_actual_fixed_fit_saved_proposal_guards_and_finite_risks_disentangled',reports=reports,actual_training=dict(fits=6,head_calls=sum(z['head_calls'] for z in reports),opinion_feature_calls=sum(z['head_calls'] for z in reports),full_class_gradients=sum(z['full_class_gradients'] for z in reports),iterations=sum(z['iterations'] for z in reports),proposals=sum(z['proposals'] for z in reports),accepted_updates=sum(z['accepted_updates'] for z in reports)),own_official_heads=0,own_official_features=0,own_official_gradients=0,own_fits=0,own_updates=0,first_training_issue_passed=False,quality_acceptance=False,source_sha256=binding)
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=result['status'],training=result['actual_training'],terminal=[dict(fold=z['fold'],arm=z['arm'],last=z['terminal_iteration_proposals'][-1]) for z in reports]),ensure_ascii=False))
if __name__=='__main__':main()
