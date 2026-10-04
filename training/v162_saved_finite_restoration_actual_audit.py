"""Recompute every complete displacement, original row, source and cost."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,finite_step_review
from v160_saved_diagnostic_actual_result_audit import audit_scope
from v160_independent_saved_direction_certificate import dot,norm
from v161_saved_fixed_error_result_audit import risks,COHORT
from v161_independent_all_finite_results_review import parameter_hash
TRIAL=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002'
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v162_saved_finite_restoration_actual_audit_20261002'

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir()
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),ROOT/'training/v160_saved_diagnostic_actual_result_audit.py',ROOT/'training/v161_saved_fixed_error_result_audit.py',ROOT/'training/v160_independent_saved_direction_certificate.py',ROOT/'training/v161_independent_all_finite_results_review.py'}
    files|={p for p in COHORT.rglob('*') if p.is_file()}
    files|={DIAG/f'role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy' for role in range(3) for c in [1,2]}
    files|={ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_source_sum_repeat_policy.py'}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));plan=read(ROOT/'training/review_policy/v162_fixed_endpoint_finite_restoration_contract.json');reports=[]
    for role in range(3):
        folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');spec=plan['roles'][role]
        assert diag['exception'] is None and all(diag[k]==0 for k in ['new_fits','permanent_updates','new_fixed_error_target_gradients','new_full_original_class_gradients'])
        state=torch.load(ROOT/f'artifacts/v159_class_boundary_numeric_trial_20261002/fold{role}_B/endpoint.pt',map_location='cpu',weights_only=True)['state'];assert parameter_hash(state)==diag['initial_parameter_sha256']==diag['restored_parameter_sha256']
        calls=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
        for kind,prefix in [('head','head'),('feature','feature'),('full_class_gradient','gradient'),('full_parameter_margin_gradient','margin')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                ev=[c for c in calls if c['kind']==kind and c['event']==event];assert [c['ordinal'] for c in ev]==list(range(1,len(ev)+1));counts[prefix+'_'+suffix]=len(ev)
        assert counts==diag['counts'] and counts['gradient_attempts']==0 and counts['head_attempts']<=spec['head_cap'] and counts['margin_attempts']<=spec['fresh_margin_gradient_cap']
        assert counts['head_attempts']==(2+diag['finite_proposals'])*(spec['OOF_chunks']+12)+counts['margin_attempts']
        baseline={scope:audit_scope(folder/'baseline',scope)[0] for scope in ['OOF','deployment']};b=baseline['OOF'];targets=pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet');before=risks(b,targets);mass=b.truth.value_counts().reindex([1,2]).to_numpy()
        gs=[np.load(DIAG/f'role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1,2]]
        for scope in baseline:
            restored,_=audit_scope(folder/'restored',scope);fixed=baseline[scope];assert np.array_equal(restored.pred,fixed.pred) and repeat_values(restored[['p0','p1','p2']].to_numpy(),fixed[['p0','p1','p2']].to_numpy(),'probability')['passed']
        for path in folder.glob('fresh_normals/*/measurement_repeat_review.json'):assert repeat_gradient(np.load(path.parent/'repeat0_gradient.npy'),np.load(path.parent/'repeat1_gradient.npy'))['passed']
        proposals=[];restorations=[]
        for path in sorted(folder.glob('restoration*/original_unit_restoration_review.json')):
            r=read(path);u=np.load(path.parent/'current_displacement.npy');a=np.load(path.parent/'active_complete_margin_normals.npy');bm=np.load(path.parent/'base_margins.npy');cm=np.load(path.parent/'actual_current_margins.npy')
            correction=np.load(path.parent/'correction.npy');candidate=np.load(path.parent/'displacement.npy');assert np.array_equal(candidate,u+correction)
            for i,row in enumerate(a):
                value,error=dot(row,correction);reported=r['residual_reviews'][i];assert value==reported['linear_recovery'] and float(bm[i]-cm[i])==reported['target'];assert (abs(value-reported['target'])<=reported['residual_limit'])==reported['passed']
                limit=max(error,16*np.finfo(float).eps*max(abs(reported['target']),norm(row)*norm(correction)));assert limit==reported['residual_limit'] and error==reported['arithmetic_error']
            slopes=[dot(g,candidate)[0] for g in gs];assert repeat_values(slopes,[v['linear_change'] for v in r['class_reviews']],'linear_change')['passed']
            for g,review in zip(gs,r['class_reviews']):
                value,error=dot(g,candidate);resolution=16*np.finfo(float).eps*norm(g)*norm(candidate);assert value==review['linear_change'] and error==review['arithmetic_error'] and resolution==review['resolution'] and (value<-max(error,resolution))==review['resolved_negative']
            restorations.append(dict(path=path.relative_to(ROOT).as_posix(),status=r['status'],normal_rank=r['normal_rank'],active_functions=len(a),class_reviews=r['class_reviews'],all_residuals_passed=all(v['passed'] for v in r['residual_reviews'])))
            probepath=path.parent/'finite_probe/probe.json'
            if not probepath.exists():continue
            p=read(probepath);context=read(probepath.parent/'finite_restoration_context.json');assert p['step']==context['Armijo_step']==1. and context['class_slopes_are_gradient_dot_complete_displacement']
            assert p['probe_parameter_sha256']==parameter_hash(state,candidate,1.)
            sr={}
            for scope in ['OOF','deployment']:
                frame,sr[scope]=audit_scope(probepath.parent,scope);fixed=baseline[scope];assert np.array_equal(frame.row_position,fixed.row_position) and np.array_equal(frame.truth,fixed.truth)
                assert sr[scope]['M_errors']==p[scope+'_stats']['M_errors'] and sr[scope]['S_errors']==p[scope+'_stats']['S_errors']
                oldwrong=fixed.pred.ne(fixed.truth).to_numpy();wrong=frame.pred.ne(frame.truth).to_numpy();pure=fixed.pure_current_input.to_numpy();sr[scope].update(repairs_vs_endpoint=int((oldwrong&~wrong).sum()),pure_repairs_vs_endpoint=int((oldwrong&~wrong&pure).sum()),new_errors_vs_endpoint=int((~oldwrong&wrong).sum()),classification_changes_vs_endpoint=int(frame.pred.ne(fixed.pred).sum()))
                if scope=='OOF':after=risks(frame,targets)
            recorded=dict(fixed_pure_error_contribution=np.load(probepath.parent/'fixed_error_risk.npy'),full_original_class_CE=np.load(probepath.parent/'full_original_class_risk.npy'))
            for key in recorded:assert repeat_values(after[key],recorded[key],'risk')['passed']
            count_guard=all(sr['OOF'][key]<=int((b.pred.ne(b.truth)&b.truth.eq(c)).sum()) for key,c in [('M_errors',1),('S_errors',2)])
            guard=count_guard and sr['OOF']['protected_regressions']==0 and sr['deployment']['new_errors_vs_initial']==0 and p['deployment_stats']['mastered'] and p['joint_TRAIN_retention']['passed'];assert guard==p['classification_guard']
            finite=finite_step_review(before['fixed_pure_error_contribution'],recorded['fixed_pure_error_contribution'],slopes,*mass,'B',1.,guard)
            assert bool(finite['accepted'] and p['actual_parameter_change'])==p['accepted']
            proposals.append(dict(path=probepath.relative_to(ROOT).as_posix(),accepted=p['accepted'],scopes=sr,fixed_target_drop=(before['fixed_pure_error_contribution']-recorded['fixed_pure_error_contribution']).tolist(),full_original_CE_change=(recorded['full_original_class_CE']-before['full_original_class_CE']).tolist(),finite_review=finite))
        assert len(restorations)==diag['restoration_solves'] and len(proposals)==diag['finite_proposals'] and sum(p['accepted'] for p in proposals)==int(diag['actual_finite_restoration_pass'])
        reports.append(dict(role=role,status=diag['status'],actual_finite_restoration_pass=diag['actual_finite_restoration_pass'],counts=counts,restorations=restorations,proposals=proposals,parameter_hash_restored=True,original_joint_retention_passed=True))
    check_bindings(bindings);total={key:sum(r['counts'][key] for r in reports) for key in reports[0]['counts']};save(OUT/'audit.json',dict(status='saved_all_generic_finite_restoration_actual_vectors_original_rows_risks_guards_parameter_hashes_and_costs_passed',reports=reports,actual_new_counts=total,all_three_actual_finite_restoration_pass=all(r['actual_finite_restoration_pass'] for r in reports),official_audit_calls=0,new_fits=0,permanent_updates=0,quality_acceptance=False,source_sha256=bindings))
    print(json.dumps(dict(status='V162_saved_actual_restoration_audit_passed',actual_new_counts=total,official_audit_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0),indent=2)+'\n',encoding='utf-8')
        raise
