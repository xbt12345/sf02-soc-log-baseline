"""Recompute every original tail row, target/full risk, guard and cost."""
import json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v160_saved_diagnostic_actual_result_audit import audit_scope
from v161_saved_fixed_error_result_audit import risks,COHORT
from v159_float64_repeat_policy_v2 import repeat_values,finite_step_review
TRIAL=ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002'
OUT=ROOT/'artifacts/v161_cached_tail_actual_result_audit_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();folder=TRIAL/'role2';diag=read(folder/'diagnostic.json');assert diag['exception'] is None
    files={Path(__file__).resolve(),ROOT/'training/v161_saved_fixed_error_result_audit.py',ROOT/'training/v160_saved_diagnostic_actual_result_audit.py',ROOT/'training/v159_float64_repeat_policy_v2.py',COHORT/'role2/fixed_pure_error_targets.parquet'}|{p for p in TRIAL.rglob('*') if p.is_file()}
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(OUT/'pre_bindings.json',dict(source_sha256=binding,official_calls=0))
    assert diag['initial_parameter_sha256']==diag['restored_parameter_sha256'] and diag['restored_joint_TRAIN_retention']['passed'] and all(diag[k]==0 for k in ['new_fixed_error_target_gradients','new_full_original_class_gradients','new_margin_gradients','QP_solves','new_fits','permanent_updates'])
    calls=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
    for kind,prefix in [('head','head'),('feature','feature'),('full_class_gradient','gradient')]:
        for event,suffix in [('attempt','attempts'),('completed','completed')]:
            ev=[r for r in calls if r['kind']==kind and r['event']==event];assert [r['ordinal'] for r in ev]==list(range(1,len(ev)+1));counts[prefix+'_'+suffix]=len(ev)
    assert counts==diag['counts'] and counts['gradient_attempts']==0
    baseline={scope:audit_scope(folder/'baseline',scope)[0] for scope in ['OOF','deployment']};b=baseline['OOF'];targets=pd.read_parquet(COHORT/'role2/fixed_pure_error_targets.parquet');before=risks(b,targets);mass=b.truth.value_counts().reindex([1,2]).to_numpy()
    assert len(b)==92320 and b.protected_correct.sum()==90528
    for scope in baseline:
        r,_=audit_scope(folder/'restored',scope);fixed=baseline[scope];assert np.array_equal(r.pred,fixed.pred) and repeat_values(r[['p0','p1','p2']].to_numpy(),fixed[['p0','p1','p2']].to_numpy(),'probability')['passed']
    probes=[]
    for path in sorted(folder.glob('probe*/probe.json')):
        p=read(path);sr={}
        for scope in ['OOF','deployment']:
            r,sr[scope]=audit_scope(path.parent,scope);fixed=baseline[scope];assert np.array_equal(r.row_position,fixed.row_position) and np.array_equal(r.truth,fixed.truth)
            oldwrong=fixed.pred.ne(fixed.truth).to_numpy();wrong=r.pred.ne(r.truth).to_numpy();pure=fixed.pure_current_input.to_numpy();sr[scope].update(classification_changes_vs_endpoint=int(r.pred.ne(fixed.pred).sum()),repairs_vs_endpoint=int((oldwrong&~wrong).sum()),pure_repairs_vs_endpoint=int((oldwrong&~wrong&pure).sum()),new_errors_vs_endpoint=int((~oldwrong&wrong).sum()))
            if scope=='OOF':actual=risks(r,targets)
        saved=dict(fixed_pure_error_contribution=np.load(path.parent/'fixed_error_risk.npy'),full_original_class_CE=np.load(path.parent/'full_original_class_risk.npy'))
        for key in saved:assert repeat_values(actual[key],saved[key],'risk')['passed']
        count_guard=all(sr['OOF'][key]<=int((b.pred.ne(b.truth)&b.truth.eq(c)).sum()) for key,c in [('M_errors',1),('S_errors',2)])
        guard=count_guard and sr['OOF']['protected_regressions']==0 and sr['deployment']['new_errors_vs_initial']==0 and p['deployment_stats']['mastered'] and p['joint_TRAIN_retention']['passed'];assert guard==p['classification_guard']
        review=finite_step_review(before['fixed_pure_error_contribution'],saved['fixed_pure_error_contribution'],p['class_slopes'],*mass,'B',p['step'],guard);assert bool(review['accepted'] and p['actual_parameter_change'])==p['accepted']
        progress=read(path.parent/'fixed_endpoint_progress.json')
        for scope in sr:assert progress[scope]['classification_changes_vs_fixed_endpoint']==sr[scope]['classification_changes_vs_endpoint'] and progress[scope]['repairs_vs_fixed_endpoint']==sr[scope]['repairs_vs_endpoint']
        probes.append(dict(path=path.relative_to(ROOT).as_posix(),step=p['step'],accepted=p['accepted'],scopes=sr,fixed_error_target_drop=(before['fixed_pure_error_contribution']-saved['fixed_pure_error_contribution']).tolist(),full_original_CE_change=(saved['full_original_class_CE']-before['full_original_class_CE']).tolist(),finite_error_target_review=review))
    assert len(probes)==diag['finite_proposals'] and sum(p['accepted'] for p in probes)==int(diag['finite_error_target_pass']) and counts['head_attempts']==44+22*len(probes)
    check_bindings(binding);save(OUT/'audit.json',dict(status='all_actual_cached_error_target_tail_rows_risks_guards_costs_and_restore_passed',finite_error_target_pass=diag['finite_error_target_pass'],counts=counts,probes=probes,new_gradients=0,new_fits=0,permanent_updates=0,official_audit_calls=0,quality_acceptance=False,source_sha256=binding))
    print(json.dumps(dict(status='V161_actual_tail_result_audit_passed',counts=counts,official_audit_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():(OUT/'failure.json').write_text(json.dumps(dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0),indent=2)+'\n',encoding='utf-8')
        raise
