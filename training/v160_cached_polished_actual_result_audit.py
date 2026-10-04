"""Every cached-polish probe row/source/risk/progress/cost, no model calls."""
import json,math,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values,finite_step_review
from v160_saved_diagnostic_actual_result_audit import audit_scope
TRIAL=ROOT/'artifacts/v160_cached_polished_direction_finite_probe_20261002'
OLD=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v160_cached_polished_actual_result_audit_20261002'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in [1,2]);OUT.mkdir()
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),ROOT/'training/v160_saved_diagnostic_actual_result_audit.py',ROOT/'training/v160_cached_polished_direction_finite_probe.py'}
    binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_saved_result_bindings.json',dict(source_sha256=binding,official_calls=0));reports=[]
    for role in [1,2]:
        folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');assert diag['finite_probe_pass'] and diag['initial_parameter_sha256']==diag['restored_parameter_sha256'] and diag['restored_joint_TRAIN_retention']['passed'] and diag['new_fits']==diag['permanent_updates']==diag['new_class_gradients']==diag['new_margin_gradients']==0
        calls=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
        for kind,prefix in [('head','head'),('feature','feature'),('full_class_gradient','gradient')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                events=[c for c in calls if c['kind']==kind and c['event']==event];assert [c['ordinal'] for c in events]==list(range(1,len(events)+1));counts[prefix+'_'+suffix]=len(events)
        assert counts==diag['counts'] and counts['gradient_attempts']==0
        baseline={scope:audit_scope(folder/'baseline',scope)[0] for scope in ['OOF','deployment']};restored={scope:audit_scope(folder/'restored',scope)[0] for scope in ['OOF','deployment']}
        for scope in baseline:
            assert np.array_equal(baseline[scope].pred,restored[scope].pred) and repeat_values(baseline[scope][['p0','p1','p2']].to_numpy(),restored[scope][['p0','p1','p2']].to_numpy(),'probability')['passed']
        protection=pd.read_parquet(folder/'fixed_endpoint_correct_protection_rows.parquet');b=baseline['OOF'];expected=b[b.pred.eq(b.truth)&b.pure_current_input]
        assert np.array_equal(protection.row_position,expected.row_position) and protection.pure_current_input.all()
        probes=[];mass=b.truth.value_counts().reindex([1,2]).to_numpy();before=np.load(folder/'baseline_risk.npy')
        for path in sorted(folder.glob('probe*/probe.json')):
            p=read(path);progress=read(path.parent/'fixed_endpoint_progress.json');scope_reports={};risk=[]
            for scope in ['OOF','deployment']:
                r,sr=audit_scope(path.parent,scope);fixed=baseline[scope]
                assert np.array_equal(r[['row_position','truth']].to_numpy(),fixed[['row_position','truth']].to_numpy())
                oldwrong=fixed.pred.ne(fixed.truth).to_numpy();newwrong=r.pred.ne(r.truth).to_numpy();pure=fixed.pure_current_input.to_numpy()
                actual=dict(classification_changes_vs_fixed_endpoint=int(r.pred.ne(fixed.pred).sum()),repairs_vs_fixed_endpoint=int((oldwrong&~newwrong).sum()),new_errors_vs_fixed_endpoint=int((~oldwrong&newwrong).sum()),pure_repairs_vs_fixed_endpoint=int((oldwrong&~newwrong&pure).sum()),pure_new_errors_vs_fixed_endpoint=int((~oldwrong&newwrong&pure).sum()))
                assert all(actual[k]==progress[scope][k] for k in actual)
                oldlp=fixed[['logp0','logp1','logp2']].to_numpy();newlp=r[['logp0','logp1','logp2']].to_numpy();ii=np.arange(len(fixed));truth=fixed.truth.to_numpy(np.int64);rival=fixed.pred.to_numpy(np.int64)
                oldmargin=(oldlp[ii,truth]-oldlp[ii,rival])[oldwrong];newmargin=(newlp[ii,truth]-newlp[ii,rival])[oldwrong];delta=newmargin-oldmargin
                ledger=pd.read_parquet(path.parent/f'{scope}_endpoint_error_margin_progress.parquet');assert np.array_equal(ledger.row_position,fixed.loc[oldwrong,'row_position']) and np.array_equal(ledger.margin_change,delta)
                assert progress[scope]['old_error_margin_improved_rows']==int((delta>0).sum())
                if len(delta):assert progress[scope]['median_old_error_margin_change']==float(np.median(delta))
                scope_reports[scope]=dict(**sr,**actual,old_error_margin_improved_rows=int((delta>0).sum()),old_error_margin_median_change=float(np.median(delta)) if len(delta) else None)
                if scope=='OOF':risk=[math.fsum(r.loc[r.truth.eq(c),'stable_CE'])/int(r.truth.eq(c).sum()) for c in [1,2]]
            saved_risk=np.load(path.parent/'risk.npy');assert repeat_values(risk,saved_risk,'risk')['passed']
            guard=scope_reports['OOF']['protected_regressions']==0 and scope_reports['deployment']['new_errors_vs_initial']==0 and p['deployment_stats']['mastered'] and p['joint_TRAIN_retention']['passed'];assert guard==p['classification_guard']
            review=finite_step_review(before,saved_risk,p['class_slopes'],*mass,'B',p['step'],guard);assert review['accepted']==p['finite_numeric_review']['accepted'] and bool(review['accepted'] and p['actual_parameter_change'])==p['accepted']
            probes.append(dict(path=path.relative_to(ROOT).as_posix(),accepted=p['accepted'],step=p['step'],scopes=scope_reports,finite_numeric_review=review))
        assert len(probes)==diag['finite_proposals'] and sum(p['accepted'] for p in probes)==1
        reports.append(dict(role=role,counts=counts,finite_proposals=len(probes),fixed_endpoint_pure_protected_rows=len(protection),parameter_hash_restored=True,all_joint_TRAIN_retention_passed=True,probes=probes))
    check_bindings(binding);counts={key:sum(r['counts'][key] for r in reports) for key in reports[0]['counts']}
    assert counts['head_attempts']==410 and counts['gradient_attempts']==0
    save(OUT/'audit.json',dict(status='all_cached_polished_actual_rows_sources_risk_incremental_protection_margin_progress_costs_passed',reports=reports,actual_new_counts=counts,actual_new_finite_proposals=19,new_gradients=0,new_fits=0,permanent_updates=0,three_fixed_endpoints_finite_mechanism_passed=True,training_mastery_passed=False,quality_acceptance=False,official_audit_calls=0,source_sha256=binding,scope='Own saved-result recomputation, no independent actor or trained-model quality claim.'))
    print(json.dumps(dict(status='cached_polished_actual_result_audit_passed',actual_heads=410,new_gradient_or_fit_or_update=0,official_audit_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_audit_calls=0))
        raise
