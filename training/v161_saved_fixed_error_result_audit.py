"""Recompute complete original finite-target results and costs, no model calls."""
import json,math,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,finite_step_review
from v160_saved_diagnostic_actual_result_audit import audit_scope
from v160_saved_vector_numeric_polish import original_float64_review
TRIAL=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
COHORT=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT=ROOT/'artifacts/v161_saved_fixed_error_result_audit_20261002'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def risks(r,targets):
    mask=r.row_position.isin(targets.row_position)
    mass=[int(r.truth.eq(c).sum()) for c in [1,2]]
    return dict(fixed_pure_error_contribution=np.array([math.fsum(r.loc[mask&r.truth.eq(c),'stable_CE'])/mass[c-1] for c in [1,2]]),full_original_class_CE=np.array([math.fsum(r.loc[r.truth.eq(c),'stable_CE'])/mass[c-1] for c in [1,2]]))

def main():
    assert not OUT.exists() and all((TRIAL/f'role{r}/diagnostic.json').exists() for r in range(3));OUT.mkdir()
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{p for p in COHORT.rglob('*') if p.is_file()}|{Path(__file__).resolve(),ROOT/'training/v160_saved_diagnostic_actual_result_audit.py',ROOT/'training/v160_saved_vector_numeric_polish.py',ROOT/'training/v159_float64_repeat_policy_v2.py'}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_saved_result_bindings.json',dict(source_sha256=bindings,official_calls=0));reports=[]
    plan=read(ROOT/'training/review_policy/v161_fixed_error_endpoint_diagnostic_contract.json')
    for role in range(3):
        folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');spec=plan['roles'][role]
        assert diag['exception'] is None and diag['initial_parameter_sha256']==diag['restored_parameter_sha256']==spec['endpoint_parameter_sha256'] and diag['restored_joint_TRAIN_retention']['passed']
        assert diag['new_fits']==diag['permanent_updates']==diag['new_full_original_class_gradients']==0
        calls=[json.loads(s) for s in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
        assert not any(c['kind']=='full_class_gradient' for c in calls)
        for kind,prefix in [('head','head'),('feature','feature'),('fixed_error_target_gradient','gradient'),('full_parameter_margin_gradient','margin')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                events=[c for c in calls if c['kind']==kind and c['event']==event];assert [c['ordinal'] for c in events]==list(range(1,len(events)+1));counts[prefix+'_'+suffix]=len(events)
        assert counts==diag['counts'] and counts['gradient_attempts']==counts['gradient_completed']==4 and counts['head_attempts']<=spec['head_cap'] and counts['margin_attempts']<=spec['fresh_margin_gradient_cap']
        baseline={scope:audit_scope(folder/('baseline_error_class1_repeat0' if scope=='OOF' else 'baseline_deployment'),scope)[0] for scope in ['OOF','deployment']}
        targets=pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet');b=baseline['OOF'];pure=b.pure_current_input.to_numpy();correct=b.pred.eq(b.truth).to_numpy();assert np.array_equal(b.loc[~correct&pure,'row_position'],targets.row_position)
        protection=pd.read_parquet(folder/'fixed_correct_protection.parquet');assert np.array_equal(protection.row_position,b.loc[pure&correct,'row_position']) and np.array_equal(b.protected_correct,pure&correct)
        mass=b.truth.value_counts().reindex([1,2]).to_numpy();base_risks=risks(b,targets);gs=[]
        for cls in [1,2]:
            g=[np.load(folder/f'baseline_error_class{cls}_repeat{j}/complete_fixed_error_target_gradient.npy') for j in range(2)]
            gr=repeat_gradient(*g);assert gr['passed'];gs.append(g[0])
            for j in range(2):
                p=folder/f'baseline_error_class{cls}_repeat{j}';r,_=audit_scope(p,'OOF');assert np.array_equal(r.pred,b.pred)
                recomputed=risks(r,targets)
                for key,value in recomputed.items():assert repeat_values(value,np.load(p/f'{key}.npy'),'risk')['passed']
        for scope in ['OOF','deployment']:
            restored,_=audit_scope(folder/'restored',scope);fixed=baseline[scope]
            assert np.array_equal(restored.pred,fixed.pred) and repeat_values(restored[['p0','p1','p2']].to_numpy(),fixed[['p0','p1','p2']].to_numpy(),'probability')['passed']
        probes=[]
        for path in sorted(folder.glob('round*/probe*/probe.json')):
            p=read(path);scope_reports={}
            for scope in ['OOF','deployment']:
                r,sr=audit_scope(path.parent,scope);fixed=baseline[scope];assert np.array_equal(r[['row_position','truth']].to_numpy(),fixed[['row_position','truth']].to_numpy())
                oldwrong=fixed.pred.ne(fixed.truth).to_numpy();wrong=r.pred.ne(r.truth).to_numpy();pure=fixed.pure_current_input.to_numpy()
                sr.update(repairs_vs_endpoint=int((oldwrong&~wrong).sum()),pure_repairs_vs_endpoint=int((oldwrong&~wrong&pure).sum()),new_errors_vs_endpoint=int((~oldwrong&wrong).sum()),classification_changes_vs_endpoint=int(r.pred.ne(fixed.pred).sum()))
                progress=read(path.parent/'fixed_endpoint_progress.json')[scope];assert progress['repairs_vs_fixed_endpoint']==sr['repairs_vs_endpoint'] and progress['classification_changes_vs_fixed_endpoint']==sr['classification_changes_vs_endpoint']
                scope_reports[scope]=sr
                if scope=='OOF':oo=r
            recomputed=risks(oo,targets);saved=dict(fixed_pure_error_contribution=np.load(path.parent/'fixed_error_risk.npy'),full_original_class_CE=np.load(path.parent/'full_original_class_risk.npy'))
            for key in saved:assert repeat_values(recomputed[key],saved[key],'risk')['passed']
            count_guard=all(scope_reports['OOF'][key]<=int(((b.pred!=b.truth)&b.truth.eq(c)).sum()) for key,c in [('M_errors',1),('S_errors',2)])
            guard=count_guard and scope_reports['OOF']['protected_regressions']==0 and scope_reports['deployment']['new_errors_vs_initial']==0 and p['deployment_stats']['mastered'] and p['joint_TRAIN_retention']['passed']
            assert guard==p['classification_guard'] and count_guard==p['full_original_M_S_error_count_guard']
            finite=finite_step_review(base_risks['fixed_pure_error_contribution'],saved['fixed_pure_error_contribution'],p['class_slopes'],*mass,'B',p['step'],guard)
            assert finite['accepted']==p['finite_error_target_review']['accepted'] and bool(finite['accepted'] and p['actual_parameter_change'])==p['accepted']
            # Reporting separates residual wrong targets from repaired target confidence.
            target_mask=oo.row_position.isin(targets.row_position);decomposition=oo.loc[target_mask].assign(current_wrong=oo.loc[target_mask].pred.ne(oo.loc[target_mask].truth)).groupby(['root','truth','current_wrong']).agg(original_rows=('truth','size'),stable_CE_sum=('stable_CE','sum')).reset_index()
            decomposition.to_parquet(OUT/f'role{role}_{path.parent.parent.name}_{path.parent.name}_fixed_target_residual_sources.parquet',index=False)
            probes.append(dict(path=path.relative_to(ROOT).as_posix(),step=p['step'],accepted=p['accepted'],scopes=scope_reports,fixed_error_target_drop=(base_risks['fixed_pure_error_contribution']-saved['fixed_pure_error_contribution']).tolist(),full_original_CE_change=(saved['full_original_class_CE']-base_risks['full_original_class_CE']).tolist(),finite_error_target_review=finite))
        assert len(probes)==diag['finite_proposals'] and sum(p['accepted'] for p in probes)==int(diag['finite_error_target_pass'])
        certificates=[]
        for path in sorted(folder.glob('round*/polished_certificate.json')):
            p=read(path)
            if 'raw' not in p and not (path.parent/'polished_raw.npy').exists():certificates.append(dict(path=path.relative_to(ROOT).as_posix(),status=p['status']));continue
            a=np.load(path.parent/'raw_margin_normals.npy');raw=np.load(path.parent/'polished_raw.npy');review=original_float64_review(*gs,a,p['alpha'],np.array(p['multipliers']),raw)
            assert review['passed']==p['passed'];certificates.append(dict(path=path.relative_to(ROOT).as_posix(),status=p['status'],original_certificate_passed=review['passed'],inward_corrections=len(p['inward_trace'])))
        for path in folder.glob('fresh_normals/*/measurement_repeat_review.json'):assert repeat_gradient(np.load(path.parent/'repeat0_gradient.npy'),np.load(path.parent/'repeat1_gradient.npy'))['passed']
        reports.append(dict(role=role,status=diag['status'],finite_error_target_pass=diag['finite_error_target_pass'],counts=counts,probes=probes,certificates=certificates,parameter_restored=True,joint_retention_passed=True))
    check_bindings(bindings);total={key:sum(r['counts'][key] for r in reports) for key in reports[0]['counts']}
    save(OUT/'audit.json',dict(status='all_V161_saved_original_rows_error_targets_full_CE_gradient_repeats_finite_guards_certificates_and_costs_passed',reports=reports,actual_new_counts=total,all_roles_finite_error_target_pass=all(r['finite_error_target_pass'] for r in reports),official_audit_calls=0,new_fits=0,permanent_updates=0,quality_acceptance=False,source_sha256=bindings))
    print(json.dumps(dict(status='V161_saved_actual_audit_passed',actual_new_counts=total,official_audit_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as err:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(err).__name__,error=str(err),traceback=traceback.format_exc(),official_calls=0))
        raise
