"""Recompute every saved diagnostic row and vector; zero model calls."""
import json,math,traceback
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,finite_step_review
from v159_source_sum_repeat_policy import source_sum_repeat_review
from v160_independent_saved_direction_certificate import certificate
TRIAL=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v160_saved_diagnostic_actual_result_audit_20261002'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def audit_scope(folder,scope):
    q=np.load(folder/f'{scope}_q.npy');lp=np.load(folder/f'{scope}_logq.npy');r=pd.read_parquet(folder/f'{scope}_original_rows.parquet');s=pd.read_parquet(folder/f'{scope}_source_rows.parquet')
    idx=r.local.to_numpy(np.int64);truth=r.truth.to_numpy(np.int64);pred=q[idx].argmax(1)
    assert np.array_equal(pred,r.pred) and np.array_equal(q[idx],r[['p0','p1','p2']].to_numpy()) and np.array_equal(lp[idx],r[['logp0','logp1','logp2']].to_numpy())
    stable=-lp[idx,truth];clip=-np.log(np.maximum(q[idx,truth],1e-300))
    assert repeat_values(stable,r.stable_CE.to_numpy(),'CE')['passed'] and repeat_values(clip,r.probability_clip_CE.to_numpy(),'CE')['passed']
    recomputed=r.assign(stable_CE=stable,probability_clip_CE=clip,wrong=pred!=truth).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
    sr=source_sum_repeat_review(r,r,recomputed,s);assert sr['passed']
    return r,dict(original_rows=len(r),source_groups=len(s),M_errors=int(((pred!=truth)&(truth==1)).sum()),S_errors=int(((pred!=truth)&(truth==2)).sum()),protected_regressions=int(((pred!=truth)&r.protected_correct).sum()),new_errors_vs_initial=int(((pred!=truth)&r.initial_correct).sum()))

def main():
    assert not OUT.exists() and all((TRIAL/f'role{f}/diagnostic.json').exists() for f in range(3));OUT.mkdir()
    files={p for p in TRIAL.rglob('*') if p.is_file()}|{Path(__file__).resolve(),ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v159_source_sum_repeat_policy.py',ROOT/'training/v160_independent_saved_direction_certificate.py'}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)};save(OUT/'pre_saved_result_bindings.json',dict(source_sha256=bindings,official_calls=0))
    reports=[]
    for role in range(3):
        folder=TRIAL/f'role{role}';diag=read(folder/'diagnostic.json');assert diag['initial_parameter_sha256']==diag['restored_parameter_sha256'] and diag['official_fits']==diag['permanent_updates']==0 and diag['restored_joint_TRAIN_retention']['passed']
        calls=[json.loads(line) for line in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()];counts={}
        for kind,prefix in [('head','head'),('feature','feature'),('full_class_gradient','gradient'),('full_parameter_margin_gradient','margin')]:
            for event,suffix in [('attempt','attempts'),('completed','completed')]:
                events=[c for c in calls if c['kind']==kind and c['event']==event];assert [c['ordinal'] for c in events]==list(range(1,len(events)+1));counts[prefix+'_'+suffix]=len(events)
        assert counts==diag['counts']
        baseline={scope:audit_scope(folder/('baseline_class1' if scope=='OOF' else 'baseline_deployment'),scope)[0] for scope in ['OOF','deployment']}
        gs=[np.load(folder/f'baseline_class{c}/gradient.npy') for c in [1,2]];assert all(g.shape==(1060832,) and np.isfinite(g).all() for g in gs)
        probes=[]
        for path in sorted(folder.rglob('probe.json')):
            p=read(path);scopes={};risk=[]
            for scope in ['OOF','deployment']:
                r,scopes[scope]=audit_scope(path.parent,scope);b=baseline[scope]
                assert np.array_equal(r.row_position,b.row_position) and np.array_equal(r.truth,b.truth)
                scopes[scope]['classification_changes_vs_endpoint']=int(r.pred.ne(b.pred).sum());scopes[scope]['repairs_vs_endpoint']=int((r.pred.eq(r.truth)&b.pred.ne(b.truth)).sum());scopes[scope]['new_errors_vs_endpoint']=int((r.pred.ne(r.truth)&b.pred.eq(b.truth)).sum())
                assert scopes[scope]['M_errors']==p[scope+'_stats']['M_errors'] and scopes[scope]['S_errors']==p[scope+'_stats']['S_errors']
                if scope=='OOF':risk=[math.fsum(r.loc[r.truth.eq(c),'stable_CE'])/int(r.truth.eq(c).sum()) for c in [1,2]]
            recorded_risk=np.load(path.parent/'risk.npy');assert repeat_values(risk,recorded_risk,'risk')['passed']
            guard=scopes['OOF']['protected_regressions']==0 and scopes['deployment']['new_errors_vs_initial']==0 and p['deployment_stats']['mastered'] and p['joint_TRAIN_retention']['passed'];assert guard==p['classification_guard']
            rv=np.load(folder/'baseline_class1/risk.npy');fr=finite_step_review(rv,recorded_risk,p['class_slopes'],*pd.read_parquet(folder/'baseline_class1/OOF_original_rows.parquet').truth.value_counts().reindex([1,2]).to_numpy(),'B',p['step'],guard)
            assert fr['accepted']==p['finite_numeric_review']['accepted'] and bool(fr['accepted'] and p['actual_parameter_change'])==p['accepted']
            probes.append(dict(path=path.relative_to(ROOT).as_posix(),accepted=p['accepted'],step=p['step'],scopes=scopes,finite_numeric_review=fr))
        assert len(probes)==diag['finite_proposals'] and sum(p['accepted'] for p in probes)==int(diag['finite_probe_pass'])
        qp=[]
        for path in sorted(folder.glob('round*/QP_certificate.json')):
            q=read(path);d=np.load(path.parent/'direction.npy');a=np.load(path.parent/'raw_margin_normals.npy');c=certificate(*gs,a,d)
            assert c['eligible_for_finite_trial_only']==q['independent_saved_vector_certificate']['eligible_for_finite_trial_only'];qp.append(dict(path=path.relative_to(ROOT).as_posix(),status=q['status'],independent_saved_vector_certificate=c))
        assert len(qp)==diag['QP_solves']
        normals=0
        for path in sorted(folder.glob('normals/*/measurement_repeat_review.json')):
            gr=repeat_gradient(np.load(path.parent/'repeat0_gradient.npy'),np.load(path.parent/'repeat1_gradient.npy'));assert gr['passed'];normals+=1
        assert normals==diag['margin_normals']
        reports.append(dict(role=role,status=diag['status'],counts=counts,finite_proposals=len(probes),QP_solves=len(qp),margin_normals=normals,probes=probes,QP=qp,parameter_hash_restored=True,joint_TRAIN_retention_passed=True))
    check_bindings(bindings)
    total={name:sum(r['counts'][name] for r in reports) for name in reports[0]['counts']}
    save(OUT/'audit.json',dict(status='all_saved_fixed_endpoint_diagnostic_outputs_vectors_rows_sources_costs_recomputed',reports=reports,actual_new_counts=total,all_roles_finite_pass=all(any(p['accepted'] for p in r['probes']) for r in reports),quality_acceptance=False,official_audit_heads=0,official_audit_gradients=0,official_audit_fits=0,source_sha256=bindings))
    print(json.dumps(dict(status='saved_actual_diagnostic_audit_passed',actual_new_counts=total,official_audit_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),official_audit_calls=0))
        raise
