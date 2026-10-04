"""Saved-source consistency and failure costs only; no classifier replay."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
OUT=ROOT/'artifacts/v159_independent_group_CE_failure_scope_audit_20261002'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    failure=json.loads((RUN/'evaluation_failure.json').read_text(encoding='utf-8'))
    assert 'source_CE_sum' in failure['traceback'] and 'v159_boundary_evaluate_v4.py' in failure['traceback']
    files=[Path(__file__).resolve(),RUN/'evaluation_failure.json',RUN/'run_seal.json']
    cost=dict(head=0,feature=0,gradients=0,training_head=0,training_gradient=0,updates=0,proposals=0)
    scopes=[];next_replay=0
    for folder in sorted(RUN.glob('fold*_*')):
        r=json.loads((folder/'fit.json').read_text(encoding='utf-8'));files.append(folder/'fit.json')
        k=[10,4,10][r['fold']];replay=(len(r['last5'])+1)*(12+k);next_replay+=replay
        cost['training_head']+=r['counts']['head_attempts'];cost['training_gradient']+=r['full_class_gradients'];cost['updates']+=r['accepted_updates'];cost['proposals']+=r['proposal_evaluations']
        ec=folder/'evaluation_calls.jsonl'
        if ec.exists():
            files.append(ec);ev=[json.loads(s) for s in ec.read_text(encoding='utf-8').splitlines()]
            for kind,key in [('head','head'),('feature','feature'),('full_class_gradient','gradients')]:
                attempts=sum(e['kind']==kind and e['event']=='attempt' for e in ev);completed=sum(e['kind']==kind and e['event']=='completed' for e in ev)
                assert attempts==completed;cost[key]+=attempts
        consistency=[]
        for item in r['last5']:
            rp=folder/f"accepted{item['update']}_OOF_rows.parquet";sp=folder/f"accepted{item['update']}_OOF_sources.parquet";files.extend([rp,sp])
            rows=pd.read_parquet(rp);saved=pd.read_parquet(sp)
            rebuilt=rows.assign(wrong=rows.pred.ne(rows.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index()
            assert np.array_equal(rebuilt[['root','truth','support','errors']],saved[['root','truth','support','errors']])
            gap=float(np.abs(rebuilt[['stable_CE_sum','probability_clip_CE_sum']].to_numpy()-saved[['stable_CE_sum','probability_clip_CE_sum']].to_numpy()).max());assert gap==0
            logtrue=rows[['logp0','logp1','logp2']].to_numpy()[np.arange(len(rows)),rows.truth]
            scale=rows.assign(row_scale=np.maximum(1.,np.abs(logtrue))).groupby(['root','truth']).row_scale.sum().reset_index(name='row_scale_sum')
            rebuilt=rebuilt.merge(scale,on=['root','truth']);rebuilt['old_single_value_envelope']=8*np.finfo(float).eps*np.maximum(1.,np.abs(rebuilt.stable_CE_sum));rebuilt['sum_of_row_log_envelopes']=8*np.finfo(float).eps*rebuilt.row_scale_sum
            example=rebuilt.sort_values('support',ascending=False).iloc[0]
            consistency.append(dict(update=item['update'],saved_source_self_consistency_gap=gap,
                                    largest_group_example={key:float(example[key]) for key in ['support','stable_CE_sum','old_single_value_envelope','sum_of_row_log_envelopes']}))
        prop=[json.loads(s) for s in (folder/'proposals.jsonl').read_text(encoding='utf-8').splitlines()]
        terminal=[e for e in prop if e['iteration']==r['gradient_iterations']]
        scopes.append(dict(fold=r['fold'],arm=r['arm'],last_window_states=len(r['last5']),new_frozen_replay_calls=replay,
                           OOF=r['OOF_stats'],termination=r['termination'],terminal_proposal_guard_failures=sum(not e['classification_guard'] for e in terminal),
                           terminal_deployment_guard_failures=sum(not e['deployment_stats']['mastered'] or e['deployment_stats']['new_errors_vs_initial']>0 for e in terminal),
                           terminal_OOF_protection_failures=sum(e['OOF_stats']['protected_regressions']>0 for e in terminal),saved_source_consistency=consistency))
    assert cost['head']==cost['feature']==110 and cost['gradients']==0 and next_replay==640
    result=dict(status='saved_source_records_consistent_failed_evaluation_cost_and_recovery_scope_recomputed',
                completed_fits=6,cost=cost,scopes=scopes,new_frozen_replay_head_and_feature_cap=640,
                total_technical_evaluation_head_calls=110+640,old_evaluation_cap=720,technical_cap_increment=30,
                actual_failed_source_difference_unknown=True,original_actual_replay_arrays_not_saved=True,
                numerical_rule_problem='A sum of per-row log-value error envelopes is generally larger than one envelope on a near-zero class/source CE sum.',
                probability_clip_log_requires_its_own_error_propagation=True,
                official_model_queries=0,official_feature_calls=0,official_gradients=0,official_fits=0,
                first_issue_training_passed=False,quality_acceptance=False,additional_training_permission=False,
                source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','completed_fits','cost','new_frozen_replay_head_and_feature_cap','technical_cap_increment','actual_failed_source_difference_unknown','quality_acceptance']},ensure_ascii=False))


if __name__=='__main__':main()
