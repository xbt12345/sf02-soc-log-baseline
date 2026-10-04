"""Read saved real preflight arrays, gold and call journals; no official calls."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review_v159_v3 import require_run_seal,check_bindings,sha
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,repeat_direction

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
PREP=ROOT/'artifacts/v159_boundary_input_preparation_20261002'
OUT=ROOT/'artifacts/v159_independent_numeric_real_preflight_audit_20261002'


def main():
    assert not OUT.exists()
    plan=require_run_seal(RUN/'run_seal.json',ROOT/'training/v159_boundary_train_v4.py')
    pre=json.loads((RUN/'preflight.json').read_text(encoding='utf-8'));check_bindings(pre['source_sha256'])
    assert pre['status']=='actual_all_role_origin_and_repeated_full_class_gradient_preflight_passed'
    assert pre['official_fits']==pre['official_updates']==0 and pre['joint_TRAIN_retention']['passed']
    gold=pd.read_parquet(ROOT/'data/official/train.parquet',columns=['label_binary']).label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    assert len(gold)==2056871 and np.isfinite(gold).all()
    old=torch.load(ROOT/'artifacts/v159_class_boundary_trial_20261002/initial.pt',map_location='cpu',weights_only=True)
    new=torch.load(RUN/'initial.pt',map_location='cpu',weights_only=True)
    assert old.keys()==new.keys() and old['seed']==new['seed']==15901
    assert old['state'].keys()==new['state'].keys() and all(torch.equal(t,new['state'][k]) for k,t in old['state'].items())
    info=json.loads((PREP/'qualification.json').read_text(encoding='utf-8'))
    reports=[];head=features=gradients=0
    for f in range(3):
        ref=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet')
        ids=np.sort(ref.local.unique());mass=np.bincount(ref.truth,minlength=3)
        assert np.array_equal(ref.truth,gold[ref.row_position])
        budget=plan['role_call_budgets'][f]
        assert mass.tolist()==budget['original_class_mass'] and len(ref)==budget['original_rows']
        qs={}
        for arm in ['A','B']:
            folder=RUN/f'preflight{f}_{arm}'
            events=[json.loads(line) for line in (folder/'calls.jsonl').read_text(encoding='utf-8').splitlines()]
            counts={}
            for kind in ['head','feature','full_class_gradient']:
                starts=[e for e in events if e['kind']==kind and e['event']=='attempt']
                ends=[e for e in events if e['kind']==kind and e['event']=='completed']
                assert len(starts)==len(ends) and [e['ordinal'] for e in starts]==list(range(1,len(starts)+1))
                counts[kind]=len(starts)
                if kind=='full_class_gradient':
                    assert [e['class_id'] for e in starts]==([1,2,1,2] if arm=='A' else [])
                    assert all(e['original_class_mass']==mass.tolist() for e in starts)
            assert counts['head']==counts['feature']==budget['preflight_classifier_cap_'+arm]
            assert counts['full_class_gradient']==(4 if arm=='A' else 0)
            head+=counts['head'];features+=counts['feature'];gradients+=counts['full_class_gradient']
            scopes={};qs[arm]={}
            for scope,active in [('OOF',ids),('deployment',np.arange(22546))]:
                q=np.load(folder/f'{scope}_probability.npy');repeat=np.load(folder/f'{scope}_repeated_probability.npy')
                assert repeat_values(q[active],repeat[active],'probability')['passed']
                bank=np.load(BANK/f'fold{f}/{scope}_probabilities.npy',mmap_mode='r')[active,:16,:].mean(1)
                assert np.abs(q[active]-bank).max()<=3e-12 and np.array_equal(q[active].argmax(1),bank.argmax(1))
                rows=pd.read_parquet(PREP/f'fold{f}/{scope}_visible_input_rows.parquet')
                assert np.array_equal(rows.row_position,ref.row_position) and np.array_equal(rows.truth,gold[rows.row_position])
                pred=q[rows.local].argmax(1);wrong=pred!=rows.truth.to_numpy()
                stats=dict(original_rows=len(rows),M_errors=int((wrong&rows.truth.eq(1)).sum()),S_errors=int((wrong&rows.truth.eq(2)).sum()),
                           pure_errors=int((wrong&rows.pure_current_input).sum()),protected_regressions=int((wrong&rows.protected_correct).sum()),
                           new_errors_vs_initial=int((wrong&rows.initial_correct).sum()))
                assert stats['protected_regressions']==stats['new_errors_vs_initial']==0
                if scope=='deployment':assert stats['pure_errors']==0 and int(wrong.sum())<=info['roles'][f]['scopes'][scope]['current_input_minimum_errors']
                scopes[scope]=stats;qs[arm][scope]=q
            assert np.array_equal(qs['A']['OOF'][ids].argmax(1),qs[arm]['OOF'][ids].argmax(1))
            report=dict(fold=f,arm=arm,original_class_mass=mass.tolist(),counts=counts,scopes=scopes)
            if arm=='A':
                gs=[];point=None
                for rep in [0,1]:
                    pair=[]
                    for cls in [1,2]:
                        prefix=folder/f'repetition{rep}_class{cls}'
                        g=np.load(str(prefix)+'_complete_gradient.npy');q=np.load(str(prefix)+'_probability.npy')[ids]
                        lp=np.load(str(prefix)+'_log_probability.npy')[ids];rv=np.load(str(prefix)+'_risk.npy')
                        assert g.shape==(1060832,) and np.isfinite(g).all() and np.count_nonzero(g[:-48])==0 and np.linalg.norm(g[-48:])>0
                        if point is None:point=(q,lp,rv)
                        assert all(repeat_values(a,b,'probability' if j==0 else 'values')['passed'] for j,(a,b) in enumerate(zip(point,(q,lp,rv))))
                        assert repeat_values(q,qs['A']['OOF'][ids],'probability')['passed']
                        pair.append(g)
                    gs.append(pair)
                gradient_reports=[repeat_gradient(gs[0][j],gs[1][j]) for j in [0,1]]
                direction_reports={a:repeat_direction(*gs,*mass[1:],a) for a in ['A','B']}
                assert all(r['passed'] for r in gradient_reports+list(direction_reports.values()))
                report.update(gradient_repeats=gradient_reports,initial_directions=direction_reports)
            reports.append(report)
        assert all(repeat_values(qs['A'][s][ids],qs['B'][s][ids],'probability')['passed'] for s in ['OOF','deployment'])
    assert head==features==pre['actual_head_forward_calls']==pre['actual_opinion_feature_calls']==336 and gradients==pre['actual_full_class_gradients']==12
    assert not list(RUN.glob('fold*_*/started.json')), 'Audit before training starts'
    paths=[Path(__file__).resolve(),RUN/'preflight.json',RUN/'run_seal.json',RUN/'initial.pt',ROOT/'data/official/train.parquet']
    result=dict(status='real_three_role_preflight_independently_recomputed_from_saved_arrays_and_gold',reports=reports,
                initial_parameters_bitwise_unchanged=True,head_calls_recomputed=head,feature_calls_recomputed=features,class_gradients_recomputed=gradients,
                cumulative_head_calls=124+head,cumulative_feature_calls=124+features,cumulative_full_class_gradients=8+gradients,
                worker_fits_started=0,worker_updates=0,root_official_heads=0,root_official_gradients=0,root_official_fits=0,
                permission_scope='Only the registered six fits; retain finite stop, guards, budgets, fixed endpoints and full official-row evaluation.',
                quality_acceptance=False,first_training_issue_passed=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    OUT.mkdir();(OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','head_calls_recomputed','class_gradients_recomputed','cumulative_head_calls','cumulative_full_class_gradients','permission_scope','quality_acceptance']},ensure_ascii=False))


if __name__=='__main__':main()
