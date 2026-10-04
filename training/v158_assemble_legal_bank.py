"""Verify all 54 completed supervised fits and materialize legal OOF arrays."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from experiment_review import ROOT,read,sha,check_bindings
from v158_nested_runtime_v2 import OUT as BASE,PLAN,require
from v158_fusion_contract import STAGES,role,counts,inner_fold
from v131_common import load_data
from v158_conditions import conditions
from v142_retention_check import check as retention

OUT=ROOT/'artifacts/v158_legal_fusion_bank_20261001'
CURRENT=ROOT/'artifacts/v157_complete_function_logit_audit_v2_20261001'
LEGACY=ROOT/'artifacts/v107_matched_training_20260928'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def lines(path):return [json.loads(z) for z in path.read_text(encoding='utf-8').splitlines()]

def main():
    p=require(ROOT/'training/v158_nested_base_train_v2.py');require(ROOT/'training/v158_legacy_nested_train_v2.py')
    assert not OUT.exists() and (BASE/'base_phase_completion.json').exists() and (BASE/'legacy_phase_completion.json').exists()
    paths={PLAN,Path(__file__).resolve(),ROOT/'training/v158_conditions.py',BASE/'run_seal.json',BASE/'run_seal_v2.json',BASE/'base_phase_completion.json',BASE/'legacy_phase_completion.json',CURRENT/'run_seal.json',CURRENT/'audit.json',CURRENT/'v2_verification.json'}
    frozen_sources=read(CURRENT/'run_seal.json')['source_sha256'];check_bindings(frozen_sources)
    paths|={ROOT/k for k in frozen_sources}
    receipts=[]
    for f in range(3):
        for j in range(3):
            for name in STAGES:
                folder=BASE/f'outer{f}_inner{j}'/name
                for file in ['fit.json','started.json','endpoint.pt','member_logits.npy','mean_probability.npy','progress.json','FIT_reference.parquet','classifier_calls.jsonl']:
                    paths.add(folder/file)
                if name==STAGES[0]:
                    paths.update(folder/file for file in ['steps.jsonl','h1.npy','h2.npy'])
                else:paths.update(folder/file for file in ['gradients.jsonl','proposals.jsonl'])
                receipts.append(read(folder/'fit.json'))
            folder=BASE/f'legacy_outer{f}_inner{j}'
            paths.update(folder/file for file in ['fit.json','started.json','endpoint.joblib','ASA_logits.npy','gradient_calls.jsonl','ASA_forward_calls.jsonl'])
        paths.update(CURRENT/f'fold{f}'/file for file in ['logits.npy','probabilities.npy'])
        paths.add(LEGACY/f'fold{f}_N1_teacher'/'scores_all_input_ids.npy')
    from v107_matched_training import ASA_IDS
    prior_path=ROOT/'artifacts/v158_legacy_expert_bank_qualification_20261001/initial_legacy_priors.json'
    paths|={ASA_IDS,prior_path,ROOT/'training/v158_bank_initialization.py'}
    OUT.mkdir();bindings={q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q):sha(q) for q in sorted(paths)}
    save(OUT/'pre_saved_array_bindings.json',dict(status='bound_before_any_array_recombination',source_sha256=bindings,official_model_calls=0))
    x,d=load_data();cond=conditions(x);np.save(OUT/'conditions.npy',cond);asa_ids=np.load(ASA_IDS)
    priors=read(prior_path);fit_summary=[];role_ledger=[];guard_rows=[];audit=[]
    for f in range(3):
        legal=d[d.fold.ne(f)].copy();used=np.unique(legal.local);legal['inner_fold']=legal.root.map(lambda r:inner_fold(f,r));legal['training_role']=f
        assert legal.groupby('local').inner_fold.nunique().max()==1
        current=softmax(np.load(CURRENT/f'fold{f}/logits.npy'),axis=-1)
        old_current=np.load(CURRENT/f'fold{f}/probabilities.npy')
        assert np.max(np.abs(current.mean(1)-old_current))<=1e-12 and np.array_equal(current.mean(1).argmax(1),old_current.argmax(1))
        legacy=softmax(np.load(LEGACY/f'fold{f}_N1_teacher/scores_all_input_ids.npy',mmap_mode='r')[asa_ids],axis=-1)
        deploy=np.concatenate([current,legacy[:,None,:]],1);oof=np.full_like(deploy,np.nan)
        inner_rows=[]
        for j in range(3):
            frame,query,held=role(d,f,j);expected=p['role_budgets'][f*3+j];mass=counts(frame)
            assert len(frame)==expected['fit_rows'] and mass.sum(0).tolist()==expected['fit_class_mass']
            for name in STAGES:
                folder=BASE/f'outer{f}_inner{j}'/name;r=read(folder/'fit.json');started=read(folder/'started.json')
                assert (r['outer_fold'],r['excluded_inner'],r['stage'])==(f,j,name)
                assert r['fit_rows']==len(frame) and r['class_mass']==mass.sum(0).tolist() and r['canonical_label_collision_floor']==expected['canonical_floor']
                assert set(started['fit_roots'])==set(frame.root) and not set(started['fit_roots'])&set(query.root) and not set(started['fit_roots'])&set(held.root)
                assert sha(folder/'endpoint.pt')==r['source_model_sha256'] and sha(folder/'member_logits.npy')==r['logits_sha256']
                calls=lines(folder/'classifier_calls.jsonl');assert sum(z['event']=='attempt' for z in calls)==sum(z['event']=='completed' for z in calls)==r['classifier_forward_calls']
                if name==STAGES[0]:
                    logs=lines(folder/'steps.jsonl');done=[z for z in logs if z['event']=='batch_gradient_and_update_completed']
                    assert len(done)==r['accepted_updates']==expected['full_network_gradient_updates']
                    assert np.array_equal(np.sum([z['original_class_mass'] for z in done],axis=0),100*mass.sum(0))
                else:
                    logs=lines(folder/'gradients.jsonl');done=[z for z in logs if z['event']=='full_gradient_completed']
                    assert len(done)==sum(z['event']=='full_gradient_attempt' for z in logs)==r['full_gradients']<=200
                    assert all(z['class_mass']==mass.sum(0).tolist() for z in done)
                fit_summary.append(dict(outer_fold=f,excluded_inner=j,stage=name,fit_rows=len(frame),
                    endpoint_FIT_stats=r['endpoint_FIT_stats'] if 'endpoint_FIT_stats' in r else r['last5_FIT_stats'][-1],
                    canonical_floor=r['canonical_label_collision_floor'],accepted_updates=r['accepted_updates'],
                    full_gradients=r['full_gradients'],batch_gradients=r['batch_gradients'],classifier_calls=r['classifier_forward_calls']))
            final=BASE/f'outer{f}_inner{j}'/STAGES[-1]
            prob=softmax(np.load(final/'member_logits.npy'),axis=-1)
            assert np.isfinite(prob).all() and np.max(np.abs(prob.mean(1)-np.load(final/'mean_probability.npy')))<=1e-12
            lin=BASE/f'legacy_outer{f}_inner{j}';lr=read(lin/'fit.json');lp=softmax(np.load(lin/'ASA_logits.npy'),axis=-1)
            assert lr['class_mass']==p['legacy_role_budgets'][f*3+j]['fit_class_mass'] and lr['fit_rows']==p['legacy_role_budgets'][f*3+j]['fit_rows']
            assert sha(lin/'endpoint.joblib')==lr['model_sha256'] and sha(lin/'ASA_logits.npy')==lr['scores_sha256']
            logs=lines(lin/'gradient_calls.jsonl');assert sum(z['event']=='gradient_completed' for z in logs)==lr['full_gradient_evaluations']<=1000
            assert all(z['original_class_mass']==lr['class_mass'] for z in logs if z['event']=='gradient_attempt')
            calls=lines(lin/'ASA_forward_calls.jsonl');assert sum(z['event']=='attempt' for z in calls)==sum(z['event']=='completed' for z in calls)==lr['ASA_classifier_forward_chunks']==3
            ii=np.unique(query.local);oof[ii]=np.concatenate([prob[ii],lp[ii,None,:]],1)
            qr=query[['row_position','local','root','fold','truth','canonical_key']].copy();qr['training_role']=f;qr['excluded_inner']=j
            qr['pred_current16']=prob[query.local].mean(1).argmax(1);qr['pred_new_legacy']=lp[query.local].argmax(1)
            inner_rows.append(qr)
        assert np.isfinite(oof[used]).all() and np.allclose(oof[used].sum(-1),1,atol=1e-12,rtol=0)
        folder=OUT/f'fold{f}';folder.mkdir();np.save(folder/'OOF_probabilities.npy',oof);np.save(folder/'deployment_probabilities.npy',deploy)
        legal.to_parquet(folder/'legal_FIT_reference.parquet',index=False)
        original_rows=pd.concat(inner_rows,ignore_index=True).sort_values('row_position').reset_index(drop=True)
        assert np.array_equal(original_rows.row_position,legal.row_position) and np.array_equal(original_rows.truth,legal.truth)
        role_ledger.append(original_rows)
        # Earlier qualification fixed these legal FIT priors before new OOF.
        entry=priors['folds'][f] if 'folds' in priors else priors['roles'][f]
        alpha=entry['legacy_prior']
        prior=np.array([(1-alpha)/16]*16+[alpha]);np.save(folder/'prior.npy',prior)
        q=(deploy*prior[None,:,None]).sum(1);oldpred=old_current[legal.local].argmax(1);protected=oldpred==legal.truth.to_numpy()
        assert np.all(q[legal.local].argmax(1)[protected]==legal.truth.to_numpy()[protected])
        guard_rows.append(legal[['row_position','truth']].assign(training_role=f,pred=q[legal.local].argmax(1)))
        audit.append(dict(fold=f,original_FIT_rows=len(legal),OOF_query_original_rows=len(original_rows),
            protected_current_correct_rows=int(protected.sum()),protected_init_regressions=0,legacy_alpha=float(alpha),
            score_summary_not_quality=True))
    combined=pd.concat(role_ledger,ignore_index=True);assert len(combined)==225614;combined.to_parquet(OUT/'all_legal_OOF_original_rows.parquet',index=False)
    pd.DataFrame(fit_summary).to_parquet(OUT/'all_45_stage_FIT_quality.parquet',index=False)
    joint=retention(pd.concat(guard_rows,ignore_index=True));assert joint['passed']
    check_bindings(bindings)
    outputs=[OUT/'conditions.npy',OUT/'all_legal_OOF_original_rows.parquet',OUT/'all_45_stage_FIT_quality.parquet']
    outputs.extend(q for f in range(3) for q in (OUT/f'fold{f}').glob('*'))
    save(OUT/'qualification.json',dict(status='all_54_supervised_fit_identities_and_legal_OOF_bank_verified_before_fusion',
        base_fits=45,legacy_fits=9,fusion_fits=0,official_classifier_calls=0,new_gradients=0,new_updates=0,
        original_FIT_roles=225614,condition_width=527,expert_count=17,init_joint_TRAIN_retention=joint,folds=audit,
        quality_acceptance=False,source_sha256={q.relative_to(ROOT).as_posix():sha(q) for q in outputs+[OUT/'pre_saved_array_bindings.json']}))
    print(dict(legal_bank_qualified=True,supervised_fits=54,fusion_fits=0,original_FIT_roles=225614,quality_acceptance=False))

if __name__=='__main__':main()
