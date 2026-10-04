"""Post-result diagnosis of the fixed V158 trial, with no model evaluation.

All source outputs are bound before decoding. This does not select a model,
change its priors, or authorize any further fitting.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiment_review import ROOT,read,sha,check_bindings

TRIAL=ROOT/'artifacts/v158_fusion_trial_20261001'
BANK=ROOT/'artifacts/v158_legal_fusion_bank_v2_20261001'
OUT=ROOT/'artifacts/v158_saved_result_transfer_audit_20261001'

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def stats(frame,prob):
    y=frame.truth.to_numpy(); p=prob[frame.local.to_numpy()]
    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-12,rtol=0)
    pred=p.argmax(1); ce=-np.log(np.maximum(p[np.arange(len(y)),y],1e-300))
    brier=((p-np.eye(3)[y])**2).sum(1)
    return dict(original_rows=len(y),CE=float(ce.mean()),Brier=float(brier.mean()),
        classes={str(c):dict(support=int((y==c).sum()),errors=int(((y==c)&(pred!=c)).sum()),
            CE=float(ce[y==c].mean()) if (y==c).any() else None,
            Brier=float(brier[y==c].mean()) if (y==c).any() else None) for c in range(3)})

def main():
    assert not OUT.exists(); OUT.mkdir()
    paths=[Path(__file__).resolve(),TRIAL/'final_delivery.json',TRIAL/'verification.json',TRIAL/'quality.json',
        TRIAL/'ASA_prediction_ledger.parquet',TRIAL/'all_fixed_cohort_quality.json',
        BANK/'qualification.json',BANK/'all_legal_OOF_original_rows.parquet',
        ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet']
    for f in range(3):
        paths.extend(BANK/f'fold{f}'/name for name in ['OOF_probabilities.npy','deployment_probabilities.npy','prior.npy','legal_FIT_reference.parquet'])
        for arm in ['A','B']:
            paths.extend(TRIAL/f'preflight{f}_{arm}'/name for name in ['OOF_probability.npy','deployment_probability.npy'])
            paths.extend(TRIAL/f'fold{f}_{arm}'/name for name in ['fit.json','endpoint_OOF_probability.npy','endpoint_deployment_probability.npy'])
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    save(OUT/'pre_saved_result_bindings.json',dict(status='bound_before_saved_probability_decoding',source_sha256=bindings,
        official_model_calls=0,new_gradients=0,new_fits=0,new_updates=0))
    delivery=read(TRIAL/'final_delivery.json'); assert delivery['classifier_fits']==60 and not delivery['quality_acceptance']
    asa=pd.read_parquet(TRIAL/'ASA_prediction_ledger.parquet'); original_oof=pd.read_parquet(BANK/'all_legal_OOF_original_rows.parquet')
    assert len(asa)==112807 and len(original_oof)==225614
    summaries=[]; oof_ledgers=[]; asa['pred_registered_origin']=np.zeros(len(asa),np.int8)
    for f in range(3):
        frame=pd.read_parquet(BANK/f'fold{f}/legal_FIT_reference.parquet'); ids=frame.local.to_numpy()
        bank=np.load(BANK/f'fold{f}/OOF_probabilities.npy'); prior=np.load(BANK/f'fold{f}/prior.npy')
        used=np.unique(ids); assert np.isfinite(bank[used]).all()
        # Only saved-array arithmetic. The registered origin is also checked
        # against the prospectively sealed real zero-step forward output.
        q=np.full((22546,3),np.nan); q[used]=(bank[used]*prior[None,:,None]).sum(1)
        origin=np.load(TRIAL/f'preflight{f}_A/OOF_probability.npy')
        assert np.max(np.abs(q[used]-origin[used]))<=1e-12
        values=dict(current16=bank[:,:16,:].mean(1),new_legacy=bank[:,16,:],registered_origin=origin)
        for arm in ['A','B']:values[arm]=np.load(TRIAL/f'fold{f}_{arm}/endpoint_OOF_probability.npy')
        summaries.append(dict(training_role=f,original_class_mass=[int(frame.truth.eq(c).sum()) for c in range(3)],
            fixed_alpha=float(prior[-1]),OOF={name:stats(frame,prob) for name,prob in values.items()}))
        base=frame.copy()
        for name,prob in values.items():
            base['pred_'+name]=prob[ids].argmax(1)
            base[name+'_CE']=-np.log(np.maximum(prob[ids,frame.truth.to_numpy()],1e-300))
        oof_ledgers.append(base)
        outer=asa.fold.eq(f).to_numpy(); deploy_origin=np.load(TRIAL/f'preflight{f}_A/deployment_probability.npy')
        asa.loc[outer,'pred_registered_origin']=deploy_origin[asa.loc[outer,'local']].argmax(1)
    ledger=pd.concat(oof_ledgers,ignore_index=True); assert len(ledger)==225614
    old=original_oof.sort_values(['training_role','row_position']).reset_index(drop=True)
    new=ledger.sort_values(['training_role','row_position']).reset_index(drop=True)
    assert np.array_equal(old.row_position,new.row_position) and np.array_equal(old.truth,new.truth)
    assert np.array_equal(old.pred_current16,new.pred_current16) and np.array_equal(old.pred_new_legacy,new.pred_new_legacy)
    ledger.to_parquet(OUT/'all_original_OOF_trajectory_endpoints.parquet',index=False)
    asa.to_parquet(OUT/'all_ASA_origin_and_actual_endpoints.parquet',index=False)
    fixed=pd.read_parquet(paths[8]); assert np.array_equal(fixed.row_position,asa.row_position)
    changes=[]
    for before,after in [('pred_registered_origin','pred_A'),('pred_registered_origin','pred_B'),('pred_A','pred_B'),('pred_A0','pred_B'),('pred_V146_A','pred_B')]:
        mask=asa[before].ne(asa[after]); part=asa[mask].copy(); part['before_correct']=part[before].eq(part.truth);part['after_correct']=part[after].eq(part.truth)
        part['facts_json']=fixed.loc[mask,'facts_json'].to_numpy(); part['both_ports_observed']=fixed.loc[mask,'both_ports_observed'].to_numpy()
        part.to_parquet(OUT/f'all_changed_rows_{after}_vs_{before}.parquet',index=False)
        source=part.groupby(['fold','root','truth']).agg(changed=('truth','size'),before_correct=('before_correct','sum'),after_correct=('after_correct','sum')).reset_index()
        source.to_parquet(OUT/f'all_changed_source_classes_{after}_vs_{before}.parquet',index=False)
        changes.append(dict(before=before,after=after,changed_original_rows=len(part),
            classes={str(c):dict(repairs=int((part.truth.eq(c)&~part.before_correct&part.after_correct).sum()),
                regressions=int((part.truth.eq(c)&part.before_correct&~part.after_correct).sum())) for c in range(3)},
            affected_source_class_records=source.to_dict('records')))
    scores={name:{str(c):dict(support=int(asa.truth.eq(c).sum()),errors=int((asa.truth.eq(c)&asa[name].ne(c)).sum())) for c in range(3)}
        for name in ['pred_A0','pred_V146_A','pred_registered_origin','pred_A','pred_B']}
    # Test the actual whole-class and source result; never call lower OOF CE
    # a success on the newly inspected outer population.
    assert scores['pred_B']['1']['errors']==414 and scores['pred_B']['2']['errors']==1872
    matched=next(z for z in changes if z['before']=='pred_A' and z['after']=='pred_B')
    assert matched['classes']['1']==dict(repairs=10,regressions=0)
    assert matched['classes']['2']==dict(repairs=0,regressions=10)
    check_bindings(bindings)
    save(OUT/'audit.json',dict(status='actual_60_fit_saved_result_transfer_diagnosed_no_new_model_evaluation',
        OOF_original_roles=225614,outer_ASA_original_rows=112807,OOF=summaries,outer_errors=scores,paired_changes=changes,
        array_origin_recombinations=3,official_classifier_forward_calls=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0,
        quality_acceptance=False,model_promoted=False,automatic_extra_fits=False,
        limits=['This is a post-result diagnosis, not causal identification or new qualification.',
            'The 527 conditions change capacity as well as information; the text sketch is lossy.',
            'The registered nonnegative convex bank cannot repair the separately proven 188 all-expert-negative S rows.',
            'TRAIN guards protect legal FIT behavior, not outer sources or full bank observational collision floors.'],
        source_bindings_sha256=sha(OUT/'pre_saved_result_bindings.json')))
    print(json.dumps(dict(outer_errors=scores,matched_changes=matched,official_classifier_calls=0),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
