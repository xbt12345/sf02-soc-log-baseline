"""Independent official-label replay for V120 primary A/B fits. No fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from experiment_review import ROOT, read, sha, require_run_seal, require_checkpoint, evaluate_primary, class_counts
from v104_phase_b import SparseTabM, predict_all, DEVICE
from v107_matched_training import FOLDS, ROWS, FID, DEST as TEACHERS
from v116_preflight import VIEW
from v121_train import DEST, PLAN, MANIFEST, OFFICIAL, TRACE


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    plan = require_run_seal(DEST/'run_seal.json', ROOT/'training/v121_train.py')
    if plan != read(PLAN):
        raise ValueError('Plan changed.')
    if (DEST/'primary_evaluation.json').exists():
        raise FileExistsError('Existing primary evaluation must not be overwritten.')
    reg=read(DEST/'registration.json')
    x=sparse.load_npz(VIEW)
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    if len(trace)!=plan['evaluation_contract']['raw_ASA_rows'] or trace.row_position.duplicated().any():
        raise ValueError('Incomplete ASA reference.')
    raw={a:np.empty(len(trace),dtype=np.int8) for a in ('A','B')}
    prob_s={a:np.empty(len(trace),dtype=np.float32) for a in ('A','B')}
    fits=[];replays=[]
    for fold in range(3):
        for arm in ('A','B'):
            folder=DEST/f'fold{fold}_{arm}'
            fit=read(folder/'fit.json')
            require_checkpoint(plan,fit)
            if (fit['fold'],fit['arm'],fit['seed']) != (fold,arm,plan['primary_seed']):
                raise ValueError('Fit belongs to a different arm/fold/seed.')
            if fit['seal_sha256']!=sha(DEST/'run_seal.json') or fit['optimizer_steps']!=reg['schedule'][fold]['registered_optimizer_steps_per_arm']:
                raise ValueError('Run seal or step count changed.')
            checks=read(folder/'checkpoints.json')
            if len(checks)!=len(plan['diagnostic_checkpoints']) or [e['epoch'] for e in checks]!=plan['diagnostic_checkpoints']:
                raise ValueError('Diagnostic checkpoint set incomplete.')
            if (fit['checkpoint_report_sha256']!=sha(folder/'checkpoints.json')
                    or fit['progress_sha256']!=sha(folder/'progress.json')
                    or fit['model_sha256']!=sha(folder/'epoch25_model.pt')
                    or fit['prob_sha256']!=sha(folder/'epoch25_prob.npy')):
                raise ValueError('Fit evidence identity changed.')
            for entry in checks:
                if (entry['model_sha256']!=sha(folder/f"epoch{entry['epoch']}_model.pt")
                        or entry['prob_sha256']!=sha(folder/f"epoch{entry['epoch']}_prob.npy")):
                    raise ValueError('Checkpoint identity changed.')
            state=torch.load(folder/'epoch25_model.pt',map_location='cpu',weights_only=True)
            if (state['seed'],state['fold'],state['arm'],state['epoch'],state['seal_sha256']) != (
                    plan['primary_seed'],fold,arm,25,sha(DEST/'run_seal.json')):
                raise ValueError('Model provenance mismatch.')
            model=SparseTabM().to(DEVICE);model.load_state_dict(state['state_dict'])
            replay=predict_all(model,'TabM',x,DEVICE)
            stored=np.load(folder/'epoch25_prob.npy')
            if not np.array_equal(replay.argmax(1),stored.argmax(1)) or not np.allclose(replay,stored,atol=2e-6,rtol=2e-6):
                raise ValueError('Stored predictions do not replay from model.')
            mask=trace.fold.eq(fold).to_numpy();ids=trace.loc[mask,'local'].to_numpy(dtype=np.int64)
            raw[arm][mask]=replay[ids].argmax(1)
            prob_s[arm][mask]=replay[ids,2]
            replays.append({'fold':fold,'arm':arm,'max_probability_difference':float(np.max(np.abs(replay-stored)))})
            fits.append({'fold':fold,'arm':arm,'fit_sha256':sha(folder/'fit.json'),
                         'initial_state_sha256':fit['initial_state_sha256'],
                         'steps':fit['optimizer_steps'],'seconds':fit['seconds'],
                         'fit_class_25':checks[-1]['fit_by_class']})
            del model
        if fits[-1]['initial_state_sha256'] != fits[-2]['initial_state_sha256']:
            raise ValueError('Matched A/B models did not begin at identical parameters.')
    d=trace.copy()
    for arm in ('A','B'):
        d[f'expert_pred_{arm}']=raw[arm]
        d[f'expert_S_probability_{arm}']=prob_s[arm]
    d.to_parquet(DEST/'expert_ASA_predictions.parquet',index=False)
    # Independent labels/identity come from the official parquet, not model output or prior prediction ledgers.
    official=pd.read_parquet(OFFICIAL,columns=['event_id','label_binary'])
    r=pd.read_parquet(ROWS,columns=['row_position','event_id','label_index','route'])
    f=pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold'])
    n=len(r)
    if not (len(official)==len(f)==n==plan['evaluation_contract']['full_task_rows']):
        raise ValueError('Incomplete official population.')
    if not np.array_equal(r.row_position.to_numpy(),np.arange(n)) or not np.array_equal(f.row_position.to_numpy(),r.row_position.to_numpy()):
        raise ValueError('Reference row order changed.')
    if not np.array_equal(official.event_id.to_numpy(),r.event_id.to_numpy()):
        raise ValueError('Original event identities changed.')
    true=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2})
    if true.isna().any() or not np.array_equal(true.to_numpy(dtype=np.int8),r.label_index.to_numpy(dtype=np.int8)):
        raise ValueError('Official labels differ from derived reference.')
    positions=np.flatnonzero(r.route.eq('asa'))
    if not np.array_equal(positions,trace.row_position.to_numpy()) or not np.array_equal(true.iloc[positions].to_numpy(),trace.truth.to_numpy()):
        raise ValueError('ASA row identity or labels changed.')
    if not np.array_equal(f.proposed_fold.to_numpy()[positions],trace.fold.to_numpy()) or not np.array_equal(f.root.to_numpy()[positions],trace.root.to_numpy()):
        raise ValueError('Source-closed folds changed.')
    ids=np.load(FID,mmap_mode='r')
    if len(ids)!=n:
        raise ValueError('Teacher feature mapping incomplete.')
    teacher=np.empty(n,dtype=np.int8)
    for fold in range(3):
        folder=TEACHERS/f'fold{fold}_N1_teacher';file=folder/'scores_all_input_ids.npy'
        if sha(file)!=read(folder/'fit.json')['scores_sha256']:
            raise ValueError('Frozen teacher changed.')
        scores=np.load(file,mmap_mode='r')
        q=np.flatnonzero(f.proposed_fold.eq(fold))
        teacher[q]=scores[np.asarray(ids[q],np.int64)].argmax(1)
    gate=teacher[positions]!=0
    if int((~gate).sum()) != plan['evaluation_contract']['current_teacher_excluded_ASA_rows']:
        raise ValueError('ASA gate population changed.')
    final={arm:teacher.copy() for arm in ('A','B')}
    for arm in ('A','B'):
        final[arm][positions[gate]]=raw[arm][gate]
    if not np.array_equal(final['A'][~r.route.eq('asa')],final['B'][~r.route.eq('asa')]):
        raise ValueError('Non-ASA route was changed by new expert.')
    ref=pd.DataFrame({'row_position':r.row_position,'truth':true.astype(np.int8),
                      'route':r.route,'root':f.root,'fold':f.proposed_fold})
    final_ledger=pd.DataFrame({'row_position':r.row_position,'teacher_prediction':teacher,
                               'gate_open':r.route.eq('asa') & np.isin(r.row_position.to_numpy(),positions[gate]),
                               'final_pred_A':final['A'],'final_pred_B':final['B']})
    final_ledger.to_parquet(DEST/'full_prediction_ledger.parquet',index=False)
    quality=evaluate_primary(ref,final_ledger.rename(columns={'final_pred_A':'pred_A','final_pred_B':'pred_B'}),
                             plan['quality_profile'])
    all_asaref=ref.iloc[positions].copy();all_asaref['raw_A']=raw['A'];all_asaref['raw_B']=raw['B']
    expert_metrics={arm:class_counts(all_asaref,f'raw_{arm}') for arm in ('A','B')}
    # Raw expert and routed output agree for this registered frozen gate; keep both accounts explicit.
    for arm in ('A','B'):
        if expert_metrics[arm]['1']['missed']!=quality['ASA'][arm]['1']['missed'] or expert_metrics[arm]['2']['missed']!=quality['ASA'][arm]['2']['missed']:
            raise ValueError('Expert/routed ASA loss was conflated.')
    other=(~r.route.eq('asa')).to_numpy()
    frozen_other_errors=int((teacher[other]!=true.to_numpy()[other]).sum())
    if frozen_other_errors != plan['evaluation_contract']['frozen_non_ASA_errors']:
        raise ValueError('Frozen non-ASA errors changed.')
    manifest=pd.read_parquet(MANIFEST)
    support=[]
    for fold in range(3):
        fit=manifest[(manifest.outer_fold.eq(fold)) & (manifest.fold.ne(fold))]
        q=d[d.fold.eq(fold)].copy()
        seen=set(fit.parameter.dropna())
        roots=fit[fit.parameter.notna()].groupby(['parameter','truth']).root.nunique().to_dict()
        def bucket(row):
            if pd.isna(row.parameter):return 'parameter_missing'
            if row.parameter not in seen:return 'parameter_unseen'
            count=roots.get((row.parameter,row.truth),0)
            return 'parameter_seen_class_absent' if count==0 else ('same_class_one_root' if count==1 else 'same_class_multiple_roots')
        param=manifest[(manifest.outer_fold.eq(fold)) & (manifest.fold.eq(fold))][['row_position','parameter']]
        q=q.merge(param,on='row_position',validate='one_to_one')
        if len(q)!=(d.fold==fold).sum():raise ValueError('Missing support rows.')
        q['support_bin']=q.apply(bucket,axis=1)
        for (truth,name),v in q.groupby(['truth','support_bin']):
            support.append({'fold':fold,'truth':int(truth),'support_bin':name,'rows':len(v),
                            'roots':int(v.root.nunique()),'A_errors':int(v.expert_pred_A.ne(v.truth).sum()),
                            'B_errors':int(v.expert_pred_B.ne(v.truth).sum())})
    if sum(q['rows'] for q in support)!=len(d):
        raise ValueError('Support diagnosis omitted ASA rows.')
    source=d.assign(A_wrong=raw['A']!=d.truth.to_numpy(),B_wrong=raw['B']!=d.truth.to_numpy()).groupby(['root','truth']).agg(
        rows=('local','size'),A_errors=('A_wrong','sum'),B_errors=('B_wrong','sum')).reset_index()
    source.to_csv(DEST/'source_group_changes.csv',index=False)
    save(DEST/'support_results.json',support)
    old=read(ROOT/'artifacts/v116_nested_selection_20260929/evaluation.json')
    old_A=old['ASA']['A']['errors']
    new_A=int(sum(quality['ASA']['A'][str(c)]['missed'] for c in (1,2)))
    result={'stage':'primary_six_paired_fits_evaluated','classifier_fits_new':len(fits),
            'optimizer_steps':sum(q['steps'] for q in fits),'raw_ASA_rows':len(d),
            'full_rows':n,'raw_expert_metrics':expert_metrics,'full_primary_recount':quality,
            'frozen_ASA_gate_exclusions':int((~gate).sum()),'frozen_non_ASA_errors':frozen_other_errors,
            'historical_fixed25_A_ASA_errors':old_A,'fresh_A_ASA_errors':new_A,
            'fresh_A_minus_historical_A_errors':new_A-old_A,
            'fit_receipts':fits,'model_replays':replays,'support_results_path':'support_results.json',
            'primary_quality_passed':bool(quality['primary_quality_passed']),
            'confirmation_allowed':bool(quality['primary_quality_passed']),
            'quality_acceptance':False,'model_promoted':False,
            'limitations':['Previously inspected source-closed development folds, not an independent external test.',
                           'A full-task primary gate is not seed confirmation or model promotion.',
                           'Support bins are postfit diagnosis and not inference-time information.']}
    save(DEST/'primary_evaluation.json',result)
    print(json.dumps({'stage':'primary_evaluated','fresh_A_ASA_errors':new_A,
       'B_ASA_errors':int(sum(quality['ASA']['B'][str(c)]['missed'] for c in (1,2))),
       'gates':quality['gates'],'confirmation_allowed':result['confirmation_allowed']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
