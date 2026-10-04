"""Independent V107 same-fold OOF readout, replay and frozen continuation gates."""
import json
import re

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from scipy import sparse

from run_v75 import ROOT, OUT, load_sparse, save, sha
from v104_phase_a import group_class_recall
from v104_phase_b import (ASA_IDS, DEVICE, FID, PREP, ROWS, X,
                          SparseTabM, predict_all)
from v107_matched_training import DEST, FOLDS, check


def report_metric(y, pred):
    cm=np.zeros((3,3),np.int64)
    np.add.at(cm,(y,pred),1)
    names={}
    for c,label in enumerate(('benign','malicious','suspicious')):
        support=int(cm[c].sum());called=int(cm[:,c].sum());tp=int(cm[c,c])
        names[label]={'support':support,'correct':tp,'missed':support-tp,
                      'false_called':called-tp,'precision':tp/called if called else None,
                      'recall':tp/support if support else None,
                      'f1':2*tp/(support+called) if support+called else None}
    return {'rows':len(y),'errors':int(len(y)-np.trace(cm)),
            'confusion_matrix_true_rows_pred_columns':cm.tolist(),
            'predicted_counts':cm.sum(0).astype(int).tolist(),
            'class':names,
            'MS_equal_F1':sum(names[n]['f1'] or 0.0 for n in ('malicious','suspicious'))/2,
            'macro_F1':sum(names[n]['f1'] or 0.0 for n in ('benign','malicious','suspicious'))/3}


def group_recall(ledger, pred, c):
    return group_class_recall(ledger, pred, c)


def replay_all():
    check()
    x = load_sparse(X)
    asa = {v: sparse.load_npz((PREP if v=='N1' else DEST) / (v+'_ASA.npz'))
           for v in ('N1','N2')}
    for fold in (0,1,2):
        for view in ('N1','N2'):
            tf = DEST / f'fold{fold}_{view}_teacher'
            tr = json.loads((tf/'fit.json').read_text(encoding='utf-8'))
            assert tr['converged'] and tr['heldout_gradient_rows'] == 0
            assert sha(tf/'teacher.joblib') == tr['model_sha256']
            assert sha(tf/'scores_all_input_ids.npy') == tr['scores_sha256']
            model = joblib.load(tf/'teacher.joblib')
            scores = np.load(tf/'scores_all_input_ids.npy', mmap_mode='r')
            delta = sparse.load_npz((PREP if view=='N1' else DEST) / (view+'_delta.npz'))
            # Check every ASA input, plus a fixed random sample across all formats.
            rng = np.random.default_rng(10700+fold)
            rows = np.unique(np.r_[np.load(ASA_IDS), rng.choice(x.shape[0], 8192, replace=False)])
            replay = np.asarray((x[rows]+delta[rows]) @ model['coef'])+model['intercept']
            assert np.allclose(replay, scores[rows], atol=1e-8, rtol=1e-8)
            cf = DEST / f'fold{fold}_{view}_TabM25_seed10201'
            cr = json.loads((cf/'fit.json').read_text(encoding='utf-8'))
            assert cr['epochs'] == 25 and cr['heldout_gradient_rows'] == 0
            assert sha(cf/'model.pt') == cr['model_sha256']
            assert sha(cf/'ASA_input_prob.npy') == cr['prob_sha256']
            checkpoint = torch.load(cf/'model.pt', map_location='cpu', weights_only=True)
            assert checkpoint['fold'] == fold and checkpoint['view'] == view and checkpoint['epoch'] == 25
            net = SparseTabM().to(DEVICE)
            net.load_state_dict(checkpoint['state_dict'])
            probability = predict_all(net, 'TabM', asa[view], DEVICE)
            assert np.allclose(probability, np.load(cf/'ASA_input_prob.npy'), atol=2e-6, rtol=2e-6)
            print(json.dumps({'stage':'replay_pass', 'fold':fold,'view':view}), flush=True)


def main():
    assert not (DEST/'evaluation.json').exists()
    replay_all()
    f = pd.read_parquet(FOLDS, columns=['proposed_fold','root'])
    r = pd.read_parquet(ROWS, columns=['route','label_index'])
    fid = np.load(FID, mmap_mode='r')
    ids = np.load(ASA_IDS)
    lookup = np.full(int(fid.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    asa = r.route.eq('asa').to_numpy()
    pos = np.flatnonzero(asa)
    local = lookup[fid[pos]]
    assert (local >= 0).all() and len(pos)==112807
    folds = f.proposed_fold.to_numpy()
    afold = folds[pos]
    yall = r.label_index.to_numpy(dtype=np.int8)
    y = yall[pos]
    roots = f.root.to_numpy()[pos]
    old = pd.read_parquet(ROOT/'artifacts/v106_frozen_audit_20260928/paired_OOF_predictions.parquet',
                          columns=['row_position','root','nested_pattern'])
    assert np.array_equal(old.row_position.to_numpy(), pos)
    nested = old.nested_pattern.to_numpy(dtype=bool)
    old_roots = old.root.to_numpy()
    ledger = pd.DataFrame({'row_position':pos,'root':roots,'fold':afold,'truth':y})
    preds={}; full={}
    for view in ('N1','N2'):
        teacher_all = np.empty(len(f), np.int8)
        for fold in (0,1,2):
            scores=np.load(DEST/f'fold{fold}_{view}_teacher'/'scores_all_input_ids.npy',mmap_mode='r')
            take=folds==fold
            teacher_all[take]=scores[fid[take]].argmax(1).astype(np.int8)
        full[f'{view}_teacher']=teacher_all
        preds[f'{view}_teacher']=teacher_all[asa]
        probability=np.empty((len(pos),3),np.float32)
        for fold in (0,1,2):
            take=afold==fold
            p=np.load(DEST/f'fold{fold}_{view}_TabM25_seed10201'/'ASA_input_prob.npy',mmap_mode='r')
            probability[take]=p[local[take]]
        preds[f'{view}_TabM25']=probability.argmax(1).astype(np.int8)
        ledger[f'{view}_S_probability']=probability[:,2]
    for name,pred in preds.items():
        ledger[name]=pred
    # Composite rule was frozen before fitting: teacher retains non-ASA and
    # ASA rows it calls benign. No true-label-dependent routing.
    base_all=full['N1_teacher']
    for view in ('N1','N2'):
        cp=base_all.copy()
        gate=base_all[pos]!=0
        cp[pos[gate]]=preds[f'{view}_TabM25'][gate]
        full[f'{view}_TabM25_composite']=cp
    masks={'all_ASA':np.ones(len(y),bool),
           'no_nested_wrapper':~nested,'nested_wrapper':nested,
           'outside_historic_637660_2868':~np.isin(old_roots,[637660,2868]),
           'historic_637660':old_roots==637660,'historic_2868':old_roots==2868,
           'historic_2300':old_roots==2300}
    # These are lexical descriptions of visible evidence, not extra features.
    hard={s:np.zeros(len(pos),bool) for s in ('icmp_type3_code13','udp_514','tcp_6514')}
    rawfile=pq.ParquetFile(ROOT/'data/official/train.parquet')
    offset=0
    for batch in rawfile.iter_batches(batch_size=65536,columns=['message_sanitized']):
        lo=np.searchsorted(pos,offset);hi=np.searchsorted(pos,offset+batch.num_rows)
        if hi>lo:
            loc=pos[lo:hi]-offset
            texts=batch.column(0).take(pa.array(loc)).to_pylist()
            for j,text in enumerate(texts,start=lo):
                text=(text or '').lower()
                hard['icmp_type3_code13'][j]=('icmp' in text and bool(re.search(r'(?:type\s*[=: ]\s*3|type3)',text)) and bool(re.search(r'(?:code\s*[=: ]\s*13|code13)',text)))
                hard['udp_514'][j]='udp' in text and bool(re.search(r'(?<!\d)514(?!\d)',text))
                hard['tcp_6514'][j]='tcp' in text and bool(re.search(r'(?<!\d)6514(?!\d)',text))
        offset+=batch.num_rows
    assert offset==len(f)
    masks.update(hard)
    summary={'status':'twelve_fits_evaluated_developmental', 'first_wave_fits':12,
             'data_scope':'official original rows, body-source-closed OOF; no fresh blind or private answers',
             'source_sha256':sha(__file__),'training_source_sha256':sha(ROOT/'training/v107_matched_training.py'),
             'registration_sha256':sha(DEST/'registration.json'),
             'classes_full':np.bincount(yall,minlength=3).tolist(),
             'classes_ASA':np.bincount(y,minlength=3).tolist(),
             'variants':{},'full_task':{},'comparisons':{}}
    invariant=json.loads((ROOT/'artifacts/v106_frozen_audit_20260928/verification_and_N2_preflight.json').read_text(encoding='utf-8'))
    assert invariant['all_checks_passed'] and invariant['checks']['N2_collapse_expand_invariance_all_unique_raw']
    summary['paired_wrapper_variant_invariance']={
       'tested_unique_raw':invariant['N2_invariance_unique_raw_checked'],
       'same_N2_input_for_original_and_expanded_after_collapse':True,
       'N2_model_prediction_flips_by_construction':0,
       'quality_scope':'Input equality and deterministic prediction only; not correct classification.'}
    for name,pred in preds.items():
        d={'all_ASA':report_metric(y,pred),
           'per_fold':{str(k):report_metric(y[afold==k],pred[afold==k]) for k in (0,1,2)},
           'group_M':group_recall(ledger,pred,1),
           'group_S':group_recall(ledger,pred,2),
           'slices':{s:report_metric(y[m],pred[m]) for s,m in masks.items() if m.any()}}
        summary['variants'][name]=d
    for name,pred in full.items():
        stats=report_metric(yall,pred)
        stats['benign_false_alerts_per_10000']=stats['class']['benign']['missed']*10000/stats['class']['benign']['support']
        summary['full_task'][name]=stats
    ref=preds['N1_teacher']
    for name,pred in preds.items():
        if name=='N1_teacher': continue
        correct_ref=ref==y; correct=pred==y
        d={'repaired':int((~correct_ref&correct).sum()),
           'regressed':int((correct_ref&~correct).sum()),
           'by_class':{label:{'repaired':int((~correct_ref&correct&(y==c)).sum()),
                              'regressed':int((correct_ref&~correct&(y==c)).sum())}
                       for c,label in ((1,'M'),(2,'S'))}}
        if name=='N2_TabM25':
            other=preds['N1_TabM25']; othercorrect=other==y
            d['vs_same_arch_N1']={'repaired':int((~othercorrect&correct).sum()),
                                  'regressed':int((othercorrect&~correct).sum()),
                                  'by_class':{label:{'repaired':int((~othercorrect&correct&(y==c)).sum()),
                                                     'regressed':int((othercorrect&~correct&(y==c)).sum())}
                                              for c,label in ((1,'M'),(2,'S'))}}
        summary['comparisons'][name]=d
    base=summary['variants']['N1_teacher']; candidate=summary['variants']['N2_TabM25']
    no=base['slices']['no_nested_wrapper']; no_new=candidate['slices']['no_nested_wrapper']
    checks={
      'M_recall_at_least_N1_teacher':candidate['all_ASA']['class']['malicious']['recall']>=base['all_ASA']['class']['malicious']['recall'],
      'M_F1_at_least_N1_teacher':candidate['all_ASA']['class']['malicious']['f1']>=base['all_ASA']['class']['malicious']['f1'],
      'S_recall_at_least_N1_teacher':candidate['all_ASA']['class']['suspicious']['recall']>=base['all_ASA']['class']['suspicious']['recall'],
      'S_F1_at_least_N1_teacher':candidate['all_ASA']['class']['suspicious']['f1']>=base['all_ASA']['class']['suspicious']['f1'],
      'MS_F1_improved':candidate['all_ASA']['MS_equal_F1']>base['all_ASA']['MS_equal_F1'],
      'no_wrapper_S_recall_improved':no_new['class']['suspicious']['recall']>no['class']['suspicious']['recall'],
      'no_wrapper_MS_F1_improved':no_new['MS_equal_F1']>no['MS_equal_F1'],
      'S_group_recall_improved':candidate['group_S']['mean_recall']>base['group_S']['mean_recall'],
      'S_zero_group_count_decreased':candidate['group_S']['zero_recall_groups']<base['group_S']['zero_recall_groups'],
      'at_least_two_folds_MS_F1_improved':sum(candidate['per_fold'][str(k)]['MS_equal_F1']>base['per_fold'][str(k)]['MS_equal_F1'] for k in (0,1,2))>=2,
      'outside_major_roots_MS_F1_improved':candidate['slices']['outside_historic_637660_2868']['MS_equal_F1']>base['slices']['outside_historic_637660_2868']['MS_equal_F1'],
      'benign_boundary_no_regression':summary['full_task']['N2_TabM25_composite']['class']['benign']['missed']<=summary['full_task']['N1_teacher']['class']['benign']['missed'],
      'non_ASA_exactly_frozen':bool(np.array_equal(full['N2_TabM25_composite'][~asa],base_all[~asa])),
    }
    summary['frozen_gate_checks']=checks
    summary['continue_replication']=bool(all(checks.values()))
    summary['model_promoted']=False
    ledger.to_parquet(DEST/'OOF_ASA_ledger.parquet',index=False)
    summary['OOF_ledger_sha256']=sha(DEST/'OOF_ASA_ledger.parquet')
    save(DEST/'evaluation.json',summary)
    print(json.dumps({'stage':'evaluated', 'continue_replication':summary['continue_replication'],
          'gate_checks':checks,
          'ASA':{n:{'M_correct':v['all_ASA']['class']['malicious']['correct'],
                     'S_correct':v['all_ASA']['class']['suspicious']['correct'],
                     'MS_F1':v['all_ASA']['MS_equal_F1']} for n,v in summary['variants'].items()}},
          ensure_ascii=False),flush=True)


if __name__=='__main__': main()
