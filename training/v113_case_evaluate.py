"""Paired developmental OOF evaluation and full-task frozen-route replay."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import sparse

from run_v75 import ROOT, save, sha
from v104_phase_a import group_class_recall
from v104_phase_b import DEVICE, FID, ROWS, SEED, SparseTabM, predict_all
from v107_evaluate import report_metric
from v113_case_train import DEST, VIEW, OLD, FOLDS, check


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    reg = check()
    assert not (DEST/'evaluation.json').exists()
    row = pd.read_parquet(ROOT/'artifacts/v112_fine_supervision_review_20260929/interface_row_audit.parquet')
    reference = pd.read_parquet(OLD/'OOF_ASA_ledger.parquet')
    assert np.array_equal(row.row_position, reference.row_position)
    assert np.array_equal(row.truth, reference.truth)
    assert np.array_equal(row.root, reference.root)
    assert np.array_equal(row.fold, reference.fold)
    local = row.local.to_numpy(dtype=np.int64)
    fold = row.fold.to_numpy(dtype=np.int8)
    truth = row.truth.to_numpy(dtype=np.int8)
    base = reference.N1_TabM25.to_numpy(dtype=np.int8)
    x = sparse.load_npz(VIEW)
    probs = np.empty((len(row),3),np.float32)
    source = []
    for k in range(3):
        folder = DEST/f'fold{k}_case_TabM25_seed{SEED}'
        fit = read(folder/'fit.json')
        assert fit['status'] == 'fit_executed' and fit['epochs'] == 25 and fit['heldout_gradient_rows'] == 0
        assert fit['source_sha256'] == reg['source_sha256']
        assert sha(folder/'model.pt') == fit['model_sha256']
        assert sha(folder/'ASA_input_prob.npy') == fit['prob_sha256']
        checkpoint = torch.load(folder/'model.pt',map_location='cpu',weights_only=True)
        assert checkpoint['fold'] == k and checkpoint['view'] == 'case' and checkpoint['epoch'] == 25
        model = SparseTabM().to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        replay = predict_all(model,'TabM',x,DEVICE)
        saved = np.load(folder/'ASA_input_prob.npy')
        assert np.array_equal(replay.argmax(1),saved.argmax(1))
        assert np.allclose(replay,saved,rtol=2e-6,atol=2e-6)
        mask = fold == k
        probs[mask] = saved[local[mask]]
        source.append({'fold':k, 'fit_sha256':sha(folder/'fit.json'),
            'model_sha256':fit['model_sha256'], 'prob_sha256':fit['prob_sha256'],
            'replay_max_abs_diff':float(np.max(np.abs(replay-saved))),
            'last_online_CE':read(folder/'progress.json')[-1]['online_original_row_CE'],
            'train_errors_by_class':fit['train_errors_by_class'],
            'train_original_errors':fit['train_original_errors'], 'seconds':fit['seconds']})
        del model
        if DEVICE == 'cuda':torch.cuda.empty_cache()
    pred = probs.argmax(1).astype(np.int8)
    # Recount the reused A arm from the bound probability arrays.
    a_prob = np.empty_like(probs)
    for k in range(3):
        p = np.load(OLD/f'fold{k}_N1_TabM25_seed{SEED}'/'ASA_input_prob.npy',mmap_mode='r')
        take = fold==k
        a_prob[take] = p[local[take]]
    assert np.array_equal(a_prob.argmax(1),base)
    rootstat = []
    for root,g in row.assign(A=base,B=pred).groupby('root'):
        y = g.truth.to_numpy(); a = g.A.to_numpy(); b = g.B.to_numpy()
        item = {'root':int(root), 'rows':len(g)}
        for c,name in [(1,'M'),(2,'S')]:
            m = y==c
            item[name+'_support'] = int(m.sum())
            item[name+'_A_errors'] = int((m & (a!=c)).sum())
            item[name+'_B_errors'] = int((m & (b!=c)).sum())
            item[name+'_repaired'] = int((m & (a!=c) & (b==c)).sum())
            item[name+'_regressed'] = int((m & (a==c) & (b!=c)).sum())
        rootstat.append(item)
    roots = pd.DataFrame(rootstat).sort_values('root')
    roots.to_csv(DEST/'source_group_changes.csv',index=False,encoding='utf-8')
    top3 = row.loc[truth==2].groupby('root').size().nlargest(3).index.to_numpy()
    outside = (truth==2) & ~np.isin(row.root.to_numpy(),top3)
    audit = row[['row_position','root','fold','local','truth','behavior','coarse','interface_literal_pair','interface_parse_verified']].copy()
    audit['A_N1_prediction'] = base
    audit['B_case_prediction'] = pred
    audit['A_S_probability'] = a_prob[:,2]
    audit['B_S_probability'] = probs[:,2]
    audit['repaired'] = (base!=truth)&(pred==truth)
    audit['regressed'] = (base==truth)&(pred!=truth)
    audit.to_parquet(DEST/'OOF_ASA_decisions.parquet',index=False)
    asa_a = report_metric(truth,base)
    asa_b = report_metric(truth,pred)
    per_fold = {str(k):{'A':report_metric(truth[fold==k],base[fold==k]),
        'B':report_metric(truth[fold==k],pred[fold==k])} for k in range(3)}
    # Replay the 2,056,871 original three-class rows. Only ASA predictions
    # may change; the existing teacher and gate remain frozen and label-free.
    r = pd.read_parquet(ROWS,columns=['route','label_index'])
    f = pd.read_parquet(FOLDS,columns=['proposed_fold'])
    fid = np.load(FID,mmap_mode='r')
    assert len(r)==len(f)==len(fid)==2056871
    fids=np.asarray(fid,dtype=np.int64)
    all_y=r.label_index.to_numpy(dtype=np.int8)
    full_fold=f.proposed_fold.to_numpy(dtype=np.int8)
    teacher=np.empty(len(r),dtype=np.int8)
    for k in range(3):
        fit=read(OLD/f'fold{k}_N1_teacher'/'fit.json')
        path=OLD/f'fold{k}_N1_teacher'/'scores_all_input_ids.npy'
        assert fit['scores_sha256']==sha(path)
        scores=np.load(path,mmap_mode='r')
        take=np.flatnonzero(full_fold==k)
        teacher[take]=scores[fids[take]].argmax(1).astype(np.int8)
    asa=r.route.eq('asa').to_numpy()
    pos=np.flatnonzero(asa)
    assert np.array_equal(pos,row.row_position.to_numpy())
    full_a=teacher.copy();full_b=teacher.copy()
    gate=teacher[pos]!=0
    full_a[pos[gate]]=base[gate]
    full_b[pos[gate]]=pred[gate]
    assert np.array_equal(full_a[~asa],full_b[~asa])
    full_a_stats=report_metric(all_y,full_a)
    full_b_stats=report_metric(all_y,full_b)
    prior=read(OLD/'evaluation.json')['full_task']['N1_TabM25_composite']
    assert full_a_stats['confusion_matrix_true_rows_pred_columns']==prior['confusion_matrix_true_rows_pred_columns']
    assert asa_a['confusion_matrix_true_rows_pred_columns']==read(OLD/'evaluation.json')['variants']['N1_TabM25']['all_ASA']['confusion_matrix_true_rows_pred_columns']
    del full_a,full_b,teacher
    comparison={}
    for c,name in [(1,'M'),(2,'S')]:
        take=truth==c
        comparison[name]={'support':int(take.sum()),
            'A_correct':int((take&(base==c)).sum()),'B_correct':int((take&(pred==c)).sum()),
            'A_errors':int((take&(base!=c)).sum()),'B_errors':int((take&(pred!=c)).sum()),
            'repaired':int((take&(base!=c)&(pred==c)).sum()),
            'regressed':int((take&(base==c)&(pred!=c)).sum()),
            'groups_A_errors':int((roots[name+'_A_errors']>0).sum()),
            'groups_B_errors':int((roots[name+'_B_errors']>0).sum())}
    group_a={name:group_class_recall(reference,base,c) for c,name in [(1,'M'),(2,'S')]}
    group_b={name:group_class_recall(reference,pred,c) for c,name in [(1,'M'),(2,'S')]}
    target=math_target=int(np.floor(asa_a['errors']*.9))
    gates={
        'ASA_total_errors_at_most_90pct_matched_A':asa_b['errors']<=target,
        'ASA_M_errors_not_increase':comparison['M']['B_errors']<=comparison['M']['A_errors'],
        'ASA_S_errors_not_increase':comparison['S']['B_errors']<=comparison['S']['A_errors'],
        'full_B_false_alerts_not_increase':full_b_stats['class']['benign']['missed']<=full_a_stats['class']['benign']['missed'],
        'full_each_class_F1_not_decrease':all(full_b_stats['class'][name]['f1']>=full_a_stats['class'][name]['f1'] for name in ('benign','malicious','suspicious')),
        'S_recall_outside_top3_not_decrease':int((outside&(pred==2)).sum())>=int((outside&(base==2)).sum()),
        'M_S_group_mean_recall_not_decrease':all(group_b[name]['mean_recall']>=group_a[name]['mean_recall'] for name in ('M','S')),
        'at_least_two_folds_ASA_errors_decrease':sum(per_fold[str(k)]['B']['errors']<per_fold[str(k)]['A']['errors'] for k in range(3))>=2,
    }
    report={'status':'three_new_fits_evaluated_developmental','source_sha256':sha(__file__),
        'training_source_sha256':reg['source_sha256'],'registration_sha256':sha(DEST/'registration.json'),
        'classifier_fits_new':3,'baseline_fits_reused':3,'calibration_fits':0,
        'official_unknown_test_used':False,'external_training_data_used':False,
        'input_semantic_invariance':'verified ASA interface name lowercasing only; case-equivalent inputs canonicalize identically',
        'ASA':{'A':asa_a,'B':asa_b,'per_fold':per_fold,'class_comparison':comparison,
            'group_class_recall':{'A':group_a,'B':group_b},
            'S_outside_top3':{'support':int(outside.sum()),
                'A_correct':int((outside&(base==2)).sum()),'B_correct':int((outside&(pred==2)).sum())},
            'top3_S_roots':[int(v) for v in top3]},
        'full_task':{'A':full_a_stats,'B':full_b_stats},
        'suggested_engineering_ASA_max_errors':target,'gates':gates,
        'all_gates_passed':bool(all(gates.values())),
        'quality_acceptance':False,'model_promoted':False,
        'fits':source,'prediction_sha256':sha(DEST/'OOF_ASA_decisions.parquet'),
        'group_table_sha256':sha(DEST/'source_group_changes.csv'),
        'limits':['Current outer folds have been repeatedly inspected; this is source-held-out development, not blind transfer.',
            'No missing official M/S semantic rule was inferred from predictions.',
            'Case invariance is restricted to verified ASA interface name spans; suffix and unparsed records are preserved.',
            'Matched A reuses a V107 checkpoint only after identity and full task decision replay.']}
    save(DEST/'evaluation.json',report)
    print(json.dumps({'stage':'evaluated','A_errors':asa_a['errors'],'B_errors':asa_b['errors'],
        'class_comparison':comparison,'gates':gates,'all_gates_passed':report['all_gates_passed']},
        ensure_ascii=False),flush=True)


if __name__=='__main__':main()
