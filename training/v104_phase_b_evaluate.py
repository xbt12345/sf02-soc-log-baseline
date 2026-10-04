"""Frozen grouped OOF readout and declared v10.3 continuation guards."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_v75 import ROOT, OUT, read, save, sha
from v89_common import LAST
from v104_phase_a import DEST as ADEST, FOLDS, class_metrics, group_class_recall
from v104_phase_b import DEST, PREP, ASA_IDS, check


def main():
    check()
    output = DEST / 'phase_B_evaluation.json'
    if output.exists():
        raise FileExistsError(output)
    for fold in range(3):
        teacher = DEST / f'fold{fold}_N1_teacher'
        report = read(teacher / 'fit.json')
        assert report['converged'] and report['heldout_gradient_rows'] == 0
        assert report['model_sha256'] == sha(teacher / 'teacher.joblib')
        assert report['scores_sha256'] == sha(teacher / 'scores_all_input_ids.npy')
        for arm in ('MLP', 'TabM'):
            folder = DEST / f'fold{fold}_C_{arm}'
            fit = read(folder / 'fit.json')
            assert fit['epochs'] == 100 and fit['heldout_gradient_rows'] == 0
            for epoch, binding in fit['checkpoint_sha256'].items():
                assert binding['model'] == sha(folder / f'epoch{epoch}_model.pt')
                assert binding['prob'] == sha(folder / f'epoch{epoch}_ASA_input_prob.npy')
    f = pd.read_parquet(FOLDS, columns=['proposed_fold', 'root'])
    r = pd.read_parquet(OUT / 'rows.parquet', columns=['route', 'label_index'])
    fid = np.load(LAST / 'row_feature_id.npy', mmap_mode='r')
    ids = np.load(ASA_IDS)
    lookup = np.full(457566, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    asa = r.route.eq('asa').to_numpy()
    apos = np.flatnonzero(asa)
    local = lookup[fid[asa]]
    assert (local >= 0).all()
    folds = f.proposed_fold.to_numpy()
    truth_all = r.label_index.to_numpy()
    y = truth_all[asa]
    afold = folds[asa]
    ledger = pd.DataFrame({'row_position': apos, 'root': f.root.to_numpy()[asa],
                           'fold': afold, 'truth': y})
    reference_all = np.load(ADEST / 'full_all_original_OOF_pred.npy')
    reference_asa = reference_all[asa]
    assert len(y) == 112807
    variants = {}
    n1_all_pred = np.empty(len(f), np.int8)
    n1_scores = np.empty((len(y), 3), np.float64)
    for fold in range(3):
        scores = np.load(DEST / f'fold{fold}_N1_teacher' / 'scores_all_input_ids.npy', mmap_mode='r')
        take = folds == fold
        n1_all_pred[take] = scores[fid[take]].argmax(1).astype(np.int8)
        n1_scores[afold == fold] = scores[fid[asa][afold == fold]]
    variants['N1_teacher'] = {'pred': n1_all_pred[asa],
                              'rank_score': n1_scores[:, 2]-n1_scores[:, 1],
                              'all_pred': n1_all_pred}
    ledger['R0_full_teacher'] = reference_asa
    ledger['N1_teacher'] = n1_all_pred[asa]
    for arm in ('MLP','TabM'):
        for epoch in (25,50,100):
            probability = np.empty((len(y),3), np.float32)
            for fold in range(3):
                take = afold == fold
                path = DEST / f'fold{fold}_C_{arm}' / f'epoch{epoch}_ASA_input_prob.npy'
                probability[take] = np.load(path, mmap_mode='r')[local[take]]
            name = f'C_{arm}_epoch{epoch}'
            pred = probability.argmax(1).astype(np.int8)
            variants[name] = {'pred': pred,
                              'rank_score': probability[:,2]-probability[:,1]}
            ledger[name] = pred
            for c, label in enumerate(('B','M','S')):
                ledger[f'{name}_{label}_prob'] = probability[:,c]
    ledger.to_parquet(DEST / 'phase_B_ASA_OOF_ledger.parquet', index=False)
    first = class_metrics(y, reference_asa)
    base = class_metrics(y, variants['N1_teacher']['pred'])
    variant_audit = read(PREP / 'source_variant_audit.json')
    assert variant_audit['N1_feature_equivalence_checks'] == 6693
    assert variant_audit['N1_prediction_flips_by_construction'] == 0
    base_fold = {fold: class_metrics(y[afold==fold], variants['N1_teacher']['pred'][afold==fold])
                 for fold in range(3)}
    base_group = group_class_recall(ledger, variants['N1_teacher']['pred'], 2)
    report = {'status':'phase_B_nine_fits_evaluated',
              'classifier_fits':9, 'calibration_fits':0,
              'ASA_rows':len(y), 'metric_scope':'fixed grouped official OOF, observed repeatedly; developmental',
              'R0_full_teacher':first, 'N1_teacher':base,
              'candidates':{}, 'ranked':[], 'qualified':[], 'selected':None,
              'model_promoted':False,
              'OOF_ledger_sha256':sha(DEST/'phase_B_ASA_OOF_ledger.parquet')}
    giant = ledger[ledger.truth==2].groupby('root').size().idxmax()
    for name, item in variants.items():
        pred = item['pred']
        metrics = class_metrics(y, pred)
        fold_metric = {str(k): class_metrics(y[afold==k],pred[afold==k]) for k in range(3)}
        group = {'M':group_class_recall(ledger,pred,1),
                 'S':group_class_recall(ledger,pred,2)}
        neg = (variants['N1_teacher']['pred']==y)&(pred!=y)
        fix = (variants['N1_teacher']['pred']!=y)&(pred==y)
        old_neg = (reference_asa==y)&(pred!=y)
        old_fix = (reference_asa!=y)&(pred==y)
        miss_giant = ledger.root.to_numpy()!=giant
        metrics.update(fold=fold_metric,group=group,
            ranking_S_vs_M_AUC=float(roc_auc_score(y==2,item['rank_score'])),
            excluding_largest_S_group=class_metrics(y[miss_giant],pred[miss_giant]),
            largest_S_group_rows=int(((ledger.root==giant)&(ledger.truth==2)).sum()),
            vs_N1_teacher={'repaired':int(fix.sum()),'regressed':int(neg.sum()),
                'by_class':{str(c):{'repaired':int((fix&(y==c)).sum()),
                                    'regressed':int((neg&(y==c)).sum())} for c in (1,2)}},
            vs_R0_full_teacher={'repaired':int(old_fix.sum()),'regressed':int(old_neg.sum()),
                'by_class':{str(c):{'repaired':int((old_fix&(y==c)).sum()),
                                    'regressed':int((old_neg&(y==c)).sum())} for c in (1,2)}})
        report['candidates'][name]=metrics
    target = pd.read_parquet(ROOT/'artifacts/v103_plan_preflight_20260928/113_target_support_after_split.parquet',
                             columns=['row_position'])
    is_target = ledger.row_position.isin(target.row_position).to_numpy()
    support = pd.read_parquet(ROOT/'artifacts/v103_plan_preflight_20260928/ASA_support_preflight.parquet',
                              columns=['row_position','small_exact_support','full_exact_support'])
    support = ledger[['row_position']].merge(support,on='row_position',validate='1:1')
    strict_gain = (support.small_exact_support==0).to_numpy() & (support.full_exact_support>0).to_numpy()
    report['diagnostic_slices']={}
    for name,item in variants.items():
        pred=item['pred']
        report['diagnostic_slices'][name]={
            'historic_113_correct':int((is_target&(pred==y)).sum()),
            'historic_101_newly_supported_correct':int((is_target&strict_gain&(pred==y)).sum()),
            'all_newly_supported_S_correct':int((strict_gain&(y==2)&(pred==y)).sum()),
            'all_newly_supported_S_total':int((strict_gain&(y==2)).sum())}
    assert int((is_target&strict_gain).sum())==101
    for name,metrics in report['candidates'].items():
        if name=='N1_teacher':
            continue
        checks={
            'M_recall':metrics['class']['malicious']['recall']>=base['class']['malicious']['recall']-1e-12,
            'M_F1':metrics['class']['malicious']['f1']>=base['class']['malicious']['f1']-1e-12,
            'S_recall':metrics['class']['suspicious']['recall']>=base['class']['suspicious']['recall']-1e-12,
            'S_F1':metrics['class']['suspicious']['f1']>=base['class']['suspicious']['f1']-1e-12,
            'MS_F1_improved':metrics['MS_equal_F1']>base['MS_equal_F1'],
            'S_root_recall_improved':metrics['group']['S']['mean_recall']>base_group['mean_recall'],
            'worst_fold_non_decreasing':min(x['MS_equal_F1'] for x in metrics['fold'].values())>=
                min(x['MS_equal_F1'] for x in base_fold.values())-1e-12,
            'at_least_two_folds_net_improved':sum(metrics['fold'][str(k)]['MS_equal_F1']>
                base_fold[k]['MS_equal_F1'] for k in range(3))>=2,
            'legal_wrapper_invariance':variant_audit['N1_prediction_flips_by_construction']==0,
        }
        metrics['continuation_checks']=checks
        metrics['research_continue']=bool(all(checks.values()))
        metrics['official_old_correct_gate']=metrics['vs_R0_full_teacher']['regressed']==0
        metrics['quality_acceptance']=False
    report['ranked']=sorted([n for n in report['candidates'] if n!='N1_teacher'],
        key=lambda n:(-report['candidates'][n]['MS_equal_F1'],
                      int(n.rsplit('epoch',1)[1]),n))
    report['qualified']=[n for n in report['ranked'] if report['candidates'][n]['research_continue']]
    report['selected']=report['qualified'][0] if report['qualified'] else None
    # Candidate raw ASA output has no independent benign supervision. The
    # composition keeps the same-fold full N1 teacher outside ASA, and on ASA
    # where it predicts benign. No truth-dependent per-row routing.
    report['full_task_composition']={}
    for name in ['N1_teacher']+report['ranked']:
        pred=n1_all_pred.copy()
        if name!='N1_teacher':
            pred[apos[n1_all_pred[asa]!=0]]=variants[name]['pred'][n1_all_pred[asa]!=0]
        report['full_task_composition'][name]=class_metrics(truth_all,pred)
    report['fixed_R0_full_task_reference']=class_metrics(truth_all,reference_all)
    report['normal_boundary_held_ASA_rows']=int((n1_all_pred[asa]==0).sum())
    report['N1_variant_audit_sha256']=sha(PREP / 'source_variant_audit.json')
    report['source_sha256']=sha(__file__)
    report['input_sha256']={str(p.relative_to(ROOT)):sha(p) for p in
        [DEST/'registration.json', ADEST/'phase_A_evaluation.json',
         ADEST/'phase_A_extended_diagnosis.json', FOLDS, ROWS, LAST/'row_feature_id.npy']}
    save(output,report)
    print(json.dumps({'stage':'phase_B_evaluated','selected':report['selected'],
        'teacher_errors':base['errors'],
        'ranked':[(n,report['candidates'][n]['errors'],
                   report['candidates'][n]['MS_equal_F1'],
                   report['candidates'][n]['research_continue']) for n in report['ranked']],
        'full_task_reference_errors':report['fixed_R0_full_task_reference']['errors']},
        ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
