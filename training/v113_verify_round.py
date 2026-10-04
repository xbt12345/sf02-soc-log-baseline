"""Independent V113 label, paired-score, and source-group receipt checks."""
import json
import numpy as np
import pandas as pd
from scipy import sparse
from run_v75 import ROOT, save, sha
from v75_views import BYTE_FEATURES, byte_matrix
from v113_case_train import DEST, VIEW, AUDIT, OLD, PREP, SEED, check


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def main():
    target = DEST/'verification.json'
    assert not target.exists()
    reg = check()
    report = read(DEST/'evaluation.json')
    assert report['registration_sha256']==sha(DEST/'registration.json')
    assert report['source_sha256']==reg['evaluation_source_sha256']
    checks={}
    official=pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary'])
    labels=official.label_binary.map({'benign':0,'malicious':1,'suspicious':2}).to_numpy()
    d=pd.read_parquet(DEST/'OOF_ASA_decisions.parquet')
    checks['all_official_rows_and_ASA_labels']=(len(labels)==2056871 and len(d)==112807 and d.row_position.is_unique
        and np.array_equal(labels[d.row_position.to_numpy()],d.truth.to_numpy()))
    checks['same_old_fold_and_source_rows']=bool(np.array_equal(d.row_position,
        pd.read_parquet(OLD/'OOF_ASA_ledger.parquet',columns=['row_position']).row_position))
    checks['all_three_fits_bound_and_no_heldout_gradient']=len(report['fits'])==3 and all(
        read(DEST/f'fold{k}_case_TabM25_seed{SEED}'/'fit.json')['heldout_gradient_rows']==0 and
        read(DEST/f'fold{k}_case_TabM25_seed{SEED}'/'fit.json')['source_sha256']==reg['source_sha256']
        for k in range(3))
    pred_a=d.A_N1_prediction.to_numpy();pred_b=d.B_case_prediction.to_numpy();y=d.truth.to_numpy();fold=d.fold.to_numpy()
    checks['M_S_support_recount']=int((y==1).sum())==78748 and int((y==2).sum())==34059
    checks['per_class_correct_recount']=all(
        int(((y==c)&(pred_a==c)).sum())==report['ASA']['class_comparison'][name]['A_correct'] and
        int(((y==c)&(pred_b==c)).sum())==report['ASA']['class_comparison'][name]['B_correct']
        for c,name in [(1,'M'),(2,'S')])
    checks['all_repair_regression_flags_recount']=(np.array_equal(d.repaired.to_numpy(),(pred_a!=y)&(pred_b==y)) and
        np.array_equal(d.regressed.to_numpy(),(pred_a==y)&(pred_b!=y)))
    checks['fold_error_recount']=all(
        int(((fold==k)&(pred_b!=y)).sum())==report['ASA']['per_fold'][str(k)]['B']['errors'] and
        int(((fold==k)&(pred_a!=y)).sum())==report['ASA']['per_fold'][str(k)]['A']['errors']
        for k in range(3))
    checks['A_and_B_total_recount']=(int((pred_a!=y).sum())==report['ASA']['A']['errors'] and
        int((pred_b!=y).sum())==report['ASA']['B']['errors'])
    source=pd.read_csv(DEST/'source_group_changes.csv')
    checks['source_group_error_totals_recount']=all(
        int(source[name+'_A_errors'].sum())==int(((y==c)&(pred_a!=c)).sum()) and
        int(source[name+'_B_errors'].sum())==int(((y==c)&(pred_b!=c)).sum()) and
        int(source[name+'_support'].sum())==int((y==c).sum())
        for c,name in [(1,'M'),(2,'S')])
    checks['data_outputs_match_report_hashes']=(sha(DEST/'OOF_ASA_decisions.parquet')==report['prediction_sha256'] and
        sha(DEST/'source_group_changes.csv')==report['group_table_sha256'])
    checks['full_task_A_exact_historical_and_fixed_B_population']=(
        report['full_task']['A']['confusion_matrix_true_rows_pred_columns']==
        read(OLD/'evaluation.json')['full_task']['N1_TabM25_composite']['confusion_matrix_true_rows_pred_columns']
        and report['full_task']['B']['rows']==2056871)
    x=sparse.load_npz(VIEW);old=sparse.load_npz(PREP/'N1_ASA.npz')
    checks['facts_exact_and_full_matrix_shape']=x.shape==old.shape==(22546,66287) and (x[:,BYTE_FEATURES:]!=old[:,BYTE_FEATURES:]).nnz==0
    v=pd.read_parquet(AUDIT/'interface_views.parquet').sort_values('local')
    # Every case-swapped verified interface token must canonicalize to the
    # actual B input. This checks the advertised inference behavior, rather
    # than assuming a small change in prediction implies invariance.
    swapped=[]
    for item in v.itertuples(index=False):
        s=item.before
        for start,end,role in reversed(json.loads(item.name_spans)):
            s=s[:start]+s[start:end].swapcase()+s[end:]
        for start,end,role in reversed(json.loads(item.name_spans)):
            s=s[:start]+s[start:end].lower()+s[end:]
        swapped.append(s)
    xm=byte_matrix(swapped)
    checks['same_canonical_input_for_case_swaps']=(xm!=x[:,:BYTE_FEATURES]).nnz==0
    checks['last_actual_fits_and_quality_status']=report['classifier_fits_new']==3 and report['baseline_fits_reused']==3 and report['calibration_fits']==0 and report['model_promoted']==False
    checks['gate_boolean_consistency']=report['all_gates_passed']==all(report['gates'].values())
    assert all(checks.values()), checks
    files=[ROOT/'training/v113_case_train.py',ROOT/'training/v113_case_evaluate.py',
        ROOT/'training/v113_verify_round.py',DEST/'registration.json',DEST/'evaluation.json',
        DEST/'case_ASA.npz',DEST/'OOF_ASA_decisions.parquet',DEST/'source_group_changes.csv']
    for k in range(3):
        folder=DEST/f'fold{k}_case_TabM25_seed{SEED}'
        files += [folder/n for n in ['started.json','progress.json','fit.json','model.pt','ASA_input_prob.npy']]
    output={'status':'three_new_fits_independently_verified','all_checks_passed':True,'checks':checks,
        'new_classifier_fits':3,'baseline_fits_reused':3,'calibration_fits':0,
        'quality_acceptance':False,'model_promoted':False,
        'source_sha256':sha(__file__),
        'artifact_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in files},
        'scope':'Official label and class counts, OOF predictions, old baseline full replay, group counts, fit identity, case-swap equivalence. No private answers or unseen-environment result.'}
    save(target,output)
    print(json.dumps({'all_checks_passed':True,'checks':len(checks),'new_classifier_fits':3,
        'all_quality_gates_passed':report['all_gates_passed']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
