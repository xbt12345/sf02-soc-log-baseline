"""Independent receipt, exposure, and selected-model replay for v10.4."""
import hashlib
import json
import numpy as np
import pandas as pd
import joblib
import torch
from scipy import sparse

from run_v75 import ROOT, OUT, load_sparse, read, save, sha
from v89_common import LAST
from v104_phase_a import DEST as ADEST, FOLDS, ROWS, X, class_metrics
from v104_phase_b import DEST as BDEST, PREP, ASA_IDS, SparseTabM, IndependentMLP, predict_all, DEVICE


def main():
    output = BDEST / 'verification.json'
    if output.exists():
        raise FileExistsError(output)
    a_reg = read(ADEST / 'registration.json')
    b_reg = read(BDEST / 'registration.json')
    checks = {}
    checks['phase_A_source_unchanged'] = sha(ROOT/'training/v104_phase_a.py') == a_reg['source_sha256']
    checks['phase_B_source_unchanged'] = sha(ROOT/'training/v104_phase_b.py') == b_reg['source_sha256']
    checks['phase_A_inputs_unchanged'] = all(sha(ROOT/p)==h for p,h in a_reg['input_sha256'].items())
    checks['phase_B_inputs_unchanged'] = all(sha(ROOT/p)==h for p,h in b_reg['input_sha256'].items())
    old_eval = (ROOT/'training/v104_phase_b_evaluate.py').read_text(encoding='utf-8')
    new_eval = (ROOT/'training/v104_phase_b_evaluate_r2.py').read_text(encoding='utf-8')
    expected = old_eval.replace('DEST as ADEST, FOLDS, class_metrics',
                                'DEST as ADEST, FOLDS, ROWS, class_metrics')
    checks['evaluation_repair_only_missing_import'] = old_eval!=new_eval and expected==new_eval
    checks['evaluation_ledger_same_as_failed_first_pass'] = (
        sha(BDEST/'phase_B_ASA_OOF_ledger.parquet') ==
        '6347ddef7e347714896f555214d720abbfd671f00dfa4d9cc219ba096e525aca')
    f = pd.read_parquet(FOLDS,columns=['row_position','root','proposed_fold','old_source'])
    r = pd.read_parquet(ROWS,columns=['row_position','route','label_index'])
    fid = np.load(LAST/'row_feature_id.npy',mmap_mode='r')
    assert len(f)==len(r)==len(fid)==2056871
    fold = f.proposed_fold.to_numpy()
    truth = r.label_index.to_numpy()
    checks['all_rows_exact_once'] = bool(np.array_equal(f.row_position.to_numpy(),np.arange(len(f))) and
                                         np.array_equal(r.row_position.to_numpy(),np.arange(len(r))))
    checks['all_roots_one_fold'] = bool(f.groupby('root').proposed_fold.nunique().max()==1)
    x=load_sparse(X)
    delta=sparse.load_npz(PREP/'N1_delta.npz')
    rng=np.random.default_rng(10401)
    probe=np.sort(rng.choice(x.shape[0],size=1000,replace=False))
    a_reports=0;b_reports=0;candidate_reports=0
    for k in range(3):
        for pop in ('small','full'):
            folder=ADEST/f'fold{k}_{pop}'
            t=read(folder/'fit.json')
            mask=fold!=k
            if pop=='small':mask &= f.old_source.to_numpy()
            assert t['converged'] and t['heldout_gradient_rows']==0
            assert t['train_original_rows']==int(mask.sum())
            assert t['train_class_rows']==np.bincount(truth[mask],minlength=3).tolist()
            assert t['model_sha256']==sha(folder/'teacher.joblib')
            assert t['scores_sha256']==sha(folder/'scores_all_input_ids.npy')
            model=joblib.load(folder/'teacher.joblib')
            predicted=np.asarray(x[probe]@model['coef'])+model['intercept']
            saved=np.load(folder/'scores_all_input_ids.npy',mmap_mode='r')[probe]
            assert np.allclose(predicted,saved,rtol=1e-10,atol=1e-10)
            a_reports+=1
        folder=BDEST/f'fold{k}_N1_teacher'
        t=read(folder/'fit.json')
        mask=fold!=k
        assert t['converged'] and t['heldout_gradient_rows']==0
        assert t['train_original_rows']==int(mask.sum())
        assert t['train_class_rows']==np.bincount(truth[mask],minlength=3).tolist()
        assert t['model_sha256']==sha(folder/'teacher.joblib')
        assert t['scores_sha256']==sha(folder/'scores_all_input_ids.npy')
        model=joblib.load(folder/'teacher.joblib')
        predicted=np.asarray((x[probe]+delta[probe])@model['coef'])+model['intercept']
        saved=np.load(folder/'scores_all_input_ids.npy',mmap_mode='r')[probe]
        assert np.allclose(predicted,saved,rtol=1e-10,atol=1e-10)
        b_reports+=1
        for arm in ('MLP','TabM'):
            folder=BDEST/f'fold{k}_C_{arm}'
            report=read(folder/'fit.json')
            assert report['heldout_gradient_rows']==0 and report['epochs']==100
            assert report['train_ASA_original_rows']==int(((fold!=k)&r.route.eq('asa').to_numpy()).sum())
            for ep in (25,50,100):
                assert report['checkpoint_sha256'][str(ep)]['model']==sha(folder/f'epoch{ep}_model.pt')
                assert report['checkpoint_sha256'][str(ep)]['prob']==sha(folder/f'epoch{ep}_ASA_input_prob.npy')
            candidate_reports+=1
    checks['six_R0_phase_A_fits_and_probe_replays']=a_reports==6
    checks['three_N1_teacher_fits_and_probe_replays']=b_reports==3
    checks['six_neural_fits_and_checkpoint_hashes']=candidate_reports==6
    asax=sparse.load_npz(PREP/'N1_ASA.npz')
    for k in range(3):
        folder=BDEST/f'fold{k}_C_TabM'
        saved=np.load(folder/'epoch25_ASA_input_prob.npy')
        state=torch.load(folder/'epoch25_model.pt',map_location='cpu',weights_only=True)
        assert state['fold']==k and state['arm']=='TabM' and state['epoch']==25
        model=SparseTabM().to(DEVICE)
        model.load_state_dict(state['state_dict'])
        replay=predict_all(model,'TabM',asax,DEVICE)
        assert np.allclose(replay,saved,atol=2e-6,rtol=2e-6)
        del model,replay
        if DEVICE=='cuda':torch.cuda.empty_cache()
    checks['TabM_25_all_ASA_scores_reproduced_from_saved_models']=True
    ledger=pd.read_parquet(BDEST/'phase_B_ASA_OOF_ledger.parquet')
    assert len(ledger)==112807 and ledger.row_position.is_unique
    aa=np.flatnonzero(r.route.eq('asa').to_numpy())
    assert np.array_equal(ledger.row_position.to_numpy(),aa)
    assert np.array_equal(ledger.truth.to_numpy(),truth[aa])
    ids=np.load(ASA_IDS)
    local=np.full(x.shape[0],-1,np.int32)
    local[ids]=np.arange(len(ids))
    from_saved=np.empty(len(aa),np.int8)
    for k in range(3):
        take=fold[aa]==k
        scores=np.load(BDEST/f'fold{k}_C_TabM'/'epoch25_ASA_input_prob.npy',mmap_mode='r')
        from_saved[take]=scores[local[fid[aa[take]]]].argmax(1)
    assert np.array_equal(from_saved,ledger.C_TabM_epoch25.to_numpy())
    measured=class_metrics(truth[aa],from_saved)
    report=read(BDEST/'phase_B_evaluation.json')
    checks['ASA_OOF_and_metrics_recomputed']=(measured['errors']==report['candidates']['C_TabM_epoch25']['errors']==2980
        and abs(measured['MS_equal_F1']-report['candidates']['C_TabM_epoch25']['MS_equal_F1'])<1e-12)
    checks['no_candidate_passed_promotion']=report['selected'] is None and report['model_promoted'] is False
    checks['all_candidate_failure_is_M_recall']=all(
        [key for key,val in item['continuation_checks'].items() if not val]==['M_recall']
        for name,item in report['candidates'].items() if name!='N1_teacher')
    assert all(checks.values()),checks
    result={'status':'v104_independent_verification_complete',
            'all_checks_passed':True,'checks':checks,'phase_A_fits':a_reports,
            'phase_B_fits':b_reports+candidate_reports,'calibration_fits':0,
            'source_sha256':sha(__file__),
            'verified_model_replay':'TabM25 all ASA inputs and six teacher random probes',
            'limitations':['Internal grouped OOF already observed repeatedly; no external validation.',
                           'Post-hoc threshold frontier is diagnostic and not a chosen classifier.']}
    save(output,result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
