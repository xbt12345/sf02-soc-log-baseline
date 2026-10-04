"""V110 diagnostic recount: training error, held-out ranking and root concentration; no fits."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from run_v75 import ROOT, save, sha
from v108_root_evidence_audit import oracle
from v110_layer_probes import DEST, LEDGER, PREV, check_registration, read_json


def errors(y,pred):
    return {name:{'rows':int((y==c).sum()),'correct':int(((y==c)&(pred==c)).sum()),
      'errors':int(((y==c)&(pred!=c)).sum())} for c,name in ((1,'M'),(2,'S'))}


def main():
    check_registration()
    target=DEST/'postfit_diagnosis_r2.json'
    assert not target.exists()
    d=pd.read_parquet(LEDGER)
    t=pd.read_parquet(DEST/'OOF_probe_comparison.parquet')
    assert np.array_equal(t.row_position.to_numpy(),d.row_position.to_numpy())
    y=d.truth.to_numpy();fold=d.fold.to_numpy();local=d.local.to_numpy();root=d.root.to_numpy()
    top3=np.asarray(d.loc[d.truth==2,'root'].value_counts().head(3).index)
    outside=~np.isin(root,top3)
    result={'status':'postfit_no_training_diagnosis_corrected_r2','classifier_fits':0,'calibration_fits':0,
      'source_sha256':sha(__file__),'registration_sha256':sha(DEST/'registration.json'),
      'OOF_sha256':sha(DEST/'OOF_probe_comparison.parquet'),'fold_train_vs_holdout':{},
      'fixed_score_ranking_diagnostic':{},'root_error_concentration':{},'support_slices':{},
      'supersedes':'postfit_diagnosis.json; its destination_port_unavailable slice was incorrectly empty because an unavailable port is omitted from the behavior key, not represented as 65536.',
      'limitations':['Threshold optima use already inspected labels and are not deployable calibration.',
        'Root-level observations are source-proxy diagnostics, not independent customer environments.',
        'Training errors compare fixed trained checkpoints on their own training rows; training and held-out populations differ.',
        'A behavior key without a fixed destination port includes not-applicable protocols as well as redaction; it does not count confirmed redactions.']}
    for arm in ('P1','P2'):
        result['fold_train_vs_holdout'][arm]={}
        for k in (0,1,2):
            sc=np.load(DEST/f'fold{k}_{arm}'/'ASA_input_S_probability.npy')
            oldp=np.load(PREV/f'fold{k}_N2_TabM25_seed10201'/'ASA_input_prob.npy')
            train=fold!=k;held=~train
            assert np.array_equal((oldp[local[held]].argmax(1)),d.loc[held,'N2_TabM25'].to_numpy())
            result['fold_train_vs_holdout'][arm][str(k)]={
              'train_new':errors(y[train],np.where(sc[local[train]]>=.5,2,1)),
              'train_old_N2':errors(y[train],oldp[local[train]].argmax(1)),
              'heldout_new':errors(y[held],np.where(sc[local[held]]>=.5,2,1)),
              'heldout_old_N2':errors(y[held],oldp[local[held]].argmax(1))}
        scores=t[arm+'_S_probability'].to_numpy()
        pred=t[arm+'_prediction'].to_numpy();ref=d.N1_TabM25.to_numpy()
        result['fixed_score_ranking_diagnostic'][arm]={
          'OOF_S_AUC':float(roc_auc_score(y==2,scores)),
          'optimistic_global_fixed_score':oracle(y,scores,318,31965),
          'per_fold_at_N1_M_error_budgets':[
              {'fold':k,**oracle(y[fold==k],scores[fold==k],
                int(((fold==k)&(y==1)&(ref!=1)).sum()),
                int(((fold==k)&(y==2)&(ref==2)).sum()))} for k in (0,1,2)]}
        out={}
        for refname in ('N1','N2'):
            ref=d[refname+'_TabM25'].to_numpy()
            newcorrect=pred==y;oldcorrect=ref==y
            regress=(oldcorrect&~newcorrect);repair=(~oldcorrect&newcorrect)
            out[refname]={
              'M_regressed_fold1_fraction':float(((regress)&(y==1)&(fold==1)).sum()/max(1,((regress)&(y==1)).sum())),
              'S_repairs_top3_roots':int((repair&(y==2)&~outside).sum()),
              'S_regressions_top3_roots':int((regress&(y==2)&~outside).sum()),
              'S_repairs_other_roots':int((repair&(y==2)&outside).sum()),
              'S_regressions_other_roots':int((regress&(y==2)&outside).sum())}
        result['root_error_concentration'][arm]=out
        groups=pd.read_parquet(DEST/f'{arm}_vs_N2_source_group_delta.parquet')
        groups['M_net_loss']=groups.M_old_correct-groups.M_new_correct
        groups['S_net_gain']=groups.S_new_correct-groups.S_old_correct
        result['root_error_concentration'][arm]['N2']['largest_M_net_loss_roots']=groups.nlargest(8,'M_net_loss')[['root','fold','M_rows','M_net_loss','S_rows','S_net_gain']].to_dict('records')
        result['root_error_concentration'][arm]['N2']['largest_S_net_gain_roots']=groups.nlargest(8,'S_net_gain')[['root','fold','S_rows','S_net_gain','M_rows','M_net_loss']].to_dict('records')
        support={}
        for key,mask in {
          'exact_same_class_source_zero':d.behavior_same_roots.to_numpy()==0,
          'exact_same_class_source_one':d.behavior_same_roots.to_numpy()==1,
          'exact_same_class_source_two_plus':d.behavior_same_roots.to_numpy()>=2,
          'N2_wrapper_changed':d.changed.to_numpy(),
          'N2_wrapper_unchanged':~d.changed.to_numpy(),
          'no_fixed_destination_port_in_behavior':~d.behavior.str.contains('"dst_port_fixed":',regex=False,na=False).to_numpy(),
          'outside_top3_S_roots':outside}.items():
            ss=mask&(y==2)
            support[key]={'S_rows':int(ss.sum()),'N1_correct':int(((d.N1_TabM25.to_numpy()==2)&ss).sum()),
              'N2_correct':int(((d.N2_TabM25.to_numpy()==2)&ss).sum()),
              'new_correct':int(((pred==2)&ss).sum())}
        result['support_slices'][arm]=support
    save(target,result)
    print(json.dumps({'status':result['status'],'rankings':result['fixed_score_ranking_diagnostic'],
      'root_concentration':result['root_error_concentration']},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
