"""Post-fit V125 attribution checks. Read-only model replay; no optimizer or selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from v125_evaluate import OUT, TRACE, VIEW, load_model
from v125_experiment_review import ROOT, sha, read, require_run_seal
from v125_model import probabilities


def main():
    require_run_seal(OUT/'run_seal.json', ROOT/'training/v125_train.py')
    primary=read(OUT/'primary_evaluation.json')
    trace=pd.read_parquet(TRACE,columns=['row_position','local','root','fold','truth'])
    saved=pd.read_parquet(OUT/'expert_ASA_predictions.parquet')
    if not trace.row_position.equals(saved.row_position) or len(trace)!=112807:
        raise ValueError('Primary original-row alignment changed')
    x=sparse.load_npz(VIEW)
    length=np.load(OUT/'body_lengths.npy')
    y=trace.truth.to_numpy(dtype=np.int8)
    local=trace.local.to_numpy(dtype=np.int32)
    folds=trace.fold.to_numpy(dtype=np.int8)
    roots=trace.root.to_numpy(dtype=np.int32)
    data={}
    for arm in 'BC':
        base_pred=np.empty(len(trace),dtype=np.int8)
        base_s=np.empty(len(trace),dtype=np.float32)
        for fold in range(3):
            _,_,model=load_model(fold,arm,10201,OUT/f'fold{fold}_{arm}')
            model.branch=None
            p=probabilities(model,x,None,length,'cuda')
            q=folds==fold
            base_pred[q]=p[local[q]].argmax(1)
            base_s[q]=p[local[q],2]
            del model
        direct=saved[f'expert_pred_{arm}'].to_numpy(dtype=np.int8)
        a=saved.expert_pred_A.to_numpy(dtype=np.int8)
        by_fold=[]
        for fold in range(3):
            q=folds==fold
            by_fold.append({'fold':fold,'A_errors':int((a[q]!=y[q]).sum()),
                            'joint_errors':int((direct[q]!=y[q]).sum()),
                            'base_only_errors':int((base_pred[q]!=y[q]).sum()),
                            'joint_vs_base_only_flips':int((direct[q]!=base_pred[q]).sum())})
        r29=roots==29
        q29s=r29&(y==2)
        qreg=q29s&(a==2)&(direct==1)
        data[arm]={'by_fold':by_fold,
                   'root29':{'rows':int(r29.sum()),'S_rows':int(q29s.sum()),
                             'A_S_errors':int((a[q29s]!=2).sum()),
                             'joint_S_errors':int((direct[q29s]!=2).sum()),
                             'base_only_S_errors':int((base_pred[q29s]!=2).sum()),
                             'A_correct_joint_M':int(qreg.sum()),
                             'A_correct_base_only_M':int((q29s&(a==2)&(base_pred==1)).sum()),
                             'joint_M_base_only_S':int((qreg&(base_pred==2)).sum()),
                             'joint_M_base_only_M':int((qreg&(base_pred==1)).sum()),
                             'S_probability_base_only_mean':float(base_s[q29s].mean()),
                             'S_probability_joint_mean':float(saved.loc[q29s,f'expert_S_probability_{arm}'].mean())}}
        if arm=='B':
            np.save(OUT/'postmortem_B_base_only_pred.npy',base_pred)
        else:
            np.save(OUT/'postmortem_C_base_only_pred.npy',base_pred)
    out={'status':'postfit_ablation_no_training','primary_evaluation_sha256':sha(OUT/'primary_evaluation.json'),
         'expert_predictions_sha256':sha(OUT/'expert_ASA_predictions.parquet'),
         'base_only':data,'limitations':['Post-fit intervention identifies branch contribution to these saved models, not a randomized causal training experiment.',
                                       'Development-fold labels and outcomes have been inspected.'],
         'classifier_fits':0,'optimizer_steps':0,'model_promoted':False}
    (OUT/'postmortem_ablation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(out,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
