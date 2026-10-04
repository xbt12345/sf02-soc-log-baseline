"""Frozen endpoint CE, feature drift and old residual member diagnostics."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import numpy as np
import pandas as pd
import torch
from v142_runtime import ROOT,OUT,read,save,sha,require_run_seal
from v142_train import tensors
from v138_train import configure
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash


def main():
    require_run_seal(ROOT/'training/v142_train.py');configure();_,d=load_data();reports=[];residuals=[]
    if (OUT/'postfit_diagnosis.json').exists():raise FileExistsError('Keep previous diagnostics')
    for f in range(3):
        frame,c,pure,_,ids=fit_context(d,f);h,facts,m=tensors(f);oldq=np.load(OUT/f'fold{f}_zero_probability.npy')
        st=torch.load(OUT/f'fold{f}_S2/endpoint.pt',map_location='cpu',weights_only=True);m.load_state_dict(st['state']);before=tensor_hash(m.state_dict())
        perclass={cl:{'member_CE':0.,'ensemble_CE':0.,'feature_delta_L2':0.,'hidden_zero_fraction':0.} for cl in [1,2]};bad=set(frame.local[(pure[frame.local].astype(bool))&(oldq[frame.local].argmax(1)!=frame.truth)].to_list())
        with torch.no_grad():
            for start in range(0,len(ids),2048):
                ii=ids[start:start+2048];z=m(h[ii],facts[ii]);lp=torch.log_softmax(z,-1);q=lp.exp().mean(1);logq=torch.logsumexp(lp,1)-np.log(16)
                feature=m.features(h[ii]);delta=(feature-h[ii,:,128:]).square().sum((1,2)).sqrt();zeros=feature.eq(0).double().mean((1,2));member=z.argmax(-1)
                for cl in [1,2]:
                    w=torch.as_tensor(c[ii,cl],device='cuda',dtype=torch.float64);total=float(c[:,cl].sum())
                    perclass[cl]['member_CE']+=float((-lp[:,:,cl].mean(1)*w).sum())/total
                    perclass[cl]['ensemble_CE']+=float((-logq[:,cl]*w).sum())/total
                    perclass[cl]['feature_delta_L2']+=float((delta*w).sum())/total
                    perclass[cl]['hidden_zero_fraction']+=float((zeros*w).sum())/total
                for k,i in enumerate(ii):
                    if i in bad:
                        residuals.append({'fold':f,'local':int(i),'truth':int(c[i].argmax()),'original_mass':int(c[i].sum()),
                                          'before_pred':int(oldq[i].argmax()),'after_pred':int(q[k].argmax()),'before_p_true':float(oldq[i,2]),'after_p_true':float(q[k,2]),
                                          'correct_members':int(member[k].eq(2).sum()),'after_member_CE':float(-lp[k,:,2].mean()),
                                          'after_ensemble_CE':float(-logq[k,2]),'feature_delta_L2':float(delta[k])})
        if tensor_hash(m.state_dict())!=before:raise ValueError('Readonly diagnosis modified state')
        reports.append({'fold':f,'endpoint_perclass_original_row_metrics':perclass,'original_class_mass':c.sum(0).tolist(),
                        'endpoint_member_CE':sum(perclass[cl]['member_CE']*c[:,cl].sum() for cl in [1,2])/c.sum()})
        del h,facts,m;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(residuals).to_parquet(OUT/'original_residual_feature_repair.parquet',index=False)
    save(OUT/'postfit_diagnosis.json',{'status':'frozen_endpoint_TRAIN_only_diagnostics','new_fits':0,'new_updates':0,'new_gradient_evaluations':0,
                                     'folds':reports,'residual_inputs':len(residuals),'source_sha256':sha(__file__),'residual_sha256':sha(OUT/'original_residual_feature_repair.parquet'),
                                     'limits':'Feature drift and CE are diagnostic; actual training mastery and complete quality are independently decided.'})
    print(reports,flush=True)


if __name__=='__main__':main()
