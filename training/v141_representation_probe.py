"""TRAIN-only actual first-layer replay and second-layer gradients, zero updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha
from v138_runtime import save,OUT as BACKBONE
from v140_runtime import OUT as CURRENT,require_run_seal
from v138_train import configure,historical
from v135_model import Classifier,batch,tensor_hash
from v135_runtime import load_data,fit_context
from v138_readout import full_gradient,probabilities
from v141_second_model import SecondRepresentation

OUT=ROOT/'artifacts/v141_representation_evidence_20261001'


def main():
    require_run_seal(ROOT/'training/v140_train.py');configure()
    if OUT.exists():raise FileExistsError('Preserve previous probe')
    OUT.mkdir();x,d=load_data();reports=[];slopes=[];neighbors=[]
    for fold in range(3):
        st=torch.load(historical(fold)/'epoch100_model.pt',map_location='cpu',weights_only=True)['model']
        body=Classifier(128).to('cuda');body.load_state_dict(st);body.eval();hs=[];h2s=[]
        with torch.no_grad():
            for start in range(0,len(x.indptr)-1,256):
                ids=np.arange(start,min(start+256,x.shape[0]));_,h1,h2=batch(body,x,ids,True);hs.append(h1.cpu().numpy());h2s.append(h2.cpu().numpy())
        h1=np.concatenate(hs);h2=np.concatenate(h2s)
        if not np.array_equal(h2,np.load(BACKBONE/f'fold{fold}_hidden.npy')):raise ValueError('Actual h2 changed')
        np.save(OUT/f'fold{fold}_h1.npy',h1)
        del body;gc.collect();torch.cuda.empty_cache()
        rs=torch.load(CURRENT/f'fold{fold}_C/endpoint_readout.pt',map_location='cpu',weights_only=True)['readout']
        model=SecondRepresentation(st,rs).to('cuda');before=tensor_hash(model.state_dict())
        packed=torch.as_tensor(np.concatenate([h1,h2],axis=-1),device='cuda',dtype=torch.float64)
        facts=torch.as_tensor(np.load(BACKBONE/'facts.npy'),device='cuda',dtype=torch.float64)
        frame,c,pure,_,ids=fit_context(d,fold);q=probabilities(model,packed,facts,np.arange(len(c)))
        oldq=np.load(CURRENT/f'fold{fold}_C/sealed_all_prob.npy')
        if not np.array_equal(q,oldq):raise ValueError('Anchored actual zero step changed')
        mass=torch.as_tensor(c,device='cuda',dtype=torch.float64)
        loss,seen=full_gradient(model,packed,facts,mass,ids);full=[p.grad.detach().clone() for p in model.parameters()]
        norm=float(sum(g.square().sum() for g in full).sqrt())
        rows=pd.read_parquet(CURRENT/f'fold{fold}_C/endpoint_original_rows.parquet');bad=rows[rows.pure_TRAIN_input&rows.pred.ne(rows.truth)]
        pure_ids=ids[pure[ids].astype(bool)];classes=c[pure_ids].argmax(1)
        for row in bad.drop_duplicates('local').itertuples():
            i=int(row.local);z=model(packed[i:i+1],facts[i:i+1]);ce=-torch.log_softmax(z,-1)[:,:,row.truth].mean()
            gs=torch.autograd.grad(ce,list(model.parameters()),retain_graph=True)
            direction=float(sum((a*b).sum() for a,b in zip(gs,full)))/norm
            qq=torch.softmax(z,-1).mean(1);competitor=int(q[i].argmax());margin=qq[0,row.truth]-qq[0,competitor]
            gm=torch.autograd.grad(margin,list(model.parameters()))
            margin_slope=-float(sum((a*b).sum() for a,b in zip(gm,full)))/norm
            slopes.append({'fold':fold,'local':i,'truth':int(row.truth),'original_rows':int((bad.local==i).sum()),'full_original_CE_gradient_norm':norm,
                           'local_member_CE_gradient_norm':float(sum(a.square().sum() for a in gs).sqrt()),
                           'CE_descent_slope_along_full_risk':direction,'probability_margin_slope_along_full_risk':margin_slope})
            for name,arr in [('input',x),('h1',h1),('h2',h2)]:
                if name=='input':
                    a=arr[pure_ids];v=arr[i];an=np.sqrt(a.multiply(a).sum(1)).A1;vn=np.sqrt(v.multiply(v).sum());similarity=(a@v.T).toarray().ravel()/(an*vn+1e-30)
                else:
                    a=arr[pure_ids].reshape(len(pure_ids),-1).astype(np.float64);v=arr[i].ravel().astype(np.float64);similarity=(a@v)/(np.linalg.norm(a,axis=1)*np.linalg.norm(v)+1e-30)
                for cl in [1,2]:
                    mask=(classes==cl)&(pure_ids!=i);pool=np.flatnonzero(mask);best=pool[np.argmax(similarity[pool])];j=int(pure_ids[best])
                    neighbors.append({'fold':fold,'query_local':i,'space':name,'neighbor_truth':cl,'neighbor_local':j,'cosine':float(similarity[best]),
                                      'neighbor_original_rows':int(c[j,cl]),'query_original_rows':int(c[i,row.truth])})
        if tensor_hash(model.state_dict())!=before:raise ValueError('Gradient probe mutated parameters')
        reports.append({'fold':fold,'start':'all_matching_V140_C_fixed_endpoints_TRAIN_retention_only_not_candidate_adoption','actual_h2_cache_exact':True,
                        'zero_step_probability_exact':True,'full_original_CE':loss,'original_class_mass':seen,'second_parameter_gradient_norm':norm,
                        'trainable_existing_second_parameters':sum(p.numel() for p in model.parameters()),'pure_TRAIN_errors':len(bad),
                        'parameter_updates':0,'full_role_diagnostic_gradients':1,'frozen':['first','head','facts_direct']})
        print(reports[-1],flush=True);del model,packed,facts,mass,h1,h2,st,rs;gc.collect();torch.cuda.empty_cache()
    pd.DataFrame(slopes).to_parquet(OUT/'TRAIN_residual_gradient_slopes.parquet',index=False)
    pd.DataFrame(neighbors).to_parquet(OUT/'TRAIN_input_feature_neighbors.parquet',index=False)
    save(OUT/'probe.json',{'status':'actual_representation_and_TRAIN_only_diagnostic_gradients','classifier_fits':0,'parameter_updates':0,
                         'full_role_gradient_evaluations':3,'partial_input_derivative_calls':len(slopes)*2,'folds':reports,
                         'positive_CE_descent_inputs':sum(s['CE_descent_slope_along_full_risk']>0 for s in slopes),
                         'positive_actual_probability_margin_descent_inputs':sum(s['probability_margin_slope_along_full_risk']>0 for s in slopes),
                         'residual_inputs':len(slopes),'source_sha256':sha(__file__),
                         'limitations':['Local derivatives do not prove finite-step repairs, separability, convergence or transfer.',
                                        'Nearest-neighbor geometry is descriptive, not an authorized member router or new training target.',
                                        'C remains historical control; choosing all its fixed endpoints for TRAIN retention does not promote it as V140 candidate.']})


if __name__=='__main__':main()
