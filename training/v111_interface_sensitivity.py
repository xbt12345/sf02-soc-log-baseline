"""Frozen-model text intervention; interface aliases are NOT assumed label-preserving."""
import json,re
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,save,sha
from v75_views import byte_matrix,BYTE_FEATURES
from v104_phase_b import SparseTabM,csr_tensor,DEVICE
from v110_layer_probes import LEDGER,N2,PREV,DEST as V110,check_registration

DEST=ROOT/'artifacts/v111_root_review_20260929'

def main():
    check_registration()
    dest=DEST/'interface_sensitivity.json';assert not dest.exists()
    d=pd.read_parquet(LEDGER)
    key=d.loc[d.root==2868,'behavior'].iloc[0]
    d=d[d.behavior==key].copy().reset_index(drop=True)
    ids=np.sort(d.local.unique())
    t=pd.read_parquet(PREV/'N2_text_pairs.parquet').set_index('local').loc[ids,'after']
    x=sparse.load_npz(N2);facts=x[ids,BYTE_FEATURES:]
    originals=x[ids]
    assert (sparse.hstack([byte_matrix(t.tolist()),facts],format='csr')!=originals).nnz==0
    variants={'original':t.tolist(),
      'dst_interface_lowercase':[re.sub(r'(dst )(DMZ(?:-\d+)?)(:)',lambda m:m[1]+m[2].lower()+m[3],s) for s in t],
      'dst_interface_to_DMZ':[re.sub(r'(dst )(?:DMZ|dmz)(?:-\d+)?(:)',r'\1DMZ\2',s) for s in t],
      'dst_interface_to_dmz_1':[re.sub(r'(dst )(?:DMZ|dmz)(?:-\d+)?(:)',r'\1dmz-1\2',s) for s in t]}
    folder=PREV/'fold1_N2_TabM25_seed10201'
    model=SparseTabM().to(DEVICE)
    model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict']);model.eval()
    p=np.load(V110/'fold1_P2/probe.npz');captured={}
    hook=model.second.register_forward_hook(lambda _m,_a,v:captured.update(h=torch.relu(v)))
    result={'status':'no_fit_hypothesis_generating_text_intervention','classifier_fits':0,'calibration_fits':0,
      'source_sha256':sha(__file__),'fold':1,'behavior':key,'rows':len(d),'unique_inputs':len(ids),
      'models_frozen':True,'parsed_fact_and_record_port_block_unchanged':True,'variants':{},
      'interpretation_limit':'Changing an interface name or case is not proven label-preserving: DMZ and dmz-1 may denote different network subzones. These are feature sensitivity probes, not accuracy, robustness improvement or valid augmented training examples.',
      'input_hashes':{p0.relative_to(ROOT).as_posix():sha(p0) for p0 in [LEDGER,N2,PREV/'N2_text_pairs.parquet',folder/'model.pt',V110/'fold1_P2/probe.npz']}}
    out=[]
    with torch.inference_mode(),threadpool_limits(limits=4):
        for name,ss in variants.items():
            xx=sparse.hstack([byte_matrix(ss),facts],format='csr')
            f=torch.as_tensor(facts.toarray(),device=DEVICE,dtype=torch.float32)
            z=model(csr_tensor(xx,DEVICE),f)
            prob=torch.softmax(z,-1).mean(1).cpu().numpy()
            h=np.c_[captured['h'].flatten(1).cpu().numpy(),f.cpu().numpy()].astype('f8')
            margin=((h-p['mean'])/p['std'])@p['coef']+float(p['intercept'])
            loc=np.searchsorted(ids,d.local.to_numpy())
            if name=='original':
                ref=np.load(folder/'ASA_input_prob.npy')[ids]
                assert np.allclose(prob,ref,atol=2e-6,rtol=2e-6)
                assert np.array_equal(prob.argmax(1),ref.argmax(1))
                refhead=np.load(V110/'fold1_P2/ASA_input_S_probability.npy')[ids]
                assert np.array_equal(margin>=0,refhead>=.5)
                original_pred=prob.argmax(1);original_head=margin>=0
            frame=d[['row_position','root','local','fold','truth']].copy()
            frame['variant']=name;frame['N2_prediction']=prob.argmax(1)[loc]
            frame['P2_prediction']=np.where(margin[loc]>=0,2,1);frame['P2_margin']=margin[loc]
            out.append(frame)
            result['variants'][name]=[
                {'root':int(root),'role_for_fold1':'heldout' if int(g.fold.iloc[0])==1 else 'train',
                 'rows':len(g),'changed_text_rows':int(sum(variants['original'][j]!=ss[j] for j in np.searchsorted(ids,g.local))),
                 'N2_M_predictions':int((g.N2_prediction==1).sum()),'P2_M_predictions':int((g.P2_prediction==1).sum()),
                 'N2_label_flips_from_original':int(np.sum(g.N2_prediction.to_numpy()!=original_pred[np.searchsorted(ids,g.local)])),
                 'P2_label_flips_from_original':int(np.sum((g.P2_prediction.to_numpy()==2)!=original_head[np.searchsorted(ids,g.local)])),
                 'P2_mean_S_minus_M_margin':float(g.P2_margin.mean())}
                for root,g in frame.groupby('root')]
    hook.remove()
    pd.concat(out).to_parquet(DEST/'interface_intervention_predictions.parquet',index=False)
    save(dest,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
