"""Matched frozen N1 control for the observed interface-name sensitivity."""
import re,json
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_v75 import ROOT,save,sha
from v75_views import byte_matrix,BYTE_FEATURES
from v104_phase_b import SparseTabM,csr_tensor,DEVICE
from v110_layer_probes import LEDGER,PREV,check_registration

def main():
    check_registration()
    dest=ROOT/'artifacts/v111_root_review_20260929/N1_interface_control.json';assert not dest.exists()
    d=pd.read_parquet(LEDGER); key=d.loc[d.root==2868,'behavior'].iloc[0]
    d=d[d.behavior==key].copy();ids=np.sort(d.local.unique())
    pairs=PREV/'N2_text_pairs.parquet'
    t=pd.read_parquet(pairs).set_index('local').loc[ids,'before']
    xp=ROOT/'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
    x=sparse.load_npz(xp)[ids];facts=x[:,BYTE_FEATURES:]
    assert (sparse.hstack([byte_matrix(t.tolist()),facts],format='csr')!=x).nnz==0
    variants={'original':t.tolist(),
      'dst_interface_lowercase':[re.sub(r'(dst )(DMZ(?:-\d+)?)(:)',lambda m:m[1]+m[2].lower()+m[3],s) for s in t],
      'dst_interface_to_DMZ':[re.sub(r'(dst )(?:DMZ|dmz)(?:-\d+)?(:)',r'\1DMZ\2',s) for s in t],
      'dst_interface_to_dmz_1':[re.sub(r'(dst )(?:DMZ|dmz)(?:-\d+)?(:)',r'\1dmz-1\2',s) for s in t]}
    folder=PREV/'fold1_N1_TabM25_seed10201'
    model=SparseTabM().to(DEVICE);model.load_state_dict(torch.load(folder/'model.pt',map_location='cpu',weights_only=True)['state_dict']);model.eval()
    r={'status':'no_fit_N1_interface_control','classifier_fits':0,'calibration_fits':0,'fold':1,
       'source_sha256':sha(__file__),'variants':{},'input_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in [LEDGER,xp,pairs,folder/'model.pt',folder/'ASA_input_prob.npy']},
       'scope':'Frozen N1 input sensitivity only. Interface aliases are not established label-preserving transformations; no claimed corrected events or robust accuracy.'}
    with torch.inference_mode(),threadpool_limits(limits=4):
        for name,ss in variants.items():
            xx=sparse.hstack([byte_matrix(ss),facts],format='csr')
            f=torch.as_tensor(facts.toarray(),device=DEVICE,dtype=torch.float32)
            p=torch.softmax(model(csr_tensor(xx,DEVICE),f),-1).mean(1).cpu().numpy()
            if name=='original':
                old=np.load(folder/'ASA_input_prob.npy')[ids]
                assert np.allclose(p,old,atol=2e-6,rtol=2e-6)
                assert np.array_equal(p.argmax(1),old.argmax(1))
                ref=p.argmax(1)
            r['variants'][name]=[]
            for root,g in d.groupby('root'):
                ii=np.searchsorted(ids,g.local)
                r['variants'][name].append({'root':int(root),'rows':len(g),
                  'M_predictions':int((p[ii].argmax(1)==1).sum()),
                  'label_flips':int((p[ii].argmax(1)!=ref[ii]).sum())})
    save(dest,r);print(json.dumps(r,ensure_ascii=False))

if __name__=='__main__':main()
