"""Population frozen controls separate placeholder remnants from whitespace effects."""
import json
import re
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from v89_common import ROOT, OUT, data, read, save, sha
from v75_views import byte_matrix, BYTE_FEATURES
from v75_corrective import stable
from v92_train import Branch, csr
from v98_frozen_audit import DEST, V97, V92


def main():
    assert not (DEST/'population_surface_controls.json').exists()
    r,y,fid,_,_,_,_=data();isa=r.route.eq('asa').to_numpy()
    roles=pd.read_parquet(ROOT/'artifacts/v95_cross_component_training_20260928/full_format_roles.parquet',columns=['role']).role.to_numpy().astype(object)
    for name,fold in [('inner',1),('C',2),('H',0)]:roles[r.fold.eq(fold).to_numpy()]=name
    ids=np.load(V92/'ASA_input_ids.npy');x=sparse.load_npz(V92/'ASA_R0.npz')
    rr=r[isa].copy();rr['R0']=fid[isa]
    representatives=rr.drop_duplicates('R0').set_index('R0').loc[ids]
    texts=pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    original=[stable(texts.loc[t]) for t in representatives.new_text_id]
    pat=re.compile(r'((?:CRED|HOST|USER|ORG)-)(\s*)(?=<IDENTITY>)')
    used=np.array([i for i,s in enumerate(original) if pat.search(s)],int)
    base=x[used];bt=byte_matrix([original[i] for i in used]);diff=bt-base[:,:BYTE_FEATURES]
    assert not diff.nnz or abs(diff.data).max()<1e-7
    lin=joblib.load(V97/'teacher.joblib')
    net=Branch(x.shape[1]);net.load_state_dict(torch.load(V97/'MAG_model.pt',map_location='cpu',weights_only=True)['state_dict']);net.eval()
    def score(xx):
        t=np.asarray(xx@lin['coef'])+lin['intercept']
        with torch.no_grad():return t+net(csr(xx,'cpu')).double().numpy()
    old=score(base);assert np.array_equal(old.argmax(1),np.load(V97/'MAG_prediction.npy')[ids[used]])
    modes={'literal_only':lambda m:m.group(2),'whitespace_only':lambda m:m.group(1),'literal_and_whitespace':lambda m:''}
    records={};traces=[];all_pred=np.load(V97/'MAG_prediction.npy')
    for name,replacement in modes.items():
        text=[pat.sub(replacement,original[i]) for i in used]
        xx=sparse.hstack([byte_matrix(text),base[:,BYTE_FEATURES:]],format='csr');z=score(xx)
        pred=all_pred.copy();pred[ids[used]]=z.argmax(1)
        population=[]
        for role in ['A','B','V','inner','C','H']:
            for cls in [1,2]:
                mask=isa&(roles==role)&(y==cls);p=pred[fid[mask]];op=all_pred[fid[mask]]
                population.append({'role':role,'class':cls,'rows':int(mask.sum()),
                    'modified_view_rows':int((mask&np.isin(fid,ids[used])).sum()),
                    'original_correct':int((op==cls).sum()),'control_correct':int((p==cls).sum()),
                    'repairs':int(((op!=cls)&(p==cls)).sum()),'regressions':int(((op==cls)&(p!=cls)).sum())})
        records[name]={'distinct_inputs':len(used),'prediction_flips_unique_inputs':int((z.argmax(1)!=old.argmax(1)).sum()),'populations':population}
        np.save(DEST/f'surface_{name}_ASA_prediction.npy',pred[ids])
        for row in [160415,459736]:
            j=int(np.flatnonzero(ids[used]==fid[row])[0])
            traces.append({'mode':name,'row_position':row,'original_MS_margin':float(old[j,2]-old[j,1]),
                'control_MS_margin':float(z[j,2]-z[j,1]),'original_pred':int(old[j].argmax()),'control_pred':int(z[j].argmax()),
                'derived_view':text[j]})
    result={'status':'frozen_population_controls_only','source_sha256':sha(__file__),
        'ASA_rows_in_population':int(isa.sum()),'ASA_unique_inputs':len(ids),'modified_unique_inputs':len(used),
        'modified_rows':int((isa&np.isin(fid,ids[used])).sum()),'facts_metadata_changed':0,
        'controls':records,'paired_examples':traces,'classifier_fits':0,'calibration_fits':0,
        'scope':'Frozen surface sensitivity, not a trained parser fix. Whole-population counts are posthoc development diagnostics; no selected transformation or gain claim.'}
    save(DEST/'population_surface_controls.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['controls','paired_examples']},ensure_ascii=False))
    for name,rec in records.items():print(json.dumps({'mode':name,**rec},ensure_ascii=False))


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
