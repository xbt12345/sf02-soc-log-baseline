"""Explicit original-row pair recount checks the analytical geometry formula."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import json
import numpy as np
import pandas as pd
import torch
from v143_pair_geometry_probe import ROOT,OUT,RUN,observed_key
from v142_runtime import read,save,sha,require_run_seal
from v142_train import tensors
from v138_train import configure
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash


def main():
    if (OUT/'verification.json').exists():raise FileExistsError('Keep completed verification')
    require_run_seal(ROOT/'training/v142_train.py');configure();_,d=load_data()
    t=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['facts_json'])
    d['pair_key']=t.facts_json.map(json.loads).map(observed_key).to_numpy()
    groups=pd.read_parquet(OUT/'legal_TRAIN_pair_geometry.parquet')
    row=groups[groups.training_role.eq(1)&groups.truth.eq(2)].sort_values('rows').iloc[0]
    frame,_,pure,_,_=fit_context(d,1);g=frame[frame.pair_key.eq(row.key)&pure[frame.local].astype(bool)]
    h,f,m=tensors(1);before=tensor_hash(m.state_dict());checks=[]
    with torch.no_grad():
        for state in ['C','S2']:
            if state=='S2':m.load_state_dict(torch.load(RUN/'fold1_S2/endpoint.pt',map_location='cpu',weights_only=True)['state'])
            if state=='C':z=h[g.local.to_numpy(),:,128:].mean(1).cpu().numpy()
            else:z=m.features(h[g.local.to_numpy()]).mean(1).cpu().numpy()
            z=z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
            q=g.truth.eq(2).to_numpy();a=z[q];b=z[~q];roots=g.loc[q,'root'].to_numpy()
            mask=roots[:,None]!=roots[None,:]
            # Enumerates the actual ordered positive pairs, unlike aggregate sums.
            positive=float((1-a@a.T)[mask].mean())
            negative=None if not len(b) else float((1-a@b.T).mean())
            assert int(mask.sum())==int(row.cross_root_positive_pairs)
            assert abs(positive-row[state+'_mean_cross_root_same_class_cosine_distance'])<1e-12
            if negative is not None:assert abs(negative-row[state+'_mean_opposite_class_cosine_distance'])<1e-12
            checks.append(dict(state=state,positive_rows=len(a),opposite_rows=len(b),explicit_cross_root_pairs=int(mask.sum()),positive_distance=positive,negative_distance=negative))
    assert tensor_hash(m.state_dict())==read(RUN/'fold1_S2/fit.json')['endpoint_parameter_sha256'] if 'endpoint_parameter_sha256' in read(RUN/'fold1_S2/fit.json') else tensor_hash(m.state_dict())==read(RUN/'fold1_S2/progress.json')[-1]['parameter_sha256']
    save(OUT/'verification.json',dict(status='explicit_original_pair_model_replay_passed',fold=1,key=row.key,checks=checks,
        new_fits=0,new_updates=0,new_gradients=0,source_sha256=sha(__file__),geometry_sha256=sha(OUT/'legal_TRAIN_pair_geometry.parquet'),
        limits='Checks one bounded real S group in two registered states; not full geometry or model acceptance.'))
    print(checks)


if __name__=='__main__':main()
