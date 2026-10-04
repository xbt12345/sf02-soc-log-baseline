"""Registered seed confirmation, only after all primary quality gates pass."""
import json

import numpy as np
import pandas as pd
import torch
from scipy import sparse

from experiment_review import ROOT, read, sha, require_run_seal, require_checkpoint, class_counts
from v104_phase_b import SparseTabM, predict_all, DEVICE
from v116_preflight import VIEW
from v121_train import DEST, PLAN, TRACE


def main():
    plan=require_run_seal(DEST/'run_seal.json',ROOT/'training/v121_train.py')
    if plan!=read(PLAN):raise ValueError('Plan changed.')
    if (DEST/'confirmation_evaluation.json').exists():raise FileExistsError('Confirmation already evaluated.')
    primary=read(DEST/'primary_evaluation.json')
    if not primary['confirmation_allowed'] or primary['classifier_fits_new']!=6:
        raise ValueError('Primary quality did not authorize confirmation.')
    trace=pd.read_parquet(TRACE,columns=['row_position','local','fold','truth'])
    if len(trace)!=plan['evaluation_contract']['raw_ASA_rows'] or trace.row_position.duplicated().any():
        raise ValueError('Incomplete ASA reference.')
    x=sparse.load_npz(VIEW)
    outcome=[]; fits=[]
    for seed in plan['confirmation_seeds']:
        predicted={a:np.empty(len(trace),dtype=np.int8) for a in ('A','B')}
        for fold in range(3):
            start=len(fits)
            for arm in ('A','B'):
                folder=DEST/f'seed{seed}_fold{fold}_{arm}'
                receipt=read(folder/'fit.json');require_checkpoint(plan,receipt)
                started=read(folder/'started.json')
                if (receipt['seed'],receipt['fold'],receipt['arm'])!=(seed,fold,arm):
                    raise ValueError('Confirmation fit identity changed.')
                if started['primary_evaluation_sha256']!=sha(DEST/'primary_evaluation.json'):
                    raise ValueError('Confirmation was not bound to primary result.')
                if (receipt['seal_sha256']!=sha(DEST/'run_seal.json')
                        or receipt['model_sha256']!=sha(folder/'epoch25_model.pt')
                        or receipt['prob_sha256']!=sha(folder/'epoch25_prob.npy')
                        or receipt['checkpoint_report_sha256']!=sha(folder/'checkpoints.json')
                        or receipt['progress_sha256']!=sha(folder/'progress.json')):
                    raise ValueError('Confirmation artifact identity changed.')
                state=torch.load(folder/'epoch25_model.pt',map_location='cpu',weights_only=True)
                if (state['seed'],state['fold'],state['arm'],state['epoch'])!=(seed,fold,arm,25):
                    raise ValueError('Model provenance changed.')
                model=SparseTabM().to(DEVICE);model.load_state_dict(state['state_dict'])
                replay=predict_all(model,'TabM',x,DEVICE)
                stored=np.load(folder/'epoch25_prob.npy')
                if not np.array_equal(replay.argmax(1),stored.argmax(1)) or not np.allclose(replay,stored,atol=2e-6,rtol=2e-6):
                    raise ValueError('Confirmation model predictions do not replay.')
                mask=trace.fold.eq(fold).to_numpy()
                predicted[arm][mask]=replay[trace.loc[mask,'local'].to_numpy(dtype=np.int64)].argmax(1)
                fits.append({'seed':seed,'fold':fold,'arm':arm,'fit_sha256':sha(folder/'fit.json'),
                             'initial_state_sha256':receipt['initial_state_sha256'],
                             'optimizer_steps':receipt['optimizer_steps']})
                del model
            if fits[start]['initial_state_sha256']!=fits[start+1]['initial_state_sha256']:
                raise ValueError('Confirmation A/B starts differed.')
        y=trace.truth.to_numpy();byclass={}
        for c in (1,2):
            mask=y==c
            byclass[str(c)]={'support':int(mask.sum()),'A_errors':int(np.sum(mask & (predicted['A']!=c))),
                             'B_errors':int(np.sum(mask & (predicted['B']!=c)))}
        total_A=int(np.sum(predicted['A']!=y));total_B=int(np.sum(predicted['B']!=y))
        outcome.append({'seed':seed,'ASA_A_errors':total_A,'ASA_B_errors':total_B,
                        'M':byclass['1'],'S':byclass['2'],
                        'paired_total_nonregression':total_B<=total_A})
    gates={'all_registered_12_fits':len(fits)==12,
           'every_seed_total_nonregression':all(q['paired_total_nonregression'] for q in outcome),
           'summed_M_nonregression':sum(q['M']['B_errors'] for q in outcome)<=sum(q['M']['A_errors'] for q in outcome),
           'summed_S_nonregression':sum(q['S']['B_errors'] for q in outcome)<=sum(q['S']['A_errors'] for q in outcome)}
    result={'stage':'conditional_seed_confirmation','fits':fits,'seed_outcomes':outcome,'gates':gates,
            'confirmation_passed':all(gates.values()),'model_promoted':False,'quality_acceptance':False,
            'scope':'Source-closed development-fold seed confirmation, not external generalization proof.'}
    (DEST/'confirmation_evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'stage':result['stage'],'gates':gates},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
