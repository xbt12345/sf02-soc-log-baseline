"""Strict date-only follow-up; preserves PRI, clock token, event id and body."""
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from v122_clock_gap_audit import DEST, PREV, ROOT, VIEW, canonical, SparseTabM, predict_all, DEVICE, BYTE_FEATURES, byte_matrix, sha, save


def main():
    target = DEST/'date_only_probe.json'
    if target.exists(): raise FileExistsError(target)
    q = pd.read_parquet(DEST/'clock_gap_rows.parquet')
    x = sparse.load_npz(VIEW)
    strings = []
    header_date = re.compile(r'^(<\d{1,3}>)(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+(?:\d{4}|USER-\d+)\s+')
    for raw in q.raw_message:
        m = header_date.match(raw)
        assert m
        raw_variant = m[1]+'Jan 01 2000 '+raw[m.end():]
        assert raw_variant.endswith(raw[m.end():])
        strings.append(canonical(raw_variant))
    variant = sparse.hstack([byte_matrix(strings),x[q.local.to_numpy(),BYTE_FEATURES:]],format='csr')
    metrics=[]
    for arm in ('A','B'):
        preds=np.full(len(q),-1,dtype=int)
        for fold in range(3):
            model=SparseTabM().to(DEVICE)
            model.load_state_dict(torch.load(PREV/f'fold{fold}_{arm}/epoch25_model.pt',map_location='cpu',weights_only=True)['state_dict'])
            model.eval();mask=q.fold.eq(fold).to_numpy()
            preds[mask]=predict_all(model,'TabM',variant[mask],DEVICE).argmax(1)
            del model
        q[f'{arm}_strict_date_only_prediction']=preds
        for label in (1,2):
            mask=q.truth.eq(label).to_numpy();old=q.loc[mask,f'expert_pred_{arm}'].to_numpy();new=preds[mask]
            metrics.append({'arm':arm,'truth':label,'rows':int(mask.sum()),'old_errors':int((old!=label).sum()),
                'new_errors':int((new!=label).sum()),'repaired':int(((old!=label)&(new==label)).sum()),
                'regressed':int(((old==label)&(new!=label)).sum())})
    q[['row_position','A_strict_date_only_prediction','B_strict_date_only_prediction']].to_parquet(DEST/'strict_date_only_predictions.parquet',index=False)
    result={'status':'strict_date_only_fixed_model_probe','new_fits':0,'rows':len(q),'metrics':metrics,
        'preserved':['PRI','clock token','event id','raw body','all facts and metadata'],
        'why_followup':'The first broad header variant also reset PRI to 164. It remains a broad-header diagnostic; only this follow-up isolates date changes.',
        'source_sha256':{str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in [Path(__file__),ROOT/'training/v122_clock_gap_audit.py',DEST/'clock_gap_rows.parquet',VIEW]}}
    save(target,result);print(metrics)


if __name__=='__main__':main()
