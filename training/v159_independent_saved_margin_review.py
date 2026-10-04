"""Describe margins of persisted rows; no head, feature or gradient evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
OUT=ROOT/'artifacts/v159_independent_saved_margin_review_20261002'
def main():
    assert not OUT.exists();reports=[];bindings={}
    for fold in range(3):
        for arm in ['A','B']:
            folder=RUN/f'fold{fold}_{arm}';fit=json.loads((folder/'fit.json').read_text())
            for scope in ['OOF','deployment']:
                for state in ['accepted0','endpoint']:
                    file=folder/f'{state}_{scope}_rows.parquet';frame=pd.read_parquet(file)
                    lp=frame[['logp0','logp1','logp2']].to_numpy();truth=frame.truth.to_numpy(dtype=int)
                    own=lp[np.arange(len(lp)),truth];other=lp.copy();other[np.arange(len(lp)),truth]=-np.inf
                    margin=own-other.max(1);correct=frame.pred.to_numpy()==truth
                    protected=frame.initial_correct.to_numpy(dtype=bool)&correct
                    part=margin[protected]
                    reports.append(dict(fold=fold,arm=arm,scope=scope,state=state,accepted_updates=fit['accepted_updates'],rows=len(frame),
                      correct=int(correct.sum()),initial_correct_now_correct=int(protected.sum()),
                      correct_initial_margin_min=float(part.min()),correct_initial_margin_quantiles=np.quantile(part,[0,.001,.01,.1,.5,1]).tolist(),
                      protected_at_or_below_margin={str(t):int((part<=t).sum()) for t in [0,1e-14,1e-12,1e-10,1e-8,1e-6,.001,.01]},
                      wrong_margin_median=float(np.median(margin[~correct])) if (~correct).any() else None))
                    bindings[file.relative_to(ROOT).as_posix()]=hashlib.sha256(file.read_bytes()).hexdigest()
    report=dict(status='saved_log_probability_margins_described_no_causal_guard_claim',reports=reports,
      official_heads=0,official_features=0,official_gradients=0,official_fits=0,
      scope='Current correct margins identify possible active constraints; rejected proposal row predictions were not saved, so margin proximity does not identify actual blockers.',source_sha256=bindings)
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([r for r in reports if r['state']=='endpoint' and r['scope']=='deployment']))
if __name__=='__main__':main()
