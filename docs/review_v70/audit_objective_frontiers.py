"""Outer-answer oracle frontiers for saved v69 scores; diagnostic only, no fit."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'artifacts/v69_support_control_20260921'


def summarize(d):
    d=d.sort_values('score',ascending=False,kind='stable').reset_index(drop=True)
    s=d.score.to_numpy(); y=d.label.eq(2).to_numpy()
    ends=np.r_[-1,np.flatnonzero(np.r_[np.diff(s)!=0,True])]
    total_S=int(y.sum()); total_M=int((~y).sum())
    tp=np.r_[0,np.cumsum(y)][ends+1]; fp=np.r_[0,np.cumsum(~y)][ends+1]
    fn=total_S-tp;tn=total_M-fp
    f1s=np.divide(2*tp,2*tp+fp+fn,out=np.zeros(len(tp),float),where=(2*tp+fp+fn)>0)
    f1m=np.divide(2*tn,2*tn+fp+fn,out=np.zeros(len(tp),float),where=(2*tn+fp+fn)>0)
    sizes=d.groupby(['label','group']).label.transform('size').to_numpy()
    ng=d.groupby('label').group.nunique()
    weight=1/sizes/d.label.map(ng).to_numpy()
    sw=np.r_[0,np.cumsum(weight*y)][ends+1]
    mw=np.r_[0,np.cumsum(weight*(~y))][ends+1]
    cut=np.r_[np.nextafter(s[0],np.inf),s[ends[1:]]]
    def point(i):
        return {'threshold_NOT_FOR_DEPLOYMENT':float(cut[i]),'M_errors':int(fp[i]),'S_correct':int(tp[i]),
                'S_row_recall':float(tp[i]/total_S),'M_row_error':float(fp[i]/total_M),
                'S_source_recall':float(sw[i]),'M_source_error':float(mw[i]),
                'S_precision_original_row_prior':float(tp[i]/max(1,tp[i]+fp[i])),
                'row_macro_F1_M_S':float((f1s[i]+f1m[i])/2),'total_errors':int(fp[i]+fn[i])}
    # Highest threshold breaks metric ties; all equal score blocks kept intact.
    ans={'M_rows':total_M,'S_rows':total_S,'M_source_symbols':int(ng[1]),'S_source_symbols':int(ng[2]),
         'oracle_best_row_macro_F1':point(int(np.argmax((f1s+f1m)/2))),
         'oracle_best_row_accuracy':point(int(np.argmin(fp+fn))),
         'oracle_best_source_balanced_accuracy':point(int(np.argmax(sw-mw))),
         'budget_frontier':{}}
    for budget in [.001,.005,.01,.02,.05,.1,.2]:
        eligible=np.flatnonzero((fp/total_M<=budget+1e-12)&(mw<=budget+1e-12))
        i=int(eligible[-1]);ans['budget_frontier'][str(budget)]=point(i)
    return ans


def main():
    out={'new_fits':0,'scope':'Oracle cutoffs read outer evaluation answers; fixed-score descriptive ceilings only, no model selection or deployment.',
         'original_row_prior_note':'Precision and row macro F1 retain actual M/S counts; no class balancing.',
         'cells':[],'input_sha256':{}}
    for arm in ['A_original_S','B_small_sources','C_all_sources']:
        for fold in range(3):
            p=RUN/f'fold{fold}'/arm/'evaluation.parquet';d=pd.read_parquet(p)
            out['input_sha256'][str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
            for protocol in ['tcp','udp']:
                out['cells'].append({'arm':arm,'fold':fold,'protocol':protocol,
                                     **summarize(d[d.transport_protocol.eq(protocol)])})
    out['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dest=Path(__file__).with_suffix('.json');dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    for c in out['cells']:
        if c['arm']=='B_small_sources':
            print(c['fold'],c['protocol'],'best F1',c['oracle_best_row_macro_F1'],
                  'budget1',c['budget_frontier']['0.01']['S_source_recall'],
                  'budget5',c['budget_frontier']['0.05']['S_source_recall'],flush=True)


if __name__=='__main__':main()
