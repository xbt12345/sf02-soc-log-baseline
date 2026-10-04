"""Separate decoder/ranking diagnostics, never fit a target-derived threshold."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from run_v75 import ROOT, NAMES, read, save, sha

DEST=ROOT/'evidence/2026-09-22/v77_attribution'


def main():
    receipt=read(DEST/'inference_receipt.json')
    assert sha(DEST/'predictions.parquet')==receipt['predictions_sha256']
    p=pd.read_parquet(DEST/'predictions.parquet')
    ap=ROOT/'data/official/valid_answer_private.parquet'
    a=pd.read_parquet(ap)
    d=p.merge(a,on='event_id',validate='one_to_one',how='left',sort=False)
    assert len(d)==2014052 and d.label_binary.isin(NAMES).all()
    records=[];quantiles=[];cms=[]
    for route,z in d.groupby('route'):
        y=z.label_binary.map(dict(zip(NAMES,range(3)))).to_numpy()
        for model in ['D_inner','D_full','F_inner','F_full']:
            q=z[[model+'_p'+str(i) for i in range(3)]].to_numpy()
            pred=z[model+'_pred'].to_numpy()
            for yi in range(3):
                take=y==yi
                cms.append({'route':route,'model':model,'label':NAMES[yi],'rows':int(take.sum()),
                    **{'pred_'+NAMES[k]:int((pred[take]==k).sum()) for k in range(3)}})
                if take.any():
                    quantiles.append({'route':route,'model':model,'label':NAMES[yi],
                        'pM_quantiles':np.quantile(q[take,1],[0,.1,.5,.9,1]).tolist()})
            for task in ['M_vs_all','M_vs_B','M_vs_S']:
                other={'M_vs_B':0,'M_vs_S':2}.get(task)
                take=np.ones(len(y),bool) if other is None else ((y==1)|(y==other))
                truth=(y[take]==1)
                score=q[take,1] if other is None else q[take,1]/np.maximum(q[take,1]+q[take,other],1e-300)
                both=len(np.unique(truth))==2
                records.append({'route':route,'model':model,'task':task,'positive_rows':int(truth.sum()),
                    'negative_rows':int((~truth).sum()),'auroc':float(roc_auc_score(truth,score)) if both else None,
                    'average_precision':float(average_precision_score(truth,score)) if both else None,
                    'positive_prevalence':float(truth.mean()) if len(truth) else None})
    pd.DataFrame(records).to_csv(DEST/'pairwise_ranking.csv',index=False)
    pd.DataFrame(cms).to_csv(DEST/'route_confusions.csv',index=False)
    save(DEST/'score_quantiles.json',quantiles)
    save(DEST/'score_diagnostic_receipt.json',{'new_fits':0,'thresholds_fitted':0,'source_sha256':sha(__file__),
        'predictions_sha256':sha(DEST/'predictions.parquet'),'answers_sha256':sha(ap),
        'outputs_sha256':{name:sha(DEST/name) for name in ['pairwise_ranking.csv','route_confusions.csv','score_quantiles.json']},
        'scope':'Inspected-development descriptive ranking only. Pairwise tasks condition on true labels and are not deployed classifiers. AUROC/AP are not three-class accuracy, low-FPR guarantee or a calibrated operating point.'})
    print(pd.DataFrame(records).query("route in ['asa','unsupported','vpc_v2']").to_string(index=False))


if __name__=='__main__': main()
