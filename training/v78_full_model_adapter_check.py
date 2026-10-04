"""Apply fixed parser recovery to BOTH previously fully trained models.

All unaffected predictions are immutable verified V77 receipts. This is a
post-result transfer diagnostic, not model selection or new blind validation.
"""
import json
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, read, save, sha, adapter, metrics, NAMES
from v78_boundary import DEST
from v78_denial_adapter_v4 import parse
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode


def main():
    target=DEST/'full_model_adapter';assert not target.exists();target.mkdir()
    refs=ROOT/'evidence/2026-09-22/v77_attribution'
    receipt=read(refs/'inference_receipt.json')
    assert sha(refs/'predictions.parquet')==receipt['predictions_sha256']
    paths={'D_full':OUT/'full/FULL.joblib','F_full':OUT/'F_diagnostic/model.joblib'}
    models={n:joblib.load(p) for n,p in paths.items()}
    save(target/'contract.json',{'new_fits':0,'answers_read':False,'model_sha256':{n:sha(p) for n,p in paths.items()},
        'baseline_predictions_sha256':receipt['predictions_sha256'],'source_sha256':sha(__file__),
        'adapter_sha256':sha(ROOT/'training/v78_denial_adapter_v4.py'),
        'scope':'Both earlier all-official-train models, frozen decoder, only already fixed input grammar. Not a new fit or blind test. No selection between models.'})
    p=pd.read_parquet(refs/'predictions.parquet',columns=['event_id','route','D_full_pred','F_full_pred'])
    wanted=set(pd.read_parquet(DEST/'denial_adapter_v4/changed_predictions.parquet').row_position)
    old=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');results=[];offset=0
    with threadpool_limits(limits=4):
        for b in pq.ParquetFile(ROOT/'data/official/valid_input.parquet').iter_batches(batch_size=4096,columns=['event_id','message_sanitized','src_port'],use_threads=False):
            positions=[i for i in range(offset,offset+len(b)) if i in wanted]
            if positions:
                df=b.to_pandas().iloc[np.array(positions)-offset]
                np.testing.assert_array_equal(df.event_id,p.event_id.iloc[positions])
                beforefacts=[old.prepare_record({'message_sanitized':s})['facts'] for s in df.message_sanitized]
                afterfacts=[parse(s)['facts'] for s in df.message_sanitized]
                tx=[view(s)[0] for s in df.message_sanitized];newpred={}
                for kind,facts in [('before',beforefacts),('after',afterfacts)]:
                    fx=normalize(enc.transform(facts).astype(np.float32),copy=False)
                    extra,_,_=encode(df.src_port.tolist(),[f.get('src_port_fixed',65536) for f in facts])
                    for n,m in models.items():
                        texts=tx if n=='D_full' else [stable(s) for s in tx]
                        x=sparse.hstack([byte_matrix(texts),fx,extra],format='csr',dtype=np.float32)
                        pred=(x@m.coef_.T+m.intercept_).argmax(1)
                        if kind=='before':np.testing.assert_array_equal(pred,p[n+'_pred'].iloc[positions])
                        else:newpred[n]=pred
                results.extend({'row_position':int(pos),'event_id':str(df.event_id.iloc[j]),
                    **{n+'_pred':int(q[j]) for n,q in newpred.items()}} for j,pos in enumerate(positions))
            offset+=len(b)
    changed=pd.DataFrame(results);assert len(changed)==6170
    changed.to_parquet(target/'changed_predictions.parquet',index=False)
    save(target/'inference_receipt.json',{'new_fits':0,'answers_read':False,'changed_rows':len(changed),
        'old_predictions_reproduced_on_all_changed_rows':True,'changed_sha256':sha(target/'changed_predictions.parquet')})
    # Only now join historical inspected answers.
    a=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet').set_index('event_id').label_binary
    y=a.reindex(p.event_id).map(dict(zip(NAMES,range(3)))).to_numpy();totals={};cells=[]
    for n in models:
        before=p[n+'_pred'].to_numpy();after=before.copy();after[changed.row_position]=changed[n+'_pred']
        totals[n]={'before':metrics(confusion_matrix(y,before,labels=[0,1,2])),
            'after':metrics(confusion_matrix(y,after,labels=[0,1,2])),
            'fixed':int(((before!=y)&(after==y)).sum()),'broken':int(((before==y)&(after!=y)).sum())}
        for route in ['ALL']+sorted(p.route.unique()):
            mask=np.ones(len(p),bool) if route=='ALL' else p.route.eq(route).to_numpy()
            c=confusion_matrix(y[mask],after[mask],labels=[0,1,2])
            for k in range(3):
                support=int(c[k].sum());pp=int(c[:,k].sum());tp=int(c[k,k])
                cells.append({'model':n,'route':route,'label':NAMES[k],'support':support,'correct':tp,'errors':support-tp,
                    'recall':tp/support if support else None,'precision':tp/pp if pp else None,
                    'f1':2*tp/(support+pp) if support+pp else None,
                    **{'pred_'+NAMES[j]:int(c[k,j]) for j in range(3)}})
    pd.DataFrame(cells).to_csv(target/'classwise.csv',index=False)
    save(target/'scoring.json',{'new_fits':0,'quality_acceptance':False,'totals':totals,
        'scope':'Adaptive inspected-development input repair diagnostic on both prior full-data models. No fresh blind or target-format benign/suspicious acceptance.'})
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
