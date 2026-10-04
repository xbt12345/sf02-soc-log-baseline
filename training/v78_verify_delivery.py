"""Independent metric recomputation and raw-input paired prediction replay."""
import json
import collections
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, OUT, adapter, sha, read, save, NAMES
from v78_boundary import DEST
from v78_denial_adapter_v4 import parse, self_check
from v75_views import view, byte_matrix
from v75_corrective import stable
from v75_metadata import encode


def main():
    target=DEST/'denial_adapter_v4';contract=read(target/'contract.json')
    assert sha(ROOT/'training/v78_denial_adapter_v4.py')==contract['adapter_sha256']
    assert sha(ROOT/'training/v78_denial_replay_v4.py')==contract['source_sha256']
    assert sha(ROOT/'training/v78_boundary.py')==read(DEST/'contract.json')['source_sha256']
    models={n:joblib.load(DEST/(n+'.joblib')) for n in contract['model_sha256']}
    for n,h in contract['model_sha256'].items(): assert sha(DEST/(n+'.joblib'))==h
    p=pd.read_parquet(DEST/'frozen_development/predictions.parquet')
    changes=pd.read_parquet(target/'changed_predictions.parquet').set_index('row_position')
    assert changes.dataset.eq('development').all()
    old=adapter();enc=joblib.load(OUT/'facts_encoder.joblib');wanted=set(changes.index)
    counts=collections.Counter();verified=0;provenance=[]
    with threadpool_limits(limits=4):
        for dataset,filename in [('train','train.parquet'),('development','valid_input.parquet')]:
            offset=0
            for b in pq.ParquetFile(ROOT/'data/official'/filename).iter_batches(batch_size=4096,columns=['event_id','message_sanitized','src_port'],use_threads=False):
                df=b.to_pandas();ix=[];newfacts=[];oldfacts=[]
                for j,s in enumerate(df.message_sanitized):
                    q=parse(s)
                    if q is None:continue
                    prior=old.prepare_record({'message_sanitized':s})
                    counts[(dataset,q['audit']['rule'],q['audit'].get('type','sentence'),prior['route'])]+=1
                    if dataset=='development' and offset+j in wanted:
                        ix.append(j);newfacts.append(q['facts']);oldfacts.append(prior['facts'])
                        provenance.append({'row_position':offset+j,'rule':q['audit']['rule'],'type':q['audit'].get('type','sentence'),
                            'src_port_observed':q['facts']['src_port_fixed']!=65536,'dst_port_observed':q['facts']['dst_port_fixed']!=65536})
                if ix:
                    sub=df.iloc[ix];positions=np.array(ix)+offset
                    np.testing.assert_array_equal(sub.event_id,p.event_id.iloc[positions])
                    tx=byte_matrix([stable(view(s)[0]) for s in sub.message_sanitized])
                    for mode,facts in [('before',oldfacts),('after',newfacts)]:
                        fx=normalize(enc.transform(facts).astype(np.float32),copy=False)
                        extra,_,_=encode(sub.src_port.tolist(),[f.get('src_port_fixed',65536) for f in facts])
                        x=sparse.hstack([tx,fx,extra],format='csr',dtype=np.float32)
                        for n,m in models.items():
                            z=x@m.coef_.T+m.intercept_ if hasattr(m,'coef_') else x@m['coef']+m['intercept']
                            expected=p[n+'_pred'].iloc[positions].to_numpy() if mode=='before' else changes.loc[positions,n+'_pred'].to_numpy()
                            np.testing.assert_array_equal(z.argmax(1),expected)
                    verified+=len(ix)
                offset+=len(b)
    assert verified==6170
    ans=pd.read_parquet(ROOT/'data/official/valid_answer_private.parquet').set_index('event_id').label_binary
    y=ans.reindex(p.event_id).map(dict(zip(NAMES,range(3)))).to_numpy()
    result=read(target/'scoring.json');rows=[];fixed={}
    for n in models:
        before=p[n+'_pred'].to_numpy();after=before.copy();after[changes.index]=changes[n+'_pred'].to_numpy()
        cm=confusion_matrix(y,after,labels=[0,1,2]);np.testing.assert_array_equal(cm,result['totals'][n]['cm'])
        outside=np.ones(len(y),bool);outside[changes.index]=False
        assert np.array_equal(before[outside],after[outside])
        fixed[n]={'fixed':int(((before!=y)&(after==y)).sum()),'broken':int(((before==y)&(after!=y)).sum())}
        for route in ['ALL']+sorted(p.route.unique()):
            mask=np.ones(len(y),bool) if route=='ALL' else p.route.eq(route).to_numpy()
            c=confusion_matrix(y[mask],after[mask],labels=[0,1,2])
            for k in range(3):
                support=int(c[k].sum());predicted=int(c[:,k].sum());tp=int(c[k,k])
                rows.append({'model':n,'route':route,'label':NAMES[k],'support':support,'correct':tp,'errors':support-tp,
                    'recall':tp/support if support else None,'precision':tp/predicted if predicted else None,
                    'f1':2*tp/(support+predicted) if support+predicted else None,
                    **{'pred_'+NAMES[j]:int(c[k,j]) for j in range(3)}})
    pd.DataFrame(rows).to_csv(target/'all_routes_classwise.csv',index=False)
    pd.DataFrame(provenance).to_parquet(target/'field_provenance.parquet',index=False)
    save(target/'independent_verification.json',{'source_sha256':sha(__file__),'all_checks_passed':True,
        'raw_rows_replayed_before_and_after':verified,'models_replayed':len(models),'self_checks':self_check(),
        'all_model_confusion_matrices_independently_recomputed':True,'unaffected_predictions_identical':True,
        'new_fits':0,'quality_acceptance':False,'paired':fixed,
        'input_only_grammar_coverage':[{'dataset':k[0],'rule':k[1],'type':k[2],'old_route':k[3],'rows':v} for k,v in counts.items()],
        'scope':'Implementation and inspected-development evidence only. No negative examples of these newly parsed formats in current train/development; no external transfer proof.'})
    print(json.dumps({'replayed':verified,'paired':fixed,'coverage':dict((str(k),v) for k,v in counts.items())}))


if __name__=='__main__':main()
