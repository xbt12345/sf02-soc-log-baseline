"""Post-training encoder and calibration audit. All ablations are diagnostic only."""
import gc, hashlib, json, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
from scipy import sparse
ROOT=Path(__file__).resolve().parents[3]; OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'artifacts/v37_cloud_20260913T072309Z/runtime'))
from run_v37_train import columns
from run_v36_train import string_codes
import v37_learning as learning

def differences(a,b,path=''):
    if isinstance(a,dict) and isinstance(b,dict):
        return sum((differences(a[k],b[k],path+'/'+k) for k in a),[])
    if isinstance(a,list) and isinstance(b,list):
        return sum((differences(x,y,path+'/'+str(i)) for i,(x,y) in enumerate(zip(a,b))),[])
    if a!=b:return [{'path':path,'saved':a,'local':b,'absolute_difference':abs(a-b) if isinstance(a,(float,int)) and isinstance(b,(float,int)) else None}]
    return []

def main():
    work=ROOT/'artifacts/v37_cloud_20260913T072309Z/work'; prep=ROOT/'artifacts/v37_prepared_r13_20260913'
    meta=pq.read_table(prep/'prepared.parquet',columns=['b1','facts','label_index','product','route']).to_pandas()
    proto=pq.read_table(prep/'protocol.parquet').to_pandas(); groups=pq.read_table(prep/'groups.parquet')['union_group'].to_numpy()
    y=meta.label_index.to_numpy(); positions=np.arange(len(meta)); results=[]; drift=[]
    # All models: distinguish exact stored-score recomputation from float-sensitive re-inference.
    cached_drift=OUT/'numerical_replay_diagnosis.json'
    if cached_drift.exists():drift=json.loads(cached_drift.read_text(encoding='utf-8'))
    for f in ([] if drift else sorted((work/'models').iterdir())):
        r=json.loads((f/'report.json').read_text()); b=joblib.load(f/'model.joblib')
        c=pq.read_table(f/'calibration.parquet').to_pandas(); cp=c.row_position.to_numpy()
        oldp=c[['p_benign','p_malicious','p_suspicious']].to_numpy(); cy=c.label_index.to_numpy()
        pol=[learning.calibration_policies(cy,oldp,meta['product'].to_numpy()[cp],groups[cp],a) for a in [.0001,.001,.01]]
        ds=differences(b['policies'],pol)
        tc,fc,_=columns(b['view']); texts,tids=string_codes(prep/'prepared.parquet',cp,tc);facts,fids=string_codes(prep/'prepared.parquet',cp,fc)
        pairs,back=np.unique(tids.astype(np.int64)*len(facts)+fids,return_inverse=True)
        xx=sparse.hstack([b['tfidf'].transform([texts[i] for i in pairs//len(facts)]),b['facts'].transform([facts[i] for i in pairs%len(facts)])],format='csr')
        npred=b['model'].predict_proba(xx)[back]
        newpol=[learning.calibration_policies(cy,npred,meta['product'].to_numpy()[cp],groups[cp],a) for a in [.0001,.001,.01]]
        ds2=differences(b['policies'],newpol)
        drift.append({'model':f.name,'saved_scores_policy_differences':ds,'reinferred_policy_differences':ds2,
            'max_probability_difference':float(abs(npred-oldp).max()),
            'frozen_threshold_calibration_decision_changes':[int(((1-oldp[:,0]>=po[n])!=(1-npred[:,0]>=po[n])).sum()) for po in b['policies'] for n in ['empirical_pooled','empirical_worst_source_global']]})
        del xx,texts,facts;gc.collect()
    (OUT/'numerical_replay_diagnosis.json').write_text(json.dumps(drift,ensure_ascii=False,indent=2),encoding='utf-8')
    codes,keys=pd.factorize(pd.MultiIndex.from_frame(meta[['b1','facts']]),sort=False)
    texts=[str(k[0]) for k in keys];facts=[str(k[1]) for k in keys]
    for task in ['known_dev','source_ad','source_duo','source_waf','asa_hard']:
        folder=work/'models'/(task+'_B2_REPAIRED'); b=joblib.load(folder/'model.joblib')
        x=sparse.hstack([b['tfidf'].transform(texts),b['facts'].transform(facts)],format='csr');x.eliminate_zeros();x.sort_indices()
        hs=[]
        for i in range(x.shape[0]):
            lo,hi=x.indptr[i:i+2];hs.append(hashlib.sha256(x.indices[lo:hi].astype('<i8').tobytes()+x.data[lo:hi].astype('<f8').tobytes()).hexdigest())
        enc,enc_keys=pd.factorize(np.asarray(hs,dtype=object),sort=False);row_enc=enc[codes];roles=proto[task].to_numpy();fit=(roles==0)|(roles==1);ev=roles==3
        counts=np.bincount(row_enc[fit]*3+y[fit],minlength=len(enc_keys)*3).reshape(-1,3)
        ec=np.bincount(row_enc[ev]*3+y[ev],minlength=len(enc_keys)*3).reshape(-1,3)
        seen=counts[row_enc[ev]].sum(1)>0
        ep=np.flatnonzero(ev);evcodes=codes[ev];zero=(np.diff(x.indptr)==0)[evcodes]
        support=[]
        for (route,label),d in meta.loc[ev].assign(seen=seen,zero=zero).groupby(['route','label_index'],observed=True):
            support.append({'route':route,'label':int(label),'rows':len(d),'encoded_seen_in_final_fit':int(d.seen.sum()),'zero_vector_rows':int(d.zero.sum())})
        example=[]
        probs=b['model'].predict_proba(x);names=np.concatenate([b['tfidf'].names(),b['facts'].names()]);nt=len(b['tfidf'].names())
        threshold=next(p for p in b['policies'] if p['alpha']==.001)['empirical_worst_source_global']
        pred=learning.gated_predictions(probs,threshold)
        masks=[(ev&(y!=pred[codes]))]
        # Inspect a few unique real errors per route/class, and all bounded native JSON payloads.
        picked=set()
        for _,d in meta.loc[masks[0]].assign(pos=ep[y[ep]!=pred[codes[ep]]]).groupby(['route','label_index'],observed=True):
            picked.update(d.drop_duplicates(['b1','facts']).head(2).pos.tolist())
        if task=='known_dev':picked.update(np.flatnonzero(ev&(meta.route.to_numpy()=='bounded_payload'))[:1].tolist())
        for pos in sorted(picked):
            idx=codes[pos];row=x.getrow(idx);a=row.indices;v=row.data
            contributions={}
            for c in [1,2]:
                q=v*(b['model'].coef_[c,a]-b['model'].coef_[0,a]);order=np.argsort(-abs(q))[:10]
                contributions[str(c)]={'intercept_vs_benign':float(b['model'].intercept_[c]-b['model'].intercept_[0]),'top':[{'feature':str(names[a[j]]),'logit_contribution_vs_benign':float(q[j])} for j in order]}
            parsed=json.loads(facts[idx]);oov_facts=sorted(set(learning.fact_dictionary(parsed))-set(k.split('=')[0] for k in b['facts'].names()))
            text_words=b['tfidf'].counter.build_analyzer()(texts[idx]);known_tokens=[w for w in text_words if w in b['tfidf'].counter.vocabulary_]
            ablations={}
            for name,mask in [('text_only',row.indices<nt),('facts_only',row.indices>=nt),('without_category',np.array([str(names[j])!='category:category=authentication' for j in row.indices],dtype=bool))]:
                z=row.copy();z.data[~mask]=0;pp=b['model'].predict_proba(z)[0];ablations[name]={'p':pp.tolist(),'primary':int(learning.gated_predictions(pp[None,:],threshold)[0])}
            example.append({'row_position':int(pos),'route':str(meta.route.iloc[pos]),'label':int(y[pos]),'b1':texts[idx],'facts':parsed,
                'probabilities':probs[idx].tolist(),'threshold':threshold,'known_text_features':sorted(set(known_tokens)),'fact_feature_keys_unseen':oov_facts,
                'contributions':contributions,'ablations_not_new_models':ablations})
        results.append({'task':task,'unique_preencoder_views':len(keys),'unique_encoded_views':len(enc_keys),
            'evaluation_encoded_mixed_groups':int(((ec>0).sum(1)>1).sum()),'evaluation_encoded_minimum_empirical_errors':int((ec.sum(1)-ec.max(1)).sum()),
            'evaluation_rows_with_fit_encoded_match':int(seen.sum()),'by_route':support,'examples':example})
        print(task,'done',flush=True);del x,b;gc.collect()
    (OUT/'encoder_analysis.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'numerical_replay_diagnosis.json').write_text(json.dumps(drift,ensure_ascii=False,indent=2),encoding='utf-8')

if __name__=='__main__':main()
