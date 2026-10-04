"""Bounded official-data development: three input views, one joint classifier.

Final predictions always use the same three-class argmax used for selection.
No alarm gate, source router, resampling, duplicate discount or target tuning.
"""
import argparse,ast,datetime,gc,hashlib,json,platform,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import v38_representation as rep
import v38_learning as learn
from run_v38_prepare import sha,save

CONTRACT={'version':'v38-local-classifier-1.0','views':list(rep.VIEWS),'C_grid':[0.1,1.0],
 'C_choice':'selection joint three-class macro-F1; smaller C unless improvement > 0.001; a selected diagnostic model is not quality promotion',
 'fit_role':0,'selection_role':1,'final_fit_roles':[0,1],'evaluation_role':2,
 'decision':'three_class_argmax','alarm_gate':False,'sample_weight':'original rows equal; aggregate exact multiplicities',
 'max_features':100000,'probability_replay_atol':1e-10,'probability_replay_rtol':1e-10,
 'regression_review':{'minimum_groups_for_supported_class':2,'class_recall_drop':0.02,'normal_error_rate_increase':0.001,
                      'whole_source_all_normal_or_all_alert_with_missed_class':'always flag'},
 'inner_source_holdout':['Duo','Barracuda WAF'],'outer_known_evaluation_accessed_for_fitting':False,
 'external_or_blind_validation':False,'availability_negative_control_C':0.1,
 'stage_D':'separate invocation only after error-mechanism review; no automatic target adaptation'}

def metrics(y,p,g,source):
    pred=p.argmax(1);r=learn.cm_metrics(y,pred)
    r['normal_errors']=int(((y==0)&(pred!=0)).sum())
    r['class_groups']=[int(len(np.unique(g[y==c]))) for c in range(3)]
    r['sources']={}
    for name in sorted(set(source)):
        m=source==name;v=learn.cm_metrics(y[m],pred[m]);v['class_groups']=[int(len(np.unique(g[m&(y==c)]))) for c in range(3)]
        v['normal_errors']=int((m&(y==0)&(pred!=0)).sum());r['sources'][str(name)]=v
    return r

def actual_keys(x):
    x=x.copy().tocsr();x.eliminate_zeros();x.sort_indices();keys=[]
    for i in range(x.shape[0]):
        a,b=x.indptr[i:i+2]
        keys.append(hashlib.sha256(x.indices[a:b].astype('<i8').tobytes()+x.data[a:b].astype('<f8').tobytes()).digest())
    return np.asarray(pd.factorize(np.asarray(keys,dtype=object),sort=False)[0],dtype=np.int32)

def collision_slice(labels,ids,p):
    _,ids=np.unique(ids,return_inverse=True)
    counts=np.bincount(ids*3+labels,minlength=(ids.max()+1)*3).reshape(-1,3)
    conflict=(counts>0).sum(1)>1
    result={'unique_vectors':len(counts),'mixed_vectors':int(conflict.sum()),'mixed_rows':int(conflict[ids].sum()),
            'minimum_total_errors':int((counts.sum(1)-counts.max(1)).sum())}
    for name,m in [('conflict',conflict[ids]),('nonconflict',~conflict[ids])]:
        result[name]=learn.cm_metrics(labels[m],p[m].argmax(1)) if m.any() else None
    return result

def text_encoder(texts,ids,fit):
    codes,unique=pd.factorize(np.asarray(texts,dtype=object),sort=False)
    freq=np.bincount(codes[ids[fit]],minlength=len(unique));take=freq>0
    return learn.FrequencyTfidf().fit(list(unique[take]),freq[take])

def negative_control(facts):
    names=sorted(set(learn.ENUMS)|set(learn.NUMERIC)|set(learn.BIT_FIELDS))
    x=np.zeros((len(facts),len(names)),dtype=float)
    for i,f in enumerate(facts):
        for j,k in enumerate(names):
            if k in f and (k not in learn.BIT_FIELDS or f[k]!=learn.BIT_FIELDS[k][1]):x[i,j]=1
    return sparse.csr_matrix(x),names

def source_closure():
    root=Path(__file__).parent;pending=['run_v38_train'];files={}
    while pending:
        name=pending.pop();p=root/(name+'.py')
        if p.name in files:continue
        files[p.name]=sha(p)
        for n in ast.walk(ast.parse(p.read_text(encoding='utf-8'))):
            names=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else ([n.module.split('.')[0]] if isinstance(n,ast.ImportFrom) and n.module else [])
            pending.extend(v for v in names if (root/(v+'.py')).exists() and v+'.py' not in files)
    return files

def trial(a):
    start=time.perf_counter();prepared=Path(a.prepared);out=Path(a.output)
    if out.exists():raise FileExistsError('New training output required')
    receipt=json.loads((prepared/'complete.json').read_text(encoding='utf-8'))
    for n,h in receipt['files'].items():assert sha(prepared/n)==h,n
    for n,h in receipt['source_hashes'].items():assert sha(Path(__file__).parent/n)==h,n
    out.mkdir(parents=True)
    contract=dict(CONTRACT)
    if a.weight=='sqrt_inverse':contract['sample_weight']='fit-side inverse-square-root class frequency, mean row weight 1; stage D development'
    contract['invoked_views']=a.views.split(',');contract['stage_D_rationale']=a.rationale
    if a.weight!='equal' and not a.rationale:raise ValueError('Stage D needs an explicit diagnosis')
    save(out/'frozen_contract.json',contract)
    save(out/'binding.json',{'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'python':platform.python_version(),'packages':{n:__import__(n).__version__ for n in ['numpy','scipy','sklearn','pyarrow']},
        'prepared_receipt_sha256':sha(prepared/'complete.json'),'sources':source_closure()})
    rows=pq.read_table(prepared/'rows.parquet').to_pandas();rows=rows[rows.inner_role>=0].copy()
    projections=pq.read_table(prepared/'projections.parquet').to_pandas()
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True);projections=projections.iloc[active].reset_index(drop=True)
    texts=projections.text.tolist();role=rows.inner_role.to_numpy();y=rows.label_index.to_numpy();g=rows.union_group.to_numpy();source=rows['product'].to_numpy()
    fit=role==0;sel=role==1;final=fit|sel;ev=role==2
    pq.write_table(pa.Table.from_pandas(rows[['row_position','inner_role','union_group']],preserve_index=False),out/'roles.parquet',compression='zstd')
    # Shared lexical matrices and original-row IDF multiplicities across A/B/C.
    te=text_encoder(texts,ids,fit);tx=te.transform(texts)
    final_te=text_encoder(texts,ids,final);final_tx=final_te.transform(texts)
    summaries=[]
    for view in a.views.split(','):
        if view not in rep.VIEWS:raise ValueError(view)
        folder=out/view;folder.mkdir();facts=[json.loads(v) for v in projections[view]]
        fe=learn.FixedFacts(view).fit([facts[i] for i in np.unique(ids[fit])]);x=sparse.hstack([tx,fe.transform(facts)],format='csr')
        save(folder/'fact_schema.json',{'names':fe.names().tolist(),'fit_unknown':fe.fit_unknown,'transform_unknown':fe.last_unknown})
        candidates=[];best=None
        def weights(mask):
            if a.weight=='equal':return None
            counts=np.bincount(y[mask],minlength=3);w=1/np.sqrt(counts[y[mask]])
            return w/w.mean()
        for c in contract['C_grid']:
            print(json.dumps({'stage':'fit','view':view,'C':c,'rows':int(fit.sum()),'unique_projection_rows':x.shape[0]}),flush=True)
            t=time.perf_counter();model,info=learn.fit_aggregated(x,ids[fit],y[fit],c,weights(fit))
            probs=model.predict_proba(x);score=metrics(y[sel],probs[ids[sel]],g[sel],source[sel]);apparent=learn.cm_metrics(y[fit],probs[ids[fit]].argmax(1))
            item={'C':c,'selection':score,'apparent_fit':apparent,'optimizer':info,'seconds':time.perf_counter()-t};candidates.append(item)
            np.savez_compressed(folder/('selection_C_'+str(c)+'.npz'),coef=model.coef_,intercept=model.intercept_)
            if best is None or score['macro_f1']>best['selection']['macro_f1']+0.001:best=item
            print(json.dumps({'stage':'selected_candidate_scored','view':view,'C':c,'selection_macro_f1':score['macro_f1'],'selection_recall':score['class_recall'],'seconds':round(time.perf_counter()-t)}),flush=True)
        save(folder/'selection.json',{'trials':candidates,'chosen_C':best['C'],'evaluation_used_for_C':False})
        joblib.dump({'text_encoder':te,'fact_encoder':fe},folder/'selection_encoders.joblib',compress=3)
        del x,model,probs;gc.collect()
        ffe=learn.FixedFacts(view).fit([facts[i] for i in np.unique(ids[final])]);fx=sparse.hstack([final_tx,ffe.transform(facts)],format='csr')
        print(json.dumps({'stage':'final_refit','view':view,'C':best['C'],'rows':int(final.sum())}),flush=True)
        model,info=learn.fit_aggregated(fx,ids[final],y[final],best['C'],weights(final))
        bundle={'version':rep.VERSION,'view':view,'text_encoder':final_te,'fact_encoder':ffe,'model':model,'decision':'three_class_argmax','contract':contract}
        joblib.dump(bundle,folder/'model.joblib',compress=3)
        unique_probs=model.predict_proba(fx);p=unique_probs[ids[ev]];pred=p.argmax(1)
        table=rows.loc[ev,['row_position','label_index','union_group','product','route']].copy()
        for i,k in enumerate(['p_benign','p_malicious','p_suspicious']):table[k]=p[:,i]
        table['pred_label']=np.asarray(['benign','malicious','suspicious'])[pred]
        pq.write_table(pa.Table.from_pandas(table,preserve_index=False),folder/'evaluation.parquet',compression='zstd')
        keyids=actual_keys(fx);evaluation=metrics(y[ev],p,g[ev],source[ev])
        slices={}
        for name,m in [('all',np.ones(ev.sum(),bool)),('asa',rows.loc[ev,'route'].to_numpy()=='asa')]:
            if m.any():slices[name]=collision_slice(y[ev][m],keyids[ids[ev]][m],p[m])
        # Model roundtrip plus the raw-message inference path on frozen cases.
        loaded=joblib.load(folder/'model.joblib');cases=json.loads((prepared/'audit_cases.json').read_text(encoding='utf-8'))
        direct=[rep.view_record(rep.prepare_message(c['raw']),view) for c in cases]
        dp,di=learn.classify(loaded,[z['text'] for z in direct],[z['facts'] for z in direct])
        case_original=np.asarray([c['projection_id'] for c in cases]);loc=np.searchsorted(active,case_original);available=(loc<len(active))
        available[available]=active[loc[available]]==case_original[available]
        replay={'cases':len(cases),'cached_comparable':int(available.sum()),'direct_cached_probability_max_difference':float(np.abs(dp[available]-unique_probs[loc[available]]).max()) if available.any() else None,
                'direct_cached_decision_differences':int((di[available]!=unique_probs[loc[available]].argmax(1)).sum()),'frozen_model_sha256':sha(folder/'model.joblib')}
        if replay['direct_cached_decision_differences'] or replay['direct_cached_probability_max_difference']>1e-10:raise RuntimeError('Inference replay failed')
        report={'view':view,'chosen_C':best['C'],'sample_weight':contract['sample_weight'],'final_optimizer':info,'evaluation':evaluation,
                'apparent_final_fit':learn.cm_metrics(y[final],unique_probs[ids[final]].argmax(1)),
                'encoded_collision_slices':slices,'replay':replay,'quality_accepted':False,'fresh_blind_test':False}
        save(folder/'report.json',report);summaries.append(report)
        save(folder/'complete.json',{'model_sha256':sha(folder/'model.joblib'),'report_sha256':sha(folder/'report.json'),'predictions_sha256':sha(folder/'evaluation.parquet')})
        print(json.dumps({'stage':'evaluation_complete','view':view,'macro_f1':evaluation['macro_f1'],'recall':evaluation['class_recall'],'normal_errors':evaluation['normal_errors']}),flush=True)
        del fx,model,loaded,bundle,unique_probs,dp;gc.collect()
    # Three-class negative control exposes what feature availability alone learns.
    if a.weight=='equal':
        f=[json.loads(v) for v in projections['C_BOTH']];mx,names=negative_control(f)
        model,info=learn.fit_aggregated(mx,ids[final],y[final],.1)
        p=model.predict_proba(mx)[ids[ev]]
        save(out/'availability_control.json',{'features':names,'optimizer':info,'evaluation':metrics(y[ev],p,g[ev],source[ev]),'diagnostic_only':True})
        joblib.dump({'model':model,'names':names},out/'availability_control.joblib',compress=3)
    save(out/'complete.json',{'models':len(summaries),'primary_fits':len(summaries)*3,'negative_control_fits':int(a.weight=='equal'),
         'seconds':time.perf_counter()-start,'quality_accepted':False,'external_or_blind_validation':False,
         'views':[{k:r[k] for k in ['view','chosen_C','evaluation','replay']} for r in summaries]})
    print(json.dumps({'stage':'bounded_v38_training_completed','seconds':round(time.perf_counter()-start),'quality_accepted':False}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--output',required=True)
    p.add_argument('--views',default=','.join(rep.VIEWS));p.add_argument('--weight',choices=['equal','sqrt_inverse'],default='equal');p.add_argument('--rationale',default='')
    trial(p.parse_args())
