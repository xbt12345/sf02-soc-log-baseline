"""One frozen v41 contrast per process; unchanged rows, labels and optimizer."""
import argparse
import gc
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import v41_core as core
from run_v39_prepare import sha,save
from run_v38_train import actual_keys
from run_v39_train import metric as old_metric,conditional


def metrics(y,p):
    m=old_metric(y,p)
    m['log_loss']=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-300)).mean()) if len(y) else None
    m['brier_sum_classes']=float((np.sum(p*p,axis=1)-2*p[np.arange(len(y)),y]+1).mean()) if len(y) else None
    return m


def partition(keys,ids,y,fit):
    counts=np.bincount(keys[ids[fit]]*3+y[fit],minlength=(keys.max()+1)*3).reshape(-1,3)
    rk=keys[ids];seen=counts.sum(1)[rk]>0;conf=((counts>0).sum(1)>1)[rk];same=counts.argmax(1)[rk]==y
    result=np.full(len(y),3,dtype=np.uint8)
    result[seen&~conf&same]=0;result[seen&~conf&~same]=1;result[seen&conf]=2
    return result,counts


def main(a):
    start=time.perf_counter(); root=Path(a.root); run=Path(a.run); stage=run/'prepared'
    receipt=json.loads((stage/'complete.json').read_text(encoding='utf-8'))
    for n,h in receipt['files'].items():assert sha(stage/n)==h,n
    for n,h in receipt['runtime_sources'].items():assert sha(Path(__file__).parent/n)==h,n
    review=json.loads((stage/'near_copy_review.json').read_text(encoding='utf-8'));assert review['proceed_same_protocol']
    conf=json.loads((stage/'configuration.json').read_text(encoding='utf-8'));assert a.view in conf['views']
    prep=Path(receipt['v39_prepared']);assert sha(prep/'complete.json')==receipt['v39_prepared_receipt_sha256']
    out=run/('old_protocol_stress' if a.stress else 'primary/fold_%s/%s'%(a.fold,a.view))
    if out.exists():raise FileExistsError('Preserve previous fit')
    if a.stress:
        decision=json.loads((run/'primary_review/selection.json').read_text(encoding='utf-8'))
        assert decision['selected_view']==a.view and decision['proceed_to_stress']
    out.mkdir(parents=True)
    rows=pq.read_table(prep/'rows.parquet').to_pandas();rows=rows[rows.fold>=0].reset_index(drop=True)
    y=rows.label_index.to_numpy(dtype=np.int64);folds=rows.fold.to_numpy();body=rows.body_group.to_numpy()
    fit=rows.inner_role.to_numpy()!=2 if a.stress else folds!=a.fold; ev=~fit
    if a.stress:
        assert fit.sum()==1199575 and ev.sum()==179075
        assert not rows.loc[fit,'product'].isin(['Duo','Barracuda WAF']).any()
        assert not set(body[fit])&set(body[ev]);assert not set(rows.union_group[fit])&set(rows.union_group[ev])
        reference=prep.parent/'old_protocol_stress'
    else:
        reference=prep.parent/('primary/fold_%s/SEMANTIC'%a.fold)
    bound=json.loads((reference/'complete.json').read_text(encoding='utf-8'))
    assert sha(reference/'model.joblib')==bound['model_sha256']
    assert sha(reference/'evaluation.parquet')==bound['predictions_sha256']
    active,ids=np.unique(rows.projection_id.to_numpy(),return_inverse=True)
    base_ids=ids.copy()
    pr=pq.read_table(prep/'projections.parquet').to_pandas().iloc[active].reset_index(drop=True)
    facts=[json.loads(v) for v in pr.facts];texts=pr.text.tolist()
    b=joblib.load(reference/'model.joblib');te=b['text_encoder']
    assert te.fit_rows==int(fit.sum())
    tx=te.transform(texts);xb=sparse.hstack([tx,b['fact_encoder'].transform(facts)],format='csr')
    bk=actual_keys(xb);base_part,bc=partition(bk,ids,y,fit);bp=b['model'].predict_proba(xb)
    reference_eval=pq.read_table(reference/'evaluation.parquet').to_pandas()
    assert np.array_equal(reference_eval.row_position,rows.row_position.to_numpy()[ev])
    assert np.array_equal(reference_eval.label_index,y[ev])
    difference=float(np.abs(reference_eval[['p_benign','p_malicious','p_suspicious']].to_numpy()-bp[base_ids[ev]]).max())
    assert difference<=conf['probability_tolerance']
    del reference_eval
    coverage_encoder=core.ParameterEffects().fit(facts,ids[fit],body[fit],conf['pair_minimum_distinct_bodies'])
    cov=pd.DataFrame(coverage_encoder.coverage(facts))
    if not a.stress:
        old_i=joblib.load(root/('artifacts/v40_local_r1_20260913/primary/fold_%s/I/model.joblib'%a.fold))
        assert coverage_encoder.term_names==old_i['parameter_encoder'].term_names
        assert coverage_encoder.support_lower_bounds==old_i['parameter_encoder'].support_lower_bounds
        del old_i;gc.collect()
    baseline_names=b['fact_encoder'].names(); text_columns=tx.shape[1]
    model_pids=rows.projection_id.to_numpy().copy()
    if a.view=='P':
        fe=b['fact_encoder'];pe=coverage_encoder
        x=sparse.hstack([xb,pe.transform(facts)],format='csr')
    elif a.view in ('F','D'):
        fe=(core.FiniteOnlyFacts() if a.view=='F' else core.DeduplicatedFacts()).fit([facts[i] for i in np.unique(ids[fit])])
        pe=None;x=sparse.hstack([tx,fe.transform(facts)],format='csr')
    else:
        assert json.loads((stage/'native_configuration.json').read_text(encoding='utf-8'))['boundary_ready']
        assert (run/'primary_review_stage1/selection.json').exists(), 'Complete P/F/D review before A'
        mapping=pq.read_table(stage/'A_updates.parquet').to_pandas()
        row_index=np.searchsorted(rows.row_position.to_numpy(),mapping.row_position.to_numpy())
        assert np.array_equal(rows.row_position.to_numpy()[row_index],mapping.row_position.to_numpy())
        model_pids[row_index]=mapping.model_projection_id.to_numpy()
        active,ids=np.unique(model_pids,return_inverse=True)
        ap=pq.read_table(stage/'A_projections.parquet').to_pandas().iloc[active]
        texts=ap.text.tolist();facts=[json.loads(v) for v in ap.facts]
        freq=np.bincount(ids[fit],minlength=len(active));use=np.flatnonzero(freq)
        te=type(te)().fit([texts[i] for i in use],freq[use])
        assert te.fit_rows==int(fit.sum())
        fe=core.prior.SemanticFacts().fit([facts[i] for i in use]);pe=None
        tx=te.transform(texts);x=sparse.hstack([tx,fe.transform(facts)],format='csr')
    isolation=core.isolation_check(a.view,xb,x,baseline_names,text_columns) if a.view!='A' else {'scope':'native-only repaired text/facts; separately refitted fit-only text encoder'}
    del xb
    del b,tx;gc.collect()
    keys=actual_keys(x);new_part,nc=partition(keys,ids,y,fit)
    save(out/'binding.json',{'version':core.VERSION,'view':a.view,'fold':a.fold if not a.stress else None,
        'protocol':'old_template_source' if a.stress else 'v39_fixed_body_folds','C':conf['views'][a.view],
        'prepared_sha256':sha(stage/'complete.json'),'reference_model_sha256':bound['model_sha256'],
        'reference_full_prediction_replay_max_difference':difference,'fit_rows':int(fit.sum()),'evaluation_rows':int(ev.sum()),
        'python':platform.python_version(),'packages':{n:__import__(n).__version__ for n in ['numpy','scipy','sklearn','pyarrow']},
        'factor_isolation':isolation,'parameter_vocabulary_matches_v40':True,'matrix_shape':list(x.shape),'matrix_nnz':x.nnz,'runtime_sources':receipt['runtime_sources'],
        'selection_sha256':sha(run/'primary_review/selection.json') if a.stress else None})
    print(json.dumps({'stage':'fit_start','view':a.view,'fold':a.fold,'stress':a.stress,'fit_rows':int(fit.sum()),'columns':x.shape[1],'nnz':x.nnz}),flush=True)
    model,optimizer=core.prior.learning.fit_aggregated(x,ids[fit],y[fit],conf['views'][a.view])
    assert optimizer['sum_weights']==int(fit.sum())
    prob=model.predict_proba(x);p=prob[ids[ev]];pred=prob.argmax(1)[ids]
    bundle={'version':core.VERSION,'view':a.view,'text_encoder':te,'fact_encoder':fe,'parameter_encoder':pe,'model':model,'decision':'three_class_argmax'}
    joblib.dump(bundle,out/'model.joblib',compress=3)
    loaded=joblib.load(out/'model.joblib')
    z=x.dot(loaded['model'].coef_.T)+loaded['model'].intercept_;z-=z.max(1,keepdims=True);z=np.exp(z);z/=z.sum(1,keepdims=True)
    replay=float(np.abs(z-prob).max());assert replay<=conf['probability_tolerance']
    assert np.array_equal(z.argmax(1),prob.argmax(1));del loaded,z
    d=rows.loc[ev,['row_position','label_index','product','route','body_group','union_group','projection_id']].copy()
    for j,n in enumerate(['p_benign','p_malicious','p_suspicious']):d[n]=p[:,j]
    d['model_projection_id']=model_pids[ev]
    d['pred_label']=np.array(['benign','malicious','suspicious'])[p.argmax(1)]
    d['baseline_partition']=base_part[ev];d['new_partition']=new_part[ev]
    for col in cov:d[col]=cov[col].to_numpy()[base_ids[ev]]
    normfacts=[core.canonicalize(f)[0] for f in facts]
    ax,names=core.prior.availability(normfacts);masks=actual_keys(ax);d['observation_mask']=masks[ids[ev]]
    pq.write_table(pa.Table.from_pandas(d,preserve_index=False),out/'evaluation.parquet',compression='zstd')
    route_metrics={str(r):metrics(y[ev][d.route.to_numpy()==r],p[d.route.to_numpy()==r]) for r in sorted(set(d.route))}
    fit_asa=fit&(rows.route.to_numpy()=='asa');asa_counts=np.bincount(keys[ids[fit_asa]]*3+y[fit_asa],minlength=(keys.max()+1)*3).reshape(-1,3)
    collision={'fit_mixed_keys':int(((nc>0).sum(1)>1).sum()),'fit_empirical_minimum_errors':int((nc.sum(1)-nc.max(1)).sum()),
        'fit_asa_minimum_errors':int((asa_counts.sum(1)-asa_counts.max(1)).sum()),
        'fit_asa_pure_rows':int((fit_asa&(new_part==0)).sum()),'fit_asa_pure_errors':int((fit_asa&(new_part==0)&(pred!=y)).sum()),
        'original_distinct_keys_merged_in_new':int((pd.DataFrame({'old':bk[base_ids],'new':keys[ids]}).drop_duplicates().groupby('new').old.nunique()>1).sum())}
    ec=np.bincount(keys[ids[ev]]*3+y[ev],minlength=(keys.max()+1)*3).reshape(-1,3)
    collision['evaluation_empirical_minimum_errors']=int((ec.sum(1)-ec.max(1)).sum())
    changes={str(r):{'fixed':int(((d.route.to_numpy()==r)&(bp[base_ids[ev]].argmax(1)!=y[ev])&(p.argmax(1)==y[ev])).sum()),
        'regressed':int(((d.route.to_numpy()==r)&(bp[base_ids[ev]].argmax(1)==y[ev])&(p.argmax(1)!=y[ev])).sum())} for r in sorted(set(d.route))}
    report={'view':a.view,'fold':a.fold if not a.stress else None,'evaluation':metrics(y[ev],p),'apparent_fit':metrics(y[fit],prob[ids[fit]]),
        'baseline_evaluation':metrics(y[ev],bp[base_ids[ev]]),'routes':route_metrics,'changes_by_route':changes,
        'sources':{str(s):metrics(y[ev][d['product'].to_numpy()==s],p[d['product'].to_numpy()==s]) for s in sorted(set(d['product']))},
        'collision':collision,'conditional_comparisons':conditional(rows,y,prob,ids,masks,fit,ev),
        'parameter_support':{'main_terms':coverage_encoder.main_count,'eligible_joint_terms':coverage_encoder.pair_count,
            'observed_joint_terms':len(coverage_encoder.support_lower_bounds),'count_is_saturated_body_lower_bound':True,'used_in_model':a.view=='P'},
        'optimizer':optimizer,'manual_softmax_replay_max_difference':replay,'seconds':time.perf_counter()-start,
        'fresh_blind_test':False,'quality_accepted':False}
    save(out/'report.json',report)
    save(out/'complete.json',{'model_sha256':sha(out/'model.joblib'),'predictions_sha256':sha(out/'evaluation.parquet'),
        'report_sha256':sha(out/'report.json'),'binding_sha256':sha(out/'binding.json'),'fit_rows':int(fit.sum()),'evaluation_rows':int(ev.sum())})
    print(json.dumps({'stage':'fit_complete','view':a.view,'fold':a.fold,'stress':a.stress,'errors':report['evaluation']['errors'],
        'macro_f1':report['evaluation']['macro_f1'],'recall':report['evaluation']['class_recall'],'asa_errors':route_metrics.get('asa',{}).get('errors'),
        'fit_asa_pure_errors':collision['fit_asa_pure_errors'],'seconds':round(report['seconds'])}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True)
    p.add_argument('--view',choices=['P','F','D','A'],required=True);p.add_argument('--fold',type=int,choices=[0,1,2],default=0);p.add_argument('--stress',action='store_true');main(p.parse_args())
