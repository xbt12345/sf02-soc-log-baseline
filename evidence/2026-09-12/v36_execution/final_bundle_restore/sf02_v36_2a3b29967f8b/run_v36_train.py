"""Paired official-data development fits. Never reads official validation answers."""
import argparse
import gc
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse

import soc_v3_prepare as base
import v36_learning as learning

TASKS = ['known_dev','source_ad','source_duo','source_waf','asa_hard']
BUDGETS = [0.0001,0.001,0.01]


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def code_hashes():
    # Includes transitive local imports, not only the entry point.
    return {p.name:base.file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}


def string_codes(path,positions,column):
    """Read only selected positions; retain one Python copy of each unique value."""
    values=[]; lookup={}; codes=np.empty(len(positions),dtype=np.int32)
    offset=0
    for b in pq.ParquetFile(path).iter_batches(batch_size=4096,columns=[column],use_threads=False):
        lo,hi=np.searchsorted(positions,[offset,offset+len(b)])
        if hi>lo:
            d=b.column(0).take(pa.array(positions[lo:hi]-offset)).dictionary_encode()
            translation=[]
            for text in d.dictionary.to_pylist():
                if text not in lookup:
                    lookup[text]=len(values);values.append(text)
                translation.append(lookup[text])
            codes[lo:hi]=np.asarray(translation,dtype=np.int32)[d.indices.to_numpy()]
        offset+=len(b)
    return values,codes


def pilot_roles(roles,groups,y,products,task):
    """Deterministic bounded engineering pilot; full source/template target retained.

    Sampling changes prevalence and excludes large fit groups. It is not a
    population-quality estimate or a final FPR-budget certification.
    """
    selected=np.zeros(len(y),dtype=bool)
    gsize=np.bincount(groups)
    salt=np.uint64(20260912)
    skipped=[]
    for role in range(4):
        if role==3 and task!='known_dev':
            selected|=roles==3;continue
        limit=50 if role==0 else (20 if role==3 else 10)
        for product in np.unique(products):
            for c in range(3):
                candidates=np.unique(groups[(roles==role)&(products==product)&(y==c)])
                available=candidates[gsize[candidates]<=1000]
                # Fixed integer hash; no error/probability dependent sampling.
                keys=(available.astype(np.uint64)^salt)*np.uint64(11400714819323198485)
                chosen=available[np.argsort(keys)[:limit]]
                selected|=(roles==role)&np.isin(groups,chosen)
                if len(candidates)!=len(available):
                    skipped.append({'role':role,'product':str(product),'class':c,'oversized_groups':int(len(candidates)-len(available))})
    result=np.where(selected,roles,-1).astype(np.int8)
    return result,{'mode':'bounded_real_data_pilot','group_limit_fit':50,'group_limit_selection_calibration':10,
                   'max_fit_group_rows':1000,'whole_source_template_evaluation':task!='known_dev',
                   'skipped_oversized_groups':skipped,'population_quality_validated':False}


def representations(texts,tcodes,facts,fcodes,view,fit):
    allowed,frequency=np.unique(tcodes[fit],return_counts=True)
    tfidf=learning.FrequencyTfidf().fit([texts[i] for i in allowed],frequency)
    if view!='B2':
        matrix=tfidf.transform(texts)
        return matrix,tcodes,tfidf,None
    fact_encoder=learning.FactEncoder().fit([facts[i] for i in np.unique(fcodes[fit])])
    # Pair identity is label independent. Future values may be transformed but
    # cannot affect vocabulary, IDF, categories or numeric scaling.
    paircode=tcodes.astype(np.int64)*len(facts)+fcodes
    pairs,inverse=np.unique(paircode,return_inverse=True)
    unique_t=pairs//len(facts);unique_f=pairs%len(facts)
    matrix=sparse.hstack([tfidf.transform([texts[i] for i in unique_t]),
                         fact_encoder.transform([facts[i] for i in unique_f])],format='csr')
    return matrix,inverse,tfidf,fact_encoder


def predict_rows(model,matrix,inverse,mask):
    ids,back=np.unique(inverse[mask],return_inverse=True)
    return model.predict_proba(matrix[ids])[back]


def evaluate(y,p,groups,products,empty,factless,thresholds,view_keys):
    result={'all_rows':learning.cm_metrics(y,p.argmax(1)),
        'all_benign_control':learning.cm_metrics(y,np.zeros(len(y),dtype=int)),
        'risk':[dict(budget=b,**learning.risk_metrics(y,p,t)) for b,t in thresholds],
        'sources':{},'information_slices':{}}
    nonempty=~empty
    _,inv,count=np.unique(groups[nonempty],return_inverse=True,return_counts=True)
    result['group_equal_nonempty']=learning.cm_metrics(y[nonempty],p[nonempty].argmax(1),1/count[inv])
    _,inv,count=np.unique(view_keys,return_inverse=True,return_counts=True)
    result['view_equal_including_collapsed_empty']=learning.cm_metrics(y,p.argmax(1),1/count[inv])
    result['group_count_note']='Conservative exact-view groups; not confirmed independent incidents. Original empty rows are NOT counted as independent evidence in group-equal metrics.'
    for source in np.unique(products):
        m=products==source
        result['sources'][str(source)]={'classification':learning.cm_metrics(y[m],p[m].argmax(1)),
            'class_groups':[int(len(np.unique(groups[m&(y==c)]))) for c in range(3)],
            'risk':[dict(budget=b,**learning.risk_metrics(y[m],p[m],t)) for b,t in thresholds]}
    for name,m in [('original_empty',empty),('no_text_or_structured_facts',factless)]:
        result['information_slices'][name]=learning.cm_metrics(y[m],p[m].argmax(1))
    # Refuse precision claims at 1e-4 merely because there were zero observed FP.
    result['no_external_or_blind_validation']=True
    return result


def run_one(args,task,view,metadata,protocol,summary):
    prepared=Path(args.prepared); root=Path(args.output_dir)
    y,groups,products,empty=metadata
    roles=protocol[task].to_numpy().astype(np.int8)
    sampling={'mode':'full_official_development','population_quality_validated':False}
    if args.pilot:
        if not hasattr(args,'pilot_cache'):args.pilot_cache={}
        if task not in args.pilot_cache:args.pilot_cache[task]=pilot_roles(roles,groups,y,products,task)
        roles,sampling=args.pilot_cache[task]
    for r in (0,1,2):
        if set(np.unique(y[roles==r]))!={0,1,2}:
            raise ValueError('All three classes required in fit/selection/calibration')
    positions=np.flatnonzero(roles>=0)
    binding={'task':task,'view':view,'parameters':learning.PARAMETERS,'sampling':sampling,
             'repeat_weighting':bool(args.repeat_weight),'weight_rule':'allowed-fit sparse feature equality, inverse sqrt frequency, clip [0.2,5], mean 1' if args.repeat_weight else 'unit row weights',
             'prepared_sha256':summary['prepared_sha256'],'groups_sha256':summary['groups_sha256'],
             'roles_sha256':hashlib.sha256(roles.tobytes()).hexdigest(),'code_hashes':code_hashes(),
             'python':platform.python_version(),'sklearn':__import__('sklearn').__version__}
    stem=task+'_'+view+('_W' if args.repeat_weight else '')
    folder=root/stem
    attempts=[folder]+sorted(root.glob(stem+'_attempt*'))
    for attempt in attempts:
        if (attempt/'complete.json').exists():
            previous=json.loads((attempt/'complete.json').read_text(encoding='utf-8'))
            if previous['binding']!=binding:
                raise ValueError('Completed fit identity changed; use a new output directory')
            for name,sha in previous['artifacts'].items():
                assert base.file_hash(attempt/name)==sha
            print(json.dumps({'stage':'reuse_verified','task':task,'view':view}),flush=True)
            return
    if folder.exists():
        folder=root/(stem+'_attempt'+str(len(attempts)).zfill(3))
        if folder.exists():raise FileExistsError(folder)
    folder.mkdir(parents=True)
    save(folder/'binding.json',binding)
    pq.write_table(pa.table({'row_position':positions.astype(np.int32),'role':roles[positions]}),folder/'roles.parquet',compression='zstd')
    start=time.perf_counter()
    texts,tcodes=string_codes(prepared/'prepared.parquet',positions,'b0' if view=='B0' else 'b1')
    facts,fcodes=string_codes(prepared/'prepared.parquet',positions,'facts')
    role=roles[positions]; labels=y[positions]
    fit=role==0; selection=role==1; calibration=role==2; evaluation=role==3
    print(json.dumps({'stage':'encode_fit_only','task':task,'view':view,'rows':len(labels),'unique_texts':len(texts)}),flush=True)
    x,inverse,encoder,fencoder=representations(texts,tcodes,facts,fcodes,view,fit)
    weights=learning.repeat_weights(x,inverse[fit]) if args.repeat_weight else None
    candidates=[];best=None
    for C in learning.PARAMETERS['C']:
        model,receipt=learning.fit_aggregated(x,inverse[fit],labels[fit],C,weights)
        p=predict_rows(model,x,inverse,selection)
        pq.write_table(pa.table({'row_position':positions[selection].astype(np.int32),'label_index':labels[selection],
             'p_benign':p[:,0],'p_malicious':p[:,1],'p_suspicious':p[:,2]}),folder/('selection_C'+str(C)+'.parquet'),compression='zstd')
        score=learning.cm_metrics(labels[selection],p.argmax(1))['macro_f1']
        candidates.append({'C':C,'selection_macro_f1':score,'fit':receipt})
        if best is None or score>best[1]+learning.PARAMETERS['tie_macro_f1']:
            best=(C,score)
        print(json.dumps({'stage':'selected_candidate','task':task,'view':view,**candidates[-1]}),flush=True)
    del x,encoder,fencoder,model;gc.collect()
    final_fit=fit|selection
    x,inverse,encoder,fencoder=representations(texts,tcodes,facts,fcodes,view,final_fit)
    weights=learning.repeat_weights(x,inverse[final_fit]) if args.repeat_weight else None
    model,receipt=learning.fit_aggregated(x,inverse[final_fit],labels[final_fit],best[0],weights)
    if weights is not None:
        receipt['repeat_weight_range']=[float(weights.min()),float(weights.max())]
        receipt['repeat_weight_mean']=float(weights.mean())
        receipt['weighted_class_mass']=np.bincount(labels[final_fit],weights=weights,minlength=3).tolist()
    calp=predict_rows(model,x,inverse,calibration)
    p=predict_rows(model,x,inverse,evaluation)
    thresholds=[(b,learning.risk_threshold(1-calp[labels[calibration]==0,0],b)) for b in BUDGETS]
    factless=(np.asarray([not t for t in texts])[tcodes]) & (np.asarray([f=='{}' for f in facts])[fcodes])
    report=evaluate(labels[evaluation],p,groups[positions][evaluation],products[positions][evaluation],
                    empty[positions][evaluation],factless[evaluation],thresholds,inverse[evaluation])
    report.update(task=task,view=view,selected_C=best[0],candidates=candidates,final_fit=receipt,
        role_class_counts={str(r):np.bincount(labels[role==r],minlength=3).tolist() for r in range(4)},
        calibration_class_groups=[int(len(np.unique(groups[positions][calibration&(labels==c)]))) for c in range(3)],
        calibration=[dict(budget=b,**learning.risk_metrics(labels[calibration],calp,t)) for b,t in thresholds],
        elapsed_seconds=time.perf_counter()-start,sampling=sampling,
        audit_inspected_after_this_run=True,transfer_validated=False)
    # Format-only diagnostic, learned exclusively from final fit rows.
    formats,fmtcodes=string_codes(prepared/'prepared.parquet',positions,'format')
    counts=np.bincount(fmtcodes[final_fit]*3+labels[final_fit],minlength=len(formats)*3).reshape(-1,3)
    prior=np.bincount(labels[final_fit],minlength=3).argmax()
    fp=counts.argmax(1);fp[counts.sum(1)==0]=prior
    report['format_only_control']=learning.cm_metrics(labels[evaluation],fp[fmtcodes[evaluation]])
    # Grouping controls duplicated inputs; this is not a guarantee against all
    # similar templates, source vocabularies or omitted causal context.
    feature_names=list(encoder.names())+(list(fencoder.names()) if fencoder else [])
    report['largest_coefficients']={str(c):[{'feature':str(feature_names[i]),'coefficient':float(model.coef_[c,i])}
          for i in np.argsort(-np.abs(model.coef_[c]))[:30]] for c in range(3)}
    # B2 factual-channel removal is deliberately labeled information deletion,
    # not a label-preserving counterfactual. Report dependence, not causality.
    if view=='B2' and fencoder is not None:
        ids,back=np.unique(inverse[evaluation],return_inverse=True)
        dropped=x[ids].copy();dropped.data[dropped.indices>=len(encoder.names())]=0;dropped.eliminate_zeros()
        changed=model.predict_proba(dropped)[back]
        report['fact_channel_deletion']={'not_label_preserving':True,'prediction_flip_rate':float(np.mean(changed.argmax(1)!=p.argmax(1))),
                 'max_probability_change':float(abs(changed-p).max()),'classification':learning.cm_metrics(labels[evaluation],changed.argmax(1))}
    save(folder/'report.json',report)
    for name,mask,prob in [('calibration',calibration,calp),('evaluation',evaluation,p)]:
        pq.write_table(pa.table({'row_position':positions[mask].astype(np.int32),'label_index':labels[mask],
             'p_benign':prob[:,0],'p_malicious':prob[:,1],'p_suspicious':prob[:,2],
             'prediction':prob.argmax(1).astype(np.uint8)}),folder/(name+'.parquet'),compression='zstd')
    joblib.dump({'view':view,'tfidf':encoder,'facts':fencoder,'model':model,'thresholds':thresholds,
                 'representation_version':summary['version'],'labels':['benign','malicious','suspicious']},folder/'model.joblib',compress=3)
    files=['report.json','roles.parquet','calibration.parquet','evaluation.parquet','model.joblib']+['selection_C'+str(C)+'.parquet' for C in learning.PARAMETERS['C']]
    save(folder/'complete.json',{'binding':binding,'artifacts':{n:base.file_hash(folder/n) for n in files},
                               'full_training_quality_not_certified':True})
    print(json.dumps({'stage':'fit_complete','task':task,'view':view,'macro_f1':report['all_rows']['macro_f1'],
                     'class_recall':report['all_rows']['class_recall'],'seconds':round(report['elapsed_seconds'])}),flush=True)


def run(args):
    prepared=Path(args.prepared);out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads((prepared/'result.json').read_text(encoding='utf-8'))
    assert summary['full_official_rows']
    for f,k in [('prepared.parquet','prepared_sha256'),('groups.parquet','groups_sha256')]:
        assert base.file_hash(prepared/f)==summary[k]
    declaration=json.loads((prepared/'protocol.json').read_text(encoding='utf-8'))
    assert base.file_hash(prepared/'protocol.parquet')==declaration['protocol_sha256']
    for name,sha in summary['source_hashes'].items():
        assert base.file_hash(Path(__file__).parent/name)==sha, 'Prepared source identity changed: '+name
    g=pq.read_table(prepared/'groups.parquet')
    y=g['label_index'].to_numpy();groups=g['union_group'].to_numpy()
    products=np.asarray(pq.read_table(prepared/'prepared.parquet',columns=['product'])['product'].to_pylist(),dtype=object)
    empty=pq.read_table(prepared/'prepared.parquet',columns=['original_empty'])['original_empty'].to_numpy()
    protocol=pq.read_table(prepared/'protocol.parquet')
    for task in args.tasks.split(','):
        assert declaration['tasks'][task]['eligible']
        for view in args.views.split(','):
            if view not in ('B0','B1','B2'):raise ValueError(view)
            run_one(args,task,view,(y,groups,products,empty),protocol,summary)
            gc.collect()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--tasks',default=','.join(TASKS));p.add_argument('--views',default='B0,B1,B2')
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--repeat-weight',action='store_true',help='One preregistered B2-only weight comparison')
    args=p.parse_args()
    if args.repeat_weight and args.views!='B2':raise ValueError('Repeat candidate is B2 only')
    run(args)
