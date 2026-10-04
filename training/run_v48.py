"""One semantic repair, one pressure fit, then conditional three-fold regression.

All source bindings, input changes and acceptance rules precede model fitting.
No target weights, threshold search, relabeling or external training data.
"""
import argparse
import collections
import gc
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def save(p,v): Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def raw_rows(official,positions):
    positions=np.sort(np.asarray(positions,dtype=np.int64));offset=0
    for b in pq.ParquetFile(official).iter_batches(batch_size=16384,columns=['message_sanitized'],use_threads=False):
        take=positions[np.searchsorted(positions,offset):np.searchsorted(positions,offset+len(b))]
        for pos in take: yield int(pos),b.column(0)[int(pos)-offset].as_py() or ''
        offset+=len(b)

def prepare(root,out):
    if out.exists(): raise FileExistsError(out)
    source=root/'artifacts/v42_local_r1_20260914/frozen_training_runtime'
    receipt=read(source.parent/'prepared/complete.json')
    bindings={}
    for name,digest in receipt['runtime_sources'].items():
        assert sha(source/name)==digest
        bindings[(source/name).relative_to(root).as_posix()]=digest
    prepared=root/'artifacts/v39_local_r2_20260913/prepared'
    expected={'rows.parquet':'e1b5add85250184f4975935fcd912cde82d83abb700f1fe88f8efa6b8dfc4df5',
              'projections.parquet':'db23bf2603b938e7b502ba0061a368eda26d38e8fbfe82c8feb75da0fb8d030e'}
    for name,digest in expected.items():
        assert sha(prepared/name)==digest
        bindings[(prepared/name).relative_to(root).as_posix()]=digest
    official=root/'data/official/train.parquet'
    assert sha(official)=='6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742'
    bindings[official.relative_to(root).as_posix()]=sha(official)
    base=root/'artifacts/v39_local_r2_20260913'
    for folder in [base/'old_protocol_stress']+[base/('primary/fold_%s/SEMANTIC'%f) for f in range(3)]:
        rec=read(folder/'complete.json')
        for name,key in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256'),('report.json','report_sha256')]:
            assert sha(folder/name)==rec[key]
            bindings[(folder/name).relative_to(root).as_posix()]=rec[key]
    out.mkdir(parents=True);runtime=out/'runtime';runtime.mkdir()
    for name in receipt['runtime_sources']: shutil.copy2(source/name,runtime/name)
    for name in ['v48_input.py','test_v48_input.py','run_v48.py','verify_v48.py']:
        shutil.copy2(root/'training'/name,runtime/name)
    sys.path.insert(0,str(runtime));import v48_input as v
    rows=pd.read_parquet(prepared/'rows.parquet')
    assert np.array_equal((rows.inner_role>=0).to_numpy(),(rows.fold>=0).to_numpy())
    rows=rows[rows.inner_role>=0].copy().sort_values('row_position').reset_index(drop=True)
    assert len(rows)==1378650
    p=pd.read_parquet(prepared/'projections.parquet')[['text','facts']]
    frequencies=rows.projection_id.value_counts()
    unknown=collections.Counter(); disposition=collections.Counter()
    for pid,n in frequencies.items():
        for key,status in v.fact_dispositions(json.loads(p.iloc[pid].facts)).items():
            disposition[status]+=int(n)
            if status.startswith('unencoded'):unknown[key+':'+status]+=int(n)
    assert not unknown,dict(unknown)
    # Full native generated routes, all native audit/CEF records, and the prior
    # representative raw cases. Other raw routes are explicitly not exhaustive.
    targets=set(v.old.previous.GENERATED)|{'audit_fields','cef_fields'}
    cases=read(prepared/'audit_cases.json')
    casepos={int(c['row_position']) for c in cases}
    selected=rows[rows.route.isin(targets)|rows.row_position.isin(casepos)]
    lookup=rows.set_index('row_position');index={int(pos):i for i,pos in enumerate(rows.row_position)}
    additions=[];newids={};changes=[];audits=[];before_drop=collections.Counter();extra=collections.Counter()
    routes=collections.Counter();raw_verified=0
    rows['old_projection_id']=rows.projection_id
    for pos,raw in raw_rows(official,selected.row_position):
        row=lookup.loc[pos];old=v.old.prepare_message(raw);prior=v.old.previous.prior.prepare_message(raw)
        cached=p.iloc[int(row.projection_id)]
        assert old['text']==cached.text and v.old.canonical(old['facts'])==v.old.canonical(json.loads(cached.facts)),pos
        assert prior['route']==row.route,pos
        new=v.prepare_message(raw);raw_verified+=1;routes[row.route]+=1
        assert not any(s.startswith('unencoded') for s in v.fact_dispositions(new['facts']).values())
        projection=v.old.previous.project(prior['b1'],prior['facts'],prior['route'])
        for k in set(prior['facts'])-set(new['facts']):before_drop[str(row.route)+':'+k]+=1
        if row.route=='audit_fields':
            for f in prior['evidence']:
                if f.get('field') in ('proctitle','argc','syscall','arch'):
                    extra['audit_fields:captured_'+f['field']]+=1
        if new['route'] in ('native_flow','native_firewall'):
            audit=new['information_audit'];audit.update(row_position=pos,body_group=str(row.body_group),old_projection_id=int(row.projection_id))
            audits.append(audit)
            extra['native:'+audit['disposition']]+=1
        if (new['text'],new['facts'])!=(old['text'],old['facts']):
            assert new['route']=='native_flow'
            assert new['text']==old['text']
            assert {k:val for k,val in new['facts'].items() if k not in ('action','outcome')}==old['facts']
            key=(int(row.projection_id),new['text'],v.old.canonical(new['facts']))
            if key not in newids:
                newids[key]=len(p)+len(additions);additions.append({'text':new['text'],'facts':key[2]})
            rows.at[index[pos],'projection_id']=newids[key]
            changes.append({'row_position':pos,'old_projection_id':int(row.projection_id),'projection_id':newids[key],
                            'body_group':row.body_group,'inner_role':int(row.inner_role),'raw_sha256':audit['raw_sha256']})
        if raw_verified%25000==0:print(json.dumps({'stage':'raw_information_audit','rows':raw_verified}),flush=True)
    assert len(changes)==3188
    p=pd.concat([p,pd.DataFrame(additions)],ignore_index=True)
    rows.to_parquet(out/'rows.parquet',index=False);p.to_parquet(out/'projections.parquet',index=False)
    pd.DataFrame(changes).to_parquet(out/'input_changes.parquet',index=False)
    with (out/'native_evidence.jsonl').open('w',encoding='utf-8') as f:
        for a in audits:f.write(json.dumps(a,ensure_ascii=False)+'\n')
    save(out/'information_audit.json',{'scope':'All allowed cached fact keys/enums; all generated native raw routes plus all audit/CEF and prior representative cases. Not all raw message semantics.',
        'allowed_rows':len(rows),'active_old_projections':len(frequencies),'fact_dispositions_weighted':dict(disposition),
        'unencoded_facts':dict(unknown),'raw_records_reparsed':raw_verified,'raw_routes':dict(routes),
        'preprojection_fields_transformed_or_removed':dict(before_drop),
        'additional_findings':dict(extra),'changed_rows':len(changes),'new_projection_count':len(additions),
        'nonempty_raw_with_empty_model_text_rows':int(((~rows.original_empty)&rows.projection_id.map(p.text.str.len()) .eq(0)).sum()),
        'caveat':'Empty model text can have typed facts; no text is not no evidence. Captured fields require semantic review, not automatic risk admission.',
        'unknown_raw_preserved_in_official_file':True,'all_raw_information_retained_in_features':False})
    config={'version':v.VERSION,'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'C':0.1,'row_weight':'exact original multiplicities','labels':'unchanged official','decision':'three_class_argmax',
        'max_classifier_fits':4,'sequence':'one_old_pressure_then_three_body_folds_only_if_pressure_gates_pass',
        'gates':{'normal_errors_no_increase':True,'macro_f1_strict_increase':True,'errors_strict_decrease':True,
            'M_to_B_no_increase':True,'outside_native_flow_errors_no_increase':True,
            'each_class_recall_drop_max':0.02,'optimizer_converged':True},
        'model_selection_on_seen_development':True,'fresh_blind_or_external_validation':False,
        'same_family_normal_native_flow_rows':int(((rows.route=='native_flow')&(rows.label_index==0)).sum()),
        'source_bindings':bindings,'runtime_bindings':{q.name:sha(q) for q in runtime.glob('*.py')},
        'prepared_files':{n:sha(out/n) for n in ['rows.parquet','projections.parquet','input_changes.parquet','native_evidence.jsonl','information_audit.json']},
        'python':platform.python_version(),'packages':{n:__import__(n).__version__ for n in ['numpy','scipy','sklearn','pyarrow']}}
    save(out/'configuration.json',config)
    print(json.dumps({'stage':'prepared','raw_reparsed':raw_verified,'changed_rows':len(changes),'configuration_sha256':sha(out/'configuration.json')}),flush=True)

def verify_bindings(root,out):
    c=read(out/'configuration.json')
    for name,h in c['source_bindings'].items():assert sha(root/name)==h,name
    for name,h in c['runtime_bindings'].items():assert sha(out/'runtime'/name)==h,name
    for name,h in c['prepared_files'].items():assert sha(out/name)==h,name
    assert Path(__file__).resolve()==(out/'runtime/run_v48.py').resolve(),'Run the frozen copy'
    return c

def compare(d):
    from run_v39_train import metric
    y=d.label_index.to_numpy();p=d[['p_benign','p_malicious','p_suspicious']].to_numpy();b=d[['b_benign','b_malicious','b_suspicious']].to_numpy()
    pred=p.argmax(1);old=b.argmax(1);w=pred!=y;ow=old!=y
    newm=metric(y,p);oldm=metric(y,b)
    route=[]
    for name,g in d.groupby('route'):
        ix=g.index.to_numpy();a=w[ix];z=ow[ix]
        route.append({'route':name,'rows':len(ix),'errors_before':int(z.sum()),'errors_after':int(a.sum()),
                      'fixes':int((z&~a).sum()),'regressions':int((~z&a).sum())})
    outside=d.route.to_numpy()!='native_flow'
    gates={'normal_errors_no_increase':newm['normal_errors']<=oldm['normal_errors'],
           'macro_f1_strict_increase':newm['macro_f1']>oldm['macro_f1'],
           'errors_strict_decrease':newm['errors']<oldm['errors'],
           'M_to_B_no_increase':int(((y==1)&(pred==0)).sum())<=int(((y==1)&(old==0)).sum()),
           'outside_native_flow_errors_no_increase':int((w&outside).sum())<=int((ow&outside).sum()),
           'each_class_recall_drop_max':all(a>=b-0.02 for a,b in zip(newm['class_recall'],oldm['class_recall']))}
    bad=d[['row_position','body_group','route','label_index']].copy();bad['before']=old;bad['after']=pred
    bad=bad[(w|ow)]
    return {'before':oldm,'after':newm,'fixes':int((ow&~w).sum()),'regressions':int((~ow&w).sum()),
            'paired_body_count':int(d.body_group.nunique()),'routes':route,'gates':gates,
            'all_quality_gates_passed':all(gates.values())},bad

def fit_one(root,out,c,name,fold=None):
    import v48_input as v
    from run_v38_train import text_encoder
    folder=out/name
    if folder.exists():raise FileExistsError(folder)
    rows=pd.read_parquet(out/'rows.parquet')
    fit=rows.inner_role.to_numpy()!=2 if fold is None else rows.fold.to_numpy()!=fold
    ev=~fit; assert not set(rows.loc[fit,'body_group'])&set(rows.loc[ev,'body_group'])
    if fold is None:assert not set(rows.loc[fit,'union_group'])&set(rows.loc[ev,'union_group'])
    p=pd.read_parquet(out/'projections.parquet');active,ids=np.unique(rows.projection_id,return_inverse=True)
    p=p.iloc[active];texts=p.text.tolist();facts=[json.loads(s) for s in p.facts]
    oldpath=root/'artifacts/v39_local_r2_20260913'/('old_protocol_stress' if fold is None else 'primary/fold_%s/SEMANTIC'%fold)
    baseline=pd.read_parquet(oldpath/'evaluation.parquet').sort_values('row_position').reset_index(drop=True)
    evaluation=rows.loc[ev].sort_values('row_position').reset_index(drop=True)
    for key in ['row_position','label_index','body_group']:assert np.array_equal(evaluation[key],baseline[key]),key
    folder.mkdir();t=time.perf_counter()
    save(folder/'binding.json',{'configuration_sha256':sha(out/'configuration.json'),'fit_rows':int(fit.sum()),
        'evaluation_rows':int(ev.sum()),'old_model_sha256':sha(oldpath/'model.joblib'),'fold':fold,'C':c['C']})
    print(json.dumps({'stage':'fit_start','name':name,'fit_rows':int(fit.sum()),'evaluation_rows':int(ev.sum())}),flush=True)
    te=text_encoder(texts,ids,fit);fe=v.old.SemanticFacts().fit([facts[k] for k in np.unique(ids[fit])])
    x=sparse.hstack([te.transform(texts),fe.transform(facts)],format='csr')
    model,optimizer=v.old.learning.fit_aggregated(x,ids[fit],rows.label_index.to_numpy()[fit],c['C'])
    assert optimizer['sum_weights']==int(fit.sum())
    bundle={'adapter_version':v.VERSION,'view':'SEMANTIC','text_encoder':te,'fact_encoder':fe,'model':model,
            'configuration_sha256':sha(out/'configuration.json'),'decision':'three_class_argmax','fold':fold}
    joblib.dump(bundle,folder/'model.joblib',compress=3)
    q=model.predict_proba(x)[ids[ev]]
    for j,k in enumerate(['benign','malicious','suspicious']):
        evaluation['p_'+k]=q[:,j];evaluation['b_'+k]=baseline['p_'+k].to_numpy()
    report,bad=compare(evaluation);report.update(optimizer=optimizer,elapsed_seconds=time.perf_counter()-t,
        scope='Matched seen-development comparison; not blind/source transfer or same-family normal acceptance')
    report['gates']['optimizer_converged']=bool(optimizer['converged'])
    report['all_quality_gates_passed']=all(report['gates'].values())
    evaluation.to_parquet(folder/'evaluation.parquet',index=False);bad.to_parquet(folder/'error_changes.parquet',index=False)
    save(folder/'report.json',report)
    save(folder/'complete.json',{n:sha(folder/n) for n in ['model.joblib','evaluation.parquet','error_changes.parquet','report.json','binding.json']})
    print(json.dumps({'stage':'fit_complete','name':name,'before_errors':report['before']['errors'],'after_errors':report['after']['errors'],
                     'macro_f1':report['after']['macro_f1'],'gates':report['gates']}),flush=True)
    del x,model,bundle,te,fe;gc.collect()
    return report

def fit(root,out):
    c=verify_bindings(root,out)
    if (out/'execution.json').exists():raise FileExistsError('Execution already recorded')
    pressure=fit_one(root,out,c,'pressure')
    fits=1;primary=None
    if pressure['all_quality_gates_passed']:
        for fold in range(3):fit_one(root,out,c,'fold_'+str(fold),fold);fits+=1
        d=pd.concat([pd.read_parquet(out/('fold_'+str(f))/'evaluation.parquet') for f in range(3)],ignore_index=True)
        primary,bad=compare(d);save(out/'primary_report.json',primary);bad.to_parquet(out/'primary_error_changes.parquet',index=False)
    save(out/'execution.json',{'classifier_fits':fits,'pressure_gates_passed':pressure['all_quality_gates_passed'],
        'primary_regression_executed':primary is not None,'primary':primary,'production_replaced':False,
        'stopped_reason':None if pressure['all_quality_gates_passed'] else 'pressure_quality_gate_failed; no primary fits',
        'official_data_only':True,'external_validation':False})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','fit']);a.add_argument('--root',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    arg=a.parse_args();root=arg.root.resolve();out=arg.out.resolve()
    if arg.stage=='prepare':prepare(root,out)
    else:fit(root,out)
