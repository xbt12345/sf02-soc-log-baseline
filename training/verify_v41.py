"""Saved-model replay and raw input counterfactuals, separate from quality gates."""
import argparse
import collections
import gc
import json
import re
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pyarrow.parquet as pq


def main(a):
    start=time.perf_counter();root=Path(a.root).resolve();run=Path(a.run).resolve();out=Path(a.out).resolve();assert not out.exists()
    sys.path.insert(0,str(run/'frozen_training_runtime'))
    import v41_core as core
    import v41_native_auth as native
    from run_v39_prepare import sha,save
    from run_v40_train import metrics
    from audit_v37_prepared import variants
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    stage=run/'prepared';receipt=read(stage/'complete.json');prep=Path(receipt['v39_prepared']);verified={}
    def check(p,h):
        assert sha(p)==h,str(p);verified[p.relative_to(root).as_posix()]=h
    check(root/'data/official/train.parquet',receipt['official_sha256'])
    check(prep/'complete.json',receipt['v39_prepared_receipt_sha256'])
    for folder,files in [(stage,receipt['files']),(run/'frozen_training_runtime',receipt['runtime_sources']),(prep,read(prep/'complete.json')['files'])]:
        for n,h in files.items():check(folder/n,h)
    for name,h in read(stage/'input_audit.json')['source_bindings'].items():check(root/name,h)
    iso=read(run/'isolation_checks.json');assert iso['all_checks_passed'];check(stage/'complete.json',iso['prepared_sha256'])
    selection=read(run/'primary_review/selection.json');check(run/'primary_review/primary_review.json',selection['primary_review_sha256'])
    check(stage/'configuration.json',selection['configuration_sha256']);check(run/'frozen_training_runtime/review_v41.py',selection['script_sha256'])
    folders=sorted((run/'primary').glob('fold_*/*/model.joblib'));assert len(folders)==12
    assert selection['selected_view'] is None or selection['proceed_to_stress']
    # Every repaired native record joins the existing real source/clock probes.
    oldcases=root/'artifacts/v40_local_r1_20260913/raw_inference_cases.json'
    cases=read(oldcases);check(oldcases,sha(oldcases));existing={c['row_position'] for c in cases}
    mapping=pq.read_table(stage/'A_updates.parquet').to_pandas();positions=set(mapping.row_position)-existing
    rows=pq.read_table(prep/'rows.parquet',columns=['row_position','projection_id','route']).to_pandas()
    off=0
    for batch in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=8192,columns=['message_sanitized'],use_threads=False):
        for pos in sorted(positions.intersection(range(off,off+len(batch)))):
            r=rows.iloc[pos];cases.append({'row_position':pos,'projection_id':int(r.projection_id),'route':r.route,'raw':batch.column(0)[pos-off].as_py() or ''})
        off+=len(batch)
    pr=pq.read_table(prep/'projections.parquet',columns=['text','facts']).to_pandas();ap=pq.read_table(stage/'A_projections.parquet').to_pandas()
    amap=dict(zip(mapping.row_position,mapping.model_projection_id))
    records=[];expected=[];counts=collections.Counter();representative=[]
    # Parse every original and counterfactual once per parser, then apply EVERY
    # serialized model to the actual projected rows; also replay public API below.
    projected={'B':[],'A':[]}; originals={'B':[],'A':[]}
    for i,item in enumerate(cases):
        raw=item['raw'];pid=item['projection_id'];base=core.prior.prepare_record({'message_sanitized':raw});new=native.prepare_record({'message_sanitized':raw})
        assert base['text']==pr.text.iloc[pid] and core.prior.canonical(base['facts'])==pr.facts.iloc[pid]
        if item['row_position'] in amap:
            j=amap[item['row_position']];assert new['text']==ap.text.iloc[j] and core.prior.canonical(new['facts'])==ap.facts.iloc[j]
        originals['B'].append(base);originals['A'].append(new)
        rows_for_case=[('original',{'message_sanitized':raw}),('outside_metadata_replaced',{'message_sanitized':raw,
            'timestamp':'2099-01-01T00:00:00Z','product_name':None,'vendor_name':'unseen_vendor','src_ip':'203.0.113.255',
            'dst_ip':'192.0.2.1','username':'unseen_identity','event_id':'replacement','pipeline':'replacement','src_port':65535,
            'src_host':'new_host','dst_host':'new_host','label_binary':'suspicious'})]
        if item['route']=='clock_probe':
            changes=[('absolute_clock',core.prior.TASK_BOUNDARY.sub(lambda m:m[1]+clock+m[3],core.prior.ISO_LITERAL.sub(clock,raw))) for clock in ['2099-01-01T00:00:00Z','2011-12-31T23:59:59Z']]
        else:
            changes=variants(raw,item['route'])
            if item['route']=='asa':changes.append(('address_interface',re.sub(r'dmz[-_]\d+','dmz-999',re.sub(r'(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)','203.0.113.99',raw))))
        rows_for_case += [(n,{'message_sanitized':r}) for n,r in changes]
        for name,r in rows_for_case:
            records.append(r);expected.append(i);counts[name]+=1
            for key,parser,ref in [('B',core.prior.prepare_record,base),('A',native.prepare_record,new)]:
                p=parser(r);assert p['text']==ref['text'] and p['facts']==ref['facts'],(item['row_position'],name,key)
                projected[key].append(p)
        if i<50 or item['row_position'] in amap or item['route']=='clock_probe':representative.extend(range(len(records)-len(rows_for_case),len(records)))
    expected=np.array(expected);model_results=[];targets=folders+[p for p in (run/'old_protocol_stress').glob('model.joblib')]
    for model_path in targets:
        folder=model_path.parent;rec=read(folder/'complete.json')
        for n,k in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256'),('report.json','report_sha256'),('binding.json','binding_sha256')]:check(folder/n,rec[k])
        binding=read(folder/'binding.json');assert binding['runtime_sources']==receipt['runtime_sources'];check(stage/'complete.json',binding['prepared_sha256'])
        model=joblib.load(model_path);key='A' if model['view']=='A' else 'B'
        orig=originals[key];ref=model['model'].predict_proba(core.matrix(model,[p['text'] for p in orig],[p['facts'] for p in orig]))
        rawdiff=0.;flips=0
        for lo in range(0,len(records),512):
            p=projected[key][lo:lo+512];prob=model['model'].predict_proba(core.matrix(model,[v['text'] for v in p],[v['facts'] for v in p]))
            q=ref[expected[lo:lo+512]];rawdiff=max(rawdiff,float(np.abs(prob-q).max()));flips+=int((prob.argmax(1)!=q.argmax(1)).sum())
        api_diff=0.
        for lo in range(0,len(representative),512):
            ix=np.array(representative[lo:lo+512]);prob=core.predict_records(model,[records[j] for j in ix])
            api_diff=max(api_diff,float(np.abs(prob-ref[expected[ix]]).max()))
        assert rawdiff<=1e-10 and api_diff<=1e-10 and flips==0
        d=pq.read_table(folder/'evaluation.parquet').to_pandas();pids,inv=np.unique(d.model_projection_id.to_numpy(),return_inverse=True)
        source=ap if key=='A' else pr
        prob=model['model'].predict_proba(core.matrix(model,source.text.iloc[pids].tolist(),[json.loads(v) for v in source.facts.iloc[pids]]))[inv]
        difference=float(np.abs(prob-d[['p_benign','p_malicious','p_suspicious']].to_numpy()).max());assert difference<=1e-10
        assert metrics(d.label_index.to_numpy(dtype=int),prob)==read(folder/'report.json')['evaluation']
        result={'view':model['view'],'fold':binding['fold'],'evaluation_rows':len(d),'evaluation_probability_max_difference':difference,
                'counterfactual_rows':len(records),'raw_probability_max_difference':rawdiff,'public_inference_api_rows':len(representative),
                'public_api_probability_max_difference':api_diff,'prediction_flips':flips,'all_metrics_match':True}
        model_results.append(result);print(json.dumps(result),flush=True);del model,d,prob;gc.collect()
    out.mkdir(parents=True);save(out/'verification.json',{'all_checks_passed':True,'model_checks':model_results,'verified_files':verified,
        'raw_original_cases':len(cases),'counterfactual_counts':dict(counts),'new_primary_fits':len(folders),'new_stress_fits':len(targets)-len(folders),
        'quality_accepted':False,'scope':'Byte identity, complete evaluation replay, specified metadata/time/identity perturbations, not migration or blind acceptance.',
        'seconds':time.perf_counter()-start,'script_sha256':sha(__file__)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);main(p.parse_args())
