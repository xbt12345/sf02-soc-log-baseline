"""Reload every model, reproduce every saved score, audit real wrapper variants."""
import argparse,gc,json
from pathlib import Path
import joblib,numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import soc_v3_prepare as base
import v37_learning as learning
from v37_contract import CONTRACT,BUDGETS
from run_v36_train import string_codes,save
from run_v37_train import columns
from audit_v37_prepared import variants
from v37_infer import predict,classify

def thresholds(policies):
    result=[]
    for r in policies:
        result.extend([r['empirical_pooled'],r['empirical_worst_source_global']])
        if r['np_iid_diagnostic']['computable']:result.append(r['np_iid_diagnostic']['threshold'])
    return result

def stress(bundle,cases):
    originals=[{'message_sanitized':c['raw_message']} for c in cases]
    decisions=classify(bundle,originals);a=decisions['probabilities'];b=predict(bundle,[dict(r,timestamp='2099',product_name='NEW',vendor_name=None,label_binary='bad',event_id='new',pipeline='new',src_port=0) for r in originals])
    check={'outer_metadata_exact':bool(np.array_equal(a,b))};count=0;maxdelta=0.;failures=[]
    check['primary_inference_policy']=bool(np.array_equal(decisions['prediction'],learning.gated_predictions(a,decisions['threshold'])))
    originals=[];changes=[];identities=[]
    for c in cases:
        for name,msg in variants(c['raw_message'],c['route']):
            originals.append({'message_sanitized':c['raw_message']});changes.append({'message_sanitized':msg});identities.append((c['row_position'],name))
    for off in range(0,len(changes),2048):
        a=predict(bundle,originals[off:off+2048]);b=predict(bundle,changes[off:off+2048]);diff=abs(a-b).max(1);maxdelta=max(maxdelta,float(diff.max()));count+=len(a)
        passed=(diff<=CONTRACT['equivalent_wrapper_atol'])&(a.argmax(1)==b.argmax(1))
        for t in thresholds(bundle['policies']):passed&=((1-a[:,0]>=t)==(1-b[:,0]>=t))&(learning.gated_predictions(a,t)==learning.gated_predictions(b,t))
        failures.extend({'row_position':identities[off+i][0],'variant':identities[off+i][1],'delta':float(diff[i])} for i in np.flatnonzero(~passed))
    # Control is expected to expose the old bug, so only repaired-model
    # equivalence failures fail the run. Both branches retain all evidence.
    if bundle['view']=='B2_REPAIRED':check['all_equivalent_wrappers']=not failures
    return {'checks':check,'cases':len(cases),'wrapper_pairs':count,'max_probability_difference':maxdelta,
            'failure_count':len(failures),'failures':failures[:100],'control_failures_are_diagnostic':bundle['view']=='B2_CONTROL'}

def regression(before,after):
    flags=[];criteria=CONTRACT['regression_flags']
    for context in ['all']+sorted(set(before['sources'])|set(after['sources'])):
        b=before if context=='all' else before['sources'].get(context)
        a=after if context=='all' else after['sources'].get(context)
        if a is None or b is None:flags.append({'slice':context,'reason':'unmatched source'});continue
        assert a['classification']['class_support']==b['classification']['class_support']
        for c,(br,ar) in enumerate(zip(b['classification']['class_recall'],a['classification']['class_recall'])):
            if br is not None and ar is not None and br-ar>criteria['class_recall_absolute_drop']:flags.append({'slice':context,'reason':'class_recall_drop','class':c,'old':br,'new':ar})
        if b['risk_ranking']['supported'] and a['risk_ranking']['supported']:
            br=b['risk_ranking']['average_precision'];ar=a['risk_ranking']['average_precision']
            if br-ar>criteria['risk_AP_absolute_drop']:flags.append({'slice':context,'reason':'risk_AP_drop','old':br,'new':ar})
        bp=next(p for p in b['policies'] if p['alpha']==.001 and p['name']=='empirical_worst_source_global')
        ap=next(p for p in a['policies'] if p['alpha']==.001 and p['name']=='empirical_worst_source_global')
        for c,(br,ar) in enumerate(zip(bp['risk']['class_alert_rates'],ap['risk']['class_alert_rates'])):
            if br is None or ar is None:continue
            if c==0 and ar-br>criteria['normal_alert_rate_absolute_increase']:flags.append({'slice':context,'reason':'normal_alert_increase','old':br,'new':ar})
            if c>0 and (br-ar>criteria['class_recall_absolute_drop'] or br>0 and ar==0):flags.append({'slice':context,'reason':'risk_detection_drop','class':c,'old':br,'new':ar})
    return flags

def run(args):
    prepared=Path(args.prepared);root=Path(args.results);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    cases=json.loads(Path(args.cases).read_text(encoding='utf-8'))
    # Audit cases contain representative groups. Add every real WAF row.
    wanted=[];off=0
    for b in pq.ParquetFile(args.train).iter_batches(batch_size=4096,columns=['message_sanitized','product_name'],use_threads=False):
        for j,r in enumerate(b.to_pylist()):
            if r['product_name']=='Barracuda WAF':wanted.append({'row_position':off+j,'raw_message':r['message_sanitized'],'route':'cef_fields'})
        off+=len(b)
    assert len(wanted)==4130
    case_map={c['row_position']:c for c in cases};case_map.update({c['row_position']:c for c in wanted});cases=list(case_map.values())
    product=np.asarray(pq.read_table(prepared/'prepared.parquet',columns=['product'])['product'].to_pylist(),object)
    gt=pq.read_table(prepared/'groups.parquet');groups=gt['union_group'].to_numpy();labels=gt['label_index'].to_numpy()
    proto=pq.read_table(prepared/'protocol.parquet');all_results=[];reports={};incomplete_receipts=[]
    for complete in sorted(root.glob('*/complete.json')):
        folder=complete.parent
        try:identity=json.loads(complete.read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            incomplete_receipts.append(str(complete.relative_to(root)));continue
        for name,sha in identity['artifacts'].items():assert base.file_hash(folder/name)==sha
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'));bundle=joblib.load(folder/'model.joblib');checks={}
        task=report['task'];view=report['view'];key=(task,view)
        if key in reports:raise ValueError('Duplicate completed model')
        reports[key]=report;tcname,fcname,_=columns(view);best=None
        for item in report['candidates']:
            t=pq.read_table(folder/('selection_C'+str(item['C'])+'.parquet'));pos=t['row_position'].to_numpy()
            assert np.array_equal(pos,np.flatnonzero(proto[task].to_numpy()==1))
            p=np.column_stack([t[n].to_numpy() for n in ['p_benign','p_malicious','p_suspicious']]);y=t['label_index'].to_numpy()
            assert np.array_equal(y,labels[pos])
            score=learning.cm_metrics(y,p.argmax(1))['macro_f1'];assert abs(score-item['selection_macro_f1'])<1e-12
            if best is None or score>best[1]+.001:best=(item['C'],score)
        checks['selection_recomputed']=best[0]==report['selected_C'];rows_replayed=0
        for role,r in [('calibration',2),('evaluation',3)]:
            t=pq.read_table(folder/(role+'.parquet'));pos=t['row_position'].to_numpy();y=t['label_index'].to_numpy()
            assert np.array_equal(pos,np.flatnonzero(proto[task].to_numpy()==r)) and np.array_equal(y,labels[pos])
            saved=np.column_stack([t[n].to_numpy() for n in ['p_benign','p_malicious','p_suspicious']])
            texts,tc=string_codes(prepared/'prepared.parquet',pos,tcname);facts,fc=string_codes(prepared/'prepared.parquet',pos,fcname)
            pairs,back=np.unique(tc.astype(np.int64)*len(facts)+fc,return_inverse=True)
            x=sparse.hstack([bundle['tfidf'].transform([texts[i] for i in pairs//len(facts)]),bundle['facts'].transform([facts[i] for i in pairs%len(facts)])],format='csr')
            p=bundle['model'].predict_proba(x)[back];rows_replayed+=len(p)
            checks[role+'_probabilities']=bool(np.allclose(saved,p,atol=CONTRACT['probability_replay_atol'],rtol=CONTRACT['probability_replay_rtol']))
            checks[role+'_argmax']=bool(np.array_equal(saved.argmax(1),p.argmax(1)))
            if role=='calibration':
                policies=[learning.calibration_policies(y,p,product[pos],groups[pos],a) for a in BUDGETS]
                checks['thresholds_recomputed']=policies==bundle['policies']==report['calibration_policies']
            else:
                checks['confusion_recomputed']=learning.cm_metrics(y,p.argmax(1))==report['classification']
                for item in report['policies']:
                    checks[item['name']+str(item['alpha'])]=learning.risk_metrics(y,p,item['threshold'])==item['risk'] and learning.cm_metrics(y,learning.gated_predictions(p,item['threshold']))==item['gated_classification']
                primary=next(v for v in bundle['policies'] if v['alpha']==.001)['empirical_worst_source_global']
                pred=learning.gated_predictions(p,primary)
                decision_dir=out.parent/'primary_decisions';decision_dir.mkdir(exist_ok=True)
                decision_file=decision_dir/(task+'_'+view+'.parquet')
                decision_table=pa.table({'row_position':pos.astype(np.int32),'label_index':y,'prediction':pred,
                    'raw_argmax':p.argmax(1).astype(np.uint8),'risk':1-p[:,0],
                    'threshold':np.full(len(p),primary),'alarm':1-p[:,0]>=primary})
                if decision_file.exists():assert pq.read_table(decision_file).equals(decision_table)
                else:pq.write_table(decision_table,decision_file,compression='zstd')
                checks['primary_output_reloaded']=np.array_equal(pq.read_table(decision_file)['prediction'].to_numpy(),pred)
            del x,texts,facts;gc.collect()
        invariance=stress(bundle,cases);checks.update(invariance['checks'])
        all_results.append({'task':task,'view':view,'checks':checks,'all_checks_passed':all(checks.values()),
                            'rows_replayed':rows_replayed,'raw_invariance':invariance,
                            'primary_decisions_file':str(decision_file.relative_to(out.parent)),
                            'primary_decisions_sha256':base.file_hash(decision_file)})
        print(json.dumps({'stage':'replayed','task':task,'view':view,'all_checks_passed':all(checks.values())}),flush=True)
    paired=[]
    for task in sorted({k[0] for k in reports}):
        if (task,'B2_CONTROL') in reports and (task,'B2_REPAIRED') in reports:
            b=reports[(task,'B2_CONTROL')];a=reports[(task,'B2_REPAIRED')]
            paired.append({'task':task,'regression_flags':regression(b,a),
                           'control_classification':b['classification'],'repaired_classification':a['classification'],
                           'control_ranking':b['risk_ranking'],'repaired_ranking':a['risk_ranking']})
    result={'models':all_results,'paired_comparisons':paired,'contract':CONTRACT,'incomplete_receipts_not_accepted':incomplete_receipts,
       'all_checks_passed':bool(all_results) and all(r['all_checks_passed'] for r in all_results),
       'model_quality_accepted':False,'stage_D_status':'C results require review; no automatic target-driven interaction fitting',
       'scope':'Full saved calibration/evaluation score replay; full WAF and sampled other raw wrapper stress; software and internal regression, not external validation.'}
    save(out,result)
    if not result['all_checks_passed']:raise AssertionError('Model replay or repaired wrapper gate failed')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True)
    p.add_argument('--train',required=True);p.add_argument('--cases',required=True);run(p.parse_args())
