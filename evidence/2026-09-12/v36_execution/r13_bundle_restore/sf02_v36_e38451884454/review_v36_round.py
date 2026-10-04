"""Reload models, replay probabilities and audit single-field dependencies."""
import argparse
import json
from pathlib import Path
import re
import gc

import joblib
import numpy as np
import pyarrow.parquet as pq
from scipy import sparse
from scipy.special import softmax

import soc_v3_prepare as base
import v36_learning as learning
from run_v36_train import string_codes,save
from v36_infer import predict


def run(args):
    prepared=Path(args.prepared);root=Path(args.results);out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    all_results=[]
    cases=json.loads(Path(args.cases).read_text(encoding='utf-8')) if args.cases else []
    product=np.asarray(pq.read_table(prepared/'prepared.parquet',columns=['product'])['product'].to_pylist(),dtype=object)
    for complete in sorted(root.glob('*/complete.json')):
        folder=complete.parent;identity=json.loads(complete.read_text(encoding='utf-8'))
        for f,h in identity['artifacts'].items():
            if base.file_hash(folder/f)!=h:raise AssertionError('Artifact damaged: '+str(folder/f))
        report=json.loads((folder/'report.json').read_text(encoding='utf-8'));bundle=joblib.load(folder/'model.joblib')
        checks={};deletions={};new_thresholds=[];role_tables={}
        if (folder/'selection_C0.1.parquet').exists():
            best=None
            for item in report['candidates']:
                t=pq.read_table(folder/('selection_C'+str(item['C'])+'.parquet'))
                p=np.column_stack([t[n].to_numpy() for n in ['p_benign','p_malicious','p_suspicious']])
                score=learning.cm_metrics(t['label_index'].to_numpy(),p.argmax(1))['macro_f1']
                assert abs(score-item['selection_macro_f1'])<1e-12
                if best is None or score>best[1]+.001:best=(item['C'],score)
            checks['C_selection_recomputed']=best[0]==report['selected_C']
        for role in ['calibration','evaluation']:
            t=pq.read_table(folder/(role+'.parquet'));pos=t['row_position'].to_numpy()
            y=t['label_index'].to_numpy();saved=np.column_stack([t[n].to_numpy() for n in ['p_benign','p_malicious','p_suspicious']])
            texts,tc=string_codes(prepared/'prepared.parquet',pos,'b0' if bundle['view']=='B0' else 'b1')
            x=bundle['tfidf'].transform(texts)
            inverse=tc
            if bundle['view']=='B2':
                facts,fc=string_codes(prepared/'prepared.parquet',pos,'facts')
                keys=tc.astype(np.int64)*len(facts)+fc
                pairs,inverse=np.unique(keys,return_inverse=True)
                x=sparse.hstack([bundle['tfidf'].transform([texts[i] for i in pairs//len(facts)]),
                     bundle['facts'].transform([facts[i] for i in pairs%len(facts)])],format='csr')
            replay=bundle['model'].predict_proba(x)[inverse]
            checks[role+'_probabilities_replayed']=bool(np.allclose(saved,replay,atol=1e-10,rtol=1e-10))
            checks[role+'_decisions_replayed']=bool(np.array_equal(t['prediction'].to_numpy(),replay.argmax(1)))
            role_tables[role]=(pos,y,saved)
            if role=='calibration':
                for item in report['calibration']:
                    threshold=learning.risk_threshold(1-replay[y==0,0],item['budget'])
                    checks['calibration_threshold_'+str(item['budget'])]=threshold==item['threshold']
                # New diagnostic policy fixed by allowed calibration sources:
                # one shared threshold, never target-specific inference routing.
                # No target evaluation labels or probabilities choose it.
                for b in [.0001,.001,.01]:
                    source_values=[]
                    for source in np.unique(product[pos][y==0]):
                        m=(y==0)&(product[pos]==source)
                        source_values.append({'source':str(source),'benign_rows':int(m.sum()),
                          'threshold':learning.risk_threshold(1-replay[m,0],b)})
                    new_thresholds.append({'budget':b,'threshold':max(v['threshold'] for v in source_values),'calibration_sources':source_values})
            else:
                checks['evaluation_confusion_recomputed']=learning.cm_metrics(y,replay.argmax(1))['confusion_matrix']==report['all_rows']['confusion_matrix']
                for item in report['risk']:
                    checks['evaluation_risk_'+str(item['budget'])]=learning.risk_metrics(y,replay,item['threshold'])['class_alert_counts']==item['class_alert_counts']
                if bundle['view']=='B2':
                    fact_names=list(bundle['facts'].names());start=len(bundle['tfidf'].names())
                    field_names=sorted(set(n.split(':',1)[1].split('=',1)[0] for n in fact_names))
                    logits=bundle['model'].decision_function(x)
                    for field in field_names:
                        columns=[start+j for j,n in enumerate(fact_names) if n.split(':',1)[1].split('=',1)[0]==field]
                        influence=x[:,columns].dot(bundle['model'].coef_[:,columns].T)
                        changed=softmax(logits-influence,axis=1)[inverse]
                        deletions[field]={'prediction_flip_rate':float(np.mean(changed.argmax(1)!=replay.argmax(1))),
                            'classification':learning.cm_metrics(y,changed.argmax(1)),
                            'not_label_preserving':True,'does_not_establish_causal_effect':True}
            del x,texts;gc.collect()
        invariance={}
        if cases:
            originals=[{'message_sanitized':c['raw_message']} for c in cases]
            variants=[dict(r,product_name='NEW_PRODUCT',vendor_name=None,timestamp='2099-12-31',src_ip='198.51.100.17',
                       label_binary='not_a_feature',pipeline='NEW_COLLECTOR',event_id='different_id') for r in originals]
            p=predict(bundle,originals);q=predict(bundle,variants)
            invariance['outer_field_cases']=len(cases)
            invariance['outer_field_max_probability_difference']=float(abs(p-q).max())
            checks['real_raw_outer_metadata_predictions_invariant']=bool(np.array_equal(p,q))
        pos,y,p=role_tables['evaluation']
        worst_policy=[]
        for item in new_thresholds:
            worst_policy.append(dict(item,evaluation=learning.risk_metrics(y,p,item['threshold'])))
        all_results.append({'task':report['task'],'view':report['view'],'weighted':identity['binding'].get('repeat_weighting',False),
             'checks':checks,'all_checks_passed':all(checks.values()),'raw_invariance':invariance,
             'single_fact_deletion':deletions,'worst_calibration_source_threshold_diagnostic':worst_policy,
             'diagnostic_policy_not_selected_on_target':True,'target_quality_validated':False})
        print(json.dumps({'stage':'replayed','task':report['task'],'view':report['view'],'all_checks_passed':all(checks.values())}),flush=True)
    if not all_results:raise ValueError('No completed model outputs')
    result={'models':all_results,'all_checks_passed':all(r['all_checks_passed'] for r in all_results),
            'scope':'Every saved calibration/evaluation probability replayed from reloaded models; selection and thresholds recomputed. Does not certify parser correctness for all raw messages or transfer quality.'}
    save(out,result)
    if not result['all_checks_passed']:raise AssertionError('Replay failed')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',required=True);p.add_argument('--results',required=True);p.add_argument('--output',required=True);p.add_argument('--cases')
    run(p.parse_args())
