"""Recount all fixed OOF rows and choose at most one stress candidate."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from run_v39_prepare import sha,save
from run_v40_train import metrics

LABELS={'benign':0,'malicious':1,'suspicious':2}
PARTS=['fit_pure_same_label','fit_pure_opposite_label','fit_conflicted','unseen_input']


def probs(d):return d[['p_benign','p_malicious','p_suspicious']].to_numpy()


def main(a):
    root=Path(a.root);run=Path(a.run);out=run/'primary_review';assert not out.exists();out.mkdir()
    config=json.loads((run/'prepared/configuration.json').read_text(encoding='utf-8'))
    baseline_run=root/'artifacts/v39_local_r2_20260913'
    tables=[];baseline_folds=[]
    for fold in range(3):
        f=baseline_run/('primary/fold_%s/SEMANTIC'%fold)
        receipt=json.loads((f/'complete.json').read_text(encoding='utf-8'))
        assert sha(f/'evaluation.parquet')==receipt['predictions_sha256']
        b=pq.read_table(f/'evaluation.parquet').to_pandas();b['fold']=fold
        r=pq.read_table(run/('primary/fold_%s/R/evaluation.parquet'%fold),columns=['row_position','baseline_partition','observation_mask','has_parameter','unseen_parameter','has_joint_value','unseen_joint_value','unsupported_joint_value']).to_pandas()
        assert np.array_equal(b.row_position,r.row_position)
        for col in r:
            if col!='row_position':b[col]=r[col].to_numpy()
        tables.append(b)
        baseline_folds.append(metrics(b.label_index.to_numpy(dtype=int),probs(b)))
    b=pd.concat(tables,ignore_index=True).sort_values('row_position').reset_index(drop=True);del tables
    assert len(b)==1378650 and not b.row_position.duplicated().any()
    y=b.label_index.to_numpy(dtype=int);bp=probs(b);bpred=bp.argmax(1);routes=b.route.to_numpy();asa=routes=='asa'
    bpart=b.baseline_partition.to_numpy()
    pr=pq.read_table(baseline_run/'prepared/projections.parquet',columns=['facts']).to_pandas()
    facts=[json.loads(v) for v in pr.facts]
    icmp=np.array([f.get('transport_protocol')=='icmp' for f in facts])[b.projection_id.to_numpy()]
    code13=np.array([f.get('transport_protocol')=='icmp' and f.get('icmp_type')==3 and f.get('icmp_code')==13 for f in facts])[b.projection_id.to_numpy()]
    slices={'ASA':asa,'ASA_ICMP':asa&icmp,'ASA_non_ICMP':asa&~icmp,'ASA_code13':asa&code13,
        'Duo':b['product'].to_numpy()=='Duo','WAF':b['product'].to_numpy()=='Barracuda WAF',
        'WAF_unique_suspicious':(b['product'].to_numpy()=='Barracuda WAF')&(y==2),
        'native_flow':routes=='native_flow','bounded_payload':routes=='bounded_payload'}
    for col in ['unseen_parameter','unseen_joint_value','unsupported_joint_value']:
        slices['ASA_'+col]=asa&b[col].to_numpy(dtype=bool)
    base_metrics=metrics(y,bp)
    result={'baseline':base_metrics,'baseline_folds':baseline_folds,
        'baseline_slices':{k:metrics(y[m],bp[m]) for k,m in slices.items() if m.any()},'candidates':{},'model_bindings':[]}
    all_improvements=[]
    for view in ['R','N','I']:
        tables=[];fold_results=[];fold_asa_delta=[]
        for fold in range(3):
            f=run/('primary/fold_%s/%s'%(fold,view));rec=json.loads((f/'complete.json').read_text(encoding='utf-8'))
            for filename,key in [('model.joblib','model_sha256'),('evaluation.parquet','predictions_sha256'),('report.json','report_sha256'),('binding.json','binding_sha256')]:assert sha(f/filename)==rec[key]
            d=pq.read_table(f/'evaluation.parquet').to_pandas();d['fold']=fold;tables.append(d)
            rep=json.loads((f/'report.json').read_text(encoding='utf-8'));fold_results.append(rep)
            bf=json.loads((baseline_run/('primary/fold_%s/SEMANTIC/report.json'%fold)).read_text(encoding='utf-8'))
            fold_asa_delta.append(rep['routes']['asa']['errors']-bf['routes']['asa']['errors'])
            result['model_bindings'].append({'view':view,'fold':fold,'path':f.relative_to(root).as_posix(),'model_sha256':rec['model_sha256'],'predictions_sha256':rec['predictions_sha256']})
        d=pd.concat(tables,ignore_index=True).sort_values('row_position').reset_index(drop=True);del tables
        assert np.array_equal(d.row_position,b.row_position) and np.array_equal(d.label_index,y)
        assert np.array_equal(d.baseline_partition,bpart)
        p=probs(d);pred=p.argmax(1);m=metrics(y,p)
        byslice={k:{'metrics':metrics(y[s],p[s]),'fixed':int((s&(bpred!=y)&(pred==y)).sum()),'regressed':int((s&(bpred==y)&(pred!=y)).sum()),
                   'class_body_keys':[int(b.loc[s&(y==c),'body_group'].nunique()) for c in range(3)]} for k,s in slices.items() if s.any()}
        parts={}
        for k,name in enumerate(PARTS):
            s=asa&(bpart==k)
            parts[name]={'rows':int(s.sum()),'baseline_errors':int((s&(bpred!=y)).sum()),'new_errors':int((s&(pred!=y)).sum()),
                'fixed':int((s&(bpred!=y)&(pred==y)).sum()),'regressed':int((s&(bpred==y)&(pred!=y)).sum())}
        newparts={name:{'rows':int((asa&(d.new_partition.to_numpy()==k)).sum()),'errors':int((asa&(d.new_partition.to_numpy()==k)&(pred!=y)).sum())} for k,name in enumerate(PARTS)}
        # Match the known single protocol; zero supported rare regressions is
        # the predeclared manual-review conservative decision for this run.
        rare=(routes=='authentication')|((b['product'].to_numpy()=='Barracuda WAF')&(y==2))|(routes=='bounded_payload')
        rare_regressions=int((rare&(bpred==y)&(pred!=y)).sum())
        checks={'two_folds_improve_ASA':sum(v<0 for v in fold_asa_delta)>=2,
            'pooled_ASA_improves':byslice['ASA']['metrics']['errors']<result['baseline_slices']['ASA']['errors'],
            'malicious_recall_within_limit':m['class_recall'][1]>=base_metrics['class_recall'][1]-config['recall_drop_limit'],
            'suspicious_recall_within_limit':m['class_recall'][2]>=base_metrics['class_recall'][2]-config['recall_drop_limit'],
            'normal_false_alerts_within_limit':m['false_alerts_per_10000_normal']<=base_metrics['false_alerts_per_10000_normal']+config['normal_errors_per_10000_increase_limit'],
            'no_previously_correct_critical_rare_row_regressed':rare_regressions==0}
        conditional_slices=[]
        for (route,mask),idx in d.groupby(['route','observation_mask']).indices.items():
            iy=np.asarray(idx);classes=np.unique(y[iy])
            if len(classes)<2:continue
            conditional_slices.append({'route':route,'mask':int(mask),'rows':len(iy),'class_support':np.bincount(y[iy],minlength=3).tolist(),
                'class_body_keys':[int(d.iloc[iy].loc[y[iy]==c,'body_group'].nunique()) for c in range(3)],
                'baseline_errors':int((bpred[iy]!=y[iy]).sum()),'errors':int((pred[iy]!=y[iy]).sum())})
        rr=d.loc[(routes=='authentication')&(y==2),['row_position','body_group','fold','p_benign','p_malicious','p_suspicious','pred_label']].to_dict('records')
        item={'evaluation':m,'fold_ASA_error_changes':fold_asa_delta,'gate':checks,'eligible_for_stress':all(checks.values()),
            'slices':byslice,'fixed_baseline_ASA_partitions':parts,'new_ASA_partitions':newparts,'rare_regressions':rare_regressions,
            'Duo_suspicious_rows':rr,'conditional_observation_slices':conditional_slices,
            'fit_diagnostics':[v['collision'] for v in fold_results],'parameter_support':[v['parameter_support'] for v in fold_results],
            'per_fold':[v['evaluation'] for v in fold_results],'seconds':sum(v['seconds'] for v in fold_results)}
        result['candidates'][view]=item
        if item['eligible_for_stress']:all_improvements.append((byslice['ASA']['metrics']['errors'],m['errors'],['R','N','I'].index(view),view))
        save(out/('candidate_%s.json'%view),item)
        print(json.dumps({'view':view,'macro_f1':m['macro_f1'],'errors':m['errors'],'ASA_errors':byslice['ASA']['metrics']['errors'],'normal_errors':m['normal_errors'],'eligible':all(checks.values()),'gate':checks}),flush=True)
        del d,p;gc.collect()
    # Frozen negative control is replayed as saved decisions under identical rows.
    av=[]
    for fold in range(3):
        folder=root/('artifacts/v39_local_r1_20260913/primary/fold_%s/AVAILABILITY'%fold)
        receipt=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
        assert sha(folder/'evaluation.parquet')==receipt['predictions_sha256']
        av.append(pq.read_table(folder/'evaluation.parquet',columns=['row_position','label_index','p_benign','p_malicious','p_suspicious']).to_pandas())
    av=pd.concat(av,ignore_index=True).sort_values('row_position').reset_index(drop=True)
    assert np.array_equal(av.row_position,b.row_position)
    result['frozen_v39_availability_control']=metrics(y,probs(av))
    result['scope']='All inspected development rows; no new blind or external validation'
    result['new_primary_fits']=9;result['quality_accepted']=False
    save(out/'primary_review.json',result)
    selected=min(all_improvements)[3] if all_improvements else None
    save(out/'selection.json',{'selected_view':selected,'proceed_to_stress':selected is not None,'eligible_views':[v[3] for v in sorted(all_improvements)],
        'rule':config['selection'],'critical_rare_review':'Conservatively reject candidates with any previously correct authentication, WAF suspicious or bounded-payload row regressed.',
        'primary_review_sha256':sha(out/'primary_review.json'),'configuration_sha256':sha(run/'prepared/configuration.json'),
        'script_sha256':sha(__file__),'no_stress_results_used_for_selection':True,'quality_accepted':False})
    print(json.dumps({'selected_view':selected,'proceed_to_stress':selected is not None}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--run',required=True);main(p.parse_args())
