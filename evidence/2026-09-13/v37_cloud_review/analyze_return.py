"""Read-only v3.7 return diagnostics; no fits or target-derived deployed decisions."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import confusion_matrix, average_precision_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CLOUD = ROOT / 'artifacts/v37_cloud_20260913T072309Z/work'
PREP = ROOT / 'artifacts/v37_prepared_r13_20260913'
sys.path.insert(0, str(ROOT / 'artifacts/v37_cloud_20260913T072309Z/runtime'))
import v37_learning as learning

def save(name, data):
    (OUT/name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def quant(a):
    return dict(zip(['min','p01','median','p99','max'],np.quantile(a,[0,.01,.5,.99,1]).tolist())) if len(a) else None

def policy(r):
    return next(p for p in r['policies'] if p['name']=='empirical_worst_source_global' and p['alpha']==.001)

def main():
    meta=pq.read_table(PREP/'prepared.parquet', columns=['row_position','label_index','product','route','b1','facts','supported','original_empty','quality']).to_pandas()
    groups=pq.read_table(PREP/'groups.parquet')['union_group'].to_numpy()
    proto=pq.read_table(PREP/'protocol.parquet').to_pandas()
    cloud_proto=pq.read_table(CLOUD/'prepared_attempt001/protocol.parquet').to_pandas()
    cloud_groups=pq.read_table(CLOUD/'prepared_attempt001/groups.parquet').to_pandas()
    identity={'protocol_equal':proto.equals(cloud_proto),'group_table_equal':pq.read_table(PREP/'groups.parquet').to_pandas().equals(cloud_groups)}
    assert all(identity.values())
    result=[]; slices={}; examples=[]; matrix=[]
    for folder in sorted((CLOUD/'models').iterdir()):
        report=json.loads((folder/'report.json').read_text()); complete=json.loads((folder/'complete.json').read_text())
        hashes={n:hashlib.sha256((folder/n).read_bytes()).hexdigest()==h for n,h in complete['artifacts'].items()}
        assert all(hashes.values())
        ev=pq.read_table(folder/'evaluation.parquet').to_pandas(); cal=pq.read_table(folder/'calibration.parquet').to_pandas()
        dec=pq.read_table(CLOUD/'primary_decisions'/f'{folder.name}.parquet').to_pandas()
        pos=ev.row_position.to_numpy(); y=ev.label_index.to_numpy(); p=ev[['p_benign','p_malicious','p_suspicious']].to_numpy(); risk=1-p[:,0]
        assert np.array_equal(pos,np.flatnonzero(proto[report['task']].to_numpy()==3))
        assert np.array_equal(y,meta.label_index.to_numpy()[pos])
        threshold=policy(report)['threshold']; pred=np.where(risk>=threshold,1+p[:,1:].argmax(1),0)
        assert np.array_equal(pos,dec.row_position) and np.array_equal(pred,dec.prediction)
        cm=confusion_matrix(y,pred,labels=[0,1,2]).tolist(); assert cm==policy(report)['gated_classification']['confusion_matrix']
        cr=1-cal.p_benign.to_numpy(); cy=cal.label_index.to_numpy(); cp=cal.row_position.to_numpy()
        cal_sources=[]
        for s in sorted(meta.iloc[cp]['product'].unique()):
            mask=(meta['product'].to_numpy()[cp]==s)&(cy==0)
            if mask.any(): cal_sources.append({'source':s,'normal_rows':int(mask.sum()),'normal_groups':int(len(np.unique(groups[cp][mask]))),'scores':quant(cr[mask]),'normal_alarm_rows':int((cr[mask]>=threshold).sum())})
        primary=policy(report)
        item={'name':folder.name,'selected_C':report['selected_C'],'primary':primary,'raw_argmax':report['classification'],
              'AP':report['risk_ranking'],'score_ranges_by_label':[quant(risk[y==c]) for c in range(3)],
              'calibration_normal':quant(cr[cy==0]),'calibration_sources':cal_sources,
              'support':{k:v for k,v in report['review_support'].items() if k!='fit_relationship_class_support'},
              'hashes':hashes,'iterations':report['final_fit']['iterations']}
        slices[folder.name]={k:{'primary':policy(v),'argmax':v['classification']} for k,v in report['slices'].items()}
        # Eval-label-based separation is retrospective diagnosis, NEVER a selected threshold.
        if (y==0).any() and (y!=0).any():
            item['retrospective_separation_only']={'max_normal':float(risk[y==0].max()),'min_risk_class':float(risk[y!=0].min()),'strictly_separable':bool(risk[y==0].max()<risk[y!=0].min()),'not_a_deployable_threshold':True}
        if report['view']=='B2_REPAIRED':
            route_rows=[]; ep=meta.iloc[pos].copy(); ep['pred']=pred; ep['alarm']=risk>=threshold; ep['risk']=risk
            ep['group']=groups[pos]
            for (route,label), d in ep.groupby(['route','label_index'],observed=True):
                route_rows.append({'route':route,'label':int(label),'rows':len(d),'groups':int(d.group.nunique()),'predictions':np.bincount(d.pred,minlength=3).tolist(),'risk':quant(d.risk.to_numpy())})
            item['route_metrics']=route_rows
            bad=ep[ep.label_index!=ep.pred]
            for _,b in bad.groupby(['route','label_index','pred'],observed=True):
                b=b.drop_duplicates(['b1','facts']).head(3)
                for row in b.to_dict('records'):
                    examples.append({'task':report['task'],**{k:row[k] for k in ['row_position','label_index','pred','product','route','risk','b1','facts','quality']}})
        result.append(item)
        matrix.append({'task':report['task'],'view':report['view'],'macro_f1':primary['gated_classification']['macro_f1'],
             'class_support':primary['risk']['class_support'],'alarm_counts':primary['risk']['class_alert_counts'],
             'class_correct':np.diag(cm).tolist(),'threshold':threshold,'AP':report['risk_ranking'].get('average_precision')})
    save('independent_score_analysis.json',{'identity':identity,'models':result,'scope':'Saved-score and labels independent recomputation. Model re-execution recorded separately; no training.'})
    save('slice_analysis.json',slices);save('primary_matrix.json',matrix);save('error_examples.json',examples)
    # All official training-side coverage; no target-dependent relabelling.
    counts=meta.groupby(['route','label_index'],observed=True).size().unstack(fill_value=0)
    save('full_route_labels.json',counts.to_dict(orient='index'))
    print(json.dumps(matrix,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
