"""Read-only cloud-return audit; writes derived review evidence only."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
CLOUD = ROOT / 'artifacts/v36_cloud_20260912T191112Z'
LOCAL = ROOT / 'artifacts/v36_prepared_r13_20260912'

def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p):
    with p.open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def save(name, v): (OUT/name).write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+'\n',encoding='utf-8')
def cm(y,p): return np.bincount(y*3+p,minlength=9).reshape(3,3)
def f1(c):
    d=c.sum(0)+c.sum(1)
    return float(np.divide(2*c.diagonal(),d,out=np.zeros(3),where=d>0).mean())
def alarm(y,p,t): return [int(((1-p[:,0]>=t)&(y==i)).sum()) for i in range(3)]
def threshold(s,b):
    q=np.sort(s)[::-1];return float(np.nextafter(q[min(int(len(q)*b),len(q)-1)],np.inf))

def main():
    checks={}; issued=read(ROOT/'platform/v36_bundle_receipt.json')
    for n,h in issued['source_files'].items(): checks['source:'+n]=sha(CLOUD/'runtime'/n)==h
    checks['official_train_hash']=sha(ROOT/'data/official/train.parquet')==read(CLOUD/'work/run_complete.json')['official_input_sha256']
    gt=pq.read_table(CLOUD/'work/prepared_attempt001/groups.parquet')
    lg=pq.read_table(LOCAL/'groups.parquet'); checks['full_group_contents_match_local']=gt.equals(lg)
    prot=pq.read_table(CLOUD/'work/prepared_attempt001/protocol.parquet')
    lp=pq.read_table(LOCAL/'protocol.parquet');checks['full_protocol_contents_match_local']=prot.equals(lp)
    yy=gt['label_index'].to_numpy().astype(int);groups=gt['union_group'].to_numpy()
    labels=pq.read_table(ROOT/'data/official/train.parquet',columns=['label_binary'])['label_binary'].to_pylist()
    original_y=np.array([{'benign':0,'malicious':1,'suspicious':2}[v] for v in labels]);del labels
    checks['all_labels_match_original']=bool(np.array_equal(yy,original_y))
    products=np.asarray(pq.read_table(LOCAL/'prepared.parquet',columns=['product'])['product'].to_pylist(),dtype=object)
    empty=pq.read_table(LOCAL/'prepared.parquet',columns=['original_empty'])['original_empty'].to_numpy()
    replay=read(CLOUD/'work/model_replay.json')
    rows=[];prob_rows=0;content_hashes={};threshold_details=[]
    for folder in sorted((CLOUD/'work/models').iterdir()):
        if not folder.is_dir():continue
        name=folder.name;a=read(folder/'report.json');b=read(folder/'binding.json');complete=read(folder/'complete.json');task=a['task']
        checks[name+':binding']=complete['binding']==b
        for n,h in complete['artifacts'].items():checks[name+':'+n]=sha(folder/n)==h
        roles=prot[task].to_numpy(); checks[name+':role_hash']=hashlib.sha256(roles.tobytes()).hexdigest()==b['roles_sha256']
        rt=pq.read_table(folder/'roles.parquet');positions=rt['row_position'].to_numpy()
        checks[name+':roles']=bool(np.array_equal(positions,np.flatnonzero(roles>=0)) and np.array_equal(rt['role'].to_numpy(),roles[positions]))
        lo=np.full(groups.max()+1,127);hi=np.full(groups.max()+1,-127);m=roles>=0
        np.minimum.at(lo,groups[m],roles[m]);np.maximum.at(hi,groups[m],roles[m])
        checks[name+':group_isolation']=bool(np.all(lo[lo!=127]==hi[lo!=127]))
        tables={};select=[]
        for fn,role in [('selection_C0.1',1),('selection_C1.0',1),('calibration',2),('evaluation',3)]:
            t=pq.read_table(folder/(fn+'.parquet'));pos=t['row_position'].to_numpy();y=t['label_index'].to_numpy().astype(int)
            p=np.column_stack([t[n].to_numpy() for n in ['p_benign','p_malicious','p_suspicious']])
            checks[name+':'+fn+'_identity']=bool(np.array_equal(pos,np.flatnonzero(roles==role)) and np.array_equal(y,yy[pos]))
            checks[name+':'+fn+'_probability']=bool(np.isfinite(p).all() and (p>=0).all() and (p<=1).all() and np.allclose(p.sum(1),1,atol=1e-12))
            if 'prediction' in t.column_names:checks[name+':'+fn+'_decision']=bool(np.array_equal(t['prediction'].to_numpy(),p.argmax(1)))
            tables[fn]=(pos,y,p);prob_rows+=len(y)
            if fn.startswith('selection'):select.append((float(fn.split('C')[1]),f1(cm(y,p.argmax(1)))))
        best=select[1] if select[1][1]>select[0][1]+.001 else select[0]
        checks[name+':selected_C']=best[0]==a['selected_C']
        checks[name+':selection_scores']=all(abs(s-i['selection_macro_f1'])<1e-12 for (c,s),i in zip(select,a['candidates']))
        pos,y,p=tables['evaluation'];c=cm(y,p.argmax(1));checks[name+':confusion']=c.tolist()==a['all_rows']['confusion_matrix']
        cy,cp=tables['calibration'][1:];calpos=tables['calibration'][0];risk=1-p[:,0]
        for item,cal in zip(a['risk'],a['calibration']):
            th=threshold(1-cp[cy==0,0],item['budget'])
            checks[name+':threshold_'+str(item['budget'])]=th==item['threshold']==cal['threshold']
            checks[name+':alarms_'+str(item['budget'])]=alarm(y,p,th)==item['class_alert_counts'] and alarm(cy,cp,th)==cal['class_alert_counts']
        r=next(v for v in replay['models'] if v['task']==task and v['view']==a['view'] and v['weighted']==b['repeat_weighting'])
        worst=next(v for v in r['worst_calibration_source_threshold_diagnostic'] if v['budget']==.001)
        sources=[]
        for src in np.unique(products[calpos][cy==0]):
            m=(cy==0)&(products[calpos]==src);th=threshold(1-cp[m,0],.001)
            sources.append({'source':str(src),'normal_rows':int(m.sum()),'nonempty_groups':int(len(np.unique(groups[calpos[m&~empty[calpos]]]))),'threshold':th})
        th=max(v['threshold'] for v in sources)
        checks[name+':worst_calibration_policy']=th==worst['threshold'] and alarm(y,p,th)==worst['evaluation']['class_alert_counts']
        gated=np.where(risk>=th,1+p[:,1:].argmax(1),0)
        target=[]
        for cls in range(3):
            mask=y==cls
            target.append({'class':cls,'rows':int(mask.sum()),'risk_quantiles':np.quantile(risk[mask],[0,.01,.5,.9,.99,1]).tolist() if mask.any() else None,
                           'groups':int(len(np.unique(groups[pos[mask]])))})
        ordering={}
        for cls in [1,2]:
            if np.any(y==cls) and np.any(y==0):
                cutoff=float(risk[y==cls].min());ordering[str(cls)]={'normal_scores_at_least_min_target_score':int((risk[y==0]>=cutoff).sum()),'target_min_score':cutoff,'diagnostic_only_uses_evaluation_labels':True}
        counts=np.bincount(y,minlength=3)
        row={'model':name,'task':task,'C':a['selected_C'],'class_counts':counts.tolist(),'confusion':c.tolist(),'macro_f1':a['all_rows']['macro_f1'],
             'group_macro_f1':a['group_equal_nonempty']['macro_f1'],'format_only':a['format_only_control'],'risk_001':a['risk'][1],
             'worst_calibration_001':worst['evaluation'],'exploratory_risk_then_subtype_confusion':cm(y,gated).tolist(),
             'risk_AP':float(average_precision_score(y!=0,risk)) if (y==0).any() and (y!=0).any() else None,
             'risk_ROC_AUC':float(roc_auc_score(y!=0,risk)) if (y==0).any() and (y!=0).any() else None,
             'class_score_ranges':target,'ordering_diagnostic':ordering,'role_class_counts':a['role_class_counts'],'weighted_class_mass':a['final_fit'].get('weighted_class_mass'),
             'elapsed_seconds':a['elapsed_seconds'],'single_fact_deletion':r['single_fact_deletion'],'fact_channel_deletion':a.get('fact_channel_deletion')}
        rows.append(row);threshold_details.append({'model':name,'sources':sources})
        print(json.dumps({'model':name,'cm':c.tolist(),'risk001':a['risk'][1]['class_alert_counts'],'worst001':worst['evaluation']['class_alert_counts'],'AP':row['risk_AP'],'order':ordering},ensure_ascii=False),flush=True)
    checks['expected_model_set']=set(r['model'] for r in rows)=={t+'_'+v for t in ['known_dev','source_ad','source_duo','source_waf','asa_hard'] for v in ['B0','B2','B2_W']}
    checks['all_converged']=all(read(f/'report.json')['final_fit']['converged'] and all(i['fit']['converged'] for i in read(f/'report.json')['candidates']) for f in (CLOUD/'work/models').iterdir() if f.is_dir())
    save('independent_return_audit.json',{'checks':checks,'all_checks_passed':all(checks.values()),'probability_records_checked_including_selection':prob_rows,
        'scope':'Independent hashes, labels, roles, probability validity, decisions, metrics, thresholds and calibration-only worst-source policy; not model replay or external safety guarantee.'})
    save('model_comparison.json',rows);save('calibration_source_support.json',threshold_details)
    print('ALL_CHECKS_PASSED',all(checks.values()),'FAILED',[k for k,v in checks.items() if not v],flush=True)
if __name__=='__main__':main()
