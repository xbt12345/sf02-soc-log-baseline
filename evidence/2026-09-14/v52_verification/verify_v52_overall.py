"""Independent count/probability replay and bounded post-fit capability diagnosis."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import confusion_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/2026-09-14/v52_verification'
RUN=ROOT/'artifacts/v52_overall_pressure_20260914'
AUDIT=ROOT/'evidence/2026-09-14/v52_overall'
FIELDS=['action','outcome','transport_protocol','src_role','dst_role','src_port_fixed','dst_port_fixed','src_port_range','dst_port_range','icmp_type','icmp_code','icmp_message','icmp_unreachable']
MAXIMUM={'src_port_fixed':65536,'dst_port_fixed':65536,'icmp_type':256,'icmp_code':256}
VIEWS={'full':[],'no_src':['src_port_fixed','src_port_range'],'no_dst':['dst_port_fixed','dst_port_range'],'no_ports':['src_port_fixed','src_port_range','dst_port_fixed','dst_port_range']}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,x):(OUT/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')

def observed(f):
    out={}
    for k in FIELDS:
        v=f.get(k)
        if v is None or v=='':continue
        if k in MAXIMUM and (not isinstance(v,int) or isinstance(v,bool) or not 0<=v<MAXIMUM[k]):continue
        out[k]=str(v)
    return out

def threshold_capacity(y,q,baseM,baseS):
    # Evaluation-label oracle, ONLY asks whether any cut can meet the two error counts.
    d=pd.DataFrame({'q':q,'m':y==1,'s':y==2}).groupby('q')[['m','s']].sum().sort_index()
    m=np.r_[0,d.m.cumsum().to_numpy()];s=int(d.s.sum())-np.r_[0,d.s.cumsum().to_numpy()]
    feasible=(m<=baseM)&(s<=baseS)
    return {'status':'EVALUATION_LABEL_ORACLE_NOT_A_LEARNED_THRESHOLD',
        'any_threshold_meets_base_class_error_limits':bool(feasible.any()),
        'minimum_errors_within_constraints':int((m+s)[feasible].min()) if feasible.any() else None,
        'no_threshold_selected_saved_or_applied':True}

def main():
    assert not OUT.exists();OUT.mkdir(parents=True)
    checked=0
    for folder,receipt,key in [(RUN,'complete.json','bindings'),(AUDIT,'receipt.json','files')]:
        for p,digest in read(folder/receipt)[key].items():assert sha(folder/p)==digest,p;checked+=1
    cfg=read(RUN/'configuration.json')
    for p,digest in cfg['bindings'].items():assert sha(ROOT/p)==digest,p;checked+=1
    for p,digest in read(AUDIT/'summary.json')['bindings'].items():assert sha(ROOT/p)==digest,p;checked+=1
    rows=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/rows.parquet')
    projections=pd.read_parquet(ROOT/'artifacts/v48_information_repair_r2_20260914/projections.parquet')
    a=rows[rows.route=='asa'].sort_values('row_position').reset_index(drop=True)
    manifest=pd.read_parquet(RUN/'split_manifest.parquet')
    pd.testing.assert_frame_equal(a,manifest)
    facts={i:observed(json.loads(projections.iloc[int(i)].facts)) for i in a.projection_id.unique()}
    fit=a[a.inner_role!=2];ev=a[a.inner_role==2].reset_index(drop=True)
    assert set(fit.body_group).isdisjoint(ev.body_group)
    assert set(fit.union_group).isdisjoint(ev.union_group)
    e=pd.read_parquet(RUN/'pressure_evaluation.parquet');total=0;maxdiff=0.;modelchecks=[]
    for name,interaction in [('R',False),('I',True)]:
        for view,drop in VIEWS.items():
            model=joblib.load(RUN/(name+'_'+view+'.joblib'))
            def rec(pid):
                f={k:v for k,v in facts[pid].items() if k not in drop}
                if interaction and all(k in f for k in ['transport_protocol','src_role','dst_role']):
                    f['protocol_roles']=json.dumps([f[k] for k in ['transport_protocol','src_role','dst_role']],separators=(',',':'))
                return f
            vocab={k+'='+v for i in fit.projection_id.unique() for k,v in rec(i).items()}
            assert set(model[0].get_feature_names_out())==vocab
            for scenario,d in e.groupby('scenario'):
                d=d[d.view==view]
                if d.empty:continue
                xx=model[0].transform([rec(i) for i in d.projection_id])
                # Logistic formula replay independent of candidate prediction helper.
                q=expit(np.asarray(xx@model[-1].coef_.T).ravel()+model[-1].intercept_[0])
                diff=float(np.max(abs(q-d[name+'_qM'].to_numpy())))
                assert diff<=1e-12
                total+=len(d);maxdiff=max(maxdiff,diff)
            modelchecks.append({'name':name,'view':view,'vocabulary_fit_only':True,'classes':model[-1].classes_.tolist()})
    summary=read(AUDIT/'summary.json');diagnosis=pd.read_parquet(AUDIT/'asa_row_diagnosis.parquet')
    assert len(diagnosis)==106953 and int(diagnosis.error.sum())==5972
    # Separate dictionary counter instead of pandas crosstab implementation.
    counts={}
    for f,k,y in zip(diagnosis.fold,diagnosis.input_key,diagnosis.label_index):
        item=counts.setdefault((int(f),int(k)),[0,0]);item[int(y)-1]+=1
    floor=sum(min(z) for z in counts.values());assert floor==summary['ASA_finite_input_floor_total']==1970
    for k,g in diagnosis.groupby('support_category'):
        assert int(g.error.sum())==summary['ASA_support_partition'][k]['errors']
    base=pd.read_parquet(ROOT/'artifacts/v51_fact_residual_20260914/pressure/evaluation.parquet')
    base=base[base.route=='asa'].sort_values('row_position').reset_index(drop=True)
    natural=e[e.scenario=='original'].sort_values('row_position').reset_index(drop=True)
    np.testing.assert_array_equal(base.row_position,natural.row_position)
    bpred=base[['p_benign','p_malicious','p_suspicious']].to_numpy().argmax(1);y=base.label_index.to_numpy()
    primary=pd.read_parquet(RUN/'paired_primary_ASA.parquet')
    report=read(RUN/'report.json');cap={};changes={}
    for name in ['R','I']:
        pred=np.where(natural[name+'_qM']>=.5,1,2)
        assert confusion_matrix(y,pred,labels=[0,1,2]).tolist()==report['pressure'][name]['confusion_B_M_S']
        cap[name]=threshold_capacity(y,natural[name+'_qM'].to_numpy(),142,685)
        z=natural[['projection_id','body_group','label_index']].copy();z['before_error']=bpred!=y;z['after_error']=pred!=y
        change=z.groupby(['projection_id','label_index']).agg(rows=('label_index','size'),before_errors=('before_error','sum'),after_errors=('after_error','sum'),body_groups=('body_group','nunique')).reset_index()
        change=change[change.before_errors!=change.after_errors]
        changes[name]=change.astype(int).to_dict(orient='records')
        pm=primary[name+'_pred'].to_numpy()
        assert confusion_matrix(primary.label_index,pm,labels=[0,1,2]).tolist()==report['primary_ASA'][name]['confusion_B_M_S']
    # Proposed behavioral network scope: audit class support before any new model design.
    ff=[json.loads(s) for s in projections.facts]
    gate=np.array([f.get('action') in {'allow','deny','open','close'} and f.get('transport_protocol') in {'tcp','udp','icmp','icmp6'} for f in ff])
    net=rows[gate[rows.projection_id.to_numpy()]].copy()
    support=[]
    for (role,route,label),g in net.groupby(['inner_role','route','label_index']):
        support.append({'role':int(role),'route':route,'label':int(label),'rows':len(g),'bodies':int(g.body_group.nunique())})
    save('network_behavior_support.json',{'gate':'actual action in allow/deny/open/close AND transport tcp/udp/icmp/icmp6; audit only, not new model routing','rows':len(net),'class_counts_B_M_S':[int((net.label_index==k).sum()) for k in range(3)],'by_role_route_label':support})
    save('pressure_changes.json',changes);save('threshold_capacity_diagnostic_only.json',cap)
    save('verification.json',{'file_hash_checks':checked,'models_checked':modelchecks,'probability_values_replayed':total,'probability_max_absolute_diff':maxdiff,
        'fit_and_eval_body_and_union_disjoint':True,'original_split_rows_labels_unchanged':True,'finite_same_input_error_floor_independent_count':floor,
        'classification_counts_recomputed':True,'scope':'Verifies fixed development execution and artifacts. No blind test, normal-boundary validation, learned thresholds or real-world calibration.'})
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save('receipt.json',{'files':{p.name:sha(p) for p in OUT.iterdir() if p.is_file()}})
    print(json.dumps({'probability_values_replayed':total,'maxdiff':maxdiff,'threshold_capacity':cap,'network_behavior_class_counts':[int((net.label_index==k).sum()) for k in range(3)]}),flush=True)

if __name__=='__main__':main()
