"""Full matched old-B2/new-B2 training; no downweighting or population sampler."""
import argparse,gc,hashlib,json,platform,time
from pathlib import Path
import joblib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import sparse
import soc_v3_prepare as base
import v36_learning as old
import v37_learning as new
from v37_contract import CONTRACT,TASKS,VIEWS,BUDGETS
from run_v36_train import string_codes,predict_rows,save

def code_hashes():
    # Exact local import closure, shared with bundle builder; no unrelated files.
    import ast
    root=Path(__file__).parent;pending=['run_v37_train','v37_representation'];seen={}
    while pending:
        name=pending.pop()
        if name+'.py' in seen:continue
        p=root/(name+'.py');seen[name+'.py']=base.file_hash(p)
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for n in ast.walk(tree):
            imports=[a.name.split('.')[0] for a in n.names] if isinstance(n,ast.Import) else ([n.module.split('.')[0]] if isinstance(n,ast.ImportFrom) and n.module else [])
            pending.extend(s for s in imports if (root/(s+'.py')).exists() and s+'.py' not in seen)
    return seen

def columns(view):
    if view=='B2_CONTROL':return 'b0','b0facts',old
    if view=='B2_REPAIRED':return 'b1','facts',new
    raise ValueError(view)

def encode(texts,tc,facts,fc,fit,learning):
    ids,counts=np.unique(tc[fit],return_counts=True)
    te=learning.FrequencyTfidf().fit([texts[i] for i in ids],counts)
    fe=learning.FactEncoder().fit([facts[i] for i in np.unique(fc[fit])])
    pairs,inverse=np.unique(tc.astype(np.int64)*len(facts)+fc,return_inverse=True)
    x=sparse.hstack([te.transform([texts[i] for i in pairs//len(facts)]),
                    fe.transform([facts[i] for i in pairs%len(facts)])],format='csr')
    return x,inverse,te,fe

def relationship(value):
    f=json.loads(value) if isinstance(value,str) else value
    # Fit-support diagnostic only. Does not route scores, labels or thresholds.
    keys=['category','action','outcome','protocol','src_role','dst_role','src_zone','dst_zone',
          'dst_port_category','icmp_type','icmp_code']
    return json.dumps({k:f[k] for k in keys if k in f},sort_keys=True,separators=(',',':'))

def review_flags(limited,supported,factless,relation_support,projection_support):
    # A coarse relation can legitimately contain several classes with different
    # messages. Only identical projected input is an observed label conflict.
    return limited|(~supported)|factless|(relation_support.sum(1)==0)|((projection_support>0).sum(1)>1)

def assessment(y,p,g,product,policies):
    result={'classification':new.cm_metrics(y,p.argmax(1)),'risk_ranking':new.risk_ranking(y,p),'policies':[]}
    for policy in policies:
        for name in ['empirical_pooled','empirical_worst_source_global']:
            t=policy[name];alarm=1-p[:,0]>=t
            result['policies'].append({'alpha':policy['alpha'],'name':name,'threshold':t,
               'risk':new.risk_metrics(y,p,t),'gated_classification':new.cm_metrics(y,new.gated_predictions(p,t)),
               'normal_alert_groups':int(len(np.unique(g[alarm&(y==0)]))),
               'normal_groups':int(len(np.unique(g[y==0]))),
               'largest_normal_alert_group_rows':int(np.unique(g[alarm&(y==0)],return_counts=True)[1].max()) if np.any(alarm&(y==0)) else 0})
    return result

def run_one(args,task,view,metadata,protocol,summary):
    prepared=Path(args.prepared);root=Path(args.output_dir);y,groups,products,empty,supported,asa,limited=metadata
    roles=protocol[task].to_numpy().astype(np.int8)
    for r in (0,1,2):assert set(np.unique(y[roles==r]))=={0,1,2}
    binding={'version':CONTRACT['version'],'task':task,'view':view,'contract':CONTRACT,
       'parameters':new.PARAMETERS,'row_weight':1,'repeat_weighting':False,
       'prepared_sha256':summary['prepared_sha256'],'groups_sha256':summary['groups_sha256'],
       'roles_sha256':hashlib.sha256(roles.tobytes()).hexdigest(),'code_hashes':code_hashes(),
       'python':platform.python_version(),'sklearn':__import__('sklearn').__version__}
    stem=task+'_'+view;attempts=[root/stem]+sorted(root.glob(stem+'_attempt*'))
    for folder in attempts:
        if (folder/'complete.json').exists():
            try:previous=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
            except json.JSONDecodeError:continue
            if previous['binding']!=binding:raise ValueError('Completed fit identity changed')
            for name,sha in previous['artifacts'].items():assert base.file_hash(folder/name)==sha
            print(json.dumps({'stage':'reuse_verified','task':task,'view':view}),flush=True);return
    folder=root/stem
    if folder.exists():folder=root/(stem+'_attempt'+str(len(attempts)).zfill(3))
    folder.mkdir(parents=True);save(folder/'binding.json',binding);start=time.perf_counter()
    positions=np.flatnonzero(roles>=0);role=roles[positions];labels=y[positions]
    fit=role==0;select=role==1;cal=role==2;ev=role==3;final_fit=fit|select
    pq.write_table(pa.table({'row_position':positions.astype(np.int32),'role':role}),folder/'roles.parquet',compression='zstd')
    textcol,factcol,learning=columns(view)
    texts,tc=string_codes(prepared/'prepared.parquet',positions,textcol)
    facts,fc=string_codes(prepared/'prepared.parquet',positions,factcol)
    print(json.dumps({'stage':'encode_fit_only','task':task,'view':view,'rows':len(labels),'unique_texts':len(texts)}),flush=True)
    x,inverse,te,fe=encode(texts,tc,facts,fc,fit,learning)
    candidates=[];best=None
    for C in CONTRACT['C_grid']:
        model,receipt=learning.fit_aggregated(x,inverse[fit],labels[fit],C)
        p=predict_rows(model,x,inverse,select)
        write_predictions(folder/('selection_C'+str(C)+'.parquet'),positions[select],labels[select],p)
        score=new.cm_metrics(labels[select],p.argmax(1))['macro_f1']
        candidates.append({'C':C,'selection_macro_f1':score,'fit':receipt})
        if best is None or score>best[1]+.001:best=(C,score)
        print(json.dumps({'stage':'candidate_fitted','task':task,'view':view,**candidates[-1]}),flush=True)
    del x,te,fe,model;gc.collect()
    x,inverse,te,fe=encode(texts,tc,facts,fc,final_fit,learning)
    model,receipt=learning.fit_aggregated(x,inverse[final_fit],labels[final_fit],best[0])
    calp=predict_rows(model,x,inverse,cal)
    policies=[new.calibration_policies(labels[cal],calp,products[positions][cal],groups[positions][cal],a) for a in BUDGETS]
    # All fit choices and decision thresholds frozen before reading target scores.
    save(folder/'frozen_decisions.json',{'C':best[0],'policies':policies,'contract':CONTRACT})
    p=predict_rows(model,x,inverse,ev);target=labels[ev];pg=groups[positions][ev];pp=products[positions][ev]
    report=assessment(target,p,pg,pp,policies)
    report.update(task=task,view=view,selected_C=best[0],candidates=candidates,final_fit=receipt,
      role_class_counts={str(r):np.bincount(labels[role==r],minlength=3).tolist() for r in range(4)},
      policies=report['policies'],calibration_policies=policies,all_benign_control=new.cm_metrics(target,np.zeros(len(target),int)),
      elapsed_seconds=time.perf_counter()-start,training_quality_accepted=False,transfer_validated=False)
    report['sources']={str(s):assessment(target[pp==s],p[pp==s],pg[pp==s],pp[pp==s],policies) for s in np.unique(pp)}
    nonempty=~empty[positions][ev]
    _,gi,gn=np.unique(pg[nonempty],return_inverse=True,return_counts=True)
    report['group_equal_nonempty']=new.cm_metrics(target[nonempty],p[nonempty].argmax(1),1/gn[gi])
    report['group_metric_note']='Diagnostic equal group mass only; training remains equal original rows. Exact-view groups are not proven independent incidents.'
    for s in report['sources']:
        report['sources'][s]['nonempty_class_groups']=[int(len(np.unique(pg[(pp==s)&nonempty&(target==c)]))) for c in range(3)]
    factless=np.asarray([not t for t in texts])[tc] & np.asarray([f=='{}' for f in facts])[fc]
    # Relation support is learned from fit+selection only, with row multiplicity.
    rels=[relationship(f) for f in facts];rel_values,rel_ids=np.unique(rels,return_inverse=True);row_rel=rel_ids[fc]
    support=np.bincount(row_rel[final_fit]*3+labels[final_fit],minlength=len(rel_values)*3).reshape(-1,3)
    fit_support=support[row_rel[ev]]
    pair_support=np.bincount(inverse[final_fit]*3+labels[final_fit],minlength=x.shape[0]*3).reshape(-1,3)
    review=review_flags(limited[positions][ev],supported[positions][ev],factless[ev],fit_support,pair_support[inverse[ev]])
    report['review_support']={'rows':int(review.sum()),'coverage_without_review':float((~review).mean()),
        'review_class_counts':np.bincount(target[review],minlength=3).tolist(),
        'all_rows_retained':True,'not_a_new_class_or_alarm_rule':True,'not_calibrated_uncertainty':True,
        'coarse_relationship_label_mixture_is_not_input_conflict':True,
        'definition':'limited/unsupported/empty evidence, unseen fit relation, or identical projected fit input with mixed labels; absence of a flag is not verified correctness',
        'fit_relationship_class_support':{str(r):c.tolist() for r,c in zip(rel_values,support)}}
    empirical=np.bincount(inverse[ev]*3+target,minlength=x.shape[0]*3).reshape(-1,3)
    mixed=(empirical>0).sum(1)>1
    report['evaluation_projection_conflicts']={'mixed_groups':int(mixed.sum()),'mixed_rows':int(empirical[mixed].sum()),
        'minimum_empirical_errors':int((empirical.sum(1)-empirical.max(1)).sum()),'diagnostic_only':True}
    report['slices']={}
    for name,m in [('original_empty',empty[positions][ev]),('information_empty',factless[ev]),('unsupported',~supported[positions][ev]),
       ('ASA_projection_conflict',asa[positions][ev]&mixed[inverse[ev]]),
       ('ASA_projection_nonconflict',asa[positions][ev]&~mixed[inverse[ev]]),
       ('ASA_relation_unseen',asa[positions][ev]&(fit_support.sum(1)==0)),
       ('review',review),('covered',~review)]:
        report['slices'][name]=assessment(target[m],p[m],pg[m],pp[m],policies)
    names=list(te.names())+list(fe.names())
    report['largest_coefficients']={str(c):[{'feature':str(names[i]),'coefficient':float(model.coef_[c,i])} for i in np.argsort(-abs(model.coef_[c]))[:40]] for c in range(3)}
    report['stage_D_status']='await_matched_C_review; target errors cannot automatically choose interactions'
    save(folder/'report.json',report)
    for name,mask,prob in [('calibration',cal,calp),('evaluation',ev,p)]:write_predictions(folder/(name+'.parquet'),positions[mask],labels[mask],prob)
    pq.write_table(pa.table({'row_position':positions[ev].astype(np.int32),'review':review,
        'fit_relation_benign_rows':fit_support[:,0],'fit_relation_malicious_rows':fit_support[:,1],'fit_relation_suspicious_rows':fit_support[:,2],
        'fit_projection_classes':(pair_support[inverse[ev]]>0).sum(1).astype(np.int8)}),folder/'support.parquet',compression='zstd')
    joblib.dump({'view':view,'tfidf':te,'facts':fe,'model':model,'policies':policies,
         'representation_version':summary['version'],'labels':['benign','malicious','suspicious']},folder/'model.joblib',compress=3)
    files=['binding.json','roles.parquet','frozen_decisions.json','report.json','calibration.parquet','evaluation.parquet','support.parquet','model.joblib']+['selection_C'+str(C)+'.parquet' for C in CONTRACT['C_grid']]
    save(folder/'complete.json',{'binding':binding,'artifacts':{n:base.file_hash(folder/n) for n in files},'quality_accepted':False})
    print(json.dumps({'stage':'fit_complete','task':task,'view':view,'seconds':round(report['elapsed_seconds']),
                      'macro_f1':report['classification']['macro_f1']}),flush=True)

def write_predictions(path,pos,y,p):
    pq.write_table(pa.table({'row_position':pos.astype(np.int32),'label_index':y,'p_benign':p[:,0],
       'p_malicious':p[:,1],'p_suspicious':p[:,2],'prediction':p.argmax(1).astype(np.uint8)}),path,compression='zstd')

def run(args):
    folder=Path(args.prepared);Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    summary=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    audit=json.loads(Path(args.audit).read_text(encoding='utf-8'))
    assert summary['full_official_rows'] and audit['all_checks_passed']
    for name,key in [('prepared.parquet','prepared_sha256'),('groups.parquet','groups_sha256'),('protocol.parquet','protocol_sha256')]:
        assert base.file_hash(folder/name)==audit[key]
    for name,sha in summary['source_hashes'].items():assert base.file_hash(Path(__file__).parent/name)==sha
    for name,sha in audit['transitive_parser_hashes'].items():assert base.file_hash(Path(__file__).parent/name)==sha
    decl=json.loads((folder/'protocol.json').read_text(encoding='utf-8'));assert decl['all_tasks_eligible']
    g=pq.read_table(folder/'groups.parquet');m=pq.read_table(folder/'prepared.parquet',columns=['product','original_empty','supported','route','quality'])
    limited=np.array([bool((lambda q:q.get('information_limited') or q.get('only_record_family_observed'))(json.loads(s))) for s in m['quality'].to_pylist()])
    metadata=(g['label_index'].to_numpy(),g['union_group'].to_numpy(),np.asarray(m['product'].to_pylist(),object),m['original_empty'].to_numpy(),m['supported'].to_numpy(),np.isin(np.asarray(m['route'].to_pylist(),object),['asa','asa_acl','asa_protocol']),limited)
    protocol=pq.read_table(folder/'protocol.parquet')
    for task in args.tasks.split(','):
        assert task in TASKS
        for view in args.views.split(','):run_one(args,task,view,metadata,protocol,summary);gc.collect()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--audit',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--tasks',default=','.join(TASKS));p.add_argument('--views',default=','.join(VIEWS));run(p.parse_args())
