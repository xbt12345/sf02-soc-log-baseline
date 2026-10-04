"""Full actual-local, root-excluded conditional geometry, zero fitting."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,sys,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
import torch
from experiment_review import ROOT,read,sha,check_bindings,ReviewError
from v138_train import configure
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v155_train import initialize
from v155_runtime import require as require_old,OUT as LAST
from v156_neighborhood_geometry import condition,nearest,normalize_dense,TOLERANCE
from v148_observed_relation import FIELDS

OUT=ROOT/'artifacts/v156_conditional_representation_neighborhood_20261001'
PLAN=ROOT/'training/review_policy/v156_conditional_neighborhood_plan.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)

def validate(p):
    expected=dict(version='V156-conditional-neighborhood-diagnostic',classifier_fits=0,gradients=0,updates=0,
        folds=[0,1,2],spaces=['actual_CSR_input','all16_H1','all16_V146_A_H2'],query_population='all_112807_originals_each_role_fold',
        neighbor_population='all_legal_TRAIN_original_mass_all_classes_including_mixed_actual_locals',
        source_exclusion='exclude_entire_query_root_for_both_TRAIN_and_outer_queries',
        condition='exact_protocol_and_13_typed_field_observation_states_no_value_equivalence',
        metric='L2_normalized_full_vectors_cosine_float64_no_member_pooling',tie_tolerance=TOLERANCE,
        tie_policy='retain_all_tied_local_original_class_mass_and_union_of_distinct_roots_select_min_local_for_witness',
        canonical_aliases='retain_actual_numeric_locals_no_canonical_merge',root_is_real_environment=False,
        labels_used_for='postfit_class_conditional_diagnostic_only_no_vote_router_or_synthetic_labels',
        endpoint='all_registered_folds_spaces_and_original_rows_no_best_space_selection',automatic_training=False,
        feature_function_calls=3,model_forward_calls=0,batch_size=256,quality_acceptance=False)
    for k,v in expected.items():
        if p.get(k)!=v:raise ReviewError('Changed diagnostic scope: '+k)
    check_bindings(p['evidence_sha256']);return p

def review():return validate(read(PLAN))

def negative_cases(p):
    out=[]
    for k,v in [('classifier_fits',6),('source_exclusion','exclude_query_row_only'),('condition','collapse_unknown'),
        ('canonical_aliases','merge_canonical'),('metric','mean_pool_members'),('labels_used_for','nearest_label_router'),('automatic_training',True)]:
        q=dict(p);q[k]=v
        try:validate(q)
        except ReviewError:out.append(dict(field=k,rejected=True))
        else:raise ReviewError('Forbidden diagnostic variant accepted')
    return out

def reference():
    x,d=load_data()
    trace=ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
    t=pd.read_parquet(trace,columns=['row_position','facts_json']);assert np.array_equal(t.row_position,d.row_position)
    d=d.reset_index(drop=True);d['condition']=t.facts_json.map(json.loads).map(condition)
    for col in ['root','fold','condition']:
        assert d.groupby('local')[col].nunique().eq(1).all(),col
    nodes=d[['local','root','fold','condition']].drop_duplicates().sort_values('local').reset_index(drop=True)
    assert len(nodes)==22546 and np.array_equal(nodes.local,np.arange(22546))
    return x,d,nodes

def register():
    p=review();require_old(ROOT/'training/v155_train.py');configure();assert not OUT.exists()
    x,d,nodes=reference();OUT.mkdir()
    d.to_parquet(OUT/'original_reference.parquet',index=False);nodes.to_parquet(OUT/'local_reference.parquet',index=False)
    # Resolve actual CUDA/model loading dependencies without calling any model or feature.
    h,ff,m=initialize(0);assert all(v.grad is None for v in m.parameters())
    normalize_dense(np.array([[3.,4.]]));(sparse.csr_matrix([[1.]])@sparse.csr_matrix([[1.]])).toarray()
    extras={PLAN,Path(__file__).resolve(),ROOT/'training/v156_neighborhood_geometry.py',ROOT/'training/test_v156_neighborhood_geometry.py',
        OUT/'original_reference.parquet',OUT/'local_reference.parquet'}|{ROOT/z for z in p['evidence_sha256']}
    extras|={ROOT/z for z in read(LAST/'run_seal.json')['source_sha256']}
    for mod in list(sys.modules.values()):
        name=getattr(mod,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.suffix=='.pyc' and q.with_suffix('.py').is_file():q=q.with_suffix('.py')
            if q.is_file():extras.add(q)
    extras.add(Path(sys.executable).resolve());extras.update((Path(torch.__file__).parent/'lib').glob('*.dll'))
    for n in ['python.exe','python3.dll','python311.dll','python312.dll','python313.dll']:
        q=Path(sys.base_prefix)/n
        if q.is_file():extras.add(q)
    bindings={q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q):sha(q) for q in sorted(z.resolve() for z in extras)}
    save(OUT/'run_seal.json',dict(status='sealed_before_any_feature_or_distance_audit',source_sha256=bindings,
        plan_sha256=sha(PLAN),entry=Path(__file__).relative_to(ROOT).as_posix(),python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','pyarrow']}))
    save(OUT/'registration.json',dict(status='registered_zero_training_diagnostic',physical_files=len(bindings),
        original_rows=len(d),actual_locals=len(nodes),conditions=d.condition.nunique(),expected_role_rows=338421,
        feature_function_calls=0,model_forward_calls=0,gradients=0,fits=0,updates=0,negative_cases=negative_cases(p),seal_sha256=sha(OUT/'run_seal.json')))
    emit(stage='neighborhood_registered',physical_files=len(bindings),feature_calls=0,fits=0)

def require():
    p=review();s=read(OUT/'run_seal.json')
    assert s['status']=='sealed_before_any_feature_or_distance_audit' and s['plan_sha256']==sha(PLAN)
    assert (ROOT/s['entry']).resolve()==Path(__file__).resolve() and s['python_version']==sys.version
    assert s['package_versions']=={n:importlib.metadata.version(n) for n in s['package_versions']}
    check_bindings(s['source_sha256']);return p

def geometry(x,nodes,c,space,features=None):
    n=len(nodes);roots=nodes.root.to_numpy();result=nodes[['local','root','fold','condition']].copy()
    zero=np.zeros(n,bool)
    if space=='actual_CSR_input':
        actual=x.astype(np.float64);norms=np.sqrt(actual.multiply(actual).sum(1)).A1;zero=norms==0
        vectors=sparse.diags(1/np.where(zero,1,norms))@actual;vectors=vectors.tocsr()
    else:
        a,zero=normalize_dense(features);vectors=torch.as_tensor(a,device='cuda',dtype=torch.float64)
    result['zero_vector']=zero
    for cl in [1,2]:
        for name in ['cosine','local','tied_locals','tied_original_rows','tied_roots','available_original_rows','available_locals']:
            result[f'c{cl}_{name}']=np.nan if name=='cosine' else -1
    for _,group in nodes.groupby('condition',sort=True):
        query=group.local.to_numpy();candidates=query[c[query].sum(1)>0]
        # Current source identity is one root per actual local; no own-row shortcut.
        source_roots={int(j):{int(roots[j])} for j in candidates}
        pool=vectors[candidates].T if space=='actual_CSR_input' else vectors[torch.as_tensor(candidates,device='cuda')].T
        for start in range(0,len(query),256):
            take=query[start:start+256]
            sim=(vectors[take]@pool).toarray() if space=='actual_CSR_input' else (vectors[torch.as_tensor(take,device='cuda')]@pool).cpu().numpy()
            outside=roots[take,None]!=roots[candidates][None,:]
            for cl in [1,2]:
                remaining=np.where(outside,c[candidates,cl][None,:],0)
                z=nearest(sim,candidates,remaining,source_roots,roots[take])
                for name,value in z.items():result.loc[take,f'c{cl}_{name}']=value
    assert result.filter(regex='local$|rows$|roots$|locals$').ge(-1).all().all()
    return result

def run():
    p=require();configure();assert not (OUT/'audit.json').exists() and not (OUT/'started.json').exists()
    x,d,nodes=reference();assert d.equals(pd.read_parquet(OUT/'original_reference.parquet'))
    save(OUT/'started.json',dict(status='distance_audit_started',seal_sha256=sha(OUT/'run_seal.json'),classifier_fits=0))
    allrows=[];summaries=[];feature_reports=[]
    for f in p['folds']:
        _,c,_,_,_=fit_context(d,f);h,ff,m=initialize(f);before=tensor_hash(m.state_dict())
        save(OUT/'compute_progress.json',dict(stage='before_features',fold=f,feature_function_attempts=f+1,feature_function_completed=f,model_forward_calls=0,gradients=0,fits=0,updates=0))
        with torch.no_grad():h2=m.features(h).cpu().numpy()
        h1=h[...,:128].cpu().numpy();assert tensor_hash(m.state_dict())==before and all(v.grad is None for v in m.parameters())
        feature_reports.append(dict(fold=f,feature_function_calls=1,model_forward_calls=0,feature_shape=list(h2.shape),model_state_unchanged=True))
        save(OUT/'compute_progress.json',dict(stage='features_completed',fold=f,feature_function_attempts=f+1,feature_function_completed=f+1,model_forward_calls=0,gradients=0,fits=0,updates=0))
        del h,ff,m;gc.collect();torch.cuda.empty_cache()
        baseline=np.load(LAST/f'fold{f}_zero_probability.npy')
        qA=np.load(LAST/f'fold{f}_A/sealed_all_prob.npy');qB=np.load(LAST/f'fold{f}_B/sealed_all_prob.npy')
        for space in p['spaces']:
            g=geometry(x,nodes,c,space,{'all16_H1':h1,'all16_V146_A_H2':h2}.get(space))
            g.to_parquet(OUT/f'fold{f}_{space}_local_neighbors.parquet',index=False)
            rows=d.merge(g.drop(columns=['root','fold','condition']),on='local',how='left',validate='many_to_one')
            assert np.array_equal(rows.row_position,d.row_position)
            rows['training_role']=f;rows['query_role']=np.where(rows.fold.eq(f),'outer_HELD','legal_TRAIN')
            rows['space']=space;rows['pred_baseline']=baseline[rows.local].argmax(1)
            rows['pred_V155_A']=qA[rows.local].argmax(1);rows['pred_V155_B']=qB[rows.local].argmax(1)
            one=rows.truth.eq(1).to_numpy();same=np.where(one,rows.c1_cosine,rows.c2_cosine);other=np.where(one,rows.c2_cosine,rows.c1_cosine)
            rows['same_minus_other_cosine']=same-other
            rows['relation']=np.select([np.isnan(same)&np.isnan(other),np.isnan(same),np.isnan(other),same>other+TOLERANCE,other>same+TOLERANCE],
                ['no_either_class','no_same_class','no_other_class','same_class_closer','other_class_closer'],default='class_tie')
            for (role,cl,relation),chunk in rows.groupby(['query_role','truth','relation'],sort=True):
                summaries.append(dict(fold=f,space=space,role=role,truth=int(cl),relation=relation,original_rows=len(chunk),roots=int(chunk.root.nunique()),
                    baseline_errors=int(chunk.pred_baseline.ne(chunk.truth).sum()),V155_A_errors=int(chunk.pred_V155_A.ne(chunk.truth).sum()),V155_B_errors=int(chunk.pred_V155_B.ne(chunk.truth).sum())))
            allrows.append(rows);emit(stage='conditional_geometry_completed',fold=f,space=space,original_role_rows=len(rows))
        del h1,h2;gc.collect();torch.cuda.empty_cache()
    combined=pd.concat(allrows,ignore_index=True)
    assert len(combined)==1015263 and combined.groupby('space').size().eq(338421).all()
    path=OUT/'all_original_role_neighbors.parquet';combined.to_parquet(path,index=False)
    save(OUT/'audit.json',dict(status='all_actual_local_and_original_role_conditioned_neighborhoods_completed',latest_actual_classifier='V155',
        registered_neighborhood_baseline='V146_A_common_V155_initialization_not_new_candidate',original_rows=112807,actual_locals=22546,conditions=int(nodes.condition.nunique()),
        rows_per_space=338421,legal_TRAIN_role_rows_per_space=225614,outer_original_rows_per_space=112807,total_space_role_rows=len(combined),
        feature_function_calls=3,model_forward_calls=0,gradients=0,classifier_fits=0,updates=0,feature_reports=feature_reports,summaries=summaries,
        neighbor_class_mass_includes_mixed=True,own_entire_root_excluded=True,no_canonical_alias_merge=True,no_vote_or_generated_labels=True,
        issue_solved=False,quality_acceptance=False,new_method_selected=False,
        limits=['Cosine and conditional availability describe these representations, not threat semantics or causal information loss.',
            'H1/H2 cosine omits the classifier facts@head_facts.T bypass and head projection; it is not whole-classifier evidence.',
            'Own-root exclusion is a source proxy, not a validated environment identity. Role duplicates are not new data.',
            'All inspected outer truth is diagnostic only; no layer, threshold, checkpoint, label or model is selected.'],
        seal_sha256=sha(OUT/'run_seal.json'),source_sha256={z.relative_to(ROOT).as_posix():sha(z) for z in [Path(__file__),PLAN,path,OUT/'run_seal.json']}))
    emit(stage='all_conditional_neighborhoods_complete',feature_calls=3,classifier_fits=0,rows=len(combined))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','run']);z=ap.parse_args()
    try:register() if z.mode=='register' else run()
    except Exception as e:
        if OUT.exists() and not (OUT/'audit.json').exists():
            progress=read(OUT/'compute_progress.json') if (OUT/'compute_progress.json').exists() else dict(feature_function_attempts=0,feature_function_completed=0)
            save(OUT/'failure.json',dict(status='execution_error_original_source_and_seal_preserved',error_type=type(e).__name__,error=str(e),mode=z.mode,progress=progress,classifier_fits=0,gradients=0,updates=0,entry_sha256=sha(Path(__file__))))
        raise
