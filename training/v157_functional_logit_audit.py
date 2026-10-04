"""Registered frozen complete-function decomposition, not a new classifier."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,sys,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings,ReviewError
from v135_runtime import load_data
from v135_model import tensor_hash
from v138_train import configure
from v155_train import initialize
from v155_runtime import OUT as LAST
from v156_conditional_neighborhood import OUT as PRIOR,require as require_prior
from v157_function_components import groups,components,ensemble_summary

OUT=ROOT/'artifacts/v157_complete_function_logit_audit_20261001'
PLAN=ROOT/'training/review_policy/v157_functional_logit_plan.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)

def validate(p):
    expected=dict(version='V157-frozen-complete-function-audit',classifier_fits=0,gradients=0,updates=0,
        folds=[0,1,2],reference='fixed_V146_A_common_V155_initialization_and_V156_geometry_baseline',
        population='all_22546_actual_locals_each_fold_then_all_112807_originals_each_role_fold',members=16,
        classifier_forward_chunk_calls=36,feature_function_calls=36,chunk=2048,
        field_partition='all_477_existing_fact_coordinate_names_plus_18_original_record_port_coordinates',
        real_prediction='mean_of_16_member_softmax_probabilities_then_argmax',
        direct_recomposition='same_operation_order_body_plus_bias_plus_facts',
        field_sum_tolerance=1e-12,field_recomposed_probability_tolerance=2e-12,
        mean_logit_is_diagnostic_only=True,no_field_removal_or_counterfactual_inputs=True,
        component_size_not_semantic_cause=True,automatic_new_training=False,quality_acceptance=False)
    for k,v in expected.items():
        if p.get(k)!=v:raise ReviewError('Unregistered functional diagnostic: '+k)
    check_bindings(p['evidence_sha256']);return p

def review():return validate(read(PLAN))

def negative_cases(p):
    out=[]
    for k,v in [('classifier_fits',6),('reference','best_outer_checkpoint'),('real_prediction','mean_logits'),
        ('field_partition','selected_good_fields'),('no_field_removal_or_counterfactual_inputs',False),
        ('component_size_not_semantic_cause',False),('automatic_new_training',True)]:
        q=dict(p);q[k]=v
        try:validate(q)
        except ReviewError:out.append(dict(field=k,rejected=True))
        else:raise ReviewError('Forbidden functional diagnostic accepted')
    return out

def register():
    p=review();require_prior();configure();assert not OUT.exists()
    x,d=load_data();OUT.mkdir();d.reset_index(drop=True).to_parquet(OUT/'original_reference.parquet',index=False)
    names=read(ROOT/'artifacts/v92_evidence_training_20260928/representation_contract.json')['old_fact_coordinate_names']
    partition=groups(names);save(OUT/'field_partition.json',partition)
    h,ff,m=initialize(0)
    assert np.array_equal(ff.cpu().numpy(),x[:,65792:].toarray().astype(np.float64))
    assert all(v.grad is None for v in m.parameters())
    torch.softmax(torch.zeros(1,3,dtype=torch.float64),-1) # Nonclassifier setup only.
    extras={PLAN,Path(__file__).resolve(),OUT/'original_reference.parquet',OUT/'field_partition.json'}|{ROOT/z for z in p['evidence_sha256']}
    extras|={ROOT/z for z in read(PRIOR/'run_seal.json')['source_sha256']}
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
    save(OUT/'run_seal.json',dict(status='sealed_before_any_classifier_or_feature_call',source_sha256=bindings,
        plan_sha256=sha(PLAN),entry=Path(__file__).relative_to(ROOT).as_posix(),python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','pyarrow']}))
    save(OUT/'registration.json',dict(status='registered_frozen_function_diagnostic',physical_files=len(bindings),
        classifier_forward_calls=0,feature_function_calls=0,gradients=0,fits=0,updates=0,
        classifier_forward_call_budget=36,feature_function_call_budget=36,negative_cases=negative_cases(p),seal_sha256=sha(OUT/'run_seal.json')))
    emit(stage='complete_function_registered',physical_files=len(bindings),classifier_calls=0)

def require():
    p=review();s=read(OUT/'run_seal.json')
    assert s['status']=='sealed_before_any_classifier_or_feature_call' and s['plan_sha256']==sha(PLAN)
    assert (ROOT/s['entry']).resolve()==Path(__file__).resolve() and s['python_version']==sys.version
    assert s['package_versions']=={n:importlib.metadata.version(n) for n in s['package_versions']}
    check_bindings(s['source_sha256']);return p

def run():
    p=require();configure();assert not (OUT/'started.json').exists()
    _,d=load_data();d=d.reset_index(drop=True);assert d.equals(pd.read_parquet(OUT/'original_reference.parquet'))
    save(OUT/'started.json',dict(status='registered_function_audit_started',seal_sha256=sha(OUT/'run_seal.json')))
    partition=read(OUT/'field_partition.json');allrows=[];reports=[];calls={'classifier_attempts':0,'classifier_completed':0,'feature_function_calls':0}
    for f in p['folds']:
        h,ff,m=initialize(f);before=tensor_hash(m.state_dict());captured=[];original_features=m.features
        def collect(packed):
            calls['feature_function_calls']+=1
            feature=original_features(packed);captured.append(feature);return feature
        m.features=collect
        arrays={n:[] for n in ['logits','body','facts','probabilities']};field_arrays={k:[] for k in partition};statistics=[]
        max_field_gap=0.;max_field_prob_gap=0.;bias=None
        with torch.no_grad():
            for start in range(0,len(h),2048):
                take=np.arange(start,min(start+2048,len(h)));calls['classifier_attempts']+=1
                save(OUT/'compute_progress.json',dict(stage='before_classifier_call',fold=f,**calls,gradients=0,fits=0,updates=0))
                z=m(h[take],ff[take]);calls['classifier_completed']+=1
                assert len(captured)==1;actual_h2=captured.pop()
                body,direct,bias,fields=components(actual_h2,ff[take],m,partition)
                assert torch.equal(z,body+bias+direct[:,None,:])
                fact_sum=sum(fields.values());gap=float((fact_sum-direct).abs().max());assert gap<=p['field_sum_tolerance'];max_field_gap=max(max_field_gap,gap)
                q,stats=ensemble_summary(z,body,direct,bias)
                reconstructed=torch.softmax(body+bias+fact_sum[:,None,:],-1).mean(1)
                gap=float((q-reconstructed).abs().max());assert gap<=p['field_recomposed_probability_tolerance'];max_field_prob_gap=max(max_field_prob_gap,gap)
                assert torch.equal(q.argmax(1),reconstructed.argmax(1))
                for k,v in [('logits',z),('body',body),('facts',direct),('probabilities',q)]:arrays[k].append(v.cpu().numpy())
                for k,v in fields.items():field_arrays[k].append(v.cpu().numpy())
                statistics.append(pd.DataFrame({k:v.cpu().numpy() for k,v in stats.items()}))
                save(OUT/'compute_progress.json',dict(stage='classifier_and_decomposition_completed',fold=f,**calls,gradients=0,fits=0,updates=0))
        m.features=original_features
        assert tensor_hash(m.state_dict())==before and all(v.grad is None for v in m.parameters())
        values={k:np.concatenate(v) for k,v in arrays.items()};q=values['probabilities']
        assert np.array_equal(q,np.load(LAST/f'fold{f}_zero_probability.npy'))
        folder=OUT/f'fold{f}';folder.mkdir()
        for k,v in values.items():np.save(folder/(k+'.npy'),v)
        np.save(folder/'bias.npy',bias.cpu().numpy())
        np.savez(folder/'field_logits.npz',**{k:np.concatenate(v) for k,v in field_arrays.items()})
        local=pd.concat(statistics,ignore_index=True);assert len(local)==22546;local['local']=np.arange(22546)
        for k,v in field_arrays.items():
            a=np.concatenate(v);local['field_'+k+'_S_minus_M']=a[:,2]-a[:,1]
        local.to_parquet(folder/'local_function_summary.parquet',index=False)
        rows=d.merge(local,on='local',how='left',validate='many_to_one');assert np.array_equal(rows.row_position,d.row_position)
        rows['training_role']=f;rows['query_role']=np.where(rows.fold.eq(f),'outer_HELD','legal_TRAIN')
        rows['pred']=q[rows.local].argmax(1)
        for cl in range(3):rows[f'p{cl}']=q[rows.local,cl]
        allrows.append(rows)
        reports.append(dict(fold=f,actual_locals=22546,classifier_forward_calls=12,feature_function_calls=12,
            probability_matches_V146_A_exact=True,direct_logit_recomposition_exact=True,field_sum_max_gap=max_field_gap,
            field_probability_recomposition_max_gap=max_field_prob_gap,model_tensor_identity_unchanged=True,
            mean_logit_vs_probability_original_role_disagreements=int(rows.mean_logit_pred.ne(rows.pred).sum())))
        emit(stage='complete_function_fold_audited',**reports[-1]);del h,ff,m;gc.collect();torch.cuda.empty_cache()
    assert calls==dict(classifier_attempts=36,classifier_completed=36,feature_function_calls=36)
    all=pd.concat(allrows,ignore_index=True);assert len(all)==338421
    path=OUT/'all_original_role_function_summary.parquet';all.to_parquet(path,index=False)
    save(OUT/'audit.json',dict(status='all_registered_complete_function_components_and_actual_probabilities_replayed',
        reference='V146_A_common_V155_initialization_not_new_candidate',latest_actual_classifier='V155',
        classifier_forward_calls=36,feature_function_calls=36,gradients=0,classifier_fits=0,updates=0,
        original_rows=112807,actual_locals_per_fold=22546,original_role_rows=338421,field_groups=len(partition),reports=reports,
        inputs_or_fields_removed=False,no_new_classifier_or_semantic_attribution=True,issue_solved=False,quality_acceptance=False,
        source_sha256={x.relative_to(ROOT).as_posix():sha(x) for x in [Path(__file__),PLAN,OUT/'run_seal.json',path]}))
    emit(stage='complete_function_audit_finished',**calls,fits=0,updates=0)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','run']);z=ap.parse_args()
    try:register() if z.mode=='register' else run()
    except Exception as e:
        if OUT.exists() and not (OUT/'audit.json').exists():
            progress=read(OUT/'compute_progress.json') if (OUT/'compute_progress.json').exists() else dict(classifier_attempts=0,classifier_completed=0,feature_function_calls=0)
            save(OUT/'failure.json',dict(status='execution_error_source_and_seal_preserved',mode=z.mode,error_type=type(e).__name__,error=str(e),progress=progress,entry_sha256=sha(Path(__file__)),fits=0,updates=0))
        raise
