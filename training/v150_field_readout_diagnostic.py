"""Six registered factual readout fits; all classifier/backbone weights frozen."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,sys,importlib.metadata
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import sha,read,check_bindings,ReviewError
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v138_train import configure
from v142_train import tensors
from v146_runtime import require as require_V146
from v148_observed_relation import observed
from v150_field_decoder import FIELDS,RIDGE,inner_held,schema,decode,solve,predict

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v150_field_readout_diagnostic_20261001'
PLAN=ROOT/'training/review_policy/v150_field_readout_plan.json'
OLD=ROOT/'artifacts/v146_guarded_pair_training_20261001'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)

def validate(p):
    expected=dict(version='V150-field-diagnostic',latest_actual_classifier='V146',kind='frozen_behavior_field_readability_diagnostic',decoder_fits=6,classifier_fits=0,
        layers=['H1','V146_B_H2'],folds=[0,1,2],ridge=RIDGE,fit_solver='one_original_frequency_multivariate_ridge_normal_equation_per_layer_fold',
        split='legal_TRAIN_roots_SHA256_V150_field_readout_mod5_eq0_inner_held',
        targets='all_13_existing_fields_bit_and_enum_with_unknown_state_retained_no_threat_labels',
        feature='flatten_all_16_members_128_dims_no_mean_pool_canonical_min_local',
        baseline='fit_original_frequency_modal_field',selection='none_fixed_ridge_no_layer_or_field_selection',
        held_labels_used_for_fit=False,backbone_seen_inner_held_class_labels=True,
        new_classifier_parameters=0,new_classifier_updates=0,automatic_repeat=False,automatic_class_training=False,
        endpoint='single_fixed_ridge_normal_equation_solve_no_tuning',
        target_thresholds=dict(bits=.5,enum='argmax_if_max_ge_0.5_else_unknown'))
    for k,v in expected.items():
        if p.get(k)!=v:raise ReviewError('Unregistered factual diagnostic: '+k)
    check_bindings(p['evidence_sha256'])
    return p

def review():return validate(read(PLAN))

def reference(d):
    source=pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet',columns=['row_position','facts_json'])
    assert np.array_equal(source.row_position,d.row_position)
    facts=source.facts_json.map(json.loads).tolist()
    names=read(ROOT/'artifacts/v92_evidence_training_20260928/representation_contract.json')['old_fact_coordinate_names']
    columns,entries=schema(names)
    target=(np.load(ROOT/'artifacts/v138_single_issue_round1_20260930/facts.npy')[:,columns]>0).astype(np.float64)
    keys=d[['local','canonical_key']].drop_duplicates().sort_values('local')
    assert keys.local.nunique()==22546
    mapping=keys.groupby('canonical_key').local.min().to_dict()
    reps=keys.set_index('local').canonical_key.map(mapping).reindex(range(22546)).to_numpy(np.int64)
    assert np.array_equal(target,target[reps])
    expected=decode(target,entries)
    known={f:np.array([observed(v,f) for v in facts],dtype=bool) for f in FIELDS}
    # Check field targets against all observed parsed facts, not only fit rows.
    for f in FIELDS:
        loc=d.local.to_numpy();want=np.array([v.get(f) for v in facts],dtype=object)
        assert np.all(expected[f][loc][known[f]]==want[known[f]])
    return target,entries,expected,known,reps

def manifest(d,f):
    legal=d.fold.ne(f).to_numpy();inner=np.array([inner_held(r) for r in d.root])
    roles=np.where(~legal,'outer_HELD',np.where(inner,'inner_HELD','readout_FIT'))
    fit=d[roles=='readout_FIT'];held=d[roles=='inner_HELD'];outer=d[roles=='outer_HELD']
    assert not(set(fit.root)&set(held.root)) and not((set(fit.root)|set(held.root))&set(outer.root))
    assert len(fit)+len(held)==len(fit_context(d,f)[0]) and len(d)==112807
    r=d.copy();r['readout_role']=roles
    counts=np.bincount(fit.local,minlength=22546).astype(np.float64)
    assert int(counts.sum())==len(fit)
    r['canonical_seen_by_decoder']=r.canonical_key.isin(set(fit.canonical_key))
    return r,counts

def negative_cases(p):
    results=[]
    for k,v in [('decoder_fits',12),('classifier_fits',6),('held_labels_used_for_fit',True),('ridge',.01),
        ('feature','mean_pool_members'),('backbone_seen_inner_held_class_labels',False),('automatic_class_training',True)]:
        q=dict(p);q[k]=v
        try:validate(q)
        except ReviewError:results.append(dict(field=k,rejected=True))
        else:raise ReviewError('Diagnostic rule accepted forbidden '+k)
    return results

def register():
    p=review();require_V146(ROOT/'training/v146_train.py');configure()
    assert not OUT.exists(),'Never overwrite diagnostic run'
    _,d=load_data();target,entries,_,_,reps=reference(d)
    OUT.mkdir();extras={PLAN,Path(__file__),ROOT/'training/v150_field_decoder.py',ROOT/'training/test_v150_field_decoder.py',
        ROOT/'training/experiment_review.py'}|{ROOT/z for z in p['evidence_sha256']}
    for f in range(3):
        rows,_=manifest(d,f);path=OUT/f'fold{f}_role_manifest.parquet';rows.to_parquet(path,index=False);extras.add(path)
        h,ff,m=tensors(f);m.load_state_dict(torch.load(OLD/f'fold{f}_B/endpoint.pt',map_location='cpu',weights_only=True)['state'])
        assert torch.isfinite(m.features(h[:32])).all()
        del h,ff,m;gc.collect();torch.cuda.empty_cache()
    np.save(OUT/'canonical_min_local.npy',reps);extras.add(OUT/'canonical_min_local.npy')
    save(OUT/'target_schema.json',dict(entries=entries,target_dimensions=target.shape[1]));extras.add(OUT/'target_schema.json')
    # Historical seal covers all input/model/cache dependencies; include loaded physical runtime.
    extras|={ROOT/z for z in read(OLD/'run_seal.json')['source_sha256']}
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.suffix=='.pyc' and q.with_suffix('.py').exists():q=q.with_suffix('.py')
            if q.is_file():extras.add(q)
    extras.add(Path(sys.executable).resolve());extras.update((Path(torch.__file__).parent/'lib').glob('*.dll'))
    for name in ['python.exe','python3.dll','python311.dll','python312.dll','python313.dll']:
        q=Path(sys.base_prefix)/name
        if q.is_file():extras.add(q)
    bindings={q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q):sha(q) for q in sorted(z.resolve() for z in extras)}
    save(OUT/'run_seal.json',dict(status='sealed_before_diagnostic_fit',trainer_path=Path(__file__).relative_to(ROOT).as_posix(),
        plan_sha256=sha(PLAN),source_sha256=bindings,python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','pyarrow']},
        scope='Physical loaded dependencies plus all historical V146 bindings; OS/GPU driver not byte-snapshotted.'))
    save(OUT/'registration.json',dict(status='sealed_before_diagnostic_fit',decoder_fits_max=6,classifier_fits=0,
        negative_cases=negative_cases(p),seal_sha256=sha(OUT/'run_seal.json'),registration_feature_forward_calls=3,
        registration_decoder_fits=0,role_rows=[len(pd.read_parquet(OUT/f'fold{f}_role_manifest.parquet')) for f in range(3)]))
    emit(stage='diagnostic_registered',decoder_fits_max=6,classifier_fits=0,target_dimensions=target.shape[1],physical_bindings=len(bindings))

def require():
    p=review();s=read(OUT/'run_seal.json')
    if s['status']!='sealed_before_diagnostic_fit' or s['plan_sha256']!=sha(PLAN) or (ROOT/s['trainer_path']).resolve()!=Path(__file__).resolve():raise ReviewError('Diagnostic seal mismatch')
    check_bindings(s['source_sha256'])
    if s['python_version']!=sys.version or s['package_versions']!={n:importlib.metadata.version(n) for n in s['package_versions']}:raise ReviewError('Runtime changed')
    return p

def fit(f,layer):
    p=require();configure();assert f in p['folds'] and layer in p['layers']
    folder=OUT/f'fold{f}_{layer}';assert not folder.exists() and len(list(OUT.glob('fold*_*/started.json')))<6
    _,d=load_data();target,entries,expected,known,reps=reference(d);r,mass=manifest(d,f)
    assert r.equals(pd.read_parquet(OUT/f'fold{f}_role_manifest.parquet')) and np.array_equal(reps,np.load(OUT/'canonical_min_local.npy'))
    h,ff,m=tensors(f);m.load_state_dict(torch.load(OLD/f'fold{f}_B/endpoint.pt',map_location='cpu',weights_only=True)['state']);before=tensor_hash(m.state_dict())
    for v in m.parameters():v.requires_grad_(False)
    folder.mkdir();save(folder/'started.json',dict(fold=f,layer=layer,diagnostic_fit=1,classifier_fits=0,
        seal_sha256=sha(OUT/'run_seal.json'),classifier_state_before=before))
    with torch.no_grad():
        if layer=='H1':x=h[:,:,:128].reshape(len(h),-1).clone()
        else:x=torch.cat([m.features(h[start:start+1024]).reshape(-1,2048) for start in range(0,len(h),1024)])
        x=x[torch.as_tensor(reps,device='cuda')]
        y=torch.as_tensor(target,device='cuda',dtype=torch.float64);w=torch.as_tensor(mass,device='cuda',dtype=torch.float64)
        state,receipt=solve(x,y,w);scores=predict(x,state)
    got=decode(scores,entries);loc=r.local.to_numpy();fitrows=r.readout_role.eq('readout_FIT').to_numpy()
    classifier_pred=np.load(OLD/f'fold{f}_B/sealed_all_prob.npy')[loc].argmax(1)
    ledger=r.copy();ledger['classifier_pred_frozen']=classifier_pred
    reports=[]
    for e in entries:
        field=e['field'];truth=expected[field][loc];pred=got[field][loc]
        ledger[field+'_observed']=known[field];ledger[field+'_decoded']=pred;ledger[field+'_exact']=pred==truth
        all_vals=truth[fitrows];counts=Counter(str(v) for v in all_vals)
        # Stable mode uses original occurrence counts; no class-conditioned choice.
        mode_key=sorted(counts,key=lambda z:(-counts[z],z))[0]
        mode=next(v for v in all_vals if str(v)==mode_key)
        ledger[field+'_mode_exact']=truth==mode
        for role in ['readout_FIT','inner_HELD','outer_HELD']:
            rows=r.readout_role.eq(role).to_numpy()
            obs=rows&known[field]
            reports.append(dict(field=field,role=role,original_rows=int(rows.sum()),observed_rows=int(obs.sum()),
                observed_errors=int((obs&(pred!=truth)).sum()),mode_observed_errors=int((obs&(truth!=mode)).sum()),
                unobserved_rows=int((rows&~known[field]).sum()),unobserved_state_errors=int((rows&~known[field]&(pred!=truth)).sum()),
                observed_correct_class_rows=int((obs&(classifier_pred==r.truth)).sum()),
                observed_wrong_class_rows=int((obs&(classifier_pred!=r.truth)).sum()),
                observed_wrong_class_decoder_errors=int((obs&(classifier_pred!=r.truth)&(pred!=truth)).sum())))
    ledger.to_parquet(folder/'all_original_field_ledger.parquet',index=False)
    torch.save({k:v.cpu() for k,v in state.items()},folder/'decoder.pt')
    after=tensor_hash(m.state_dict());assert before==after and all(v.grad is None for v in m.parameters())
    result=dict(status='diagnostic_decoder_fit_executed',fold=f,layer=layer,diagnostic_decoder_fits=1,classifier_fits=0,
        classifier_gradients=0,classifier_updates=0,optimizer_updates=0,normal_equation_solves=1,ridge=RIDGE,
        feature_model_forward_calls=0 if layer=='H1' else (22546+1023)//1024,
        decoder_coefficient_values=int(state['coef'].numel()),classifier_state_unchanged=True,initial_classifier_sha256=before,
        backbone_seen_inner_HELD_class_labels=True,blind_generalization=False,quality_acceptance=False,issue_solved=False,
        original_class_mass_by_role={role:r[r.readout_role.eq(role)].truth.value_counts().reindex([0,1,2],fill_value=0).astype(int).tolist() for role in ['readout_FIT','inner_HELD','outer_HELD']},
        rows_sha256=sha(folder/'all_original_field_ledger.parquet'),decoder_sha256=sha(folder/'decoder.pt'),
        feature_dimensions=2048,target_dimensions=target.shape[1],solve=receipt,field_results=reports,
        seal_sha256=sha(OUT/'run_seal.json'))
    save(folder/'fit.json',result);emit(stage='diagnostic_decoder_fit_executed',fold=f,layer=layer,solve=receipt,classifier_unchanged=True)

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('mode',choices=['register','fit']);a.add_argument('--fold',type=int);a.add_argument('--layer')
    z=a.parse_args();register() if z.mode=='register' else fit(z.fold,z.layer)
