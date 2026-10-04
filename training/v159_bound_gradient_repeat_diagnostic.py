"""One new sealed technical diagnostic: <=4 full class gradients, <=40 heads.

Preserve the failed trial, save each complete vector before comparison, never
fit/update a model and never restart the old preflight entry.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,read,sha,check_bindings
from experiment_review_v159_v2 import require_run_seal
from v158_nested_runtime_v2 import physical
from v159_boundary_train_v3 import context,model_for,Counter,risk,configure,tensor_hash
from v159_class_direction import class_direction

OUT=ROOT/'artifacts/v159_bound_gradient_repeat_diagnostic_20261002'
TRIAL=ROOT/'artifacts/v159_class_boundary_trial_20261002'
CONTRACT=ROOT/'training/review_policy/v159_gradient_repeat_diagnostic_contract.json'

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def differences(a,b):
    d=a-b;na=float(np.linalg.norm(a));nb=float(np.linalg.norm(b));scale=max(float(np.abs(a).max()),float(np.abs(b).max()),np.finfo(float).tiny)
    return dict(bitwise_equal=bool(np.array_equal(a,b)),maximum_absolute=float(np.abs(d).max()),maximum_relative_to_global_scale=float(np.abs(d).max()/scale),
        maximum_elementwise_relative=float(np.max(np.abs(d)/np.maximum(np.maximum(np.abs(a),np.abs(b)),np.finfo(float).tiny))),L2_difference=float(np.linalg.norm(d)),
        relative_L2=float(np.linalg.norm(d)/max(na,nb,np.finfo(float).tiny)),first_norm=na,second_norm=nb,cosine=float(np.dot(a,b)/(na*nb)) if na and nb else None,
        changed_components=int(np.count_nonzero(d)),sign_changed_components=int(np.count_nonzero(np.sign(a)!=np.sign(b))))

def main():
    assert not OUT.exists() and not CONTRACT.exists();OUT.mkdir()
    original=require_run_seal(TRIAL/'run_seal.json',ROOT/'training/v159_boundary_train_v3.py');configure()
    cost=read(ROOT/'artifacts/v159_preflight_failure_cost_receipt_20261002/receipt.json');assert cost['actual_complete_class_gradients']==4 and cost['official_fits']==cost['official_updates']==0
    paths={Path(__file__).resolve(),ROOT/'training/experiment_review_v159_v2.py',ROOT/'training/v159_boundary_train_v3.py',ROOT/'training/v159_class_direction.py',TRIAL/'run_seal.json',TRIAL/'initial.pt',ROOT/'artifacts/v159_preflight_failure_cost_receipt_20261002/receipt.json',ROOT/'artifacts/v159_independent_preflight_failure_audit_20261002/audit.json'}
    p=dict(protocol='V159-one-technical-repeat-gradient-diagnostic-no-fit-no-update',actual_entry=Path(__file__).resolve().relative_to(ROOT).as_posix(),fold=0,classes=[1,2],repetitions=2,full_class_gradient_cap=4,head_forward_cap=40,opinion_feature_cap=40,fits=0,updates=0,
        previous_costs=dict(head=84,feature=84,full_class_gradient=4),cumulative_after_maximum=dict(head=124,feature=124,full_class_gradient=8),original_total_caps=dict(head=78072,feature=78072,full_class_gradient=2412),
        same_entry_restart=False,new_tolerance_selected=False,quality_acceptance=False,source_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in sorted(paths)})
    save(CONTRACT,p)
    files=physical()|paths|{CONTRACT}|{ROOT/k for k in read(TRIAL/'run_seal.json')['source_sha256']}
    seal=dict(status='technical_diagnostic_physically_sealed_before_any_new_head_feature_gradient',entry=p['actual_entry'],contract_sha256=sha(CONTRACT),python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256={f.relative_to(ROOT).as_posix() if f.is_relative_to(ROOT) else str(f):sha(f) for f in sorted(q.resolve() for q in files)})
    save(OUT/'run_seal.json',seal);check_bindings(seal['source_sha256']);assert p['actual_entry']==Path(__file__).resolve().relative_to(ROOT).as_posix()
    ctx=context(0);model=model_for();initial=tensor_hash(model.state_dict());counter=Counter(model,40,OUT/'actual_calls.jsonl',4)
    features=[]
    def observe_features(_module,_args,output):
        # Observe outputs of the already-counted opinion block; no additional
        # feature function or classifier evaluation is invoked.
        features.append([__import__('hashlib').sha256(t.detach().cpu().numpy().tobytes()).hexdigest() for t in output])
    hook=model.opinions.register_forward_hook(observe_features);vectors=[];risks=[];probability_hashes=[]
    for rep in range(2):
        gs=[];values=[]
        for cls in [1,2]:
            v,q,lp,g=risk(model,ctx,'OOF',ctx['ids'],counter,cls)
            np.save(OUT/f'repetition{rep}_class{cls}_complete_gradient.npy',g)
            save(OUT/f'repetition{rep}_class{cls}_receipt.json',dict(status='complete_gradient_saved_before_repeat_comparison',gradient_sha256=sha(OUT/f'repetition{rep}_class{cls}_complete_gradient.npy'),class_id=cls,repetition=rep,original_class_mass=ctx['mass'].tolist(),stable_class_risks=v.tolist(),parameters_sha256=tensor_hash(model.state_dict()),actual_counts=counter.counts()))
            gs.append(g);values.append(v);probability_hashes.append(__import__('hashlib').sha256(q[ctx['ids']].tobytes()+lp[ctx['ids']].tobytes()).hexdigest())
        vectors.append(gs);risks.append(values)
    hook.remove();counts=counter.counts();counter.close();assert counts['head_attempts']==counts['head_completed']==counts['feature_attempts']==counts['feature_completed']==40 and counts['gradient_attempts']==counts['gradient_completed']==4 and tensor_hash(model.state_dict())==initial
    byclass=[];offsets=[];offset=0
    for name,param in model.named_parameters():offsets.append((name,offset,offset+param.numel()));offset+=param.numel()
    for j,cls in enumerate([1,2]):
        a,b=vectors[0][j],vectors[1][j];segments=[dict(parameter=name,start=start,end=end,metrics=differences(a[start:end],b[start:end])) for name,start,end in offsets]
        byclass.append(dict(class_id=cls,metrics=differences(a,b),parameter_segments=segments,stable_risk_pairs_identical=bool(np.array_equal(risks[0][j],risks[1][j]))))
    directions=[]
    for arm in ['A','B']:
        d=[class_direction(*gs,*ctx['mass'][1:],arm) for gs in vectors]
        details=dict(arm=arm,statuses=[x['status'] for x in d],M_weights=[x['M_weight'] for x in d],M_weight_absolute_difference=abs(d[0]['M_weight']-d[1]['M_weight']),direction_comparison=differences(d[0]['direction'],d[1]['direction']),class_slopes=[x['class_slopes'] for x in d])
        # Bounds only, not a new finite-step/classifier evaluation.
        details['Armijo_bound_difference_by_fixed_steps']={str(s):(1e-4*s*(np.asarray(d[0]['class_slopes'])-np.asarray(d[1]['class_slopes']))).tolist() for s in [1.,.01,2**-40]}
        directions.append(details)
    save(OUT/'qualification.json',dict(status='actual_four_complete_gradient_repeat_diagnostic_saved_and_compared_no_fit_no_update',actual_counts=counts,class_comparisons=byclass,direction_comparisons=directions,
        probability_and_log_probability_all_four_identical=len(set(probability_hashes))==1,opinion_feature_hashes_each_chunk_match_between_repetitions=features[:20]==features[20:],
        parameter_identity_preserved=True,original_initial_sha256=sha(TRIAL/'initial.pt'),runtime_flags=dict(CUBLAS_WORKSPACE_CONFIG=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cuda_matmul_tf32=torch.backends.cuda.matmul.allow_tf32,num_threads=torch.get_num_threads(),torch=torch.__version__,cuda=torch.version.cuda),
        official_fits=0,official_updates=0,cumulative_actual_head_calls=124,cumulative_actual_opinion_feature_calls=124,cumulative_actual_complete_class_gradients=8,quality_acceptance=False,new_repeat_tolerance_selected=False,
        original_failure_gradient_magnitude_still_unavailable=True,source_sha256=seal['source_sha256'],limits=['This new measurement cannot reconstruct the unsaved failed gradients.','A difference is measured; its CUDA operator cause is not certified by this diagnostic alone.','No Armijo candidate, classifier update, source score or quality effect was evaluated.']))
    check_bindings(seal['source_sha256']);print(json.dumps(dict(status='four_real_complete_gradient_diagnostic_terminal',class_metrics=[x['metrics'] for x in byclass],directions=directions,actual_counts=counts),ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),no_automatic_restart=True,source_sha256=sha(__file__)))
        raise
