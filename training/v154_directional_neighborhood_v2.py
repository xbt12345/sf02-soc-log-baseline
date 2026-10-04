"""Registered finite directional stress probe. No optimizer or persistent updates."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,sys,time,importlib.metadata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.func import functional_call
from experiment_review import read,sha,check_bindings,ReviewError
from v135_runtime import load_data,fit_context
from v135_model import tensor_hash
from v138_train import configure
from v138_readout import probabilities
from v142_train import tensors
from v142_retention_check import check as retention
from v146_runtime import require as require_old

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/v154_directional_neighborhood_v2_20261001'
OLD=ROOT/'artifacts/v146_guarded_pair_training_20261001'
PLAN=ROOT/'training/review_policy/v154_directional_neighborhood_plan_v2.json'
POINTS=['base','plus_TRAIN_gradient','minus_TRAIN_gradient','fixed_random']
CHUNK=2048

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def emit(**v):print(json.dumps(v,ensure_ascii=False),flush=True)

def validate(p):
    expected=dict(version='V154-directional-stress-v2',latest_actual_classifier='V146',kind='finite_directional_diagnostic_not_SAM_training',
        folds=[0,1,2],arms=['A','B'],points=POINTS,relative_L2_radius=.001,random_seed_base=17454,
        radius_selection='fixed_before_probe_no_scan_or_backtracking',direction_labels='legal_TRAIN_only_original_frequency_member_CE',
        full_gradient_evaluations=18,base_repeated_gradients_per_endpoint=2,plus_point_gradients_per_endpoint=1,
        persistent_classifier_updates=0,classifier_fits=0,perturbed_functional_points=18,baseline_all_local_evaluations=6,
        parameter_scope='existing_second_layer_22528_only_reference_head_facts_frozen',
        held_labels_used_for_direction=False,cohort_used_for_direction=False,model_selection=False,
        canonicalize_model_inputs=False,automatic_training=False,quality_acceptance=False,
        primary_population='all_112807_original_ASA_and_all_225614_legal_TRAIN_role_rows_per_arm',
        retention='joint_V142_V140_V138_all_correct_TRAIN_including_mixed_majority',
        stopping='exact_fixed_probe_budget_or_execution_error_no_repeat')
    for k,v in expected.items():
        if p.get(k)!=v:raise ReviewError('Unregistered neighborhood probe: '+k)
    check_bindings(p['evidence_sha256'])
    return p

def review():return validate(read(PLAN))

def physical_modules():
    result=set()
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            q=Path(name).resolve()
            if q.suffix=='.pyc' and q.with_suffix('.py').is_file():q=q.with_suffix('.py')
            if q.is_file():result.add(q)
    return result

def key(q):return q.relative_to(ROOT).as_posix() if q.is_relative_to(ROOT) else str(q)

def register():
    p=review();require_old(ROOT/'training/v146_train.py');configure()
    assert not OUT.exists(),'No diagnostic overwrite'
    # Resolve functional/autograd machinery with a separately counted dummy.
    dummy=torch.nn.Linear(1,1,dtype=torch.float64).cuda()
    dp={k:v.detach().clone().requires_grad_(True) for k,v in dummy.named_parameters()}
    z=functional_call(dummy,dp,(torch.ones((1,1),device='cuda',dtype=torch.float64),))
    torch.autograd.grad(z.sum(),tuple(dp.values()))
    del dummy,dp,z
    _,d=load_data();assert len(d)==112807
    OUT.mkdir();extras={Path(__file__),PLAN,ROOT/'training/v154_verify_neighborhood_v2.py'}|{ROOT/z for z in p['evidence_sha256']}
    manifests=[]
    for f in p['folds']:
        frame,c,pure,_,ids=fit_context(d,f)
        path=OUT/f'fold{f}_TRAIN_reference.parquet'
        r=frame[['row_position','local','root','fold','truth','canonical_key']].copy()
        r['pure_TRAIN_input']=pure[frame.local].astype(bool);r.to_parquet(path,index=False);extras.add(path)
        manifests.append(dict(fold=f,role_rows=len(frame),original_class_mass=c.sum(0).tolist(),active_locals=len(ids)))
        h,ff,m=tensors(f)
        for a in p['arms']:
            m.load_state_dict(torch.load(OLD/f'fold{f}_{a}/endpoint.pt',map_location='cpu',weights_only=True)['state'])
            assert sum(v.numel() for v in m.parameters())==22528
        del h,ff,m;gc.collect();torch.cuda.empty_cache()
    extras|={ROOT/z for z in read(OLD/'run_seal.json')['source_sha256']}
    extras|=physical_modules();extras.add(Path(sys.executable).resolve())
    extras.update((Path(torch.__file__).parent/'lib').glob('*.dll'))
    for name in ['python.exe','python3.dll','python311.dll','python312.dll','python313.dll']:
        q=Path(sys.base_prefix)/name
        if q.is_file():extras.add(q)
    bindings={key(q):sha(q) for q in sorted(z.resolve() for z in extras)}
    save(OUT/'run_seal.json',dict(status='sealed_before_classifier_probe',trainer_path=key(Path(__file__)),
        plan_sha256=sha(PLAN),source_sha256=bindings,python_version=sys.version,
        package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','pyarrow']},
        scope='All inherited physical V146 bindings and loaded runtime; OS/GPU driver not byte-snapshotted.'))
    rejected=[]
    for k,v in [('full_gradient_evaluations',36),('relative_L2_radius',.01),('held_labels_used_for_direction',True),
                ('model_selection',True),('automatic_training',True),('canonicalize_model_inputs',True)]:
        q=dict(p);q[k]=v
        try:validate(q)
        except ReviewError:rejected.append(k)
        else:raise ReviewError('Accepted forbidden probe '+k)
    save(OUT/'registration.json',dict(status='sealed_before_classifier_probe',manifests=manifests,
        setup_dummy_forward_calls=1,setup_dummy_gradients=1,classifier_registration_forward_calls=0,
        classifier_registration_gradients=0,negative_cases_rejected=rejected,seal_sha256=sha(OUT/'run_seal.json')))
    emit(stage='registered',physical_bindings=len(bindings),full_classifier_gradients=18,classifier_fits=0,updates=0)

def require():
    p=review();s=read(OUT/'run_seal.json')
    assert s['status']=='sealed_before_classifier_probe' and s['trainer_path']==key(Path(__file__))
    assert s['plan_sha256']==sha(PLAN) and s['python_version']==sys.version
    assert s['package_versions']=={n:importlib.metadata.version(n) for n in s['package_versions']}
    check_bindings(s['source_sha256'])
    return p,s

def flatten(params):return torch.cat([v.reshape(-1) for v in params.values()])

def gradient(m,h,ff,counts,ids,params):
    total=counts[ids].sum();seen=torch.zeros(3,device='cuda',dtype=torch.float64)
    grads={k:torch.zeros_like(v) for k,v in params.items()};value=0.
    for start in range(0,len(ids),CHUNK):
        take=ids[start:start+CHUNK];mass=counts[take]
        logits=functional_call(m,params,(h[take],ff[take]))
        loss=-(torch.log_softmax(logits,-1).mean(1)*mass).sum()/total
        assert bool(torch.isfinite(loss))
        g=torch.autograd.grad(loss,tuple(params.values()))
        for (k,_),v in zip(params.items(),g):grads[k].add_(v.detach())
        seen+=mass.sum(0);value+=float(loss.detach())
    assert torch.equal(seen,counts[ids].sum(0)) and all(bool(torch.isfinite(v).all()) for v in grads.values())
    return value,grads,seen.cpu().tolist()

@torch.no_grad()
def all_point(m,h,ff,counts,params):
    result=[];class_risk=torch.zeros(3,device='cuda',dtype=torch.float64)
    for start in range(0,len(h),CHUNK):
        take=slice(start,start+CHUNK)
        logits=functional_call(m,params,(h[take],ff[take]))
        logprob=torch.log_softmax(logits,-1)
        class_risk+=-(logprob.mean(1)*counts[take]).sum(0)
        result.append(torch.softmax(logits,-1).mean(1).cpu())
    prob=torch.cat(result).numpy();assert np.isfinite(prob).all()
    masses=counts.sum(0);risk=float(class_risk.sum()/masses.sum())
    return prob,risk,[None if mass==0 else float(loss/mass) for loss,mass in zip(class_risk,masses)]

def probe(f,a):
    p,sealed=require();configure();assert f in p['folds'] and a in p['arms']
    folder=OUT/f'fold{f}_{a}';assert not folder.exists(),'No overwrite/resume/repeat'
    _,d=load_data();frame,c,pure,_,ids=fit_context(d,f)
    ref=frame[['row_position','local','root','fold','truth','canonical_key']].copy()
    ref['pure_TRAIN_input']=pure[frame.local].astype(bool)
    assert ref.reset_index(drop=True).equals(pd.read_parquet(OUT/f'fold{f}_TRAIN_reference.parquet').reset_index(drop=True))
    h,ff,m=tensors(f);m.load_state_dict(torch.load(OLD/f'fold{f}_{a}/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    before=tensor_hash(m.state_dict());params={k:v.detach().clone().requires_grad_(True) for k,v in m.named_parameters()}
    counts=torch.as_tensor(c,device='cuda',dtype=torch.float64)
    oldq=np.load(OLD/f'fold{f}_{a}/sealed_all_prob.npy');oldpred=oldq[frame.local].argmax(1)
    folder.mkdir();save(folder/'started.json',dict(fold=f,arm=a,model_sha256=before,seal_sha256=sha(OUT/'run_seal.json')))
    start=time.monotonic();calls=[0]
    hook=m.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
    loss,g,mass=gradient(m,h,ff,counts,ids,params)
    loss2,g2,mass2=gradient(m,h,ff,counts,ids,params)
    assert loss==loss2 and mass==mass2 and all(torch.equal(g[k],g2[k]) for k in g)
    norm=float(flatten(params).norm());rho=p['relative_L2_radius']*norm;gnorm=float(flatten(g).norm());assert gnorm>0 and rho>0
    delta={k:v*(rho/gnorm) for k,v in g.items()}
    generator=torch.Generator(device='cpu');generator.manual_seed(p['random_seed_base']+f)
    rv=torch.randint(0,2,(22528,),generator=generator,dtype=torch.int64).double().mul_(2).sub_(1)
    rv=rv.to('cuda')*(rho/float(rv.norm()));random={};pos=0
    for k,v in params.items():random[k]=rv[pos:pos+v.numel()].reshape_as(v);pos+=v.numel()
    plus={k:(v.detach()+delta[k]).requires_grad_(True) for k,v in params.items()}
    plusloss,gplus,plusmass=gradient(m,h,ff,counts,ids,plus);assert plusmass==mass
    cosine=float(torch.nn.functional.cosine_similarity(flatten(g)[None,:],flatten(gplus)[None,:]))
    torch.save({name:{k:v.cpu() for k,v in values.items()} for name,values in
                [('base_gradient',g),('repeated_base_gradient',g2),('plus_gradient',gplus)]},folder/'gradients.pt')
    directions={'plus_TRAIN_gradient':delta,'minus_TRAIN_gradient':{k:-v for k,v in delta.items()},'fixed_random':random}
    torch.save({point:{k:v.cpu() for k,v in values.items()} for point,values in directions.items()},folder/'directions.pt')
    results=[]
    for point in POINTS:
        pp=params if point=='base' else {k:v.detach()+directions[point][k] for k,v in params.items()}
        q,risk,classrisk=all_point(m,h,ff,counts,pp)
        if point=='base':assert np.array_equal(q,oldq),'Baseline functional replay differs from actual saved endpoint'
        pred=q[frame.local].argmax(1);wrong=pred!=frame.truth.to_numpy()
        rows=ref.copy();rows['training_role']=f;rows['pred_base']=oldpred;rows['pred']=pred
        for cl in range(3):rows[f'p{cl}']=q[frame.local,cl]
        rows.to_parquet(folder/f'{point}_TRAIN_rows.parquet',index=False);np.save(folder/f'{point}_all_prob.npy',q)
        item=dict(point=point,TRAIN_member_CE=risk,TRAIN_class_member_CE=classrisk,
            TRAIN_class_errors=[int((wrong&frame.truth.eq(cl).to_numpy()).sum()) for cl in range(3)],
            pure_TRAIN_errors=int((wrong&rows.pure_TRAIN_input).sum()),
            all_base_correct_TRAIN_regressions=int((wrong&(oldpred==frame.truth)).sum()),
            TRAIN_repairs=int((~wrong&(oldpred!=frame.truth)).sum()),
            perturbation_L2=0. if point=='base' else float(flatten(directions[point]).norm()),
            probability_sha256=sha(folder/f'{point}_all_prob.npy'),TRAIN_rows_sha256=sha(folder/f'{point}_TRAIN_rows.parquet'))
        results.append(item);emit(stage='point_complete',fold=f,arm=a,**item)
    hook.remove();after=tensor_hash(m.state_dict())
    assert before==after and all(v.grad is None for v in m.parameters())
    unbound=sorted(key(z) for z in physical_modules() if key(z) not in sealed['source_sha256'])
    assert not unbound,'New physical runtime dependency after seal: '+str(unbound)
    assert calls[0]==3*((len(ids)+CHUNK-1)//CHUNK)+4*((len(h)+CHUNK-1)//CHUNK)
    report=dict(status='finite_directional_probe_executed',fold=f,arm=a,full_classifier_gradient_evaluations=3,
        classifier_forward_chunk_calls=calls[0],functional_perturbed_points=3,baseline_all_local_evaluations=1,
        classifier_fits=0,persistent_parameter_updates=0,optimizer_updates=0,parameter_values=22528,
        baseline_probability_replay_exact=True,repeated_gradients_exact=True,original_class_mass_per_gradient=mass,
        original_member_CE_gradient=loss,plus_member_CE_gradient=plusloss,base_gradient_L2=gnorm,
        plus_gradient_L2=float(flatten(gplus).norm()),base_plus_gradient_cosine=cosine,parameter_L2=norm,radius_L2=rho,
        before_model_tensor_sha256=before,after_model_tensor_sha256=after,classifier_state_unchanged=True,
        held_labels_used_for_direction=False,cohort_used_for_direction=False,quality_acceptance=False,blind_generalization=False,
        directions_sha256=sha(folder/'directions.pt'),gradients_sha256=sha(folder/'gradients.pt'),seal_sha256=sha(OUT/'run_seal.json'),seconds=time.monotonic()-start,points=results)
    save(folder/'probe.json',report);emit(stage='endpoint_probe_complete',fold=f,arm=a,gradients=3,forward_chunks=calls[0],seconds=report['seconds'])

def summarize():
    p,_=require();_,d=load_data();cohort=pd.read_parquet(ROOT/'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet')
    assert np.array_equal(d.row_position,cohort.row_position)
    summaries=[];checks={};receipts=[]
    for a in p['arms']:
        for f in p['folds']:
            r=read(OUT/f'fold{f}_{a}/probe.json');assert r['status']=='finite_directional_probe_executed';receipts.append(r)
        for point in POINTS:
            joined=[];q=np.zeros((len(d),3));b=np.zeros((len(d),3))
            for f in p['folds']:
                folder=OUT/f'fold{f}_{a}';z=np.load(folder/f'{point}_all_prob.npy');base=np.load(folder/'base_all_prob.npy')
                joined.append(pd.read_parquet(folder/f'{point}_TRAIN_rows.parquet'))
                mask=d.fold.eq(f).to_numpy();q[mask]=z[d.loc[mask,'local']];b[mask]=base[d.loc[mask,'local']]
            train=pd.concat(joined,ignore_index=True);assert len(train)==225614
            guard=retention(train);checks[a+'_'+point]=guard
            train.to_parquet(OUT/f'{a}_{point}_all_TRAIN_role_rows.parquet',index=False)
            rows=d[['row_position','local','root','fold','truth','canonical_key']].copy();rows['pred_base']=b.argmax(1);rows['pred']=q.argmax(1)
            for cl in range(3):rows[f'p{cl}']=q[:,cl]
            rows.to_parquet(OUT/f'{a}_{point}_outer_original_rows.parquet',index=False)
            for name,mask in [('all_ASA',np.ones(len(d),dtype=bool)),('known_578',cohort.known_578_cohort.to_numpy()),
                              ('strict_correct_51',cohort.same_family_and_outer_fold_control_S.to_numpy())]:
                y=d.truth.to_numpy();pred=q.argmax(1);bp=b.argmax(1);wrong=pred!=y
                summaries.append(dict(arm=a,point=point,cohort=name,original_rows=int(mask.sum()),
                    outer_class_errors=[int((mask&wrong&(y==cl)).sum()) for cl in range(3)],
                    outer_repairs=int((mask&~wrong&(bp!=y)).sum()),outer_new_errors=int((mask&wrong&(bp==y)).sum()),
                    mean_pS=float(q[mask,2].mean()),joint_TRAIN_retention_passed=bool(guard['passed'])))
            if point=='base':assert guard['passed']
    save(OUT/'joint_TRAIN_retention_checks.json',checks)
    report=dict(status='fixed_finite_directional_diagnostic_complete',latest_actual_classifier='V146',
        full_classifier_gradient_evaluations=sum(r['full_classifier_gradient_evaluations'] for r in receipts),
        classifier_forward_chunk_calls=sum(r['classifier_forward_chunk_calls'] for r in receipts),
        functional_perturbed_points=sum(r['functional_perturbed_points'] for r in receipts),baseline_all_local_evaluations=6,
        setup_dummy_forward_calls=1,setup_dummy_gradients=1,classifier_fits=0,persistent_parameter_updates=0,
        baseline_replay_and_repeated_gradients_exact=True,all_classifier_states_unchanged=True,
        ASA_original_rows_per_arm_point=112807,TRAIN_role_rows_per_arm_point=225614,
        held_labels_used_for_direction=False,cohort_used_for_direction=False,model_selection=False,
        quality_acceptance=False,blind_generalization=False,automatic_training_qualified=False,issue_solved=False,
        endpoints=receipts,outer_summaries=summaries,limitations=[
            'Three fixed directions at one predeclared radius are not maximum ball sharpness, curvature or proof of causation.',
            'Parameter L2 and sharpness depend on parameterization; no reparameterization-invariant measure was claimed.',
            'Full original-frequency gradient directions differ from stochastic SAM training, including its gradient noise.',
            'All outer truth and descriptive cohorts were previously inspected development data; no selection or blind acceptance.',
            'Functional stress points may break retention and are not accepted updates or replacement checkpoints.',
            'ASA-only stress does not establish full 2056871-row classification improvement.'],
        source_sha256={key(z):sha(z) for z in [PLAN,OUT/'run_seal.json',OUT/'registration.json',OUT/'joint_TRAIN_retention_checks.json']+
            [OUT/f'fold{f}_{a}/probe.json' for f in p['folds'] for a in p['arms']]})
    assert report['full_classifier_gradient_evaluations']==18 and report['functional_perturbed_points']==18
    save(OUT/'audit.json',report);emit(stage='summary_complete',gradients=18,forward_chunks=report['classifier_forward_chunk_calls'],fits=0,updates=0,outer_summaries=summaries)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['register','probe','summarize']);ap.add_argument('--fold',type=int);ap.add_argument('--arm')
    z=ap.parse_args()
    if z.mode=='register':register()
    elif z.mode=='probe':probe(z.fold,z.arm)
    else:summarize()
