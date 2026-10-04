"""Actual complete-parameter CPU backend wiring; all model inputs synthetic."""
import copy,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from experiment_review import ROOT,sha
import v169_prior_pair_training_entry_v13 as entry
import v159_boundary_train_v4 as original
from v161_fixed_pure_error_risk import error_risk as original_risk
from v169_pair_lifecycle_v3 import run_fit

OUT=ROOT/'artifacts/v169_actual_backend_synthetic_qualification_v4_20261002'

def cpu_inputs(ctx,scope,ids):
    x=ctx['x'][ids]
    xx=torch.sparse_csr_tensor(torch.tensor(x.indptr,dtype=torch.int64),torch.tensor(x.indices,dtype=torch.int64),torch.tensor(x.data,dtype=torch.float64),size=x.shape)
    return xx,torch.tensor(ctx[scope][ids],dtype=torch.float64)

def risk(model,ctx,ids,counter=None,gradient_class=None):return original_risk(model,ctx,ids,counter,gradient_class,inputs_builder=cpu_inputs)

def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')

def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    fixture=OUT/'synthetic_fixture';prior=fixture/'prior';safe=fixture/'safe';run=fixture/'run';run.mkdir(parents=True)
    ids=np.arange(4);truth=np.array([1,2,1,2]);counts=np.zeros((22546,3),np.int64);counts[ids,truth]=1;target=counts.copy();target[2:]=0
    x=csr_matrix((np.tile(np.linspace(.1,.8,211,dtype=np.float32),4),(np.repeat(np.arange(4),211),np.tile(np.arange(211),4))),shape=(22546,66287),dtype=np.float32);x.sort_indices()
    p=np.tile(np.array([.01,.89,.10],np.float64),(22546,16,1));p[1]=[.01,.89,.10];p[0]=[.01,.10,.89];p[2]=[.01,.89,.10];p[3]=[.01,.10,.89]
    dp=p.copy();dp[0]=[.01,.89,.10];dp[1]=[.01,.10,.89]
    frame=pd.DataFrame(dict(row_position=np.arange(4),local=ids,truth=truth,root=['synthetic']*4,pure_current_input=[True]*4,protected_correct=[False,False,True,True],initial_correct=[False,False,True,True],initial_pred=[2,1,1,2]))
    deploy=frame.copy();deploy['initial_correct']=True;deploy['protected_correct']=True;deploy['initial_pred']=truth
    info={s:dict(current_input_minimum_errors=0,unconstrained_minimum_original_errors=0,initial_correct_guard_constrained_minimum_errors=0,all_initial_correct_guard_constrained_minimum_errors=0) for s in ['OOF','deployment']}
    ctx=dict(fold=0,x=x,ids=ids,counts=counts,target_counts=target,mass=counts.sum(0),OOF=p,deployment=dp,OOF_rows=frame,deployment_rows=deploy,info=info)
    model=entry.PriorPairBoundary('A');state={k:v.clone() for k,v in model.state_dict().items()};(prior/'role0').mkdir(parents=True);torch.save(dict(state=state),prior/'role0/endpoint.pt');write(prior/'role0/fit.json',dict(permanent_updates=0))
    rv,q,lp,_=risk(model,ctx,ids)
    with patch.object(original,'inputs',cpu_inputs):dq,dlp=original.probabilities(model,ctx,'deployment',np.arange(22546),True)
    ep=prior/'role0/endpoint';ep.mkdir()
    for scope,a,b in [('OOF',q,lp),('deployment',dq,dlp)]:
        np.save(ep/f'{scope}_q.npy',a);np.save(ep/f'{scope}_logq.npy',b);original.rows(ctx,a,b,scope).to_parquet(ep/f'{scope}_original_rows.parquet',index=False)
    rr=prior/'role0/parameter_point0/class1_repeat0';rr.mkdir(parents=True)
    for name,value in rv.items():np.save(rr/f'{name}.npy',value)
    cohort=original.rows(ctx,q,lp,'OOF').query('truth==2')[['row_position','local','truth','pred']].copy();cohort['initial_low_prior']=False;cohort['initial_error']=cohort.pred.ne(2)
    cf=fixture/'artifacts/v169_learnable_prior_pair_plan_20261002/role0_all_S_initial_prior_cohort.parquet';cf.parent.mkdir(parents=True);cohort.to_parquet(cf,index=False)
    dummy=entry.PairCounter(model,dict(heads=100,fixed_target_derivatives=4),OUT/'setup_synthetic_calls.jsonl')
    gs=[risk(model,ctx,ids,dummy,c)[3] for c in [1,2]];dummy.close()
    direction=entry.solve_direction(*gs,np.empty((0,1060832)));assert direction['status']=='local_QP_certified_requires_actual_finite_guard';u=direction['direction']*.01
    base=entry.ActualBackend.snapshot(type('Holder',(),dict(model=model))());entry.assign(model,base,u,1.)
    sq,slp=risk(model,ctx,ids)[1:3]
    with patch.object(original,'inputs',cpu_inputs):sdq,sdlp=original.probabilities(model,ctx,'deployment',np.arange(22546),True)
    entry.restore(model,base);safe.mkdir();np.save(safe/'direction.npy',u);write(safe/'probe.json',dict(accepted=True))
    for scope,a,b in [('OOF',sq,slp),('deployment',sdq,sdlp)]:np.save(safe/f'{scope}_q.npy',a);np.save(safe/f'{scope}_logq.npy',b)
    caps=dict(heads=1000,features=1000,fixed_target_derivatives=100,margin_derivatives=100,QP=100,proposals=100,updates=20,fits=1);plan=dict(role_caps={'0':caps},resources=dict(fixed_free_disk_reserve_bytes=2*1024**3))
    cases=[]
    # The legacy multi-role retention oracle and official frozen helper are
    # replaced only here. Their real-data qualifications are separate.
    with patch.object(entry,'ROOT',fixture),patch.object(entry,'PRIOR',prior),patch.object(entry,'OUT',run),patch.object(entry,'SAFE',{0:safe}),patch.object(entry,'load_context',lambda r:copy.deepcopy(ctx)),patch.object(entry,'apply_all_current_correct',lambda c,r:dict(synthetic_only=True)),patch.object(torch.nn.Module,'cuda',lambda self:self),patch.object(entry,'inputs',cpu_inputs),patch.object(original,'inputs',cpu_inputs),patch.object(entry,'error_risk',risk),patch.object(entry,'joint_check',lambda *a:dict(passed=True,synthetic_stub=True)):
        for arm in ['A','B']:
            backend=entry.ActualBackend(0,arm,plan);backend.started();initial=backend.identity();ledger=backend.protection_snapshot();gs=backend.targets(None,0);trial=backend.bootstrap(None,gs);assert trial['accepted'] and backend.identity()==initial
            # Real full-width trial Jacobians and exact output references.
            chunk=ids;identity=entry.input_identity(0,'OOF',chunk,x[chunk],p[chunk],0,1,2);active={identity:dict(scope='OOF',local=0,truth=1,rival=2,input_identity=identity)}
            normals=backend.trial_normals(trial,active,0,0);assert normals.shape==(1,entry.width(arm)) and backend.identity()==initial
            # Pre-receipt failure must restore all parameters/protection.
            original_save=entry.save
            def broken_save(path,value):
                if Path(path).name=='commit.json':raise OSError('Synthetic pre-receipt disk fault')
                return original_save(path,value)
            with patch.object(entry,'save',broken_save):
                try:backend.commit(trial,1)
                except OSError:pass
                else:raise AssertionError('Receipt failure must escape')
            assert backend.identity()==initial and np.array_equal(backend.protection_snapshot(),ledger)
            # Failed accepted1 artifacts stay preserved, so new state2 folder.
            ack=backend.commit(trial,2);assert backend.current_state==2 and ack['receipt']['parameter_sha256']==backend.identity();quality=backend.state_review(2)
            backend.targets(backend.accepted_observation(),2);endpoint=backend.final_replay();assert endpoint['parameter_sha256']==backend.identity()
            # Same-argmax nonprotected drift must refuse even though guard passes.
            unchanged=backend.identity();old=backend.last_observation;bad=copy.deepcopy(old);bad['OOF_q'][0,0]+=1e-6;backend.last_observation=bad
            try:backend.final_replay()
            except RuntimeError as e:assert 'same-point final' in str(e)
            else:raise AssertionError('Final output drift must refuse')
            backend.last_observation=old;assert backend.identity()==unchanged
            # No established point cannot certify a terminal baseline.
            backend.last_observation=None
            try:backend.final_replay()
            except RuntimeError as e:assert 'No same-point' in str(e)
            else:raise AssertionError('Missing accepted observation must refuse')
            backend.last_observation=old;count=backend.counter.counts()
            backend.counter.reserve=10**30
            try:backend.counter.enter('proposal')
            except RuntimeError as e:assert 'disk reserve exhausted' in str(e)
            else:raise AssertionError('Reserve violation must stop before a model call')
            assert backend.counter.counts()==count
            backend.close_resources();assert backend.resources_closed and backend.counter.log.closed and not backend.model._forward_hooks and not backend.model._forward_pre_hooks
            cases.append(dict(arm=arm,complete_parameters=entry.width(arm),actual_synthetic_counter=count,margin_derivatives=backend.counter.margins,committed_state=backend.current_state,full_class_quality=quality['saved_state_quality']['classes'],trial_and_failed_commit_restoration_exact=True,last_commit_output_drift_refused=True,missing_last_observation_refused=True,hooks_and_log_closed=True))
    # Actual main loop must abandon remaining fits on a global fault.
    batch=OUT/'synthetic_batch_abort';batch.mkdir();created=[]
    class Dummy:
        def __init__(self,role,arm,plan):created.append((role,arm));self.folder=batch/f'{role}_{arm}';self.folder.mkdir()
    global_result=dict(exception=dict(type='OSError',message='synthetic disk fault'),terminal_failure=None,fixed_endpoint_reached=False)
    with patch.object(entry,'OUT',batch),patch.object(entry,'require',lambda *a:plan),patch.object(entry,'configure',lambda:None),patch.object(entry,'ActualBackend',Dummy),patch.object(entry,'run_fit',lambda b:global_result),patch.object(torch.cuda,'empty_cache',lambda:None):entry.main()
    assert created==[(0,'A')] and (batch/'partial_inconclusive.json').is_file()
    sources=[Path(__file__).resolve(),ROOT/'training/v169_prior_pair_training_entry_v13.py',ROOT/'training/v169_pair_lifecycle_v3.py',ROOT/'training/v169_prior_pair_model.py',ROOT/'training/v169_pair_execution_review_v4.py']
    report=dict(status='V169_actual_complete_CPU_backend_synthetic_wiring_qualified',cases=cases,actual_global_batch_fault_started_fits=created,bootstrap_and_trial_Jacobians_actual_backend=True,full_target_gradients_and_q_logq_actual=True,complete_model_parameters_preserved=True,legacy_multi_role_retention_and_initial_frozen_helper_stubbed_only_in_synthetic_fixture=True,not_official_old_skill_acceptance=True,actual_official_zero_step_still_required=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    (OUT/'qualification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],cases=len(cases),official_calls=0)))

if __name__=='__main__':main()
