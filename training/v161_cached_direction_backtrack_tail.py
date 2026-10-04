"""Complete original forty-depth backtrack on one measured fixed direction."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import gc,importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import Counter,configure,CurrentInputBoundary,tensor_hash,probabilities,restore,stats
from v159_float64_repeat_policy_v2 import repeat_values
from v161_fixed_error_endpoint_diagnostic_v2 import endpoint_context,probe,store_scope,joint_check,PRIOR,save
from v161_fixed_pure_error_risk import error_risk
DIAG=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
OUT=ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002'
PLAN=ROOT/'training/review_policy/v161_cached_direction_backtrack_tail_contract.json'
PROTOCOL='V161-role2-same-certified-direction-original40-backtrack-tail-v1'

def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v161_cached_direction_backtrack_tail.py']
    assert sha(PLAN)==seal['plan_sha256'];check_bindings(seal['source_sha256'])
    assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    assert plan['new_caps']==dict(heads=484,features=484,fixed_error_target_gradients=0,full_original_class_gradients=0,margin_gradients=0,finite_proposals=20,QP_solves=0,fits=0,permanent_updates=0)
    return plan

def run():
    plan=require();configure();folder=OUT/'role2';assert not folder.exists();folder.mkdir();ctx=endpoint_context(2)
    model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/'fold2_B/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    initial=tensor_hash(model.state_dict());assert initial==plan['endpoint_parameter_sha256'];base=tuple(p.detach().clone() for p in model.parameters());counter=Counter(model,484,folder/'calls.jsonl',0)
    direction=np.load(DIAG/'role2/round0/polished_direction.npy');cert=read(DIAG/'role2/round0/polished_certificate.json');assert cert['passed'] and cert['status']=='numeric_polish_certified_for_finite_probe_only';slopes=cert['class_slopes']
    failure=None;passed=False;proposals=0;stop='original40_backtrack_domain_exhausted_stop'
    try:
        rv,qo,lo,_=error_risk(model,ctx,ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True)
        same=dict(OOF_q=repeat_values(qo[ctx['ids']],np.load(DIAG/'role2/baseline_error_class1_repeat0/OOF_q.npy')[ctx['ids']],'probability'),deployment_q=repeat_values(qd,np.load(DIAG/'role2/baseline_deployment/deployment_q.npy'),'probability'))
        for key,value in rv.items():same[key]=repeat_values(value,np.load(DIAG/f'role2/baseline_error_class1_repeat0/{key}.npy'),'risk');np.save(folder/f'baseline_{key}.npy',value)
        same['OOF_logq']=repeat_values(lo[ctx['ids']],np.load(DIAG/'role2/baseline_error_class1_repeat0/OOF_logq.npy')[ctx['ids']],'log_probability')
        same['deployment_logq']=repeat_values(ld,np.load(DIAG/'role2/baseline_deployment/deployment_logq.npy'),'log_probability')
        save(folder/'baseline_same_point_review.json',same);assert all(v['passed'] for v in same.values())
        for scope,q,lp in [('OOF',qo,lo),('deployment',qd,ld)]:store_scope(folder/'baseline',ctx,scope,q,lp)
        ctx['baseline_stats']=stats(ctx,qo,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        save(folder/'started.json',dict(parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),direction_sha256=sha(DIAG/'role2/round0/polished_direction.npy'),zero_new_gradient_solver_fit_or_permanent_update=True))
        for j in range(20,40):
            target=folder/f'probe{j}';target.mkdir();proposals+=1
            report,_=probe(model,base,ctx,target,direction,2.**(-j),rv['fixed_pure_error_contribution'],slopes)
            if report['accepted']:passed=True;stop='same_fixed_error_direction_original40_tail_finite_pass_not_fit';break
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure);stop='execution_exception_stop'
    finally:
        restore(model,base);restored=tensor_hash(model.state_dict());assert restored==initial
        rr,rq,rl,_=error_risk(model,ctx,ctx['ids']);rd,rdl=probabilities(model,ctx,'deployment',np.arange(22546),True)
        for scope,q,lp in [('OOF',rq,rl),('deployment',rd,rdl)]:store_scope(folder/'restored',ctx,scope,q,lp)
        joint=joint_check(ctx,rd,rdl);assert joint['passed'];counts=counter.counts();counter.close()
        summary=dict(status=stop,role=2,finite_error_target_pass=passed,counts=counts,finite_proposals=proposals,new_fixed_error_target_gradients=0,new_full_original_class_gradients=0,new_margin_gradients=0,QP_solves=0,new_fits=0,permanent_updates=0,initial_parameter_sha256=initial,restored_parameter_sha256=restored,restored_joint_TRAIN_retention=joint,exception=failure,quality_acceptance=False)
        save(folder/'diagnostic.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Recorded failure; no automatic replay')

if __name__=='__main__':run()
