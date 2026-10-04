"""Fixed error objectives, complete original quality, bounded zero-fit probes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,importlib.metadata,json,sys,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import context,Counter,configure,CurrentInputBoundary,tensor_hash,probabilities,assign,restore,stats,rows
from v159_float64_repeat_policy_v2 import repeat_values,finite_step_review
from v160_fixed_endpoint_diagnostic_v3 import store_scope,actual_blockers,joint_check,add_normals
from v160_cached_polished_direction_finite_probe import record_endpoint_progress
from v160_active_margin_direction_v5 import solve_direction
from v160_saved_vector_numeric_polish import polish
from v161_fixed_pure_error_risk import error_risk

PRIOR=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
OLD=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
COHORT=ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT=ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
PLAN=ROOT/'training/review_policy/v161_fixed_error_endpoint_diagnostic_contract.json'
PROTOCOL='V161-three-fixed-B-endpoint-pure-error-target-finite-diagnostic-v1'
def save(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v161_fixed_error_endpoint_diagnostic.py']
    assert sha(PLAN)==seal['plan_sha256'];check_bindings(seal['source_sha256'])
    assert sys.flags.optimize==0 and sys.version==seal['python_version'] and {n:importlib.metadata.version(n) for n in seal['package_versions']}==seal['package_versions']
    assert plan['new_caps']==dict(heads=3878,features=3878,fixed_error_target_gradients=6,full_original_class_gradients=0,margin_gradients=134,finite_proposals=180,QP_solves=9,fits=0,permanent_updates=0)
    return plan

class ErrorCounter(Counter):
    def __init__(self,model,spec,path,target_counts):
        super().__init__(model,spec['head_cap'],path,2);self.target_mass=target_counts.sum(0);self.margin_cap=spec['fresh_margin_gradient_cap'];self.margin_attempts=self.margin_completed=0
    def gradient_before(self,cls,mass):
        if self.gradient_attempts>=2:raise RuntimeError('Fixed error target gradient cap exceeded')
        self.gradient_attempts+=1;self.write(event='attempt',kind='fixed_error_target_gradient',ordinal=self.gradient_attempts,class_id=cls,complete_original_class_mass=int(mass[cls]),fixed_error_original_rows=int(self.target_mass[cls]))
    def gradient_after(self,cls,mass):
        self.gradient_completed+=1;self.write(event='completed',kind='fixed_error_target_gradient',ordinal=self.gradient_completed,class_id=cls)
    def margin_before(self,identity):
        if self.margin_attempts>=self.margin_cap:raise RuntimeError('Fresh paired margin-gradient cap exceeded')
        self.margin_attempts+=1;self.write(event='attempt',kind='full_parameter_margin_gradient',ordinal=self.margin_attempts,input_identity=identity)
    def margin_after(self,identity):
        self.margin_completed+=1;self.write(event='completed',kind='full_parameter_margin_gradient',ordinal=self.margin_completed,input_identity=identity)
    def counts(self):return dict(super().counts(),margin_attempts=self.margin_attempts,margin_completed=self.margin_completed)

def endpoint_context(role):
    ctx=context(role);fixed=pd.read_parquet(OLD/f'role{role}/baseline_class1/OOF_original_rows.parquet')
    assert np.array_equal(ctx['OOF_rows'][['row_position','truth']].to_numpy(),fixed[['row_position','truth']].to_numpy())
    targets=pd.read_parquet(COHORT/f'role{role}/fixed_pure_error_targets.parquet');protection=pd.read_parquet(COHORT/f'role{role}/fixed_pure_correct_protection.parquet')
    expected=fixed[fixed.pure_current_input&fixed.pred.ne(fixed.truth)];correct=fixed[fixed.pure_current_input&fixed.pred.eq(fixed.truth)]
    assert np.array_equal(targets.row_position,expected.row_position) and np.array_equal(protection.row_position,correct.row_position)
    counts=np.load(COHORT/f'role{role}/target_original_counts.npy');direct=np.bincount(targets.local.to_numpy(np.int64)*3+targets.truth.to_numpy(np.int64),minlength=22546*3).reshape(22546,3)
    assert np.array_equal(counts,direct) and np.all(counts<=ctx['counts']) and np.all(counts.sum(0)[1:]>0)
    ctx['target_counts']=counts;ctx['OOF_rows']=ctx['OOF_rows'].copy();ctx['OOF_rows']['protected_correct']=ctx['OOF_rows'].protected_correct.to_numpy()|(fixed.pure_current_input&fixed.pred.eq(fixed.truth)).to_numpy()
    ctx['fixed_endpoint_rows']=fixed
    return ctx

def probe(model,base,ctx,folder,direction,step,base_error,slopes):
    try:
        assign(model,base,direction,step);changed=any(not torch.equal(p,v) for p,v in zip(model.parameters(),base))
        rv,qo,lo,_=error_risk(model,ctx,ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True)
        np.save(folder/'direction.npy',direction);np.save(folder/'fixed_error_risk.npy',rv['fixed_pure_error_contribution']);np.save(folder/'full_original_class_risk.npy',rv['full_original_class_CE'])
        store_scope(folder,ctx,'OOF',qo,lo);store_scope(folder,ctx,'deployment',qd,ld)
        ob=actual_blockers(ctx,qo,lo,'OOF');db=actual_blockers(ctx,qd,ld,'deployment');blockers=pd.concat([ob,db],ignore_index=True)
        blockers['actual_guard_origin']=np.where(blockers.initial_correct,'old_initial_correct_guard','fixed_endpoint_pure_repaired_guard')
        blockers.to_parquet(folder/'actual_blocking_original_rows.parquet',index=False)
        os_,ds=stats(ctx,qo,'OOF'),stats(ctx,qd,'deployment');joint=joint_check(ctx,qd,ld)
        full_count_guard=all(os_[key]<=ctx['baseline_stats'][key] for key in ['M_errors','S_errors'])
        guard=os_['protected_regressions']==0 and ds['mastered'] and ds['new_errors_vs_initial']==0 and joint['passed'] and full_count_guard
        finite=finite_step_review(base_error,rv['fixed_pure_error_contribution'],slopes,*ctx['mass'][1:],'B',step,guard)
        report=dict(step=float(step),class_slopes=list(slopes),actual_parameter_change=changed,classification_guard=guard,full_original_M_S_error_count_guard=full_count_guard,full_CE_not_acceptance_gate=True,OOF_stats=os_,deployment_stats=ds,joint_TRAIN_retention=joint,finite_error_target_review=finite,accepted=bool(finite['accepted'] and changed),probe_parameter_sha256=tensor_hash(model.state_dict()),new_fits=0,permanent_updates=0)
        save(folder/'probe.json',report);record_endpoint_progress(folder,ctx['fold'])
        return report,blockers
    finally:restore(model,base)

def run(role):
    plan=require();assert role in [0,1,2];configure();spec=plan['roles'][role];folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir()
    ctx=endpoint_context(role);model=CurrentInputBoundary().cuda();model.load_state_dict(torch.load(PRIOR/f'fold{role}_B/endpoint.pt',map_location='cpu',weights_only=True)['state'])
    initial=tensor_hash(model.state_dict());assert initial==spec['endpoint_parameter_sha256'];base=tuple(p.detach().clone() for p in model.parameters());counter=ErrorCounter(model,spec,folder/'calls.jsonl',ctx['target_counts'])
    ctx['OOF_rows'].loc[ctx['OOF_rows'].protected_correct].to_parquet(folder/'fixed_correct_protection.parquet',index=False);np.save(folder/'fixed_target_original_counts.npy',ctx['target_counts'])
    save(folder/'started.json',dict(role=role,parameter_sha256=initial,seal_sha256=sha(OUT/'run_seal.json'),target_original_counts=ctx['target_counts'].sum(0).tolist(),complete_original_mass=ctx['mass'].tolist(),new_fits=0,permanent_updates=0))
    proposals=solves=0;passed=False;stop='execution_exception_stop';failure=None;normals=[];records={}
    try:
        observed=[];gs=[]
        for cls in [1,2]:
            rv,q,lp,g=error_risk(model,ctx,ctx['ids'],counter,cls);target=folder/f'baseline_error_class{cls}';target.mkdir()
            np.save(target/'complete_fixed_error_target_gradient.npy',g)
            for name,value in rv.items():np.save(target/f'{name}.npy',value)
            store_scope(target,ctx,'OOF',q,lp);observed.append((rv,q,lp));gs.append(g)
        rv,qo,lo=observed[0];same={key:repeat_values(rv[key],observed[1][0][key],'risk') for key in rv}
        same['q']=repeat_values(qo[ctx['ids']],observed[1][1][ctx['ids']],'probability');same['logq']=repeat_values(lo[ctx['ids']],observed[1][2][ctx['ids']],'log_probability');same['old_q']=repeat_values(qo[ctx['ids']],np.load(OLD/f'role{role}/baseline_class1/OOF_q.npy')[ctx['ids']],'probability')
        qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True);store_scope(folder/'baseline_deployment',ctx,'deployment',qd,ld)
        same['old_deployment_q']=repeat_values(qd,np.load(OLD/f'role{role}/baseline_deployment/deployment_q.npy'),'probability');save(folder/'same_point_review.json',same);assert all(r['passed'] for r in same.values())
        ctx['baseline_stats']=stats(ctx,qo,'OOF');assert ctx['baseline_stats']['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        for norm in spec['cached_normals']:
            metadata=read(ROOT/norm['metadata']);records[metadata['input_identity']]=metadata;normals.append(np.load(ROOT/norm['gradient']))
        blockers=None;stop='fixed_round_budget_stop'
        for round_number in range(3):
            if blockers is not None:
                issue=add_normals(model,ctx,base,blockers,records,normals,counter,folder/'fresh_normals')
                if issue:stop=issue;break
            target=folder/f'round{round_number}';target.mkdir();a=np.stack(normals);np.save(target/'raw_margin_normals.npy',a);save(target/'normal_records.json',records)
            solves+=1;seed=solve_direction(*gs,a);np.save(target/'unpolished_direction.npy',seed['direction']);save(target/'unpolished_QP.json',{k:v for k,v in seed.items() if k!='direction'})
            if 'alpha' not in seed or len(seed.get('multipliers',[]))!=len(a):stop=seed['status'];break
            result=polish(*gs,a,seed)
            for key in ['direction','raw']:
                if key in result:np.save(target/f'polished_{key}.npy',result[key])
            save(target/'polished_certificate.json',{k:v for k,v in result.items() if k not in ['direction','raw']})
            if result['status']!='numeric_polish_certified_for_finite_probe_only':stop=result['status'];break
            for j in range(20):
                proposal=target/f'probe{j}';proposal.mkdir();proposals+=1
                report,blockers=probe(model,base,ctx,proposal,result['direction'],2.**(-j),rv['fixed_pure_error_contribution'],result['class_slopes'])
                if report['accepted']:passed=True;stop='fixed_error_target_finite_probe_pass_not_fit';break
            if passed:break
    except Exception as error:
        failure=dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);restored=tensor_hash(model.state_dict());assert restored==initial
        rr,rq,rl,_=error_risk(model,ctx,ctx['ids']);rd,rdl=probabilities(model,ctx,'deployment',np.arange(22546),True)
        store_scope(folder/'restored',ctx,'OOF',rq,rl);store_scope(folder/'restored',ctx,'deployment',rd,rdl);joint=joint_check(ctx,rd,rdl);assert joint['passed']
        counts=counter.counts();counter.close();summary=dict(status=stop,role=role,finite_error_target_pass=passed,counts=counts,finite_proposals=proposals,QP_solves=solves,margin_normals=len(normals),new_full_original_class_gradients=0,new_fixed_error_target_gradients=counts['gradient_attempts'],new_fits=0,permanent_updates=0,initial_parameter_sha256=initial,restored_parameter_sha256=restored,restored_joint_TRAIN_retention=joint,exception=failure,quality_acceptance=False)
        save(folder/'diagnostic.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True);del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Recorded diagnostic failure; no automatic restart')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--role',type=int,choices=[0,1,2],required=True);args=ap.parse_args();run(args.role)
