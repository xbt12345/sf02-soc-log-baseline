"""Three fixed V159 B endpoints, bounded probes and no permanent updates.

This entry cannot run without the new immutable source/data/endpoint seal.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,gc,json,traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import context,inputs,risk,probabilities,stats,rows,assign,restore,Counter,configure,tensor_hash,retention,CurrentInputBoundary
from v159_float64_repeat_policy_v2 import repeat_values,repeat_gradient,resolved_class_direction,finite_step_review,EPS,STEP_EPS
from v160_active_margin_direction_v4 import solve_direction
from v160_margin_normal import measure,input_identity

PRIOR=ROOT/'artifacts/v159_class_boundary_numeric_trial_20261002'
OUT=ROOT/'artifacts/v160_fixed_endpoint_diagnostic_20261002'
PLAN=ROOT/'training/review_policy/v160_fixed_endpoint_diagnostic_contract.json'
PROTOCOL='V160-three-fixed-B-endpoint-active-margin-finite-diagnostic-v1'

def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def require():
    seal=read(OUT/'run_seal.json');plan=read(PLAN)
    assert seal['protocol']==plan['protocol']==PROTOCOL and seal['allowed_entries']==['training/v160_fixed_endpoint_diagnostic_v2.py']
    assert seal['plan_sha256']==sha(PLAN);check_bindings(seal['source_sha256'])
    assert plan['new_caps']==dict(heads=3948,features=3948,class_gradients=6,margin_gradients=144,finite_proposals=183,QP_solves=9,fits=0,permanent_updates=0)
    return plan

class DiagnosticCounter(Counter):
    def __init__(self,model,cap,path):
        super().__init__(model,cap,path,2);self.margin_attempts=self.margin_completed=0
    def margin_before(self,identity):
        if self.margin_attempts>=48:raise RuntimeError('Sealed margin-gradient cap exceeded')
        self.margin_attempts+=1;self.write(event='attempt',kind='full_parameter_margin_gradient',ordinal=self.margin_attempts,input_identity=identity)
    def margin_after(self,identity):
        self.margin_completed+=1;self.write(event='completed',kind='full_parameter_margin_gradient',ordinal=self.margin_completed,input_identity=identity)
    def counts(self):return dict(super().counts(),margin_attempts=self.margin_attempts,margin_completed=self.margin_completed)

def store_scope(folder,ctx,scope,q,lp):
    folder.mkdir(exist_ok=True,parents=True)
    np.save(folder/f'{scope}_q.npy',q);np.save(folder/f'{scope}_logq.npy',lp)
    r=rows(ctx,q,lp,scope);r.to_parquet(folder/f'{scope}_original_rows.parquet',index=False)
    r.assign(wrong=r.pred.ne(r.truth)).groupby(['root','truth']).agg(support=('truth','size'),errors=('wrong','sum'),stable_CE_sum=('stable_CE','sum'),probability_clip_CE_sum=('probability_clip_CE','sum')).reset_index().to_parquet(folder/f'{scope}_source_rows.parquet',index=False)

def actual_blockers(ctx,q,lp,scope):
    rr=rows(ctx,q,lp,scope)
    mask=rr.protected_correct if scope=='OOF' else rr.initial_correct
    bad=rr[mask & rr.pred.ne(rr.truth)].copy()
    bad['protection_kind']='initial_pure_OOF_correct' if scope=='OOF' else 'accepted_deployment_initial_correct'
    bad['role']=ctx['fold'];bad['scope']=scope;bad['rival']=bad.pred
    bad['actual_margin']=lp[bad.local,bad.truth]-lp[bad.local,bad.pred]
    return bad

def joint_check(ctx,q,lp):
    population=[]
    for f in range(3):
        if f==ctx['fold']:population.append(rows(ctx,q,lp,'deployment'))
        else:population.append(pd.read_parquet(PRIOR/f'fold{f}_B/endpoint_deployment_rows.parquet'))
    return retention(pd.concat(population,ignore_index=True))

def probe(model,base,ctx,folder,direction,step,base_risks,slopes,counter):
    try:
        assign(model,base,direction,step)
        changed=any(not torch.equal(p,v) for p,v in zip(model.parameters(),base))
        rv,qo,lo,_=risk(model,ctx,'OOF',ctx['ids']);qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True)
        np.save(folder/'risk.npy',rv);np.save(folder/'direction.npy',direction)
        store_scope(folder,ctx,'OOF',qo,lo);store_scope(folder,ctx,'deployment',qd,ld)
        ob=actual_blockers(ctx,qo,lo,'OOF');db=actual_blockers(ctx,qd,ld,'deployment');blockers=pd.concat([ob,db],ignore_index=True)
        blockers.to_parquet(folder/'actual_blocking_original_rows.parquet',index=False)
        os_,ds=stats(ctx,qo,'OOF'),stats(ctx,qd,'deployment');joint=joint_check(ctx,qd,ld)
        guard=os_['protected_regressions']==0 and ds['mastered'] and ds['new_errors_vs_initial']==0 and joint['passed']
        review=finite_step_review(base_risks,rv,slopes,*ctx['mass'][1:],'B',step,guard)
        report=dict(step=float(step),class_slopes=list(slopes),actual_parameter_change=changed,classification_guard=guard,OOF_stats=os_,deployment_stats=ds,joint_TRAIN_retention=joint,finite_numeric_review=review,accepted=bool(review['accepted'] and changed),probe_parameter_sha256=tensor_hash(model.state_dict()),official_fits=0,permanent_updates=0)
        save(folder/'probe.json',report)
        return report,blockers
    finally:restore(model,base)

def add_normals(model,ctx,base,blockers,normal_records,normals,counter,folder):
    pending={}
    for (scope,local,truth,rival),group in blockers.groupby(['scope','local','truth','rival'],sort=True):
        ids=ctx['ids'] if scope=='OOF' else np.arange(22546)
        position=int(np.searchsorted(ids,local));assert ids[position]==local
        start=(position//2048)*2048;chunk=ids[start:start+2048];query=position-start
        rawx=ctx['x'][chunk];rawp=np.asarray(ctx[scope][chunk],dtype=np.float64)
        identity=input_identity(ctx['fold'],scope,chunk,rawx,rawp,query,int(truth),int(rival))
        if identity not in normal_records:
            pending[identity]=(scope,chunk,query,int(truth),int(rival),group)
    if len(normal_records)+len(pending)>24:return 'margin_normal_cap_stop'
    if not pending:return 'no_new_actual_blocking_constraint_stop'
    for identity,(scope,chunk,query,truth,rival,group) in pending.items():
        target=folder/identity;target.mkdir(parents=True);group.to_parquet(target/'blocking_original_rows.parquet',index=False)
        np.save(target/'chunk_local_ids.npy',chunk)
        metadata=dict(input_identity=identity,role=ctx['fold'],scope=scope,query=query,truth=truth,rival=rival,local=int(chunk[query]),base_parameter_sha256=tensor_hash(model.state_dict()),head_chunk_matches_protection_scope=True)
        save(target/'input_binding.json',metadata);results=[]
        for repetition in range(2):
            counter.margin_before(identity);result=measure(model,*inputs(ctx,scope,chunk),query,truth,rival);counter.margin_after(identity)
            for key,value in result.items():np.save(target/f'repeat{repetition}_{key}.npy',value)
            results.append(result)
        review=dict(gradient=repeat_gradient(results[0]['gradient'],results[1]['gradient']),q=repeat_values(results[0]['q'],results[1]['q'],'probability'),logq=repeat_values(results[0]['logq'],results[1]['logq'],'log_probability'),margin=repeat_values([results[0]['margin']],[results[1]['margin']],'margin'))
        save(target/'measurement_repeat_review.json',review)
        if not all(v['passed'] for v in review.values()):return 'margin_measurement_not_repeatable_stop'
        assert all(torch.equal(p,v) for p,v in zip(model.parameters(),base))
        normal_records[identity]=metadata;normals.append(results[0]['gradient'])
    return None

def diagnose(role):
    plan=require();assert role in [0,1,2];configure()
    folder=OUT/f'role{role}';assert not folder.exists();folder.mkdir()
    ctx=context(role);model=CurrentInputBoundary().cuda();checkpoint=PRIOR/f'fold{role}_B/endpoint.pt'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['state'])
    initial=tensor_hash(model.state_dict());assert initial==plan['roles'][role]['endpoint_parameter_sha256']
    base=tuple(p.detach().clone() for p in model.parameters());counter=DiagnosticCounter(model,plan['roles'][role]['head_cap'],folder/'calls.jsonl')
    save(folder/'started.json',dict(role=role,initial_parameter_sha256=initial,run_seal_sha256=sha(OUT/'run_seal.json'),permanent_updates=0))
    stopped='execution_exception_stop';finite_pass=False;proposals=solves=0;normals=[];normal_records={};failure=None
    try:
        gs=[];baseline=[]
        for cls in [1,2]:
            rv,qo,lo,g=risk(model,ctx,'OOF',ctx['ids'],counter,cls);target=folder/f'baseline_class{cls}';target.mkdir()
            np.save(target/'risk.npy',rv);np.save(target/'gradient.npy',g);store_scope(target,ctx,'OOF',qo,lo);gs.append(g);baseline.append((rv,qo,lo))
        rv,qo,lo=baseline[0];same=dict(risk=repeat_values(rv,baseline[1][0],'risk'),q=repeat_values(qo[ctx['ids']],baseline[1][1][ctx['ids']],'probability'),logq=repeat_values(lo[ctx['ids']],baseline[1][2][ctx['ids']],'log_probability'))
        previous=np.load(PRIOR/f'fold{role}_B/endpoint_OOF_probability.npy');same['saved_endpoint_q']=repeat_values(qo[ctx['ids']],previous[ctx['ids']],'probability')
        qd,ld=probabilities(model,ctx,'deployment',np.arange(22546),True);store_scope(folder/'baseline_deployment',ctx,'deployment',qd,ld)
        same['saved_deployment_q']=repeat_values(qd,np.load(PRIOR/f'fold{role}_B/endpoint_deployment_probability.npy'),'probability')
        save(folder/'baseline_same_point_review.json',same);assert all(r['passed'] for r in same.values())
        assert stats(ctx,qo,'OOF')['protected_regressions']==0 and stats(ctx,qd,'deployment')['mastered'] and joint_check(ctx,qd,ld)['passed']
        control=resolved_class_direction(*gs,*ctx['mass'][1:],'B');assert control['status']=='direction_qualified_for_finite_guarded_proposal'
        np.save(folder/'control_direction.npy',control['direction']);save(folder/'control_direction_review.json',{k:v for k,v in control.items() if k!='direction'})
        old=plan['roles'][role]['last_failed_proposal'];target=folder/'control_probe';target.mkdir();proposals+=1
        report,blockers=probe(model,base,ctx,target,control['direction'],old['step'],rv,control['class_slopes'],counter)
        if report['accepted'] or not len(blockers) or any(report[scope]['protected_regressions']!=old[scope]['protected_regressions'] for scope in ['OOF_stats','deployment_stats']):stopped='control_blocker_not_reproduced_stop'
        else:
            stopped='active_round_budget_stop'
            for round_number in range(3):
                issue=add_normals(model,ctx,base,blockers,normal_records,normals,counter,folder/'normals')
                if issue:stopped=issue;break
                normal_array=np.stack(normals);target=folder/f'round{round_number}';target.mkdir()
                np.save(target/'raw_margin_normals.npy',normal_array);save(target/'normal_records.json',normal_records)
                solves+=1;result=solve_direction(*gs,normal_array);np.save(target/'direction.npy',result['direction']);save(target/'QP_certificate.json',{k:v for k,v in result.items() if k!='direction'})
                if result['status']!='local_QP_certified_requires_actual_finite_guard':stopped=result['status'];break
                d=result['direction'];slopes=[float(g@d) for g in gs]
                for backtrack in range(20):
                    trial=target/f'probe{backtrack}';trial.mkdir();proposals+=1
                    report,blockers=probe(model,base,ctx,trial,d,2.**(-backtrack),rv,slopes,counter)
                    if report['accepted']:finite_pass=True;stopped='finite_probe_pass_not_fit';break
                if finite_pass:break
    except Exception as e:
        failure=dict(error_type=type(e).__name__,error=str(e),traceback=traceback.format_exc());save(folder/'execution_failure.json',failure)
    finally:
        restore(model,base);restored=tensor_hash(model.state_dict());assert restored==initial
        # A final full replay is reserved even on a mechanism stop.
        rv_final,q_final,lp_final,_=risk(model,ctx,'OOF',ctx['ids']);qd_final,ld_final=probabilities(model,ctx,'deployment',np.arange(22546),True)
        store_scope(folder/'restored_replay',ctx,'OOF',q_final,lp_final);store_scope(folder/'restored_replay',ctx,'deployment',qd_final,ld_final)
        counts=counter.counts();counter.close()
        summary=dict(status=stopped,role=role,finite_probe_pass=finite_pass,quality_acceptance=False,counts=counts,finite_proposals=proposals,QP_solves=solves,margin_normals=len(normals),official_fits=0,permanent_updates=0,initial_parameter_sha256=initial,restored_parameter_sha256=restored,restored_joint_TRAIN_retention=joint_check(ctx,qd_final,ld_final),exception=failure)
        save(folder/'diagnostic.json',summary);print(json.dumps(summary,ensure_ascii=False),flush=True)
        del model;gc.collect();torch.cuda.empty_cache()
    require()
    if failure:raise RuntimeError('Diagnostic stopped on recorded execution failure; no automatic restart')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--role',type=int,required=True);args=ap.parse_args();diagnose(args.role)
