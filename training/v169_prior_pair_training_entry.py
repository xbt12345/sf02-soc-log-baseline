"""Prepared two-arm entry. Official execution requires a separate reviewed seal."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import argparse,importlib.metadata,json,sys,traceback,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from experiment_review import ROOT,read,sha,check_bindings
from v159_boundary_train_v4 import Counter,configure,inputs,probabilities,stats,rows,assign,restore,tensor_hash
from v159_float64_repeat_policy_v2 import repeat_values,finite_step_review
from v160_fixed_endpoint_diagnostic_v3 import actual_blockers,store_scope,joint_check
from v160_active_margin_direction_v5 import solve_direction
from v160_margin_normal import measure,input_identity
from v161_fixed_pure_error_risk import error_risk
from v163_fixed_endpoint_one_sided_restoration_v2 import blocking_functions,margins
from v165_fixed_endpoint_decision_floor_diagnostic import load_context
from v169_prior_pair_model import PriorPairBoundary,load_origin,width,gradient_repeat,validate_schema
from v169_exact_gradient_storage import save_vector,load_vector
from v169_dynamic_trial_restoration import floors,propose
from v169_pair_lifecycle import Schedule,run_fit,paired_result

OUT=ROOT/'artifacts/v169_prior_pair_training'
PLAN=ROOT/'training/review_policy/v169_prior_pair_execution_contract.json'
PROTOCOL='V169-three-role-paired-learnable-prior-bounded-training-v1'
SAFE={0:ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002/role0/probe0',1:ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/role1/probe0',2:ROOT/'artifacts/v167_trial_point_restoration_diagnostic_20261002/role2/probe0'}
PRIOR=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'

def save(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def strict_class_step(before,after,slopes,mass,classification_guard):
    # V169 arm names NEVER select the old A mean-risk objective mode.
    return finite_step_review(before,after,slopes,*mass,'B',1.,classification_guard)

def require():
    if not PLAN.is_file() or not (OUT/'run_seal.json').is_file():raise RuntimeError('V169 has no execution contract/seal; no official call permitted')
    seal,plan=read(OUT/'run_seal.json'),read(PLAN)
    if not plan.get('execution_authority') or not plan.get('root_review_passed') or plan['protocol']!=PROTOCOL or seal['protocol']!=PROTOCOL:raise RuntimeError('New root-reviewed execution authority required')
    if seal['allowed_entries']!=['training/v169_prior_pair_training_entry.py'] or sha(PLAN)!=seal['plan_sha256']:raise RuntimeError('Entry/contract seal mismatch')
    check_bindings(seal['source_sha256'])
    if sys.flags.optimize or sys.version!=seal['python_version'] or {n:importlib.metadata.version(n) for n in seal['package_versions']}!=seal['package_versions']:raise RuntimeError('Exact execution environment required')
    if plan['schedule']!=dict(accepted_updates=20,corrections=2,working_functions=64,backtracks=8):raise RuntimeError('Unqualified schedule revision')
    if shutil.disk_usage(ROOT).free<plan['resources']['minimum_free_disk_bytes']:raise RuntimeError('Registered full-run disk prerequisite not met')
    return plan

class PairCounter(Counter):
    def __init__(self,model,caps,path):
        super().__init__(model,caps['heads'],path,caps['fixed_target_derivatives']);self.caps=caps;self.point=None;self.margins=self.qps=self.proposals=self.updates=0
    def gradient_before(self,cls,mass):
        if self.gradient_attempts>=self.gradient_cap:raise RuntimeError('Registered fixed-target derivative budget')
        self.gradient_attempts+=1;self.write(event='attempt',kind='complete_fixed_error_target_derivative',ordinal=self.gradient_attempts,class_id=cls,parameter_sha256=self.point)
    def gradient_after(self,cls,mass):self.gradient_completed+=1;self.write(event='completed',kind='complete_fixed_error_target_derivative',ordinal=self.gradient_completed,class_id=cls,parameter_sha256=self.point)
    def margin_before(self,identity):
        if self.margins>=self.caps['margin_derivatives']:raise RuntimeError('Registered margin derivative budget')
        self.margins+=1;self.write(event='attempt',kind='complete_margin_derivative',ordinal=self.margins,input_identity=identity,parameter_sha256=self.point)
    def margin_after(self,identity):self.write(event='completed',kind='complete_margin_derivative',ordinal=self.margins,input_identity=identity,parameter_sha256=self.point)
    def enter(self,kind):
        name={'QP':'qps','proposal':'proposals','update':'updates'}[kind];limit=self.caps[{'QP':'QP','proposal':'proposals','update':'updates'}[kind]]
        if getattr(self,name)>=limit:raise RuntimeError('Registered '+kind+' budget')
        setattr(self,name,getattr(self,name)+1);self.write(event='attempt',kind=kind,ordinal=getattr(self,name))

class ActualBackend:
    def __init__(self,role,arm,plan):
        self.role,self.arm,self.plan=role,arm,plan;self.folder=OUT/f'role{role}_{arm}'
        if self.folder.exists():raise FileExistsError(self.folder)
        self.folder.mkdir();self.ctx=load_context(role)
        self.model=PriorPairBoundary(arm).cuda();load_origin(self.model,torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'])
        self.ctx['baseline_stats']=stats(self.ctx,np.load(PRIOR/f'role{role}/endpoint/OOF_q.npy'),'OOF')
        self.counter=PairCounter(self.model,plan['role_caps'][str(role)],self.folder/'calls.jsonl');self.last_observation=None;self.current_state=0
    def identity(self):return tensor_hash(self.model.state_dict())
    def snapshot(self):return tuple(p.detach().clone() for p in self.model.parameters())
    def protection_snapshot(self):return self.ctx['OOF_rows'].protected_correct.to_numpy(bool).copy()
    def restore(self,state,ledger):
        restore(self.model,state);self.ctx['OOF_rows']['protected_correct']=ledger
        if not all(torch.equal(p,v) for p,v in zip(self.model.parameters(),state)):raise RuntimeError('Exact all-parameter restoration failed')
    def started(self):self.counter.write(event='attempt',kind='fit',role=self.role,arm=self.arm,parameter_sha256=self.identity())
    def baseline(self):return None
    def accepted_observation(self):return self.last_observation
    def _persist_observation(self,folder,ob):
        np.savez_compressed(folder/'complete_outputs.npz',OOF_q=ob['OOF_q'],OOF_logq=ob['OOF_logq'],deployment_q=ob['deployment_q'],deployment_logq=ob['deployment_logq'],fixed_risk=ob['risk']['fixed_pure_error_contribution'],full_risk=ob['risk']['full_original_class_CE'])
    def targets(self,ob,state):
        folder=self.folder/f'point{state}';folder.mkdir();point=self.identity();self.counter.point=point;observed=[];gradients=[]
        for cls in [1,2]:
            pairs=[]
            for repetition in range(2):
                if self.identity()!=point:raise RuntimeError('Stale target parameter identity')
                rv,q,lp,g=error_risk(self.model,self.ctx,self.ctx['ids'],self.counter,cls)
                save_vector(folder/f'class{cls}_repeat{repetition}.npz',g)
                np.savez_compressed(folder/f'class{cls}_repeat{repetition}_outputs.npz',q=q,logq=lp,**rv)
                pairs.append((rv,q,lp,g));observed.append((rv,q,lp))
            review=gradient_repeat(pairs[0][3],pairs[1][3],self.arm);save(folder/f'class{cls}_repeat.json',review)
            if not review['passed']:raise RuntimeError('Complete same-point derivative repeat failed')
            gradients.append(pairs[0][3])
        first=observed[0];used=self.ctx['ids'];reviews=[]
        for rv,q,lp in observed:
            checks=[repeat_values(rv[k],first[0][k],'risk') for k in rv]+[repeat_values(q[used],first[1][used],'probability'),repeat_values(lp[used],first[2][used],'log_probability')]
            if ob is not None:checks.extend([repeat_values(q[used],ob['OOF_q'][used],'probability'),repeat_values(lp[used],ob['OOF_logq'][used],'log_probability')])
            reviews.append(checks)
        if not all(r['passed'] for checks in reviews for r in checks):raise RuntimeError('Same-point outputs/risk/argmax drift')
        if ob is None:
            dq,dlp=probabilities(self.model,self.ctx,'deployment',np.arange(22546),True)
            ob=dict(risk=first[0],OOF_q=first[1],OOF_logq=first[2],deployment_q=dq,deployment_logq=dlp,parameter_sha256=point)
            self.ctx['baseline_stats']=stats(self.ctx,first[1],'OOF')
            if not self._guard(ob)['passed']:raise RuntimeError('Actual original zero-step full protection failed')
            # Actual A/B origin comparison uses saved full outputs, not flags.
            if self.arm=='B':
                with np.load(OUT/f'role{self.role}_A/point0/complete_outputs.npz') as a:
                    for scope in ['OOF','deployment']:
                        ids=used if scope=='OOF' else np.arange(22546)
                        for name,kind in [('q','probability'),('logq','log_probability')]:
                            if not repeat_values(ob[f'{scope}_{name}'][ids],a[f'{scope}_{name}'][ids],kind)['passed']:raise RuntimeError('A/B zero-beta baseline mismatch')
            self._persist_observation(folder,ob)
        save(folder/'identity.json',dict(parameter_sha256=point,arm=self.arm,complete_parameters=width(self.arm),state=state,reviews=reviews,old_point_derivatives_reused=False))
        self.last_observation=ob;return gradients
    def _guard(self,ob):
        oo=stats(self.ctx,ob['OOF_q'],'OOF');de=stats(self.ctx,ob['deployment_q'],'deployment');joint=joint_check(self.ctx,ob['deployment_q'],ob['deployment_logq'])
        counts=all(oo[k]<=self.ctx['baseline_stats'][k] for k in ['M_errors','S_errors'])
        protected=oo['protected_regressions']==0 and de['mastered'] and de['new_errors_vs_initial']==0 and joint['passed']
        return dict(passed=bool(counts and protected),OOF=oo,deployment=de,joint=joint,full_original_class_counts=counts)
    def _trial(self,u,slopes,label):
        self.counter.enter('proposal');folder=self.folder/label;folder.mkdir();base=self.snapshot();origin=self.identity()
        try:
            assign(self.model,base,u,1.)
            rv,q,lp,_=error_risk(self.model,self.ctx,self.ctx['ids']);dq,dlp=probabilities(self.model,self.ctx,'deployment',np.arange(22546),True)
            ob=dict(risk=rv,OOF_q=q,OOF_logq=lp,deployment_q=dq,deployment_logq=dlp,parameter_sha256=self.identity())
            self._persist_observation(folder,ob);save_vector(folder/'displacement.npz',np.asarray(u,np.float64))
            guard=self._guard(ob);blocks=pd.concat([actual_blockers(self.ctx,q,lp,'OOF'),actual_blockers(self.ctx,dq,dlp,'deployment')],ignore_index=True)
            blocks.to_parquet(folder/'all_actual_blocking_original_rows.parquet',index=False)
            finite=strict_class_step(self.last_observation['risk']['fixed_pure_error_contribution'],rv['fixed_pure_error_contribution'],slopes,self.ctx['mass'][1:],guard['passed'])
            changed=self.identity()!=origin;accepted=bool(changed and finite['accepted'])
            result=dict(ob,accepted=accepted,blockers=blocks,u=u,folder=folder,guard=guard,finite=finite)
            save(folder/'proof.json',dict(accepted=accepted,guard=guard,finite=finite,origin_parameter_sha256=origin,trial_parameter_sha256=ob['parameter_sha256'],actual_parameter_change=changed,class_slopes=list(map(float,slopes)),full_row_guard_not_working_set_only=True))
            return result
        finally:
            restore(self.model,base)
            if self.identity()!=origin:raise RuntimeError('Trial restoration identity failed')
    def bootstrap(self,ob,gs):
        old=SAFE[self.role];proof=read(old/('v168_complete_probe_review.json' if self.role==1 else 'probe.json'))
        if not proof['accepted']:raise RuntimeError('Safe original candidate not accepted')
        u=np.load(old/'direction.npy');u=np.r_[u,0.] if self.arm=='B' else u
        result=self._trial(u,[float(g@u) for g in gs],'bootstrap')
        for scope in ['OOF','deployment']:
            ids=self.ctx['ids'] if scope=='OOF' else np.arange(22546)
            for name,kind in [('q','probability'),('logq','log_probability')]:
                if not repeat_values(result[f'{scope}_{name}'][ids],np.load(old/f'{scope}_{name}.npy')[ids],kind)['passed']:raise RuntimeError('Safe initialization exact replay drift')
        return result
    def direction_trial(self,ob,gs,state):
        self.counter.enter('QP');folder=self.folder/f'point{state}'
        result=solve_direction(*gs,np.empty((0,width(self.arm))))
        save(folder/'common_direction.json',{k:v for k,v in result.items() if k!='direction'})
        if result['status']!='local_QP_certified_requires_actual_finite_guard':raise RuntimeError(result['status'])
        trial=None
        for backtrack in range(8):
            step=2.**(-backtrack);u=result['direction']*step
            trial=self._trial(u,[float(g@u) for g in gs],f'point{state}_line{backtrack}')
            save(trial['folder']/'line_search.json',dict(initial_step=1.,step=step,backtrack=backtrack,maximum_proposals=8,first_actual_Armijo_and_full_protection_pass_selected=True,beta_displacement=float(u[-1]) if self.arm=='B' else None,readout_displacement_infinity=float(np.max(np.abs(u[1060784:1060832])))))
            if trial['accepted']:break
        return trial
    def blocking_functions(self,trial):
        functions=blocking_functions(self.ctx,trial['blockers'])
        return {k:dict(v,input_identity=k) for k,v in functions.items()}
    def trial_normals(self,trial,active,state,correction):
        origin=self.identity();base=self.snapshot();folder=self.folder/f'point{state}_correction{correction}_normals';folder.mkdir();normals=[]
        try:
            assign(self.model,base,trial['u'],1.);point=self.identity()
            if point!=trial['parameter_sha256']:raise RuntimeError('Actual trial linearization identity mismatch')
            self.counter.point=point
            for identity,meta in active.items():
                scope,local,truth,rival=[meta[k] for k in ['scope','local','truth','rival']]
                ids=self.ctx['ids'] if scope=='OOF' else np.arange(22546);pos=int(np.searchsorted(ids,local));chunk=ids[pos//2048*2048:pos//2048*2048+2048];query=pos%2048
                if ids[pos]!=local or input_identity(self.role,scope,chunk,self.ctx['x'][chunk],np.asarray(self.ctx[scope][chunk],np.float64),query,truth,rival)!=identity:raise RuntimeError('Complete margin input identity mismatch')
                target=folder/identity;target.mkdir();pairs=[]
                for repetition in range(2):
                    if self.identity()!=point:raise RuntimeError('Stale local trial derivative')
                    self.counter.margin_before(identity);value=measure(self.model,*inputs(self.ctx,scope,chunk),query,truth,rival);self.counter.margin_after(identity)
                    save_vector(target/f'repeat{repetition}.npz',value['gradient']);np.savez_compressed(target/f'repeat{repetition}_outputs.npz',margin=np.array([value['margin']]),q=value['q'],logq=value['logq']);pairs.append(value)
                checks=[gradient_repeat(pairs[0]['gradient'],pairs[1]['gradient'],self.arm),repeat_values(pairs[0]['q'],pairs[1]['q'],'probability'),repeat_values(pairs[0]['logq'],pairs[1]['logq'],'log_probability'),repeat_values(pairs[0]['q'],trial[f'{scope}_q'][chunk],'probability'),repeat_values(pairs[0]['logq'],trial[f'{scope}_logq'][chunk],'log_probability')]
                save(target/'identity_repeat.json',dict(parameter_sha256=point,origin_parameter_sha256=origin,complete_parameters=width(self.arm),input_identity=identity,meta=meta,checks=checks))
                if not all(v['passed'] for v in checks):raise RuntimeError('Current trial complete derivative repeat/drift failed')
                normals.append(pairs[0]['gradient'])
            return np.stack(normals)
        finally:
            restore(self.model,base);self.counter.point=origin
            if self.identity()!=origin:raise RuntimeError('Margin trial restoration failed')
    def correction_trial(self,ob,gs,trial,active,a,state,correction):
        triallogs={scope:trial[f'{scope}_logq'] for scope in ['OOF','deployment']};originlogs={scope:self.last_observation[f'{scope}_logq'] for scope in triallogs};keys=list(active)
        b,c=margins(active,keys,originlogs),margins(active,keys,triallogs);tau=floors(active,triallogs)
        def callback(event,value):
            if event=='call':self.counter.enter('QP')
            else:self.counter.write(event='returned',kind='QP',optimizer_iterations=int(value.nit) if value is not None else 0,exception=value is None)
        result=propose(callback,trial['u'],a,b,c,*gs,tau)
        folder=self.folder/f'point{state}_correction{correction}';folder.mkdir()
        save(folder/'original_unit_review.json',{k:v for k,v in result.items() if k not in ['displacement','correction']})
        if result['status']!='one_sided_joint_restoration_requires_full_actual_finite_guard':raise RuntimeError(result['status'])
        save_vector(folder/'correction.npz',result['correction'])
        return self._trial(result['displacement'],[r['linear_change'] for r in result['class_reviews']],f'point{state}_correction{correction}_trial')
    def commit(self,trial,state):
        if not trial['accepted'] or not self._guard(trial)['passed']:raise RuntimeError('Unaccepted candidate cannot be committed')
        self.counter.enter('update');base=self.snapshot();old=self.protection_snapshot();origin=self.identity();committed=False;target=self.folder/f'accepted{state}';target.mkdir()
        try:
            previous=self.last_observation['OOF_q'];rr=self.ctx['OOF_rows'];local=rr.local.to_numpy();truth=rr.truth.to_numpy();repair=(previous[local].argmax(1)!=truth)&(trial['OOF_q'][local].argmax(1)==truth)
            assign(self.model,base,trial['u'],1.)
            if self.identity()!=trial['parameter_sha256'] or self.identity()==origin:raise RuntimeError('Commit identity mismatch')
            self.ctx['OOF_rows']['protected_correct']=old|repair
            torch.save(dict(state=self.model.state_dict(),state_index=state,parameter_sha256=self.identity(),arm=self.arm),target/'checkpoint.pt')
            for scope in ['OOF','deployment']:store_scope(target,self.ctx,scope,trial[f'{scope}_q'],trial[f'{scope}_logq'])
            self.ctx['OOF_rows'].loc[old|repair,['row_position','local','truth','root','pure_current_input']].to_parquet(target/'complete_cumulative_protection.parquet',index=False)
            save(target/'commit.json',dict(state=state,previous_parameter_sha256=origin,parameter_sha256=self.identity(),repairs=int(repair.sum()),cumulative_protected_rows=int((old|repair).sum()),bootstrap_included_in20=True))
            self.last_observation=trial;self.ctx['baseline_stats']=trial['guard']['OOF'];self.current_state=state;committed=True
        finally:
            if not committed:self.restore(base,old)
    def state_review(self,state):
        trial=self.last_observation;beta=float(self.model.beta.detach()) if self.arm=='B' else 0.
        return dict(state=state,parameter_sha256=self.identity(),beta=beta,prior_coefficient=float(np.exp(beta)),OOF=stats(self.ctx,trial['OOF_q'],'OOF'),deployment=stats(self.ctx,trial['deployment_q'],'deployment'))
    def failure(self,failure):save(self.folder/'execution_failure.json',dict(failure,traceback=traceback.format_exc()))
    def final_replay(self):
        rv,q,lp,_=error_risk(self.model,self.ctx,self.ctx['ids']);dq,dlp=probabilities(self.model,self.ctx,'deployment',np.arange(22546),True)
        ob=dict(risk=rv,OOF_q=q,OOF_logq=lp,deployment_q=dq,deployment_logq=dlp,parameter_sha256=self.identity());guard=self._guard(ob)
        if not guard['passed']:raise RuntimeError('Restored complete endpoint protection failed')
        folder=self.folder/'endpoint';folder.mkdir()
        for scope in ['OOF','deployment']:store_scope(folder,self.ctx,scope,ob[f'{scope}_q'],ob[f'{scope}_logq'])
        return dict(parameter_sha256=self.identity(),guard=guard,accepted_updates=self.current_state)
    def finished(self,stop,accepted,endpoint):
        counts=dict(self.counter.counts(),margin_derivative_attempts=self.counter.margins,QP_attempts=self.counter.qps,finite_proposals=self.counter.proposals,update_attempts=self.counter.updates)
        save(self.folder/'fit_counts.json',dict(stop=stop,accepted_updates=accepted,counts=counts,endpoint=endpoint));self.counter.close()

def main():
    plan=require();configure();results={}
    for role in [0,1,2]:
        for arm in ['A','B']:
            require();backend=ActualBackend(role,arm,plan)
            result=run_fit(backend);save(backend.folder/'fit.json',result);results[f'{role}_{arm}']=result
            del backend;torch.cuda.empty_cache()
        save(OUT/f'role{role}_paired_endpoint.json',paired_result(results[f'{role}_A'],results[f'{role}_B']))
    save(OUT/'completed_or_inconclusive.json',dict(results=results,quality_acceptance=False,full_original_row_independent_audit_required=True))
    require()

if __name__=='__main__':main()
