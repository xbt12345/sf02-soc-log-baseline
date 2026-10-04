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
from v169_structural_vector_storage import save_vector,load_vector,digest,reconstruction
from v169_factorized_row_evidence import save_scope as factorized_scope
from v169_dynamic_trial_restoration import floors,propose
from v169_pair_lifecycle_v3 import Schedule,run_fit,paired_result
from v169_saved_state_quality import review as saved_state_quality
from v169_measurement_output_references_v2 import save as save_measurement_outputs

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

def require(phase='running'):
    from v169_pair_execution_review import require_run_seal
    return require_run_seal(OUT/'run_seal.json',__file__,phase)

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
        self.global_columns=np.unique(np.r_[np.unique(self.ctx['x'].indices),np.arange(5)]) # Preserve SVD-prefix coordinates; no model/input reduction
        self.target_columns={cls:np.unique(self.ctx['x'][np.flatnonzero(self.ctx['target_counts'][:,cls])].indices) for cls in [1,2]}
        self.model=PriorPairBoundary(arm).cuda();load_origin(self.model,torch.load(PRIOR/f'role{role}/endpoint.pt',map_location='cpu',weights_only=True)['state'])
        self.ctx['baseline_stats']=stats(self.ctx,np.load(PRIOR/f'role{role}/endpoint/OOF_q.npy'),'OOF')
        self.counter=PairCounter(self.model,plan['role_caps'][str(role)],self.folder/'calls.jsonl');self.last_observation=None;self.current_state=0;self.resources_closed=False
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
                save_vector(folder/f'class{cls}_repeat{repetition}.npz',g,self.target_columns[cls])
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
            for scope in ['OOF','deployment']:
                ids=used if scope=='OOF' else np.arange(22546)
                for name,kind in [('q','probability'),('logq','log_probability')]:
                    previous=np.load(PRIOR/f'role{self.role}/endpoint/{scope}_{name}.npy')
                    if not repeat_values(ob[f'{scope}_{name}'][ids],previous[ids],kind)['passed']:raise RuntimeError('V164 original zero-step risk/argmax/output drift')
            accepted=read(PRIOR/f'role{self.role}/fit.json')['permanent_updates']
            old_risk=PRIOR/f'role{self.role}/parameter_point{accepted}/class1_repeat0'
            for name in first[0]:
                if not repeat_values(first[0][name],np.load(old_risk/f'{name}.npy'),'risk')['passed']:raise RuntimeError('V164 original target/full risk zero-step drift')
            self._persist_observation(folder,ob)
        save(folder/'identity.json',dict(parameter_sha256=point,arm=self.arm,complete_parameters=width(self.arm),state=state,reviews=reviews,old_point_derivatives_reused=False))
        self.last_observation=ob;return gradients
    def _guard(self,ob):
        oo=stats(self.ctx,ob['OOF_q'],'OOF');de=stats(self.ctx,ob['deployment_q'],'deployment');joint=joint_check(self.ctx,ob['deployment_q'],ob['deployment_logq'])
        counts=all(oo[k]<=self.ctx['baseline_stats'][k] for k in ['M_errors','S_errors'])
        protected=oo['protected_regressions']==0 and de['mastered'] and de['new_errors_vs_initial']==0 and joint['passed']
        return dict(passed=bool(counts and protected),OOF=oo,deployment=de,joint=joint,full_original_class_counts=counts)
    def _trial(self,u,slopes,label,recipe=None):
        self.counter.enter('proposal');folder=self.folder/label;folder.mkdir();base=self.snapshot();origin=self.identity()
        try:
            assign(self.model,base,u,1.)
            rv,q,lp,_=error_risk(self.model,self.ctx,self.ctx['ids']);dq,dlp=probabilities(self.model,self.ctx,'deployment',np.arange(22546),True)
            ob=dict(risk=rv,OOF_q=q,OOF_logq=lp,deployment_q=dq,deployment_logq=dlp,parameter_sha256=self.identity())
            self._persist_observation(folder,ob)
            if recipe is None:
                source=folder/'displacement.npz';save_vector(source,np.asarray(u,np.float64),self.global_columns)
                recipe=dict(kind='stored',source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),width=width(self.arm),dense_sha256=digest(u))
            if reconstruction(recipe,ROOT).tobytes()!=np.asarray(u,np.float64).tobytes():raise RuntimeError('Lossless complete displacement recipe mismatch')
            save(folder/'displacement_recipe.json',recipe)
            guard=self._guard(ob);blocks=pd.concat([actual_blockers(self.ctx,q,lp,'OOF'),actual_blockers(self.ctx,dq,dlp,'deployment')],ignore_index=True)
            oo=self.ctx['OOF_rows'];de=self.ctx['deployment_rows']
            np.savez_compressed(folder/'full_original_protection_and_blocker_masks.npz',OOF_protected=oo.protected_correct.to_numpy(bool),OOF_blocked=(oo.protected_correct.to_numpy(bool)&(q[oo.local.to_numpy()].argmax(1)!=oo.truth.to_numpy())),deployment_blocked=(de.initial_correct.to_numpy(bool)&(dq[de.local.to_numpy()].argmax(1)!=de.truth.to_numpy())))
            save(folder/'original_reference_identity.json',{scope:dict(path=(PRIOR/f'role{self.role}/endpoint/{scope}_original_rows.parquet').relative_to(ROOT).as_posix(),sha256=sha(PRIOR/f'role{self.role}/endpoint/{scope}_original_rows.parquet')) for scope in ['OOF','deployment']})
            finite=strict_class_step(self.last_observation['risk']['fixed_pure_error_contribution'],rv['fixed_pure_error_contribution'],slopes,self.ctx['mass'][1:],guard['passed'])
            changed=self.identity()!=origin;accepted=bool(changed and finite['accepted'])
            result=dict(ob,accepted=accepted,blockers=blocks,u=u,folder=folder,guard=guard,finite=finite,recipe=recipe)
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
        source=folder/'complete_common_direction.npz';save_vector(source,result['direction'],self.global_columns)
        trial=None
        for backtrack in range(8):
            step=2.**(-backtrack);u=result['direction']*step
            recipe=dict(kind='scaled',source=source.relative_to(ROOT).as_posix(),source_sha256=sha(source),step_hex=float(step).hex(),width=width(self.arm),dense_sha256=digest(u))
            trial=self._trial(u,[float(g@u) for g in gs],f'point{state}_line{backtrack}',recipe)
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
                    save_vector(target/f'repeat{repetition}.npz',value['gradient'],self.ctx['x'][[local]].indices);save_measurement_outputs(target,repetition,value,trial['folder']/'complete_outputs.npz',scope,chunk);pairs.append(value)
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
        source=folder/'correction.npz';save_vector(source,result['correction'],self.global_columns)
        recipe=dict(kind='added',parent=trial['recipe'],correction=source.relative_to(ROOT).as_posix(),correction_sha256=sha(source),width=width(self.arm),dense_sha256=digest(result['displacement']))
        return self._trial(result['displacement'],[r['linear_change'] for r in result['class_reviews']],f'point{state}_correction{correction}_trial',recipe)
    def commit(self,trial,state):
        if not trial['accepted'] or not self._guard(trial)['passed']:raise RuntimeError('Unaccepted candidate cannot be committed')
        self.counter.enter('update');base=self.snapshot();old=self.protection_snapshot();origin=self.identity();committed=False;target=self.folder/f'accepted{state}';target.mkdir()
        try:
            previous=self.last_observation['OOF_q'];rr=self.ctx['OOF_rows'];local=rr.local.to_numpy();truth=rr.truth.to_numpy();repair=(previous[local].argmax(1)!=truth)&(trial['OOF_q'][local].argmax(1)==truth)
            assign(self.model,base,trial['u'],1.)
            if self.identity()!=trial['parameter_sha256'] or self.identity()==origin:raise RuntimeError('Commit identity mismatch')
            self.ctx['OOF_rows']['protected_correct']=old|repair
            torch.save(dict(state=self.model.state_dict(),state_index=state,parameter_sha256=self.identity(),arm=self.arm),target/'checkpoint.pt')
            for scope in ['OOF','deployment']:
                rr=self.ctx[scope+'_rows'];factorized_scope(target/scope,ROOT,PRIOR/f'role{self.role}/endpoint/{scope}_original_rows.parquet',trial[f'{scope}_q'],trial[f'{scope}_logq'],rr.protected_correct.to_numpy(bool))
            np.save(target/'complete_cumulative_original_row_protection.npy',old|repair)
            acknowledged_state=self.snapshot();acknowledged_ledger=self.protection_snapshot()
            receipt=dict(state=state,previous_parameter_sha256=origin,parameter_sha256=self.identity(),repairs=int(repair.sum()),cumulative_protected_rows=int((old|repair).sum()),bootstrap_included_in20=True)
            save(target/'commit.json',receipt)
            self.last_observation=trial;self.ctx['baseline_stats']=trial['guard']['OOF'];self.current_state=state;committed=True
            return dict(state=acknowledged_state,ledger=acknowledged_ledger,receipt=receipt)
        finally:
            if not committed:self.restore(base,old)
    def state_review(self,state):
        trial=self.last_observation;beta=float(self.model.beta.detach()) if self.arm=='B' else 0.
        coefficient=float(self.model.beta.exp().detach()) if self.arm=='B' else 1.
        reference=pd.read_parquet(PRIOR/f'role{self.role}/endpoint/OOF_original_rows.parquet')
        cohort=pd.read_parquet(ROOT/f'artifacts/v169_learnable_prior_pair_plan_20261002/role{self.role}_all_S_initial_prior_cohort.parquet')
        prior=np.log(np.maximum(np.asarray(self.ctx['OOF'],np.float64).mean(1),1e-12))
        quality=saved_state_quality(reference,trial['OOF_q'],trial['OOF_logq'],cohort,prior,self.model.output_weight.detach().cpu().numpy(),coefficient)
        save(self.folder/f'accepted{state}/full_original_quality_and_fixed_S_slices.json',quality)
        return dict(state=state,parameter_sha256=self.identity(),beta=beta,prior_coefficient=coefficient,OOF=stats(self.ctx,trial['OOF_q'],'OOF'),deployment=stats(self.ctx,trial['deployment_q'],'deployment'),saved_state_quality=quality)
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
        save(self.folder/'fit_counts.json',dict(stop=stop,accepted_updates=accepted,counts=counts,endpoint=endpoint));self.close_resources()
    def terminal_failure(self,failure):
        save(self.folder/'terminal_failure.json',dict(failure,accepted_updates=self.current_state,complete_parameters=width(self.arm),protection_rows=int(self.protection_snapshot().sum())))
    def close_resources(self):
        if not self.resources_closed:
            self.counter.close();self.resources_closed=True

def main():
    plan=require('initial');configure();results={}
    for role in [0,1,2]:
        for arm in ['A','B']:
            require();backend=ActualBackend(role,arm,plan)
            result=run_fit(backend);save(backend.folder/'fit.json',result);results[f'{role}_{arm}']=result
            del backend;torch.cuda.empty_cache()
        save(OUT/f'role{role}_paired_endpoint.json',paired_result(results[f'{role}_A'],results[f'{role}_B']))
    save(OUT/'completed_or_inconclusive.json',dict(results=results,quality_acceptance=False,full_original_row_independent_audit_required=True))
    require()

if __name__=='__main__':main()
