"""Full-parameter CPU scripted lifecycle oracles; failures are not learning evidence."""
import json
from pathlib import Path
import numpy as np
import torch
from experiment_review import ROOT,sha
from v159_boundary_train_v4 import tensor_hash,restore
from v169_prior_pair_model import PriorPairBoundary
from v169_pair_lifecycle import LocalWorkingSet
from v169_pair_lifecycle_v2 import run_fit,paired_result
from v169_prior_pair_training_entry_v2 import PairCounter

OUT=ROOT/'artifacts/v169_pair_lifecycle_qualification_v2_20261002'

class Oracle:
    def __init__(self,folder,fault=None):
        self.folder=folder;folder.mkdir();self.model=PriorPairBoundary('B');self.mask=np.zeros(1024,bool);self.mask[:500]=True;self.point=0;self.closed=False;self.fault=fault;self.target_points=[];self.receipts=[];self.restoration_hashes=[]
        self.counter=PairCounter(self.model,dict(heads=100,fixed_target_derivatives=84,margin_derivatives=4864,QP=57,proposals=191,updates=20),folder/'calls.jsonl')
    def identity(self):return tensor_hash(self.model.state_dict())
    def snapshot(self):return tuple(p.detach().clone() for p in self.model.parameters())
    def protection_snapshot(self):return self.mask.copy()
    def restore(self,state,mask):restore(self.model,state);self.mask=mask.copy();self.restoration_hashes.append(self.identity())
    def started(self):self.counter.write(event='attempt',kind='synthetic_scripted_fit_oracle')
    def baseline(self):
        if self.fault=='baseline':raise RuntimeError('Actual scripted early baseline exception')
        return dict(point=self.identity())
    def targets(self,ob,state):
        if self.fault=='target' and state==1:raise RuntimeError('Actual scripted point1 target exception')
        self.target_points.append((state,self.identity()))
        return None
    def accepted_observation(self):return dict(point=self.identity())
    def bootstrap(self,*args):return dict(accepted=self.fault!='bootstrap_reject',blockers=[],parameter_sha256=self.identity())
    def direction_trial(self,*args):return dict(accepted=self.fault not in ['corrections_fail','normals_throw','working64_resource'],blockers=['actual_scripted_blocker'],parameter_sha256=self.identity())
    def blocking_functions(self,trial):
        n=65 if self.fault=='working64_resource' else 3
        return {f'function{i}':dict(input_identity=f'function{i}',row=i) for i in range(n)}
    def trial_normals(self,*args):
        if self.fault=='normals_throw':
            with torch.no_grad():self.model.beta.fill_(-999.)
            raise RuntimeError('Actual scripted exception after temporary full-parameter trial assignment')
        return None
    def correction_trial(self,ob,gs,trial,active,normals,state,correction):return dict(trial,accepted=False)
    def commit(self,trial,state):
        base=self.snapshot();mask=self.mask.copy();done=False
        try:
            self.counter.enter('update')
            with torch.no_grad():self.model.beta.fill_(-.01*state)
            self.mask[500+state]=True
            if self.fault=='before_receipt' and state==2:raise RuntimeError('Actual scripted commit exception before receipt')
            ack=self.snapshot();ledger=self.mask.copy();receipt=dict(state=state,parameter_sha256=self.identity());self.receipts.append(receipt);self.point=state;done=True
            return dict(state=ack,ledger=ledger,receipt=receipt)
        finally:
            if not done:self.restore(base,mask)
    def state_review(self,state):
        if self.fault=='postcommit_review' and state==1:raise RuntimeError('Actual scripted review failure after committed receipt')
        return dict(state=state,parameter_sha256=self.identity())
    def failure(self,value):(self.folder/'failure.json').write_text(json.dumps(value),encoding='utf-8')
    def terminal_failure(self,value):(self.folder/'terminal_failure.json').write_text(json.dumps(value),encoding='utf-8')
    def final_replay(self):
        if self.fault=='terminal_replay':raise RuntimeError('Actual scripted final replay exception')
        if self.receipts:assert self.identity()==self.receipts[-1]['parameter_sha256']
        assert np.all(self.mask[:500]) and int(self.mask.sum())==500+len(self.receipts)
        return dict(parameter_sha256=self.identity(),protected_rows=int(self.mask.sum()))
    def finished(self,stop,accepted,endpoint):
        if self.fault=='finished_throw':raise RuntimeError('Actual scripted terminal receipt exception')
        (self.folder/'finished.json').write_text(json.dumps(dict(stop=stop,accepted=accepted,endpoint=endpoint)),encoding='utf-8')
    def close_resources(self):
        self.counter.close();self.closed=True

def main():
    assert not OUT.exists();OUT.mkdir();torch.set_num_threads(4);cases=[];good=None
    for fault in [None,'baseline','target','bootstrap_reject','corrections_fail','normals_throw','working64_resource','before_receipt','postcommit_review','terminal_replay','finished_throw']:
        backend=Oracle(OUT/(fault or 'twenty_including_bootstrap'),fault);result=run_fit(backend)
        assert backend.closed and backend.counter.log.closed and not backend.model._forward_pre_hooks and not backend.model._forward_hooks and not backend.model.opinions._forward_pre_hooks and not backend.model.opinions._forward_hooks
        assert result['accepted_updates']==len(backend.receipts)<=20 and all(backend.mask[:500])
        if backend.receipts:assert backend.identity()==backend.receipts[-1]['parameter_sha256']
        if fault is None:
            assert result['fixed_endpoint_reached'] and result['accepted_updates']==20
            assert [state for state,point in backend.target_points]==list(range(21)) and len({point for state,point in backend.target_points})==21
            good=result
        else:assert not result['fixed_endpoint_reached']
        if fault=='postcommit_review':assert result['accepted_updates']==1 and backend.point==1
        if fault=='before_receipt':assert result['accepted_updates']==1 and backend.counter.updates==2
        if fault=='terminal_replay':assert (backend.folder/'terminal_failure.json').exists() and (backend.folder/'finished.json').exists() and result['terminal_failure']
        cases.append(dict(case=fault or 'twenty_including_bootstrap',status=result['status'],accepted=result['accepted_updates'],closed=True,all_old_protection_retained=True,complete_parameters=1060833,terminal_failure=result['terminal_failure']))
        (backend.folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
    # Unbounded history with a bounded current work set, not a union cap.
    work=LocalWorkingSet()
    for stage in range(3):
        active={f'f{i}':dict(row=i) for i in range(stage*32,(stage+1)*32)}
        assert len(work.replace(f'point{stage}',active))==32
    assert len(work.history)==96 and len(work.active)==32
    assert len(work.replace('new_current_point',{'f0':dict(row=0)}))==1 and 'f0' in work.history
    assert paired_result(good,dict(good,fixed_endpoint_reached=False))['status'].startswith('inconclusive')
    assert not paired_result(good,dict(good,accepted_updates=19))['supports_candidate_B']
    assert paired_result(good,good)['status']=='matched_fixed_endpoints_require_complete_original_row_quality_review'
    files=[Path(__file__).resolve(),ROOT/'training/v169_pair_lifecycle.py',ROOT/'training/v169_pair_lifecycle_v2.py',ROOT/'training/v169_prior_pair_training_entry_v2.py',ROOT/'training/v169_prior_pair_model.py',ROOT/'artifacts/v169_root_preparation_adversarial_review_20261002/review.json']
    report=dict(status='V169_full_parameter_scripted_lifecycle_v2_commit_terminal_failure_and_counter_cleanup_qualified',cases=cases,working_history96_current32_drop_and_reinsert_qualified=True,target_parameter_points0_through20_include_initial_and_terminal=True,bootstrap_in20=True,postreceipt_failure_does_not_restore_precommit_state=True,failed_commit_does_not_publish_update=True,terminal_failure_always_closes_actual_PairCounter_hooks_and_log=True,unmatched_registered_endpoint_inconclusive=True,synthetic_oracle_not_actual_training_or_finite_classification_gain=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    (OUT/'qualification.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'));print(json.dumps(dict(status=report['status'],cases=len(cases),official_calls=0)))

if __name__=='__main__':main()
