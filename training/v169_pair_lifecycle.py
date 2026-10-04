"""Bounded backend-independent orchestration; no official calls in this module."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Schedule:
    accepted_updates:int=20
    corrections:int=2
    working_functions:int=64
    backtracks:int=8
    def validate(self):
        if (self.accepted_updates,self.corrections,self.working_functions,self.backtracks)!=(20,2,64,8):raise ValueError('Unregistered pair schedule')

class LocalWorkingSet:
    def __init__(self,limit=64):self.limit=limit;self.history={};self.active={};self.point=None
    def replace(self,point,actual_blockers):
        # Full protection is owned by the backend, never by this local object.
        for k,v in actual_blockers.items():self.history[k]=v
        if len(actual_blockers)>self.limit:raise RuntimeError('Current local solver resource bound; full rows remain protected')
        self.point=point;self.active=dict(actual_blockers)
        return self.active

def run_fit(backend,schedule=Schedule()):
    schedule.validate();accepted=[];failure=None;stop='fixed_endpoint_not_reached';work=LocalWorkingSet(schedule.working_functions)
    initial=backend.identity();last=backend.snapshot();ledger=backend.protection_snapshot()
    try:
        backend.started()
        observation=backend.baseline()
        gradients=backend.targets(observation,0)
        trial=backend.bootstrap(observation,gradients)
        if not trial['accepted']:stop='safe_initialization_actual_replay_rejected'
        else:
            backend.commit(trial,1);last=backend.snapshot();ledger=backend.protection_snapshot();accepted.append(dict(state=1,parameter_sha256=backend.identity()));accepted[-1].update(backend.state_review(1))
            for state in range(1,schedule.accepted_updates+1):
                observation=backend.accepted_observation()
                gradients=backend.targets(observation,state)
                if state==schedule.accepted_updates:stop='fixed20_accepted_updates_completed';break
                trial=backend.direction_trial(observation,gradients,state)
                for correction in range(schedule.corrections):
                    if trial['accepted']:break
                    if not trial['blockers']:stop='finite_rejection_without_protected_blocker';break
                    functions=backend.blocking_functions(trial)
                    active=work.replace(trial['parameter_sha256'],functions)
                    # Remeasure every active function at this actual trial.
                    normals=backend.trial_normals(trial,active,state,correction)
                    trial=backend.correction_trial(observation,gradients,trial,active,normals,state,correction)
                if not trial['accepted']:
                    if stop=='fixed_endpoint_not_reached':stop='two_corrections_or_local_direction_exhausted'
                    break
                backend.commit(trial,state+1);last=backend.snapshot();ledger=backend.protection_snapshot();accepted.append(dict(state=state+1,parameter_sha256=backend.identity()));accepted[-1].update(backend.state_review(state+1))
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));stop='technical_or_resource_stop_inconclusive'
        backend.failure(failure)
    finally:
        backend.restore(last,ledger)
        endpoint=backend.final_replay()
        backend.finished(stop,len(accepted),endpoint)
    return dict(status=stop,initial_parameter_sha256=initial,accepted_states=accepted,accepted_updates=len(accepted),endpoint=endpoint,exception=failure,
      fixed_endpoint_reached=stop=='fixed20_accepted_updates_completed' and len(accepted)==20,
      local_function_history=len(work.history),all_protection_in_backend_full_ledger=True,
      endpoint_is_last_committed_not_best=True,bootstrap_included_in20=True)

def paired_result(a,b):
    # No retrospective matching or checkpoint selection after an early stop.
    if not all(r['fixed_endpoint_reached'] and r['accepted_updates']==20 for r in [a,b]):return dict(status='inconclusive_unmatched_registered_fixed_endpoint',supports_candidate_B=False)
    return dict(status='matched_fixed_endpoints_require_complete_original_row_quality_review',supports_candidate_B=False,quality_not_computed_by_lifecycle=True)
