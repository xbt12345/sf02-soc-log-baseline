"""New lifecycle version: commit acknowledgement and terminal failure cleanup."""
from v169_pair_lifecycle import Schedule,LocalWorkingSet,paired_result
import pandas as pd

def blocker_count(value):
    if isinstance(value,(pd.DataFrame,dict,list,tuple)):return len(value)
    raise TypeError('Unknown blocker container; explicit full original DataFrame required')

def run_fit(backend,schedule=Schedule()):
    schedule.validate();accepted=[];failure=None;terminal_failure=None;stop='fixed_endpoint_not_reached';work=LocalWorkingSet(schedule.working_functions)
    initial=None;last=None;ledger=None;endpoint=None
    def commit_and_record(trial,state):
        nonlocal last,ledger
        ack=backend.commit(trial,state)
        # Official backend captures these BEFORE publishing its commit receipt.
        if ack is not None:
            last,ledger=ack['state'],ack['ledger']
        else:
            last,ledger=backend.snapshot(),backend.protection_snapshot()
        accepted.append(dict(state=state,parameter_sha256=backend.identity()))
        accepted[-1].update(backend.state_review(state))
    try:
        initial=backend.identity();last=backend.snapshot();ledger=backend.protection_snapshot()
        backend.started();observation=backend.baseline();gradients=backend.targets(observation,0);trial=backend.bootstrap(observation,gradients)
        if not trial['accepted']:stop='safe_initialization_actual_replay_rejected'
        else:
            commit_and_record(trial,1)
            for state in range(1,schedule.accepted_updates+1):
                observation=backend.accepted_observation();gradients=backend.targets(observation,state)
                if state==schedule.accepted_updates:stop='fixed20_accepted_updates_completed';break
                trial=backend.direction_trial(observation,gradients,state)
                for correction in range(schedule.corrections):
                    if trial['accepted']:break
                    if blocker_count(trial['blockers'])==0:stop='finite_rejection_without_protected_blocker';break
                    active=work.replace(trial['parameter_sha256'],backend.blocking_functions(trial))
                    normals=backend.trial_normals(trial,active,state,correction)
                    trial=backend.correction_trial(observation,gradients,trial,active,normals,state,correction)
                if not trial['accepted']:
                    if stop=='fixed_endpoint_not_reached':stop='two_corrections_or_local_direction_exhausted'
                    break
                commit_and_record(trial,state+1)
    except Exception as error:
        failure=dict(type=type(error).__name__,message=str(error));stop='technical_or_resource_stop_inconclusive'
        try:backend.failure(failure)
        except Exception as report_error:failure['failure_record_error']=str(report_error)
    finally:
        try:
            if last is None or ledger is None:raise RuntimeError('Initial snapshot/protection not established; no endpoint certification')
            backend.restore(last,ledger);endpoint=backend.final_replay()
        except Exception as error:
            terminal_failure=dict(type=type(error).__name__,message=str(error));stop='terminal_replay_or_restore_failed_inconclusive'
            # Save exact last committed identity even when final scoring fails.
            try:
                if last is not None and ledger is not None:backend.restore(last,ledger)
                terminal_failure['restored_parameter_sha256']=backend.identity()
            except Exception as restore_error:terminal_failure['restore_error']=str(restore_error)
            try:backend.terminal_failure(terminal_failure)
            except Exception as record_error:terminal_failure['failure_record_error']=str(record_error)
        finally:
            try:backend.finished(stop,len(accepted),endpoint)
            except Exception as error:
                stop='terminal_receipt_failed_inconclusive'
                if terminal_failure is None:terminal_failure=dict(type=type(error).__name__,message=str(error))
                else:terminal_failure['receipt_error']=str(error)
            finally:backend.close_resources()
    return dict(status=stop,initial_parameter_sha256=initial,accepted_states=accepted,accepted_updates=len(accepted),endpoint=endpoint,exception=failure,terminal_failure=terminal_failure,
      fixed_endpoint_reached=stop=='fixed20_accepted_updates_completed' and len(accepted)==20 and terminal_failure is None,
      local_function_history=len(work.history),all_protection_in_backend_full_ledger=True,endpoint_is_last_committed_not_best=True,bootstrap_included_in20=True)
