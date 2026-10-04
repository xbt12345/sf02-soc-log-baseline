"""Wire206 mixed controls and abort the global batch on technical faults."""
from experiment_review import ROOT,sha
from v169_prepare_training_entry_v2 import change

def main():
    old=ROOT/'training/v169_prior_pair_training_entry_v8.py';target=ROOT/'training/v169_prior_pair_training_entry_v9.py'
    if target.exists():raise FileExistsError(target)
    text=old.read_text(encoding='utf-8')
    text=change(text,'from v169_saved_state_quality import review as saved_state_quality','from v169_saved_state_quality import review as saved_state_quality\nfrom v169_current_correct_context import apply as apply_all_current_correct')
    text=change(text,'class PairCounter(Counter):','class LocalSolverStop(RuntimeError):pass\n\nclass PairCounter(Counter):')
    text=change(text,'self.folder.mkdir();self.ctx=load_context(role)','self.folder.mkdir();self.ctx=load_context(role)\n        initial_protection=apply_all_current_correct(self.ctx,role);save(self.folder/\'all_V164_current_correct_initial_protection.json\',initial_protection)')
    text=text.replace("raise RuntimeError(result['status'])","raise LocalSolverStop(result['status'])")
    text=change(text,"            require();backend=ActualBackend(role,arm,plan)\n            result=run_fit(backend);save(backend.folder/'fit.json',result);results[f'{role}_{arm}']=result\n            del backend;torch.cuda.empty_cache()", "            try:\n                require();backend=ActualBackend(role,arm,plan)\n            except Exception as error:\n                save(OUT/'partial_inconclusive.json',dict(status='constructor_or_seal_check_failed_stop_remaining_fits',completed_results=results,current_role=role,current_arm=arm,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),quality_acceptance=False))\n                raise\n            result=run_fit(backend);save(backend.folder/'fit.json',result);results[f'{role}_{arm}']=result\n            error=result.get('exception')\n            local=error and (error['type']=='LocalSolverStop' or error['message']=='Current local solver resource bound; full rows remain protected')\n            global_stop=result.get('terminal_failure') is not None or (error is not None and not local)\n            del backend;torch.cuda.empty_cache()\n            if global_stop:\n                save(OUT/'partial_inconclusive.json',dict(status='global_technical_or_storage_fault_stop_remaining_fits',completed_results=results,current_role=role,current_arm=arm,paired_result='inconclusive_not_fixed20_matched',quality_acceptance=False))\n                return")
    target.write_bytes(text.encode('utf-8'));print(dict(status='V169_entry_v9_all206_mixed_correct_rows_and_global_fault_batch_abort_wired',old_sha256=sha(old),new_sha256=sha(target),official_calls=0))

if __name__=='__main__':main()
