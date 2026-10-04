"""Record terminal pre-model startup refusal; never restart a registered run."""
import ctypes,datetime,json,shutil
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha
import v169_prior_pair_training_entry_v14 as entry
from v169_pair_execution_review_v2 import MemoryStatus

OUT=ROOT/'artifacts/v169_registered_initial_resource_refusal_20261002'

def main():
    assert not OUT.exists()
    log=ROOT/'artifacts/v169_prior_pair_training_entry_v14_original_console_20261002.txt';text=log.read_text(encoding='utf-8')
    assert "plan=require('initial');configure();results={}" in text
    assert text.rstrip().endswith('RuntimeError: Registered physical RAM prerequisite not met')
    assert entry.OUT.is_dir() and entry.PLAN.is_file()
    contents=sorted(p.name for p in entry.OUT.iterdir())
    assert contents==['registration.json','run_seal.json']
    registration=read(entry.OUT/'registration.json');seal=read(entry.OUT/'run_seal.json');plan=read(entry.PLAN)
    assert registration['run_seal_sha256']==sha(entry.OUT/'run_seal.json') and seal['plan_sha256']==sha(entry.PLAN)
    assert all(registration[k]==0 for k in ['official_heads','official_features','official_derivatives','fits','permanent_updates'])
    free_gpu,total_gpu=torch.cuda.mem_get_info()
    state=MemoryStatus();state.length=ctypes.sizeof(state);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state))
    available=dict(physical_RAM=int(state.avail_phys),available_commit=int(state.avail_page),GPU_free=int(free_gpu),disk_free=int(shutil.disk_usage(ROOT).free))
    resource=plan['resources'];required=dict(physical_RAM=resource['minimum_free_RAM_bytes'],available_commit=resource['minimum_free_commit_bytes'],GPU_free=resource['minimum_free_GPU_bytes'],disk_free=resource['minimum_free_disk_start_bytes'])
    sources=[Path(__file__).resolve(),Path(entry.__file__),entry.PLAN,entry.OUT/'registration.json',entry.OUT/'run_seal.json',log,ROOT/'training/v169_pair_execution_review_v5.py']
    report=dict(status='V169_v14_registered_first_startup_terminal_RAM_refusal_before_backend_or_official_calls',runner_observed_exit_code=1,failure_boundary='main first require(initial), before configure and all ActualBackend construction',official_zero_step_completed=False,formal_registered_directory_contents=contents,preseal_snapshot=plan['preseal_resource_snapshot'],post_failure_CUDA_initialized_resource_snapshot=dict(timestamp_UTC=datetime.datetime.now(datetime.UTC).isoformat(),available=available,required=required,gaps={k:max(0,required[k]-available[k]) for k in required},all_passed=all(available[k]>=required[k] for k in required),snapshot_is_after_failure_not_exact_failure_instant=True),registered_failed_state_retained=True,same_failed_entry_or_sealer_not_restarted=True,source_or_threshold_not_modified=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,historical_heads=19178,historical_complete_derivatives=768,fits_since_V159=9,accepted_updates_since_V159=170,latest_actual_training='V164',latest_complete_delivery='V159',full_quality_acceptance=False,three_goals_complete=False,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    OUT.mkdir();(OUT/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],resources=report['post_failure_CUDA_initialized_resource_snapshot'],official_calls=0,fit=0,updates=0)),flush=True)

if __name__=='__main__':main()
