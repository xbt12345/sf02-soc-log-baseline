"""Post-CUDA metadata snapshot after actual physical RAM preseal refusal."""
import ctypes,datetime,json,shutil,sys
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha
import v169_prior_pair_training_entry_v13 as entry
from v169_pair_execution_review_v2 import MemoryStatus

OUT=ROOT/'artifacts/v169_actual_preseal_resource_shortfall_20261002'

def main():
    assert not OUT.exists() and not entry.OUT.exists() and not entry.PLAN.exists()
    failure_log=ROOT/'artifacts/v169_seal_prior_pair_training_v4_original_console_20261002.txt';assert 'Registered physical RAM prerequisite not met' in failure_log.read_text(encoding='utf-8')
    candidate=ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json';plan=read(candidate);resources=plan['resources']
    root_review=ROOT/'artifacts/v169_root_final_preseal_review_20261002/review.json';assert read(root_review)['supports_physical_seal'] is True
    gpu_free,gpu_total=torch.cuda.mem_get_info();torch.cuda.synchronize()
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError('Actual physical/commit snapshot unavailable')
    values=dict(physical_RAM=int(state.avail_phys),available_commit=int(state.avail_page),GPU_free=int(gpu_free),disk_free=int(shutil.disk_usage(ROOT).free))
    minimums=dict(physical_RAM=resources['minimum_free_RAM_bytes'],available_commit=resources['minimum_free_commit_bytes'],GPU_free=resources['minimum_free_GPU_bytes'],disk_free=resources['minimum_free_disk_start_bytes'])
    gaps={name:dict(actual_available_bytes=value,required_available_bytes=minimums[name],shortfall_bytes=max(0,minimums[name]-value),passed=value>=minimums[name]) for name,value in values.items()}
    report=dict(status='V169_actual_physical_RAM_preseal_refusal_recorded_no_registered_or_official_training',timestamp_UTC=datetime.datetime.now(datetime.UTC).isoformat(),metadata_snapshot_after_CUDA_initialization=True,resources=gaps,GPU_total_bytes=int(gpu_total),all_current_resource_prerequisites_met=all(v['passed'] for v in gaps.values()),formal_OUT_exists=entry.OUT.exists(),formal_PLAN_exists=entry.PLAN.exists(),registration_completed=False,failed_seal_attempt_has_no_written_run_directory=True,root_design_review_remains_valid=True,root_review_support_is_conditional_on_actual_resources=True,latest_actual_training='V164',latest_complete_quality_delivery='V159',current_direction='V169_same_v13_and_registered_candidate_wait_actual_resources',no_threshold_lowered_or_application_closed=True,no_blind_repeated_seal_attempt=True,resource_change_required_before_same_sealer_retry=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [Path(__file__).resolve(),Path(entry.__file__),candidate,root_review,ROOT/'training/v169_seal_prior_pair_training_v4.py',failure_log,ROOT/'artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json']})
    OUT.mkdir();(OUT/'snapshot.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],resources=gaps,official_calls=0)))

if __name__=='__main__':main()
