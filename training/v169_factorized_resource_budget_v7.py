"""Offline v9 whole-run worst storage and explicit physical prerequisites."""
import json,shutil,ctypes,subprocess
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from scipy.sparse import load_npz
from experiment_review import ROOT,read,sha

OUT=ROOT/'artifacts/v169_factorized_resource_budget_v7_20261002'
GiB=1024**3

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]

def memory():
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError('Actual physical memory unavailable')
    return int(state.avail_phys)

def main():
    assert not OUT.exists()
    entry=ROOT/'training/v169_prior_pair_training_entry_v13.py';text=entry.read_text(encoding='utf-8')
    assert 'Registered JSON size bound exceeded' in text and 'Final complete table storage bound exceeded' in text and 'resource_check()' in text
    assert 'factorized_scope' in text and 'save_measurement_outputs' in text and "del backend;torch.cuda.empty_cache()" in text
    input_path=ROOT/'artifacts/v124_header_trial_20260929/B_header_ASA.npz';x=load_npz(input_path);assert x.shape==(22546,66287) and np.diff(x.indptr).max()==211
    cols=np.unique(np.r_[np.unique(x.indices),np.arange(5)]);assert len(cols)==417
    source=[Path(__file__).resolve(),entry,input_path,ROOT/'training/v169_structural_vector_storage.py',ROOT/'training/v169_measurement_output_references_v2.py',ROOT/'training/v169_factorized_row_evidence.py',ROOT/'artifacts/v169_saved_budget_scope_and_callgraph_v2_20261002/review.json']
    callgraph=read(source[-1]);assert callgraph['new_prepared_entry_caps']['heads']==56328
    target_max=0;row_sum=0;logical_tables=0;role_details={}
    for role in [0,1,2]:
        path=ROOT/f'artifacts/v161_independent_frozen_error_cohort_review_20261002/role{role}/target_original_counts.npy';source.append(path);counts=np.load(path)
        widths={str(cls):len(np.unique(x[np.flatnonzero(counts[:,cls])].indices))*16+241 for cls in [1,2]};target_max=max(target_max,*widths.values())
        sizes={}
        for scope in ['OOF','deployment']:
            path=ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role{role}/endpoint/{scope}_original_rows.parquet';source.append(path);table=pq.read_table(path);sizes[scope]=dict(rows=table.num_rows,arrow_logical_bytes=table.nbytes,columns=table.num_columns);logical_tables+=2*table.nbytes
        row_sum+=sizes['OOF']['rows'];role_details[str(role)]=dict(target_coordinate_upper=widths,original_scopes=sizes)
    assert target_max==6209 and row_sum==225614
    parts=dict(target_vectors=504*(6209*16+8192),margin_vectors=29184*(3617*16+8192),QP_vectors=348*(6913*16+8192),margin_raw_outputs=29184*(2048*3*8*2+8192),chunk_ids=14592*(2048*8+128),margin_references=29184*8192,normal_identity_JSON=14592*16384,candidate_outputs=1146*(22546*3*8*4+8192),candidate_masks=2*191*3*225614+1146*8192,target_outputs=504*(22546*3*8*2+8192),accepted_outputs=120*(22546*3*8*4+8192*2)+120*225614,checkpoints=126*(1060833*8+65536),final_tables=640*1024**2,other_metadata_and_logs=256*1024**2,allocation_and_directories=640*1024**2,emergency_dense_vector_and_failure=128*1024**2)
    assert 2*logical_tables+24*1024**2<parts['final_tables']
    metadata=dict(candidate_proofs=1146*16384,candidate_recipes=1146*8192,candidate_original_reference=1146*4096,line_search=912*4096,common_QP=114*65536,correction_QP=228*32768,target_repeats=252*8192,target_identities=126*32768,commits=120*4096,accepted_quality=120*16384,accepted_reference=240*8192,initial_protection=6*65536,final_replay=6*16384,fit_results=6*524288,fit_counts=6*32768,pair_results=3*4096,batch_results=2*4194304,failure_receipts=18*65536,head_and_feature_log_lines=4*56328*128,derivative_log_lines=2*29688*512,QP_log_lines=2*342*512,proposal_update_fit_log_lines=(1146+120+6)*512)
    parts['other_metadata_and_logs']=sum(metadata.values())
    total=sum(parts.values());resources=dict(full_run_worst_storage_bytes=total,fixed_free_disk_reserve_bytes=2*GiB,minimum_free_disk_start_bytes=total+2*GiB,minimum_free_RAM_bytes=6*GiB,minimum_free_GPU_bytes=512*1024**2)
    resources['minimum_free_commit_bytes']=int(6.25*GiB)
    source.extend([ROOT/'artifacts/v169_saved_storage_filecount_review_20261002/review.json',ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json',ROOT/'artifacts/v169_cuda_host_buffer_qualification_v2_20261002/qualification.json',ROOT/'artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json',ROOT/'training/v169_pair_execution_review_v4.py'])
    # Eight full 66-row arrays is still a conservative upper after one copy removal;
    # an extra GiB covers model, CSR, opinion banks, retained output/protection.
    live_RAM=read(source[-4]);ram_bound=8*66*1060833*8+live_RAM['measured_base_live_resident_increment_after_import']+live_RAM['extra_physical_slack_bytes'];assert ram_bound==5936980096 and ram_bound<resources['minimum_free_RAM_bytes']
    gpu_bound=8*(2048*16*16*8)+16*(1060833*8)+128*1024**2;assert gpu_bound<resources['minimum_free_GPU_bytes']
    gpu=subprocess.run(['nvidia-smi','--query-gpu=memory.free,memory.total,name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip();gpu_free=int(gpu.split(',')[0])*1024**2
    snapshot=dict(free_disk_bytes=shutil.disk_usage(ROOT).free,free_RAM_bytes=memory(),free_GPU_bytes=gpu_free,GPU_query=gpu)
    report=dict(status='V169_v13_factorized_whole_run_resource_candidate_runtime_caps_implemented_pending_independent_review',entry_sha256=sha(entry),role_details=role_details,prepared_callgraph=callgraph['new_prepared_entry_caps'],storage_components_upper_bytes=parts,metadata_components_upper_bytes=metadata,maximum_files=128000,maximum_directories=17000,filesystem_padding_bound_formula="128000*4096+17000*8192 < 640MiB",accepted_full_masks_three_copies_counted=True,resources=resources,RAM_complete_live_phase_accounting_upper_bytes=ram_bound,QP_eight_matrix_accounting_upper_bytes=8*66*1060833*8,actual_RAM_live_buffer_review=source[-4].relative_to(ROOT).as_posix(),commit_increment_plus_eight_matrix_and128MiB_bytes=6568317056,minimum_commit_256MiB_rounding_bytes=6710886400,physical_RAM_prerequisite_retained6GiB=True,same_full_SVD_inputs_one_matrix_copy_removed_qualification=source[-2].relative_to(ROOT).as_posix(),GPU_tensor_accounting_upper_bytes=gpu_bound,snapshot=snapshot,current_disk_prerequisite_met=snapshot['free_disk_bytes']>=resources['minimum_free_disk_start_bytes'],current_RAM_prerequisite_met=snapshot['free_RAM_bytes']>=resources['minimum_free_RAM_bytes'],current_GPU_prerequisite_met=gpu_free>=resources['minimum_free_GPU_bytes'],worst_case_does_not_assume_any_same_point_output_reuse=True,full_vector_bits_and_full_original_row_frequency_preserved=True,initial_whole_run_check_then_fixed_reserve_each_fit=True,JSON_and_final_table_caps_runtime_enforced=True,per_operation_fixed_disk_reserve_enforced=True,RAM_workspace_estimate_requires_backend_independent_review=True,official_calls=0,fits=0,permanent_updates=0,execution_authority=False,new_fit_permission=False,supports_physical_seal=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
    OUT.mkdir();(OUT/'review.json').write_bytes((json.dumps(report,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:report[k] for k in ['status','resources','snapshot','current_disk_prerequisite_met','current_RAM_prerequisite_met','current_GPU_prerequisite_met']}))

if __name__=='__main__':main()
