"""Actual warmed CUDA host + real unforwarded backend live-buffer accounting."""
import json,ctypes
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiment_review import ROOT,read,sha,check_bindings
import v169_prior_pair_training_entry_v12 as entry
import v169_cuda_host_buffer_qualification_v2 as gpu_qualification
from v169_cuda_host_buffer_qualification_v2 import process_memory
from v169_pair_execution_review_v2 import MemoryStatus

OUT=ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002'

def frame_bytes(ctx):
    arrays={};frames={}
    for key,value in ctx.items():
        if isinstance(value,np.ndarray):arrays[key]=int(value.nbytes)
        elif hasattr(value,'memory_usage'):frames[key]=int(value.memory_usage(deep=True).sum())
    x=ctx['x'];arrays['CSR_buffers']=int(x.data.nbytes+x.indices.nbytes+x.indptr.nbytes)
    return dict(arrays=arrays,frames=frames,total_logical_bytes=sum(arrays.values())+sum(frames.values()))

def memory():
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError('Physical/commit availability unavailable')
    return dict(available_physical_bytes=int(state.avail_phys),available_commit_bytes=int(state.avail_page))

def main():
    assert not OUT.exists();OUT.mkdir();before=dict(process=process_memory(),available=memory())
    # Re-run the identical saved max-batch GPU qualification in a distinct
    # artifact directory: twelve synthetic calls, no official inputs.
    warm=OUT/'synthetic_max_batch_GPU'
    with patch.object(gpu_qualification,'OUT',warm):gpu_qualification.main()
    warm_report=read(warm/'qualification.json');assert warm_report['counts']['synthetic_heads']==12
    run=OUT/'unforwarded_real_backend';run.mkdir();plan=read(ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v2.json')
    with patch.object(entry,'OUT',run):backend=entry.ActualBackend(0,'B',plan)
    assert entry.width('B')==1060833 and sum(p.numel() for p in backend.model.parameters())==1060833
    initial=backend.identity();assert float(backend.model.beta.detach())==0.;logical=frame_bytes(backend.ctx)
    # Explicit allocation and touch of all ordinary live vectors outside the
    # eight complete 66-row matrix bound. Values are storage stand-ins only.
    vectors=[np.ones(1060833,np.float64) for _ in range(8)]
    full_outputs=[np.ones((22546,3),np.float64) for _ in range(16)]
    ledgers=[np.ones(len(backend.ctx['OOF_rows']),bool) for _ in range(3)]
    # Maximum-size current blocker frame is an exact original row view copied
    # and given actual measurement columns. It is not a changed gold ledger.
    blocker_frames=[]
    for scope in ['OOF','deployment']:
        frame=backend.ctx[scope+'_rows'].copy()
        for cls in [0,1,2]:frame[f'p{cls}']=0.;frame[f'logp{cls}']=0.
        frame['pred']=np.ones(len(frame),np.int64);frame['actual_margin']=0.;frame['scope']=scope;frame['rival']=0;frame['role']=0;frame['protection_kind']='saved_memory_shape_standin';blocker_frames.append(frame)
    touched=dict(vector_bytes=sum(v.nbytes for v in vectors),complete_output_bytes=sum(v.nbytes for v in full_outputs),ledger_bytes=sum(v.nbytes for v in ledgers),blocker_logical_bytes=sum(int(f.memory_usage(deep=True).sum()) for f in blocker_frames))
    live=dict(process=process_memory(),available=memory())
    # Guard's full-data-frame temporaries are a distinct phase, outside SVD.
    # The actual saved outputs are read-only and no model is called.
    ob={f'{scope}_{name}':np.load(entry.PRIOR/f'role0/endpoint/{scope}_{name}.npy') for scope in ['OOF','deployment'] for name in ['q','logq']}
    guard=backend._guard(ob);assert guard['passed'];after_guard=dict(process=process_memory(),available=memory())
    assert backend.identity()==initial and all(v==0 for v in backend.counter.counts().values());counts=backend.counter.counts();backend.close_resources();assert backend.resources_closed
    qp_arrays=8*66*1060833*8;increment=live['process']['working']-before['process']['working'];combined=qp_arrays+increment+128*1024**2
    quantum=256*1024**2;computed_minimum=((combined+quantum-1)//quantum)*quantum
    source=[Path(__file__).resolve(),Path(entry.__file__),Path(gpu_qualification.__file__),warm/'qualification.json',ROOT/'artifacts/v169_root_full_size_RAM_review_20261002/review.json',ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v2.json']
    report=dict(status='V169_warmed_CUDA_and_actual_unforwarded_full_backend_incremental_RAM_live_shapes_reviewed',before=before,warm_GPU_source=warm.relative_to(ROOT).as_posix(),actual_context_logical_buffers=logical,explicit_live_extra_buffers=touched,QP_phase_base_live=live,guard_phase_actual_saved_rows=after_guard,actual_saved_full_protection_guard=guard,all_complete_parameters_unchanged=True,actual_backend_official_counters=counts,CPU_QP_eight_complete_matrix_bytes=qp_arrays,measured_base_live_resident_increment_after_import=increment,extra_physical_slack_bytes=128*1024**2,prospective_total_incremental_RAM_requirement_before_rounding=combined,prospective_minimum_free_RAM_256MiB_quantum=computed_minimum,physical_check_is_after_import_baseline_not_process_total=True,QP_and_guard_temporary_phase_lifetimes_separate=True,commit_space_must_be_checked_separately_if_resource_contract_changes=True,RAM_contract_unchanged_pending_independent_root_review=True,not_official_zero_step=True,synthetic_heads=12,synthetic_features=12,synthetic_complete_derivatives=12,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
    check_bindings(report['source_sha256']);(OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],resident_increment=increment,prospective_minimum_free_RAM=computed_minimum,official_calls=0)))

if __name__=='__main__':main()
