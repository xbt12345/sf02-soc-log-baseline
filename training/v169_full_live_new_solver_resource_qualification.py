"""Independent-process full-rank QP with warmed CUDA and retained real context."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from experiment_review import ROOT, read, sha, check_bindings
import v169_prior_pair_training_entry_v13 as entry
import v169_cuda_host_buffer_qualification_v2 as warmup
import v169_working_joint_restoration_v2 as solver
from v169_cuda_host_buffer_qualification_v2 import process_memory
from v169_incremental_RAM_live_buffer_review import frame_bytes, memory
from v169_dynamic_trial_restoration_v2 import propose

def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def numpy_lapack_workspace(width):
    libraries = list((Path(np.__file__).resolve().parent.parent / 'numpy.libs').glob('*openblas*.dll'))
    assert len(libraries) == 1
    library = ctypes.CDLL(str(libraries[0]))
    function = getattr(library, 'scipy_dgesdd_64_')
    integer = ctypes.c_int64
    m, n, lda, ldu, ldvt = [integer(v) for v in [66, width, 66, 66, 66]]
    job = ctypes.c_char(b'S'); lwork = integer(-1); info = integer(0)
    dummy = ctypes.c_double(0.); work = ctypes.c_double(0.); iwork = integer(0)
    function(ctypes.byref(job), ctypes.byref(m), ctypes.byref(n), ctypes.byref(dummy), ctypes.byref(lda), ctypes.byref(dummy), ctypes.byref(dummy), ctypes.byref(ldu), ctypes.byref(dummy), ctypes.byref(ldvt), ctypes.byref(work), ctypes.byref(lwork), ctypes.byref(iwork), ctypes.byref(info))
    assert info.value == 0 and work.value > 0 and work.value.is_integer()
    return dict(library_path=str(libraries[0]), library_sha256=sha(libraries[0]), symbol='scipy_dgesdd_64_', m=66, n=width, JOBZ='S', lwork_elements=int(work.value), workspace_bytes=int(work.value)*8)

def main(arm, out):
    assert not out.exists()
    out.mkdir()
    before = dict(process=process_memory(), available=memory())
    with patch.object(warmup, 'OUT', out/'synthetic_max_batch_GPU'):
        warmup.main()
    warm_report = read(out/'synthetic_max_batch_GPU/qualification.json')
    assert warm_report['counts'] == dict(synthetic_heads=12, synthetic_features=12, synthetic_complete_derivatives=12)
    run = out/'unforwarded_real_backend'; run.mkdir()
    plan_path = ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json'
    with patch.object(entry, 'OUT', run):
        backend = entry.ActualBackend(0, arm, read(plan_path))
    width = entry.width(arm)
    assert width == 1060832 + (arm == 'B')
    assert sum(p.numel() for p in backend.model.parameters()) == width
    initial = backend.identity()
    if arm == 'B': assert float(backend.model.beta.detach()) == 0.
    context = frame_bytes(backend.ctx)
    vectors = [np.ones(width, np.float64) for _ in range(8)]
    outputs = [np.ones((22546,3), np.float64) for _ in range(16)]
    ledgers = [np.ones(len(backend.ctx['OOF_rows']), bool) for _ in range(3)]
    frames = []
    for scope in ['OOF', 'deployment']:
        frame = backend.ctx[scope+'_rows'].copy()
        for cls in [0,1,2]: frame[f'p{cls}']=0.; frame[f'logp{cls}']=0.
        frame['pred']=np.ones(len(frame),np.int64);frame['actual_margin']=0.;frame['scope']=scope
        frame['rival']=0;frame['role']=0;frame['protection_kind']='saved_memory_shape_standin'
        frames.append(frame)
    touched = dict(vector_bytes=sum(v.nbytes for v in vectors), complete_output_bytes=sum(v.nbytes for v in outputs), ledger_bytes=sum(v.nbytes for v in ledgers), blocker_logical_bytes=sum(int(f.memory_usage(deep=True).sum()) for f in frames))
    live = dict(process=process_memory(), available=memory())
    u = np.empty(width); u.fill(0.); u[:2]=1.
    gm=np.empty(width);gm.fill(0.);gm[0]=-1.
    gs=np.empty(width);gs.fill(0.);gs[1]=-1.
    if arm=='B':gm[-1]=.1;gs[-1]=.2
    a=np.empty((64,width));a.fill(0.);a[np.arange(64),np.arange(2,66)]=1.
    # Retain signed zeros without creating an extra full-P matrix.
    a[0,100]=-0.;gm[100]=-0.;gs[101]=-0.
    b=np.full(64,.2);c=np.full(64,-.1);tau=np.full(64,16*np.finfo(np.float64).eps)
    original_shapes=dict(normals=list(a.shape), displacement=list(u.shape), gm=list(gm.shape), gs=list(gs.shape))
    lapack=numpy_lapack_workspace(width)
    assert lapack['workspace_bytes'] < 66*width*8
    events=[];QP_calls=0
    def callback(event,value):
        nonlocal QP_calls
        if event=='call':QP_calls+=1
        events.append(dict(phase='minimize_'+event,process=process_memory(),available=memory()))
    real_svd=np.linalg.svd
    def observed_svd(matrix, *args, **kwargs):
        assert matrix.shape==(66,width) and matrix.dtype==np.float64 and kwargs.get('full_matrices') is False
        events.append(dict(phase='before_full_66_by_P_SVD',shape=list(matrix.shape),nbytes=matrix.nbytes,C_contiguous=matrix.flags.c_contiguous,process=process_memory(),available=memory()))
        value=real_svd(matrix,*args,**kwargs)
        assert value[2].shape==(66,width)
        events.append(dict(phase='after_full_66_by_P_SVD',output_shapes=[list(v.shape) for v in value],process=process_memory(),available=memory()))
        return value
    start=time.perf_counter()
    with patch.object(np.linalg,'svd',observed_svd):
        result=propose(callback,u,a,b,c,gm,gs,tau)
    elapsed=time.perf_counter()-start
    assert QP_calls==1 and result['normal_rank']==66
    assert result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard'
    assert len(result['inequality_reviews'])==66 and all(v['passed'] for v in result['inequality_reviews'])
    assert all(v['resolved_negative'] for v in result['class_reviews'])
    qp_end=dict(process=process_memory(),available=memory())
    # Keep the original ordinary live buffers and returned QP arrays resident
    # during a real saved-output protection check; no SOC forward occurs.
    ob={f'{scope}_{name}':np.load(entry.PRIOR/f'role0/endpoint/{scope}_{name}.npy') for scope in ['OOF','deployment'] for name in ['q','logq']}
    guard=backend._guard(ob);assert guard['passed']
    guard_end=dict(process=process_memory(),available=memory())
    assert all(v.shape==(width,) and v[0]==1. and v[-1]==1. for v in vectors)
    assert len(outputs)==16 and len(frames)==2 and len(ledgers)==3
    assert backend.identity()==initial and all(v==0 for v in backend.counter.counts().values())
    counts=backend.counter.counts();backend.close_resources();assert backend.resources_closed
    sources=[Path(__file__).resolve(),Path(entry.__file__),Path(warmup.__file__),Path(solver.__file__),ROOT/'training/v169_dynamic_trial_restoration_v2.py',ROOT/'training/v169_incremental_RAM_live_buffer_review.py',plan_path,out/'synthetic_max_batch_GPU/qualification.json']
    source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources};check_bindings(source_sha256)
    report=dict(status='independent_process_new_solver_rank66_full_width_with_warmed_CUDA_and_retained_real_backend',arm=arm,pid=os.getpid(),complete_parameters=width,current_functions=64,normal_rank=66,before=before,live_before_QP=live,actual_context_logical_buffers=context,explicit_retained_buffers=touched,original_shapes=original_shapes,all_normal_coordinates_touched=True,actual_NumPy_LAPACK_workspace_query=lapack,events=events,QP_end=qp_end,guard_end=guard_end,elapsed_seconds=elapsed,actual_saved_full_protection_guard=guard,original_unit_reviews={k:v for k,v in result.items() if k not in ['displacement','correction']},full_displacement_SHA256=__import__('hashlib').sha256(result['displacement'].tobytes()).hexdigest(),full_correction_SHA256=__import__('hashlib').sha256(result['correction'].tobytes()).hexdigest(),official_counters=counts,all_backend_parameters_unchanged=True,synthetic_heads=12,synthetic_features=12,synthetic_complete_derivatives=12,independent_CPU_QP_calls=1,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,not_full_official_training_peak=True,source_sha256=source_sha256)
    save(out/'qualification.json',report)
    print(json.dumps(dict(status=report['status'],arm=arm,pid=os.getpid(),resident_peak=guard_end['process']['peak_working'],commit_peak=guard_end['process']['peak_pagefile'],base_live_resident_increment=live['process']['working']-before['process']['working'],QP_calls=QP_calls,official_calls=0)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A','B'],required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();main(args.arm,ROOT/args.out)
