"""Actual-size numerical QPs plus real saved context, no official model calls."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
import ctypes
import json
import time
from pathlib import Path

import numpy as np
from experiment_review import ROOT, sha, check_bindings
from v165_fixed_endpoint_decision_floor_diagnostic import load_context
from v169_current_correct_context import apply
from v169_dynamic_trial_restoration import propose
from v169_factorized_resource_budget_v3 import memory

OUT = ROOT/'artifacts/v169_root_full_size_RAM_review_20261002'
MiB = 1024**2


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[
        (name,ctypes.c_size_t) for name in ['peak_working','working',
        'peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool',
        'pagefile','peak_pagefile']]


def process_memory():
    current = ctypes.windll.kernel32.GetCurrentProcess
    current.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
    value = ProcessMemory();value.cb = ctypes.sizeof(value)
    if not query(current(),ctypes.byref(value),value.cb):
        raise OSError('Process physical peak memory unavailable')
    return {name:int(getattr(value,name)) for name in ['peak_working','working','pagefile','peak_pagefile']}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    files = [Path(__file__).resolve(), ROOT/'training/v169_working_joint_restoration.py',
        ROOT/'training/v169_dynamic_trial_restoration.py',
        ROOT/'training/v165_fixed_endpoint_decision_floor_diagnostic.py',
        ROOT/'training/v169_current_correct_context.py',
        ROOT/'training/v169_factorized_resource_budget_v3.py']
    ctx = load_context(0);protection = apply(ctx,0)
    retained_outputs = []
    for scope in ['OOF','deployment']:
        for name in ['q','logq']:
            path = ROOT/f'artifacts/v164_short_supervised_trajectory_20261002/role0/endpoint/{scope}_{name}.npy'
            retained_outputs.append(np.load(path));files.append(path)
    # Explicitly touched slack for training-time buffers absent from this CPU
    # kernel test. This is not synthetic training data or a smaller QP.
    held_slack = np.ones(384*MiB//8, dtype=np.float64)
    assert float(held_slack[0]+held_slack[-1]) == 2.
    before = dict(available_RAM_bytes=memory(),process=process_memory())
    records=[];QP_calls=0
    for arm,width in [('A',1060832),('B',1060833)]:
        u=np.zeros(width);u[:2]=1.
        gm=np.zeros(width);gs=np.zeros(width);gm[0]=-1.;gs[1]=-1.
        if arm=='B':gm[-1]=.1;gs[-1]=.2
        a=np.zeros((64,width));a[np.arange(64),np.arange(2,66)]=1.
        b=np.full(64,.2);c=np.full(64,-.1)
        tau=np.full(64,16*np.finfo(np.float64).eps)
        callback_events=[]
        def callback(event,value):
            nonlocal QP_calls
            if event=='call':QP_calls+=1
            callback_events.append(dict(event=event,process=process_memory(),available_RAM_bytes=memory()))
        start=time.perf_counter()
        result=propose(callback,u,a,b,c,gm,gs,tau)
        assert result['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard'
        assert len(result['inequality_reviews'])==66
        assert all(r['passed'] for r in result['inequality_reviews'])
        assert all(r['resolved_negative'] for r in result['class_reviews'])
        records.append(dict(arm=arm,full_width=width,current_functions=64,
            elapsed_seconds=time.perf_counter()-start,process=process_memory(),
            available_RAM_bytes=memory(),callback_events=callback_events,
            original_unit_reviews_passed=True))
        del u,gm,gs,a,b,c,tau,result
        print(json.dumps(records[-1],ensure_ascii=False),flush=True)
    assert QP_calls==2
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in files}
    check_bindings(bindings)
    OUT.mkdir()
    report=dict(status='actual_full_size64_CPU_QP_peak_with_real_context_and_384MiB_slack_measured',
        all_checks_passed=True,real_context_original_rows=len(ctx['OOF_rows']),
        current_correct_protection=protection,explicit_touched_slack_bytes=384*MiB,
        before=before,cases=records,final_process=process_memory(),
        independent_preparation_CPU_QP_calls=QP_calls,
        official_heads=0,official_features=0,official_derivatives=0,
        fits=0,permanent_updates=0,supports_physical_seal=False,
        workspace_shape_not_reduced=True,source_sha256=bindings,
        scope='Actual full-dimensional numerical solver on scripted rank66 constraints, retaining actual loaded input/row context and saved q/logq. Does not measure a complete CUDA training fit, does not automatically authorize lowering the RAM prerequisite, and gives no SOC classification evidence.')
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['cases','source_sha256','current_correct_protection']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
