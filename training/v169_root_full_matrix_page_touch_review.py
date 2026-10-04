"""Measure lazy zero pages versus a fully touched full-size CPU matrix."""
import ctypes
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v169_root_full_matrix_page_touch_review_20261002'


class ProcessMemory(ctypes.Structure):
    _fields_ = [('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[
        (name,ctypes.c_size_t) for name in ['peak_working','working',
        'peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool',
        'pagefile','peak_pagefile']]


def memory():
    handle = ctypes.windll.kernel32.GetCurrentProcess
    handle.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
    value = ProcessMemory();value.cb=ctypes.sizeof(value)
    if not query(handle(),ctypes.byref(value),value.cb):
        raise OSError('Actual process memory unavailable')
    return {name:int(getattr(value,name)) for name in ['working','peak_working','pagefile','peak_pagefile']}


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    before=memory()
    matrix=np.zeros((64,1060833),np.float64)
    matrix[np.arange(64),np.arange(2,66)]=1.
    few_pages=memory()
    # Touch every logical element without changing the fixture's values.
    matrix.fill(0.)
    matrix[np.arange(64),np.arange(2,66)]=1.
    all_pages=memory()
    difference=all_pages['working']-few_pages['working']
    report=dict(status='actual_full_matrix_residency_differs_after_page_touch',
        full_shape=list(matrix.shape),logical_bytes=int(matrix.nbytes),
        before=before,zeros_plus64_elements=few_pages,fully_touched=all_pages,
        observed_resident_increase_bytes=difference,
        demonstrates_full_shape_alone_is_not_full_residency=difference>matrix.nbytes/2,
        changes_resource_qualification_to_require_touching_every_large_buffer=True,
        not_whole_training_peak_or_new_RAM_threshold_evidence=True,
        official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,
        execution_authority=False,
        source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    OUT.mkdir()
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))


if __name__=='__main__':
    main()
