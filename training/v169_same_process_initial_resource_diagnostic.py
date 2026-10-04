"""One actual sealed require(initial), observed without replacing any gate."""
import ctypes,datetime,gc,json,sys,time
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha
import v169_prior_pair_training_entry_v14 as entry
from v169_pair_execution_review_v2 import MemoryStatus

OUT=ROOT/'artifacts/v169_same_process_initial_resource_diagnostic_20261002'

class ProcessMemory(ctypes.Structure):
    _fields_=[('cb',ctypes.c_ulong),('faults',ctypes.c_ulong)]+[(name,ctypes.c_size_t) for name in ['peak_working','working','peak_paged_pool','paged_pool','peak_nonpaged_pool','nonpaged_pool','pagefile','peak_pagefile']]

def process_memory():
    current=ctypes.windll.kernel32.GetCurrentProcess;current.restype=ctypes.c_void_p
    query=ctypes.windll.psapi.GetProcessMemoryInfo;query.argtypes=[ctypes.c_void_p,ctypes.POINTER(ProcessMemory),ctypes.c_ulong]
    value=ProcessMemory();value.cb=ctypes.sizeof(value)
    if not query(current(),ctypes.byref(value),value.cb):raise OSError('Process memory unavailable')
    return {name:int(getattr(value,name)) for name in ['peak_working','working','pagefile','peak_pagefile']}

def snapshot():
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise OSError('System memory unavailable')
    return dict(timestamp_UTC=datetime.datetime.now(datetime.UTC).isoformat(),process=process_memory(),system=dict(available_physical_bytes=int(state.avail_phys),available_commit_bytes=int(state.avail_page),total_physical_bytes=int(state.total_phys),total_commit_bytes=int(state.total_page)),CUDA_initialized=torch.cuda.is_initialized())

def main():
    assert not OUT.exists();assert sorted(p.name for p in entry.OUT.iterdir())==['registration.json','run_seal.json']
    original_files={p.relative_to(ROOT).as_posix():sha(p) for p in [entry.PLAN,entry.OUT/'registration.json',entry.OUT/'run_seal.json']}
    plan=read(entry.PLAN);baseline=snapshot();events=[];start=time.perf_counter();previous=sys.gettrace()
    source_paths=[ROOT/f'training/v169_pair_execution_review_v{i}.py' for i in [2,5]]
    source_lines={str(p.resolve()):p.read_text(encoding='utf-8').splitlines() for p in source_paths}
    def trace(frame,event,arg):
        name=str(Path(frame.f_code.co_filename).resolve())
        if name not in source_lines or frame.f_code.co_name!='require_run_seal':return trace
        lines=source_lines[name];line=lines[frame.f_lineno-1] if frame.f_lineno<=len(lines) else ''
        selected=(event in ['call','exception','return']) or (event=='line' and any(token in line for token in ['GlobalMemoryStatusEx','if state.avail_phys<','check_memory(plan']))
        if selected:
            record=dict(source=Path(name).relative_to(ROOT).as_posix(),event=event,line_number=frame.f_lineno,source_line=line.strip(),elapsed_seconds=time.perf_counter()-start,observed=snapshot())
            state=frame.f_locals.get('state')
            if state is not None:record['actual_gate_state']=dict(length=int(state.length),available_physical_bytes=int(state.avail_phys),available_commit_bytes=int(state.avail_page))
            if event=='exception':record['exception']=dict(type=arg[0].__name__,message=str(arg[1]))
            events.append(record)
        return trace
    sys.settrace(trace);outcome=None
    try:
        # Exactly the same entry function used by main, called only once.
        returned=entry.require('initial')
        assert returned==plan
        outcome=dict(status='actual_same_process_initial_require_passed',execution_started=False)
    except Exception as error:
        outcome=dict(status='actual_same_process_initial_require_refused',type=type(error).__name__,message=str(error),execution_started=False)
    finally:sys.settrace(previous)
    after_require=snapshot();gc.collect();after_gc=snapshot()
    gpu_free,gpu_total=torch.cuda.mem_get_info();after_CUDA=snapshot()
    assert sorted(p.name for p in entry.OUT.iterdir())==['registration.json','run_seal.json']
    assert all(sha(ROOT/p)==value for p,value in original_files.items())
    sources=[Path(__file__).resolve(),Path(entry.__file__),*source_paths,entry.PLAN,entry.OUT/'registration.json',entry.OUT/'run_seal.json',ROOT/'artifacts/v169_prior_pair_training_entry_v14_original_console_20261002.txt',ROOT/'artifacts/v169_prior_pair_training_entry_v14_retry1_original_console_20261002.txt']
    report=dict(status='V169_one_actual_same_process_resource_gate_observed_without_changes',actual_require_calls=1,baseline=baseline,events=events,outcome=outcome,after_require=after_require,after_gc=after_gc,after_CUDA_initialization_metadata_only=after_CUDA,after_CUDA_GPU_metadata=dict(free_bytes=int(gpu_free),total_bytes=int(gpu_total)),elapsed_seconds=time.perf_counter()-start,thresholds=plan['resources'],no_gate_return_or_state_or_threshold_replaced=True,no_system_or_user_process_trim=True,main_and_backend_not_called=True,registered_files_unchanged=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    OUT.mkdir();(OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status=report['status'],outcome=outcome,baseline=baseline,events=events,after_require=after_require,after_gc=after_gc,after_CUDA=after_CUDA,elapsed_seconds=report['elapsed_seconds'],official_calls=0)),flush=True)

if __name__=='__main__':main()
