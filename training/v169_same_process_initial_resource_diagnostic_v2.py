"""One actual initial gate, code-object-only observer,180s hard diagnostic cap."""
import ctypes,datetime,gc,json,os,sys,threading,time
from pathlib import Path
import torch
from experiment_review import ROOT,read,sha
import v169_prior_pair_training_entry_v14 as entry
import v169_pair_execution_review_v2 as v2
import v169_pair_execution_review_v5 as v5
from v169_pair_execution_review_v2 import MemoryStatus

OUT=ROOT/'artifacts/v169_same_process_initial_resource_diagnostic_v2_20261002'

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

def save(path,value):
    assert not path.exists()
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');temporary.replace(path)

def main():
    assert not OUT.exists();assert sorted(p.name for p in entry.OUT.iterdir())==['registration.json','run_seal.json']
    paths=[Path(__file__).resolve(),Path(entry.__file__),Path(v2.__file__),Path(v5.__file__),entry.PLAN,entry.OUT/'registration.json',entry.OUT/'run_seal.json',ROOT/'artifacts/v169_prior_pair_training_entry_v14_original_console_20261002.txt',ROOT/'artifacts/v169_prior_pair_training_entry_v14_retry1_original_console_20261002.txt',ROOT/'training/v169_same_process_initial_resource_diagnostic.py',ROOT/'artifacts/v169_same_process_initial_resource_diagnostic_20261002/termination.json']
    sources={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    protected={p.relative_to(ROOT).as_posix():sha(p) for p in [entry.PLAN,entry.OUT/'registration.json',entry.OUT/'run_seal.json']}
    plan=read(entry.PLAN);baseline=snapshot();events=[];start=time.perf_counter();previous=sys.gettrace();stop=threading.Event()
    target_codes={v2.require_run_seal.__code__,v5.require_run_seal.__code__}
    info={code:dict(source=Path(code.co_filename).resolve().relative_to(ROOT).as_posix(),lines=Path(code.co_filename).read_text(encoding='utf-8').splitlines()) for code in target_codes}
    def trace(frame,event,arg):
        if frame.f_code not in target_codes:return None
        source=info[frame.f_code];line=source['lines'][frame.f_lineno-1]
        selected=(event in ['call','exception','return']) or (event=='line' and any(token in line for token in ['GlobalMemoryStatusEx','if state.avail_phys<','check_memory(plan']))
        if selected:
            record=dict(source=source['source'],event=event,line_number=frame.f_lineno,source_line=line.strip(),elapsed_seconds=time.perf_counter()-start,observed=snapshot())
            state=frame.f_locals.get('state')
            if state is not None:record['actual_gate_state']=dict(length=int(state.length),available_physical_bytes=int(state.avail_phys),available_commit_bytes=int(state.avail_page))
            if event=='exception':record['exception']=dict(type=arg[0].__name__,message=str(arg[1]))
            events.append(record)
        return trace
    OUT.mkdir()
    def watchdog():
        if stop.wait(180):return
        report=dict(status='V169_corrected_diagnostic180s_timeout_stop_no_retry',actual_require_calls=1,baseline=baseline,events=events,at_timeout=snapshot(),main_and_backend_not_called=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256=sources)
        save(OUT/'timeout.json',report)
        os._exit(124)
    threading.Thread(target=watchdog,daemon=True).start()
    sys.settrace(trace)
    try:
        returned=entry.require('initial')
        assert returned==plan
        outcome=dict(status='actual_same_process_initial_require_passed',execution_started=False)
    except Exception as error:
        outcome=dict(status='actual_same_process_initial_require_refused',type=type(error).__name__,message=str(error),execution_started=False)
    finally:sys.settrace(previous)
    after_require=snapshot();gc.collect();after_gc=snapshot()
    gpu_free,gpu_total=torch.cuda.mem_get_info();after_CUDA=snapshot()
    assert sorted(p.name for p in entry.OUT.iterdir())==['registration.json','run_seal.json']
    assert all(sha(ROOT/p)==value for p,value in protected.items())
    report=dict(status='V169_one_actual_same_process_resource_gate_code_filtered_observer_completed',actual_require_calls=1,diagnostic_time_limit_seconds=180,elapsed_seconds=time.perf_counter()-start,baseline=baseline,events=events,outcome=outcome,after_require=after_require,after_gc=after_gc,after_CUDA_initialization_metadata_only=after_CUDA,after_CUDA_GPU_metadata=dict(free_bytes=int(gpu_free),total_bytes=int(gpu_total)),observer_first_condition_is_code_object_set_membership=True,non_target_frames_return_None=True,no_gate_return_or_state_or_threshold_replaced=True,no_system_or_user_process_trim=True,main_and_backend_not_called=True,registered_files_unchanged=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,source_sha256=sources)
    stop.set();save(OUT/'review.json',report)
    print(json.dumps(dict(status=report['status'],elapsed_seconds=report['elapsed_seconds'],outcome=outcome,baseline=baseline,events=events,after_require=after_require,after_gc=after_gc,after_CUDA=after_CUDA,official_calls=0)),flush=True)

if __name__=='__main__':main()
