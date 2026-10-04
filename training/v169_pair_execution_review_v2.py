"""V169 dedicated review plus registered physical RAM/GPU prerequisites."""
import ctypes
import torch
from v169_pair_execution_review import review_plan,require_run_seal as prior_require

class MemoryStatus(ctypes.Structure):
    _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(n,ctypes.c_ulonglong) for n in ['total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','avail_extended']]

def require_run_seal(path,current_trainer,phase='running'):
    plan=prior_require(path,current_trainer,phase)
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise RuntimeError('Actual physical RAM unavailable')
    if state.avail_phys<plan['resources']['minimum_free_RAM_bytes']:raise RuntimeError('Registered physical RAM prerequisite not met')
    if not torch.cuda.is_available():raise RuntimeError('Registered CUDA backend unavailable')
    free,_=torch.cuda.mem_get_info()
    if free<plan['resources']['minimum_free_GPU_bytes']:raise RuntimeError('Registered free GPU prerequisite not met')
    return plan
