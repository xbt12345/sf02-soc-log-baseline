"""Separate physical RAM and OS commit prerequisites, after CUDA initialization."""
import ctypes
from v169_pair_execution_review_v3 import review_plan,require_run_seal as prior_require
from v169_pair_execution_review_v2 import MemoryStatus

def check_memory(resources,available_physical,available_commit):
    if resources['minimum_free_RAM_bytes']!=6*1024**3 or resources['minimum_free_commit_bytes']!=int(6.25*1024**3):raise ValueError('Registered evidence-backed physical/commit thresholds changed')
    if available_physical<resources['minimum_free_RAM_bytes']:raise RuntimeError('Registered physical RAM prerequisite not met')
    if available_commit<resources['minimum_free_commit_bytes']:raise RuntimeError('Registered available OS commit prerequisite not met')

def require_run_seal(path,current_trainer,phase='running'):
    plan=prior_require(path,current_trainer,phase)
    # prior_require initializes/queries CUDA; this new snapshot includes its
    # actual host driver allocations and distinguishes free RAM from commit.
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise RuntimeError('Physical/commit availability unavailable')
    check_memory(plan['resources'],state.avail_phys,state.avail_page)
    return plan
