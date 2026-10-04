"""Resource-only upgrade bound to full-touch original-environment evidence."""
import ctypes
from experiment_review import ROOT,read,sha,check_bindings
from v169_pair_execution_review_v3 import review_plan as prior_review,require_run_seal as prior_require
from v169_pair_execution_review_v2 import MemoryStatus

BASIS='artifacts/v169_factorized_resource_budget_v9_20261002/review.json'
BASIS_SHA256='4b1ba16a4200056d4cc34ffce799c6592f0651f7b9de71b58b0368f052dbfbbc'
RESOURCE='artifacts/v169_factorized_resource_budget_v10_20261002/review.json'
ENTRY='training/v169_prior_pair_training_entry_v14.py'

def registered_resources():
    if sha(ROOT/BASIS)!=BASIS_SHA256:raise ValueError('Registered full-touch resource evidence changed')
    report=read(ROOT/BASIS)
    if not report['formal_original_native_BLAS24_environment_preserved']:raise ValueError('Original native environment evidence required')
    return report['resources']

def check_memory(resources,available_physical,available_commit):
    if resources!=registered_resources():raise ValueError('Unregistered evidence-backed physical/commit/resource thresholds')
    if available_physical<resources['minimum_free_RAM_bytes']:raise RuntimeError('Registered physical RAM prerequisite not met')
    if available_commit<resources['minimum_free_commit_bytes']:raise RuntimeError('Registered available OS commit prerequisite not met')

def review_plan(plan):
    result=prior_review(plan)
    if plan['entry']!=ENTRY or plan['resource_review_path']!=RESOURCE:raise ValueError('Registered resource-only entry/review path mismatch')
    report=read(ROOT/RESOURCE);check_bindings(report['source_sha256'])
    if report.get('previous_resource_review_path')!=BASIS or report.get('previous_resource_review_sha256')!=BASIS_SHA256:raise ValueError('Full-touch resource-review lineage mismatch')
    if report['resources']!=registered_resources() or plan['resources']!=report['resources']:raise ValueError('Registered resource-review thresholds mismatch')
    if report['entry_sha256']!=sha(ROOT/ENTRY) or report['entry_path']!=ENTRY:raise ValueError('Resource-only entry identity mismatch')
    if plan['source_sha256'].get(RESOURCE)!=sha(ROOT/RESOURCE):raise ValueError('Resource review not bound by execution contract')
    if not report.get('only_entry_resource_policy_import_changed'):raise ValueError('Scientific entry equivalence evidence required')
    return result

def require_run_seal(path,current_trainer,phase='running'):
    plan=prior_require(path,current_trainer,phase)
    review_plan(plan)
    # The inherited resource check has initialized CUDA. Re-read RAM/commit
    # afterward; the original full-model numerical policy remains untouched.
    state=MemoryStatus();state.length=ctypes.sizeof(state)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)):raise RuntimeError('Physical/commit availability unavailable')
    check_memory(plan['resources'],state.avail_phys,state.avail_page)
    return plan
