"""Saved regressions and real 25-row full-width arithmetic; zero model calls."""
import ctypes,json,os,traceback
from ctypes import wintypes
from pathlib import Path
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
from v159_float64_repeat_policy_v2 import EPS,STEP_EPS,finite_step_review
from v166_coverage_joint_restoration import propose,MAX_NORMALS

TRIAL=ROOT/'artifacts/v164_short_supervised_trajectory_20261002'
PRIOR=ROOT/'artifacts/v164_independent_decision_floor_counterfactual_20261002'
OUT=ROOT/'artifacts/v166_coverage_core_qualification_20261002'
POLICY_FIELDS=['local_protection_target','actual_origin_margins_for_identity_only','actual_argmax_all_original_rows_and_retention_guards_still_required','confidence_restoration_not_required_by_this_local_policy']

def save(path,value):path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2)+'\n').encode('utf-8'))

def memory():
    class Status(ctypes.Structure):
        _fields_=[('dwLength',wintypes.DWORD),('dwMemoryLoad',wintypes.DWORD)]+[(name,ctypes.c_ulonglong) for name in ['ullTotalPhys','ullAvailPhys','ullTotalPageFile','ullAvailPageFile','ullTotalVirtual','ullAvailVirtual','ullAvailExtendedVirtual']]
    class Counters(ctypes.Structure):
        _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)]+[(name,ctypes.c_size_t) for name in ['PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage','QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage','PrivateUsage']]
    status=Status();status.dwLength=ctypes.sizeof(status);assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    process=ctypes.windll.kernel32.GetCurrentProcess;process.restype=ctypes.c_void_p;values=Counters();values.cb=ctypes.sizeof(values)
    api=ctypes.windll.psapi.GetProcessMemoryInfo;api.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),wintypes.DWORD];assert api(process(),ctypes.byref(values),values.cb)
    return dict(pid=os.getpid(),physical_total_bytes=status.ullTotalPhys,physical_available_bytes=status.ullAvailPhys,process_peak_working_set_bytes=values.PeakWorkingSetSize,process_current_working_set_bytes=values.WorkingSetSize,process_private_bytes=values.PrivateUsage)

def main():
    assert not OUT.exists() and MAX_NORMALS==25;OUT.mkdir();draft=ROOT/'training/review_policy/v166_coverage_first_diagnostic_draft.json';plan=read(draft);assert plan['execution_authority'] is False and plan['uniform_maximum_margin_functions']==25
    before=memory();save(OUT/'initial_memory.json',before)
    paths={Path(__file__).resolve(),ROOT/'training/v166_coverage_joint_restoration.py',ROOT/'training/v163_one_sided_joint_restoration.py',ROOT/'training/v159_float64_repeat_policy_v2.py',ROOT/'training/v160_independent_saved_direction_certificate.py',draft,ROOT/'training/review_policy/v165_observed_boundaries.json',ROOT/'docs/V165_ACTUAL_RESULTS_AND_COVERAGE_FIRST_NEXT_PLAN_20261002.md',PRIOR/'review.json'};inputs=[]
    for row in read(PRIOR/'review.json')['results']:
        source=ROOT/row['actual_source'];point=source.parent;cached=PRIOR/f'{source.parent.parent.name}_{point.name}_{source.name}';refs=read(source/'active_normal_references.json');normal_paths=[ROOT/v['gradient'] for v in refs.values()];gradient_paths=[point/f'class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1,2]]
        paths.update([source/'active_normal_references.json',*(source/name for name in ['current_displacement.npy','base_margins.npy','actual_current_margins.npy']),*normal_paths,*gradient_paths,*(cached/name for name in ['counterfactual_original_unit_review.json','displacement.npy','correction.npy'])]);inputs.append((source,cached,normal_paths,gradient_paths))
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(paths)};save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));cases=[];qps=0
    for source,cached,normal_paths,gradient_paths in inputs:
        u=np.load(source/'current_displacement.npy');a=np.stack([np.load(p) for p in normal_paths]);b=np.load(source/'base_margins.npy');c=np.load(source/'actual_current_margins.npy');gm,gs=[np.load(p) for p in gradient_paths];result=propose(u,a,b,c,gm,gs);qps+=int('optimizer_iterations' in result)
        assert {k:v for k,v in result.items() if k not in ['displacement','correction',*POLICY_FIELDS]}==read(cached/'counterfactual_original_unit_review.json')
        for key in ['displacement','correction']:assert np.array_equal(result[key],np.load(cached/f'{key}.npy'))
        cases.append(dict(source=source.relative_to(ROOT).as_posix(),normal_functions=len(a),old_24_function_results_bit_exact=True,old_numeric_bounds_and_strict_class_signs_unchanged=True))
    width=1060832;u=np.zeros(width);u[:25]=-.125;u[25]=-1.;a=np.zeros((25,width));a[np.arange(25),np.arange(25)]=1.;gm=np.zeros(width);gm[25]=1.;gs=gm*2.;b=np.full(25,.75);c=np.full(25,-.125)
    wide=propose(u,a,b,c,gm,gs);qps+=int('optimizer_iterations' in wide);assert wide['status']=='one_sided_joint_restoration_requires_full_actual_finite_guard' and all(v['passed'] for v in wide['inequality_reviews'])
    assert np.all(np.abs((a@wide['displacement']))<=STEP_EPS*EPS*np.linalg.norm(wide['displacement']))
    permutation=np.arange(25)[::-1];permuted=propose(u,a[permutation],b[permutation],c[permutation],gm,gs);qps+=int('optimizer_iterations' in permuted);assert permuted['status']==wide['status'];assert np.linalg.norm(permuted['displacement']-wide['displacement'])<=STEP_EPS*EPS*np.linalg.norm(wide['displacement'])
    unit=1e-24;scaled=propose(u,a*unit,b*unit,c*unit,gm,gs);qps+=int('optimizer_iterations' in scaled);assert scaled['status']==wide['status'];assert np.linalg.norm(scaled['displacement']-wide['displacement'])<=STEP_EPS*EPS*np.linalg.norm(wide['displacement'])
    oversized=False
    try:propose(u,np.vstack([a,a[:1]]),np.r_[b,b[0]],np.r_[c,c[0]],gm,gs)
    except ValueError:oversized=True
    assert oversized
    # Mathematical decision-floor signs do not permit a finite bad classifier.
    finite=finite_step_review([.3,.4],[.299,.399],[r['linear_change'] for r in wide['class_reviews']],100,100,'B',1.,False);assert not finite['accepted']
    after=memory();save(OUT/'final_memory.json',after);check_bindings(bindings)
    summary=dict(status='V166_25_function_full_width_and_saved_math_core_qualified_not_executed',cases=cases,full_parameter_count=width,actual_25_function_fixture_passed=True,actual_25_function_permutation_and_units_passed=True,26_function_request_rejected=True,zero_local_floor_does_not_authorize_bad_finite_guard=True,actual_CPU_QP_solves=qps,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,largest_single_full_joint_matrix_bytes=27*width*8,66_fresh_gradient_arrays_raw_bytes=66*width*8,physical_memory_observed_before=before,physical_memory_observed_after=after,resource_pass_is_only_CPU_core_not_actual_cuda_margin_or_complete_entry=True,entry_lifecycle_and_physical_run_seal_still_required=True,execution_authority=False,source_sha256=bindings)
    save(OUT/'qualification.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','source_sha256']},ensure_ascii=False))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
