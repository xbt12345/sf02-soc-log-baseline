"""Independently verify saved full-touch resource evidence; no official calls."""
import json
from pathlib import Path

from experiment_review import ROOT, read, sha, check_bindings

OUT = ROOT/'artifacts/v169_root_full_live_resource_review_20261002'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    resource_path = ROOT/'artifacts/v169_factorized_resource_budget_v9_20261002/review.json'
    old_path = ROOT/'artifacts/v169_factorized_resource_budget_v7_20261002/review.json'
    live_path = ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json'
    resource, old, old_live = map(read, [resource_path, old_path, live_path])
    check_bindings(resource['source_sha256'])
    unchanged = set(old['resources']) - {'minimum_free_RAM_bytes'}
    assert set(resource['resources']) == set(old['resources'])
    assert all(resource['resources'][k] == old['resources'][k] for k in unchanged)
    assert resource['storage_components_upper_bytes'] == old['storage_components_upper_bytes']
    assert resource['metadata_components_upper_bytes'] == old['metadata_components_upper_bytes']
    assert resource['execution_authority'] is False and resource['new_fit_permission'] is False
    assert resource['official_calls'] == resource['fits'] == resource['permanent_updates'] == 0
    reports, paths = [], []
    for arm, width in [('A', 1060832), ('B', 1060833)]:
        path = ROOT/f'artifacts/v169_full_live_new_solver_resource_qualification_v2_20261002/arm{arm}/qualification.json'
        q = read(path)
        check_bindings(q['source_sha256'])
        assert q['arm'] == arm and q['complete_parameters'] == width
        assert q['original_shapes']['normals'] == [64, width]
        assert q['normal_rank'] == 66 and q['current_functions'] == 64
        assert q['all_normal_coordinates_touched'] and q['all_backend_parameters_unchanged']
        assert q['independent_CPU_QP_calls'] == 1 and q['actual_saved_full_protection_guard']['passed']
        assert q['formal_v13_original_native_environment_preserved']
        assert not q['native_pool_environment_fixed_or_reduced']
        assert all(q['before']['environment'][k] is None for k in
                   ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'])
        pools = [p for p in q['native_pools_after_QP'] if p['internal_api'] == 'openblas']
        assert len(pools) == 2 and all(p['num_threads'] == 24 for p in pools)
        assert all(v == 0 for v in q['official_counters'].values())
        assert all(q[k] == 0 for k in ['official_heads', 'official_features',
                                      'official_derivatives', 'fits', 'permanent_updates'])
        result = q['original_unit_reviews']
        assert result['status'] == 'one_sided_joint_restoration_requires_full_actual_finite_guard'
        assert len(result['inequality_reviews']) == 66
        assert all(x['passed'] for x in result['inequality_reviews'])
        assert all(x['resolved_negative'] for x in result['class_reviews'])
        lapack = q['actual_NumPy_LAPACK_workspace_query']
        assert lapack['m'] == 66 and lapack['n'] == width
        assert lapack['JOBZ'] == 'S' and lapack['lwork_elements'] == 17886
        assert lapack['workspace_bytes'] == 143088
        assert sha(Path(lapack['library_path'])) == lapack['library_sha256']
        reports.append(q); paths.append(path)
    assert reports[0]['pid'] != reports[1]['pid']
    max_width = 1060833
    one_matrix = 66*max_width*8
    # Five simultaneously live arrays: caller, scaled, native A, native VT,
    # returned Vh. Two more full arrays bound native/transient uncertainty.
    matrix_bound = 7*one_matrix
    resident_base = max([old_live['QP_phase_base_live']['process']['working']-
                         old_live['before']['process']['working']]+[
                         q['live_before_QP']['process']['working']-
                         q['before']['process']['working'] for q in reports])
    commit_base = max([old_live['QP_phase_base_live']['process']['pagefile']-
                       old_live['before']['process']['pagefile']]+[
                       q['live_before_QP']['process']['pagefile']-
                       q['before']['process']['pagefile'] for q in reports])
    slack = 128*1024**2; quantum = 256*1024**2
    resident_bound = matrix_bound+resident_base+slack
    commit_bound = matrix_bound+commit_base+slack
    minimum = ((resident_bound+quantum-1)//quantum)*quantum
    assert minimum == int(5.25*1024**3) == resource['resources']['minimum_free_RAM_bytes']
    assert resource['RAM_complete_live_phase_accounting_upper_bytes'] == resident_bound
    assert resource['retained_base_resident_increment_conservative_bytes'] == resident_base
    assert resource['retained_base_commit_increment_conservative_bytes'] == commit_base
    assert commit_bound < resource['resources']['minimum_free_commit_bytes'] == int(6.25*1024**3)
    for q in reports:
        peak = q['guard_end']['process']; base = q['before']['process']
        assert peak['peak_working']-base['working']+slack < minimum
        assert peak['peak_pagefile']-base['pagefile']+slack < resource['resources']['minimum_free_commit_bytes']
    native_path = ROOT/'artifacts/v169_full_live_new_solver_resource_qualification_20261002/NumPy_v2.2.6_umath_linalg_original.cpp'
    native = native_path.read_text(encoding='utf-8')
    for token in ['malloc(a_size + s_size + u_size + vt_size + iwork_size)',
                  'mem_buff2 = (npy_uint8 *)malloc(work_size)',
                  'linearize_matrix((typ*)params.A, (typ*)args[0], &a_in)',
                  'delinearize_matrix((typ*)args[3], (typ*)params.VT, &v_out)']:
        assert token in native
    solver_path = ROOT/'training/v169_working_joint_restoration_v2.py'
    solver = solver_path.read_text(encoding='utf-8')
    assert 'scaled=np.empty((len(a)+2,len(u)),np.float64)' in solver
    assert 'np.linalg.svd(scaled,full_matrices=False)' in solver
    assert 'matrix=np.vstack' not in solver
    entry_path = ROOT/'training/v169_prior_pair_training_entry_v13.py'
    entry = entry_path.read_text(encoding='utf-8')
    assert 'return np.stack(normals)' in entry
    assert 'result=propose(callback,trial[\'u\'],a,b,c,*gs,tau)' in entry
    files = [Path(__file__).resolve(), resource_path, old_path, live_path,
             native_path, solver_path, entry_path, *paths]
    result = dict(status='V169_root_saved_full_touch_original_environment_resource_review_passed',
        all_checks_passed=True, supports_prospective_resource_only_revision=True,
        supports_physical_seal=False, execution_authority=False, new_fit_permission=False,
        previous_6GiB_refusal_and_BLAS4_confound_preserved=True,
        formal_BLAS24_environment_unchanged=True, algorithm_quality_caps_data_unchanged=True,
        source_SVD_array_overlap_units=5, additional_complete_matrix_reserve_units=2,
        maximum_normal_measurement_phase_matrix_units=3,
        conservative_resident_base_bytes=resident_base, conservative_commit_base_bytes=commit_base,
        independent_incremental_RAM_bound_bytes=resident_bound,
        independent_incremental_commit_bound_bytes=commit_bound,
        resources=resource['resources'], actual_SOC_complete_training_peak_not_measured=True,
        actual_availability_and_official_zero_step_still_required=True,
        official_heads=0, official_features=0, official_derivatives=0, fits=0, permanent_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in files})
    OUT.mkdir()
    (OUT/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'source_sha256'},ensure_ascii=False))


if __name__ == '__main__':
    main()
