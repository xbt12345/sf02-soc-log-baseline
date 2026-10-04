"""Prospective resource-only candidate; no execution contract or permission."""
import copy
import json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v169_factorized_resource_budget_v8_20261002'
Q=ROOT/'artifacts/v169_full_live_new_solver_resource_qualification_20261002'

def main():
    assert not OUT.exists()
    prior_path=ROOT/'artifacts/v169_factorized_resource_budget_v7_20261002/review.json'
    prior=read(prior_path);check_bindings(prior['source_sha256'])
    old_live_path=ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json'
    old_live=read(old_live_path)
    cases=[]
    for arm in ['A','B']:
        path=Q/f'arm{arm}/qualification.json';case=read(path);check_bindings(case['source_sha256'])
        assert case['arm']==arm and case['complete_parameters']==1060832+(arm=='B')
        assert case['current_functions']==64 and case['normal_rank']==66 and case['all_normal_coordinates_touched']
        assert case['all_backend_parameters_unchanged'] and all(v==0 for v in case['official_counters'].values())
        assert case['independent_CPU_QP_calls']==1 and case['actual_saved_full_protection_guard']['passed']
        assert case['actual_NumPy_LAPACK_workspace_query']['workspace_bytes']==143088
        cases.append(case)
    assert cases[0]['pid']!=cases[1]['pid']
    matrix=66*1060833*8;quantum=256*1024**2;slack=128*1024**2
    base_resident=max([old_live['measured_base_live_resident_increment_after_import']]+[c['live_before_QP']['process']['working']-c['before']['process']['working'] for c in cases])
    base_commit=max([old_live['QP_phase_base_live']['process']['pagefile']-old_live['before']['process']['pagefile']]+[c['live_before_QP']['process']['pagefile']-c['before']['process']['pagefile'] for c in cases])
    seven=7*matrix;physical_bound=seven+base_resident+slack
    candidate_RAM=((physical_bound+quantum-1)//quantum)*quantum
    summary=[]
    for c in cases:
        baseline=c['before']['process'];live=c['live_before_QP']['process'];peak=c['guard_end']['process']
        incremental=dict(arm=c['arm'],pid=c['pid'],complete_parameters=c['complete_parameters'],normal_rank=c['normal_rank'],resident_process_peak_bytes=peak['peak_working'],commit_process_peak_bytes=peak['peak_pagefile'],resident_peak_increment_after_import=peak['peak_working']-baseline['working'],commit_peak_increment_after_import=peak['peak_pagefile']-baseline['pagefile'],QP_plus_guard_resident_increment_over_retained_base=peak['peak_working']-live['working'],QP_plus_guard_commit_increment_over_retained_base=peak['peak_pagefile']-live['pagefile'])
        assert incremental['QP_plus_guard_resident_increment_over_retained_base']<seven
        assert incremental['QP_plus_guard_commit_increment_over_retained_base']<seven
        assert incremental['resident_peak_increment_after_import']+slack<candidate_RAM
        assert incremental['commit_peak_increment_after_import']+slack<prior['resources']['minimum_free_commit_bytes']
        summary.append(incremental)
    lifecycle=[
      dict(phase='SVD',component='caller complete current normal matrix',shape=[64,1060833],complete66_matrix_units_upper=1),
      dict(phase='SVD',component='solver C-contiguous scaled matrix',shape=[66,1060833],complete66_matrix_units_upper=1),
      dict(phase='SVD',component='NumPy native LAPACK input A copy',shape=[66,1060833],complete66_matrix_units_upper=1),
      dict(phase='SVD',component='NumPy native LAPACK VT output buffer',shape=[66,1060833],complete66_matrix_units_upper=1),
      dict(phase='SVD',component='NumPy returned full Vh output array',shape=[66,1060833],complete66_matrix_units_upper=1),
      dict(phase='SVD',component='conservative native BLAS/workspace/transient reserve, queried WORK only143088 bytes',complete66_matrix_units_upper=1),
      dict(phase='SVD',component='additional complete matrix lifecycle reserve',complete66_matrix_units_upper=1)
    ]
    source_cpp=Q/'NumPy_v2.2.6_umath_linalg_original.cpp'
    text=source_cpp.read_text(encoding='utf-8')
    for token in ['malloc(a_size + s_size + u_size + vt_size + iwork_size)','mem_buff2 = (npy_uint8 *)malloc(work_size)','linearize_matrix((typ*)params.A, (typ*)args[0], &a_in)','delinearize_matrix((typ*)args[3], (typ*)params.VT, &v_out)']:
        assert token in text
    report=copy.deepcopy(prior)
    report.update(status='V169_resource_only_prospective_candidate_full_touched_A_B_independent_processes_pending_root_review',resources={**prior['resources'],'minimum_free_RAM_bytes':candidate_RAM},RAM_complete_live_phase_accounting_upper_bytes=physical_bound,QP_eight_matrix_accounting_upper_bytes=None,QP_seven_matrix_accounting_upper_bytes=seven,physical_RAM_prerequisite_retained6GiB=False,old6GiB_failure_preserved=True,prior_resource_review=prior_path.relative_to(ROOT).as_posix(),separate_A_B_full_touched_cases=summary,source_array_lifecycle_complete66_matrix_upper=lifecycle,NumPy_native_source_URL='https://raw.githubusercontent.com/numpy/numpy/v2.2.6/numpy/linalg/umath_linalg.cpp',NumPy_native_source_sections='init_gesdd and svd_wrapper',NumPy_version='2.2.6',native_WORK_actual_query_bytes=143088,caller_previous_normals_and_new_list_plus_stack_phase_max64_matrix_units=3,normal_measurement_phase_and_SVD_native_workspace_do_not_overlap=True,no_full_vstack_matrix_in_new_solver=True,retained_base_resident_increment_conservative_bytes=base_resident,retained_base_commit_increment_conservative_bytes=base_commit,extra_physical_slack_bytes=slack,minimum_RAM_quantum_bytes=quantum,rounded_RAM_slack_above_accounting_bytes=candidate_RAM-physical_bound,commit_prerequisite_unchanged_bytes=prior['resources']['minimum_free_commit_bytes'],new_commit_increment_plus_seven_matrix_and128MiB_bytes=seven+base_commit+slack,maximum_touched_resident_peak_is_not_full_SOC_training_peak=True,scientific_algorithm_data_schedule_callgraph_caps_protection_quality_unchanged=True,existing_policy_v4_still_requires6GiB_and_cannot_execute_this_candidate=True,current_gate_not_changed=True,current_disk_prerequisite_met=None,current_RAM_prerequisite_met=None,current_GPU_prerequisite_met=None,snapshot=None,supports_physical_seal=False,execution_authority=False,new_fit_permission=False,official_calls=0,synthetic_heads=24,synthetic_features=24,synthetic_complete_derivatives=24,independent_CPU_QP_calls=2,fits=0,permanent_updates=0)
    assert report['new_commit_increment_plus_seven_matrix_and128MiB_bytes']<report['resources']['minimum_free_commit_bytes']
    sources=[Path(__file__).resolve(),prior_path,old_live_path,source_cpp,*[Q/f'arm{a}/qualification.json' for a in ['A','B']],ROOT/'training/v169_working_joint_restoration_v2.py',ROOT/'training/v169_prior_pair_training_entry_v13.py',ROOT/'training/v169_pair_lifecycle_v3.py',ROOT/'artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json',ROOT/'artifacts/v169_root_full_matrix_page_touch_review_20261002/review.json']
    report['source_sha256']={**prior['source_sha256'],**{p.relative_to(ROOT).as_posix():sha(p) for p in sources}}
    check_bindings(report['source_sha256']);OUT.mkdir()
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    note=('# V169完整触页资源资格与前瞻候选\n\n'
      'A/B各自新进程，保留CUDA最大批次合成预热、实际未前向role0后台、八个全参数向量、十六份完整输出、三份账本及两份最大阻挡帧。完整64×参数宽度法向矩阵、位移与两类梯度逐页写入，实际rank66 QP及保存原行保护通过。官方调用、fit、更新均0。\n\n'
      '源码生命周期：SVD重叠的大矩阵为原法向、scaled、NumPy native A、native VT、返回Vh，合计五份66×P上界；再留两份完整矩阵作为native workspace及其他生命周期余量，使用七份上界。已安装NumPy自身OpenBLAS ILP64符号实际WORK查询为17,886个double，即143,088 bytes。U/小坐标及整数工作区远低于额外整矩阵余量。basis切片/转置是视图；small为66×66。新法向测量最坏保留上次矩阵、当前全梯度列表及stack返回三份64×P，发生于SVD调用之前。动态求解的二次修正同样逐次执行，前次native SVD内存不跨调用保留。已有逐位等价资格证明只删除重复vstack矩阵，没有删法向或参数。\n\n'
      f'保守常驻增量仍取旧真实后台测量与新两进程的较大值{base_resident:,} bytes，而非只取较低的新测量。七矩阵{seven:,}+该增量+128MiB={physical_bound:,} bytes，向上256MiB量化候选为{candidate_RAM:,} bytes（{candidate_RAM/1024**3:g}GiB）。commit保留6.25GiB；所有磁盘、GPU、存储预算、算法/数据/日程/历史成本/保护/分类门槛不变。\n\n'
      '本报告仅为根独审候选。现有policy_v4及原6GiB失败证据未修改，尚无新执行契约/封存或正式训练。实际SOC完整训练峰值和质量仍未验证；根需核对全生命周期及前瞻政策才能决定是否使用候选。\n')
    (OUT/'review.md').write_text(note,encoding='utf-8')
    print(json.dumps(dict(status=report['status'],cases=summary,conservative_resident_increment=base_resident,seven_matrix_bytes=seven,physical_bound=physical_bound,prospective_minimum_free_RAM_bytes=candidate_RAM,minimum_commit_bytes=report['resources']['minimum_free_commit_bytes'],official_calls=0)))

if __name__=='__main__':main()
