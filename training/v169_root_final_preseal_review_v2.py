"""Independent resource-only final review; never evaluate official models."""
import argparse
import ast
import json
import time
from pathlib import Path

from experiment_review import ROOT, read, sha


def require(condition, message):
    if not condition:
        raise ValueError(message)


def closure(starts):
    pending = list(starts)
    seen = set()
    while pending:
        path = pending.pop().resolve()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = [v.name for v in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT/'training'/(name.split('.')[0]+'.py')
                if candidate.is_file() and candidate.resolve() not in seen:
                    pending.append(candidate)
    return seen


def main(bundle_path, sealer_path, out):
    bundle_path, sealer_path, out = [Path(p).resolve() for p in [bundle_path, sealer_path, out]]
    require(not out.exists(), 'Immutable review output already exists')
    bundle = read(bundle_path)
    start = time.perf_counter()
    bindings = dict(bundle['source_sha256'])
    hash_errors = []
    bytes_hashed = 0
    for name, expected in bindings.items():
        path = ROOT/name
        if not path.is_file():
            hash_errors.append(dict(path=name, reason='missing'))
            continue
        bytes_hashed += path.stat().st_size
        if sha(path) != expected:
            hash_errors.append(dict(path=name, reason='changed'))
    require(not hash_errors, 'Bound source changed: '+repr(hash_errors[:8]))
    entry_path = ROOT/bundle['entry_path']
    contract_path = ROOT/bundle['candidate_contract_path']
    require(sha(entry_path) == bundle['entry_sha256'], 'Entry identity mismatch')
    require(sha(contract_path) == bundle['candidate_contract_sha256'], 'Contract identity mismatch')
    contract = read(contract_path)
    require(contract['execution_authority'] is False and contract['new_fit_permission'] is False,
            'This review must precede execution authority')
    from v169_pair_execution_review_v5 import review_plan
    review_plan(contract)
    modules = closure([entry_path, sealer_path, ROOT/'training/v169_pair_execution_review_v3.py'])
    missing = [p.relative_to(ROOT).as_posix() for p in modules
               if p.relative_to(ROOT).as_posix() not in bindings]
    require(not missing, 'Missing recursive project source: '+repr(missing))
    previous = read(ROOT/'artifacts/v168_decision_floor_diagnostic_20261002/run_seal.json')
    require(all(bindings.get(k) == v for k, v in previous['source_sha256'].items()),
            'Previous physical source/data bindings omitted')
    require(contract['schedule'] == dict(accepted_updates=20, corrections=2,
                                        working_functions=64, backtracks=8), 'Schedule changed')
    require(contract['new_caps'] == dict(heads=56328, features=56328,
                fixed_target_derivatives=504, margin_derivatives=29184, QP=342,
                proposals=1146, fits=6, updates=120, original_class_derivatives=0,
                all_complete_derivatives=29688), 'Callgraph cap mismatch')
    require(contract['complete_parameter_widths'] == dict(A=1060832, B=1060833),
            'Complete parameter schema changed')
    require(contract['objective_policy'].startswith('strict_individual_M_S'),
            'Individual class objective missing')
    for key in ['permanent_full_guard_ledger', 'full_guard_each_candidate',
                'fresh_derivatives_at_actual_parameter_point', 'batch_global_fault_stops_remaining_fits',
                'same_point_final_replay_required', 'viewed_roles_are_development_not_blind']:
        require(contract[key] is True, 'Required constraint absent: '+key)
    require(contract['unmatched_terminal_result'] == 'inconclusive', 'Unmatched comparison policy changed')
    resource = read(ROOT/contract['resource_review_path'])
    require(resource['resources'] == contract['resources'] == bundle['resources'],
            'Resource review/contract/bundle identity mismatch')
    metadata = dict(candidate_proofs=1146*16384, candidate_recipes=1146*8192,
        candidate_original_reference=1146*4096, line_search=912*4096,
        common_QP=114*65536, correction_QP=228*32768, target_repeats=252*8192,
        target_identities=126*32768, commits=120*4096, accepted_quality=120*16384,
        accepted_reference=240*8192, initial_protection=6*65536, final_replay=6*16384,
        fit_results=6*524288, fit_counts=6*32768, pair_results=3*4096,
        batch_results=2*4194304, failure_receipts=18*65536,
        head_and_feature_log_lines=4*56328*128,
        derivative_log_lines=2*29688*512, QP_log_lines=2*342*512,
        proposal_update_fit_log_lines=(1146+120+6)*512)
    expected = dict(target_vectors=504*(6209*16+8192),
        margin_vectors=29184*(3617*16+8192), QP_vectors=348*(6913*16+8192),
        margin_raw_outputs=29184*(2048*3*8*2+8192), chunk_ids=14592*(2048*8+128),
        margin_references=29184*8192, normal_identity_JSON=14592*16384,
        candidate_outputs=1146*(22546*3*8*4+8192),
        candidate_masks=2*191*3*225614+1146*8192,
        target_outputs=504*(22546*3*8*2+8192),
        accepted_outputs=120*(22546*3*8*4+8192*2)+120*225614,
        checkpoints=126*(1060833*8+65536), final_tables=640*1024**2,
        other_metadata_and_logs=sum(metadata.values()),
        allocation_and_directories=640*1024**2, emergency_dense_vector_and_failure=128*1024**2)
    require(resource['storage_components_upper_bytes'] == expected, 'Storage bound formula changed')
    require(resource['metadata_components_upper_bytes'] == metadata, 'JSON/log formula changed')
    require(sum(expected.values()) == contract['resources']['full_run_worst_storage_bytes'],
            'Whole storage bound mismatch')
    inventory_path = ROOT/'artifacts/v169_saved_storage_filecount_review_20261002/review.json'
    inventory = read(inventory_path)
    require(inventory['derived_file_count'] == sum(inventory['complete_file_components'].values()) == 126522,
            'Worst file inventory mismatch')
    require(inventory['derived_directory_count'] == sum(inventory['complete_directory_components'].values()) == 16693,
            'Worst directory inventory mismatch')
    require(128000*4096+17000*8192 <= expected['allocation_and_directories'],
            'Filesystem padding insufficient')
    ram_path = ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json'
    ram = read(ram_path)
    resource_root_path = ROOT/'artifacts/v169_root_full_live_resource_review_20261002/review.json'
    resource_root = read(resource_root_path)
    require(resource_root['all_checks_passed'] and resource_root['supports_prospective_resource_only_revision'],
            'Independent original-environment full-touch resource review absent')
    require(resource_root['resources'] == contract['resources'], 'Independently justified resource threshold changed')
    require(resource_root_path.relative_to(ROOT).as_posix() in bindings,
            'Independent full-touch evidence not bound')
    for name, expected_sha in resource_root['source_sha256'].items():
        require(bindings.get(name) == expected_sha, 'Resource-review source omitted or changed: '+name)
    resident_bound = resource_root['independent_incremental_RAM_bound_bytes']
    commit_bound = resource_root['independent_incremental_commit_bound_bytes']
    quantum = 256*1024**2
    require(((resident_bound+quantum-1)//quantum)*quantum == contract['resources']['minimum_free_RAM_bytes'],
            'Measured lifecycle RAM bound not covered by prospective resource gate')
    commit_minimum = ((commit_bound+quantum-1)//quantum)*quantum
    require(contract['resources']['minimum_free_commit_bytes'] >= commit_minimum,
            'Independent commit-space gate missing or insufficient')
    old_contract = read(ROOT/'training/review_policy/v169_prior_pair_execution_contract_candidate_v3.json')
    non_scientific = {'entry', 'entry_sha256', 'resource_review_path', 'resources', 'source_sha256', 'status'}
    require(set(contract) == set(old_contract), 'Resource-only contract schema changed')
    require(all(contract[k] == old_contract[k] for k in contract if k not in non_scientific),
            'Scientific schedule/algorithm/data/acceptance/caps changed in resource revision')
    seal_source = sealer_path.read_text(encoding='utf-8')
    require('check_memory(plan[\'resources\'],state.avail_phys,state.avail_page)' in seal_source,
            'Sealer must pass actual physical and commit availability to the gate')
    entry_source = entry_path.read_text(encoding='utf-8')
    require('from v169_pair_execution_review_v5 import require_run_seal' in entry_source,
            'Official entry does not use the physical and commit resource gate')
    old_entry_path = ROOT/'training/v169_prior_pair_training_entry_v13.py'
    old_entry_source = old_entry_path.read_text(encoding='utf-8')
    require(entry_source.replace('from v169_pair_execution_review_v5 import require_run_seal',
                                'from v169_pair_execution_review_v4 import require_run_seal') == old_entry_source,
            'Official numerical/training entry changed beyond resource-policy import')
    from v169_pair_execution_review_v5 import check_memory
    limits = contract['resources']
    check_memory(limits, limits['minimum_free_RAM_bytes'], limits['minimum_free_commit_bytes'])
    memory_refusals = []
    for kind, physical, committed in [
            ('physical_shortfall', limits['minimum_free_RAM_bytes']-1,
             limits['minimum_free_commit_bytes']),
            ('commit_shortfall', limits['minimum_free_RAM_bytes'],
             limits['minimum_free_commit_bytes']-1)]:
        try:
            check_memory(limits, physical, committed)
        except RuntimeError:
            memory_refusals.append(kind)
        else:
            raise ValueError('Actual resource gate accepts '+kind)
    qualified = []
    for item in bundle['qualifications']:
        path = ROOT/item['path'];q = read(path)
        require(sha(path) == item['sha256'], 'Qualification changed')
        require(all(q[k] == 0 for k in ['official_heads', 'official_features',
                       'official_derivatives', 'fits', 'permanent_updates']),
                'Preparation contains official model calls')
        require(q['execution_authority'] is False, 'Qualification has execution authority')
        require(all(bindings.get(k) == v for k,v in q['source_sha256'].items()),
                'Qualification sources omitted or changed')
        qualified.append(item['path'])
    root_reports = ['v169_root_all_current_correct_wiring_review',
                    'v169_root_real_frame_lifecycle_regression_review',
                    'v169_root_saved_qp_storage_support_review', 'v169_root_full_size_RAM_review']
    for name in root_reports:
        path = ROOT/f'artifacts/{name}_20261002/review.json';q = read(path)
        require(q['all_checks_passed'] is True, 'Independent prerequisite failed: '+name)
        require(path.relative_to(ROOT).as_posix() in bindings, 'Independent evidence not bound: '+name)
    for path in [inventory_path, ram_path]:
        require(path.relative_to(ROOT).as_posix() in bindings, 'Latest physical evidence not bound')
    # Availability must be measured by the worker immediately before sealing and
    # before each fit. This conditional source review performs no CUDA call.
    files = {Path(__file__).resolve(), bundle_path, contract_path, sealer_path,
             inventory_path, ram_path, resource_root_path, old_entry_path, ROOT/contract['resource_review_path']}
    result = dict(status='V169_independent_resource_only_final_source_and_resource_design_review_passed',
        reviewed_entry_sha256=bundle['entry_sha256'], supports_physical_seal=True,
        execution_authority=False, new_fit_permission=False,
        actual_resource_gates_must_still_pass_before_seal_and_each_fit=True,
        official_zero_step_and_all_registered_real_quality_gates_still_required=True,
        hash_verified_files=len(bindings), bytes_hashed=bytes_hashed,
        recursive_project_execution_and_sealing_modules=len(modules),
        qualifications=qualified, independent_storage_upper_bytes=sum(expected.values()),
        metadata_and_logs_upper_bytes=sum(metadata.values()), maximum_files=128000,
        maximum_directories=17000, independent_incremental_RAM_bound_bytes=resident_bound,
        original_environment_full_touch_resource_basis_verified=True,
        scientific_contract_and_entry_unchanged_except_resource_import=True,
        independent_incremental_commit_bound_bytes=commit_bound,
        minimum_free_commit_256MiB_quantum=commit_minimum,
        actual_resource_gate_negative_cases_refused=memory_refusals,
        accepted_206_mixed_current_correct_protection_verified=True,
        prospective_caps=contract['new_caps'], resources=contract['resources'],
        elapsed_seconds=time.perf_counter()-start,
        official_heads=0, official_features=0, official_derivatives=0, fits=0, permanent_updates=0,
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files})
    out.mkdir()
    (out/'review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','qualifications']},ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--sealer', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    main(args.bundle, args.sealer, args.out)
