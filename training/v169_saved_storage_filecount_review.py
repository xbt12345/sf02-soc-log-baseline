"""Exact worst artifact file/directory inventory against unchanged byte reserve."""
import json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def main():
    out=ROOT/'artifacts/v169_saved_storage_filecount_review_20261002';assert not out.exists()
    resource_path=ROOT/'artifacts/v169_factorized_resource_budget_v5_20261002/review.json';resource=read(resource_path);check_bindings(resource['source_sha256'])
    entry=ROOT/'training/v169_prior_pair_training_entry_v12.py';s=entry.read_text(encoding='utf-8');assert "for repetition in range(2)" in s and 'original_reference_identity.json' in s and 'same_parameter_replay.json' in s
    files=dict(margin_functions=14592*8,target_points=126*11,point0_complete_outputs=6,common_direction_QP=114*2,all_finite_proposals_and_line_policies=1146*5+912,bootstrap_vectors=6,correction_QP=228*2,accepted_checkpoints_and_quality_and_scope_payloads=120*8,final_endpoint=6*9,per_fit_receipts_and_log=6*6,batch_and_registration=12)
    directories=dict(margin_functions=14592,normal_measurement_rounds=228,correction_QP=228,finite_proposals=1146,target_points=126,accepted_state_and_scopes=120*3,endpoints=6,fit_roles=6,run_root=1)
    actual_file_bound=sum(files.values());actual_directory_bound=sum(directories.values());file_limit=128000;dir_limit=17000
    assert actual_file_bound<=file_limit and actual_directory_bound<=dir_limit
    filesystem_bound=file_limit*4096+dir_limit*8192;reserved=resource['storage_components_upper_bytes']['allocation_and_directories'];assert filesystem_bound<=reserved==640*1024**2
    # Explicitly distribute emergency bytes rather than hiding numeric files in
    # the JSON/log upper bound: baseline/final arrays and one preserved dense
    # vector per fit, plus final receipt/storage failure slack.
    emergency=dict(point0_complete_q_logq=6*(22546*3*8*4+8192),endpoint_uncompressed_q_logq_npy=6*(22546*3*8*4+4*128),preserved_dense_support_failure_vectors=6*(1060833*8+8192),failure_receipt_slack=8*1024**2)
    assert sum(emergency.values())<=resource['storage_components_upper_bytes']['emergency_dense_vector_and_failure']
    source=[Path(__file__).resolve(),resource_path,entry,ROOT/'training/v169_measurement_output_references_v2.py',ROOT/'training/v169_factorized_row_evidence.py',ROOT/'training/v169_structural_vector_storage.py']
    report=dict(status='V169_worst_filecount_scalar_corrected_without_resource_byte_or_scientific_schedule_change',source_resource_maximum_files110000_was_underestimate=True,complete_file_components=files,complete_directory_components=directories,derived_file_count=actual_file_bound,derived_directory_count=actual_directory_bound,corrected_maximum_files=file_limit,corrected_maximum_directories=dir_limit,corrected_padding_upper_bytes=filesystem_bound,existing_padding_reserve_bytes=reserved,existing_padding_reserve_covers_corrected_count=True,emergency_numeric_components_upper_bytes=emergency,emergency_numeric_sum=sum(emergency.values()),resources_unchanged=resource['resources'],root_should_use_corrected_inventory_for_physical_preseal=True,scientific_entry_unchanged=True,official_heads=0,official_features=0,official_derivatives=0,fits=0,permanent_updates=0,execution_authority=False,new_fit_permission=False,source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in source})
    out.mkdir();(out/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(status=report['status'],files=actual_file_bound,directories=actual_directory_bound,padding_bytes=filesystem_bound,reserved_bytes=reserved,official_calls=0)))

if __name__=='__main__':main()
