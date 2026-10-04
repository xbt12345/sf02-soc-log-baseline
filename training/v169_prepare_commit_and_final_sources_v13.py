"""Final commit prerequisite, exact lower-memory kernel and current candidates."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_prior_pair_training_entry_v12.py';target=ROOT/'training/v169_prior_pair_training_entry_v13.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('from v169_dynamic_trial_restoration import floors,propose','from v169_dynamic_trial_restoration_v2 import floors,propose').replace('from v169_pair_execution_review_v3 import require_run_seal','from v169_pair_execution_review_v4 import require_run_seal');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_actual_backend_synthetic_qualification_v3.py';target=ROOT/'training/v169_actual_backend_synthetic_qualification_v4.py';assert not target.exists();s=source.read_text(encoding='utf-8').replace('v169_prior_pair_training_entry_v12','v169_prior_pair_training_entry_v13').replace('v169_actual_backend_synthetic_qualification_v3_20261002','v169_actual_backend_synthetic_qualification_v4_20261002').replace('v169_pair_execution_review_v3.py','v169_pair_execution_review_v4.py');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_factorized_resource_budget_v5.py';target=ROOT/'training/v169_factorized_resource_budget_v6.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_factorized_resource_budget_v5_20261002','v169_factorized_resource_budget_v6_20261002').replace('v169_prior_pair_training_entry_v12.py','v169_prior_pair_training_entry_v13.py').replace('V169_v12_factorized','V169_v13_factorized').replace('maximum_files=110000','maximum_files=128000').replace('110000*4096+17000*8192','128000*4096+17000*8192')
    old="    # Eight full 66-row arrays covers normals/matrix/scaled/SVD copies/workspace;"
    new="    resources['minimum_free_commit_bytes']=int(6.25*GiB)\n    source.extend([ROOT/'artifacts/v169_saved_storage_filecount_review_20261002/review.json',ROOT/'artifacts/v169_incremental_RAM_live_buffer_review_20261002/review.json',ROOT/'artifacts/v169_cuda_host_buffer_qualification_v2_20261002/qualification.json',ROOT/'artifacts/v169_memory_exact_solver_qualification_20261002/qualification.json',ROOT/'training/v169_pair_execution_review_v4.py'])\n    # Eight full 66-row arrays is still a conservative upper after one copy removal;"
    assert old in s;s=s.replace(old,new)
    s=s.replace("RAM_array_accounting_upper_bytes=ram_bound", "RAM_array_accounting_upper_bytes=ram_bound,actual_RAM_live_buffer_review=source[-4].relative_to(ROOT).as_posix(),commit_increment_plus_eight_matrix_and128MiB_bytes=6568317056,minimum_commit_256MiB_rounding_bytes=6710886400,physical_RAM_prerequisite_retained6GiB=True,same_full_SVD_inputs_one_matrix_copy_removed_qualification=source[-2].relative_to(ROOT).as_posix()")
    compile(s,str(target),'exec');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_prepare_execution_contract_candidate_v2.py';target=ROOT/'training/v169_prepare_execution_contract_candidate_v3.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_prior_pair_execution_contract_candidate_v2.json','v169_prior_pair_execution_contract_candidate_v3.json').replace('v169_execution_contract_candidate_v2_20261002','v169_execution_contract_candidate_v3_20261002').replace('v169_prior_pair_training_entry_v12','v169_prior_pair_training_entry_v13').replace('v169_factorized_resource_budget_v5_20261002','v169_factorized_resource_budget_v6_20261002').replace('v169_pair_execution_review_v3 import','v169_pair_execution_review_v4 import')
    # registry_chain remains owned by v3; v4 only adds memory prerequisite.
    s=s.replace('from v169_pair_execution_review_v4 import review_plan,registry_chain','from v169_pair_execution_review_v4 import review_plan\nfrom v169_pair_execution_review_v3 import registry_chain')
    s=s.replace("ROOT/'training/v169_pair_execution_review_v3.py',", "ROOT/'training/v169_pair_execution_review_v3.py',ROOT/'training/v169_pair_execution_review_v4.py',ROOT/'training/v169_dynamic_trial_restoration_v2.py',ROOT/'training/v169_working_joint_restoration_v2.py',")
    compile(s,str(target),'exec');target.write_text(s,encoding='utf-8');print('Prepared v13 entry, backend v4, resource v6 and contract v3')

if __name__=='__main__':main()
