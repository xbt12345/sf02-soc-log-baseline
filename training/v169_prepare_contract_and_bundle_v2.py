"""Version final storage-accounting and complete closure preparation."""
from experiment_review import ROOT

def main():
    source=ROOT/'training/v169_prepare_execution_contract_candidate.py';target=ROOT/'training/v169_prepare_execution_contract_candidate_v2.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_prior_pair_execution_contract_candidate_v1.json','v169_prior_pair_execution_contract_candidate_v2.json').replace('v169_execution_contract_candidate_20261002','v169_execution_contract_candidate_v2_20261002').replace('v169_prior_pair_training_entry_v11','v169_prior_pair_training_entry_v12').replace('v169_factorized_resource_budget_v4_20261002','v169_factorized_resource_budget_v5_20261002');target.write_text(s,encoding='utf-8')
    source=ROOT/'training/v169_prepare_full_preseal_bundle.py';target=ROOT/'training/v169_prepare_full_preseal_bundle_v2.py';assert not target.exists()
    s=source.read_text(encoding='utf-8').replace('v169_prior_pair_training_entry_v11','v169_prior_pair_training_entry_v12').replace('v169_full_preseal_bundle_20261002','v169_full_preseal_bundle_v2_20261002').replace('v169_prior_pair_execution_contract_candidate_v1.json','v169_prior_pair_execution_contract_candidate_v2.json').replace('v169_actual_backend_synthetic_qualification_v2','v169_actual_backend_synthetic_qualification_v3').replace("'v169_execution_contract_candidate'","'v169_execution_contract_candidate_v2'").replace("'v169_factorized_resource_budget_v4'","'v169_factorized_resource_budget_v5'").replace('V169_final_v11','V169_final_v12');target.write_text(s,encoding='utf-8');print('Prepared final concrete v2 contract/bundle')

if __name__=='__main__':main()
