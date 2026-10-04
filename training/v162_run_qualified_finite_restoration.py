"""Execute registered budget after binding post-seal lifecycle qualification."""
import argparse
from experiment_review import ROOT,read,sha,check_bindings
import v162_fixed_endpoint_finite_restoration as entry
PLAN=ROOT/'training/review_policy/v162_finite_restoration_activation_contract_v2.json'
SEAL=entry.OUT/'activation_seal_v2.json'

def require_activation():
    seal=read(SEAL);plan=read(PLAN)
    assert seal['allowed_activation_entries']==['training/v162_run_qualified_finite_restoration.py']
    assert sha(PLAN)==seal['activation_plan_sha256'] and sha(entry.OUT/'run_seal.json')==plan['registered_run_seal_sha256']
    assert plan['additional_budget']==0 and plan['registered_caps']==entry.require()['new_caps']
    check_bindings(seal['source_sha256'])
    return plan

def main(role):
    require_activation();entry.run(role)
    entry.save(entry.OUT/f'role{role}/activation_receipt_v2.json',dict(status='original_registered_finite_entry_executed_after_lifecycle_supplement',activation_seal_sha256=sha(SEAL),registered_run_seal_sha256=sha(entry.OUT/'run_seal.json'),diagnostic_sha256=sha(entry.OUT/f'role{role}/diagnostic.json'),additional_budget=0,no_new_fit_authority=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',type=int,choices=[0,1,2],required=True);main(p.parse_args().role)
