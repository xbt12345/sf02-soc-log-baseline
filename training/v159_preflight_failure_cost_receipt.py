"""Reconcile failed real preflight from actual journals, preserving all files."""
import json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

OUT=ROOT/'artifacts/v159_preflight_failure_cost_receipt_20261002'
TRIAL=ROOT/'artifacts/v159_class_boundary_trial_20261002'
def main():
    assert not OUT.exists();OUT.mkdir()
    paths=[Path(__file__).resolve(),TRIAL/'run_seal.json',TRIAL/'registration.json',TRIAL/'execution_failure_preflight_None_None.json',TRIAL/'preflight_original_console.txt',TRIAL/'preflight0_A/calls.jsonl',ROOT/'artifacts/v159_independent_preflight_failure_audit_20261002/audit.json']
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in paths}
    (OUT/'pre_saved_journal_bindings.json').write_text(json.dumps(dict(source_sha256=bindings),indent=2)+'\n',encoding='utf-8')
    e=[json.loads(s) for s in (TRIAL/'preflight0_A/calls.jsonl').read_text().splitlines()]
    counts={k:sum(x['kind']==k and x['event']=='attempt' for x in e) for k in ['head','feature','full_class_gradient']}
    for k in counts:assert counts[k]==sum(x['kind']==k and x['event']=='completed' for x in e)
    assert counts==dict(head=84,feature=84,full_class_gradient=4) and not list(TRIAL.glob('fold*_*/started.json')) and not (TRIAL/'preflight.json').exists()
    result=dict(status='failed_bitwise_gradient_repeat_before_first_fit_costs_conserved',actual_official_head_calls=84,actual_official_opinion_feature_calls=84,actual_complete_class_gradients=4,official_fits=0,official_updates=0,
        original_preflight_head_cap=336,original_preflight_gradient_cap=12,remaining_original_preflight_heads=252,remaining_original_preflight_gradients=8,
        cumulative_trial_head_cap=78072,cumulative_trial_class_gradient_cap=2412,remaining_total_head_cap=77988,remaining_total_class_gradient_cap=2408,
        repeated_gradient_difference_magnitude='unknown: original vectors were not saved before failed assertion; do not infer tolerance or cause',
        source_and_seal_preserved=True,no_same_entry_restart=True,quality_acceptance=False,goal_status='active',source_sha256=bindings)
    check_bindings(bindings);(OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
