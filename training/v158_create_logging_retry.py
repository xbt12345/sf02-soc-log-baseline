"""Preserve completed fit and old seal; create a logging-only entry revision."""
import json
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings

def main():
    out=ROOT/'artifacts/v158_current_pipeline_OOF_trial_20261001'
    failure=read(out/'execution_failure.json');first=out/'outer0_inner0/V135_R_decay_full_network'
    assert failure['error_type']=='TypeError' and 'multiple values' in failure['error'] and "'stage'" in failure['error']
    r=read(first/'fit.json');assert r['status']=='base_stage_fit_executed' and r['batch_gradients']==r['accepted_updates']==6300
    assert sha(first/'endpoint.pt')==r['source_model_sha256'] and sha(first/'member_logits.npy')==r['logits_sha256']
    assert len(list(out.rglob('fit.json')))==1
    check_bindings(read(out/'run_seal.json')['source_sha256'])
    old=ROOT/'training/v158_nested_runtime.py';s=old.read_text(encoding='utf-8')
    s=s.replace("PLAN=ROOT/'training/review_policy/v158_complete_nested_trial_plan.json'","PLAN=ROOT/'training/review_policy/v158_complete_nested_trial_v2_plan.json'")
    s=s.replace("version='V158-complete-nested-member-fusion'","version='V158-complete-nested-member-fusion-logging-v2'")
    s=s.replace("'training/v158_nested_base_train.py'","'training/v158_nested_base_train_v2.py'").replace("'training/v158_legacy_nested_train.py'","'training/v158_legacy_nested_train_v2.py'")
    s=s.replace("OUT/'run_seal.json'","OUT/'run_seal_v2.json'")
    target=ROOT/'training/v158_nested_runtime_v2.py';assert not target.exists();target.write_text(s,encoding='utf-8')
    old=ROOT/'training/v158_nested_base_train.py';s=old.read_text(encoding='utf-8')
    s=s.replace('from v158_nested_runtime import ','from v158_nested_runtime_v2 import ')
    s=s.replace('import v158_legacy_nested_train','import v158_legacy_nested_train_v2')
    s=s.replace("emit(stage='nested_base_stage_complete',**result)","emit(event='nested_base_stage_complete',**result)")
    s=s.replace("OUT/'run_seal.json'","OUT/'run_seal_v2.json'").replace("OUT/'registration.json'","OUT/'registration_v2.json'")
    s=s.replace('p=review();configure();assert not OUT.exists();OUT.mkdir()',
        "p=review();configure();assert OUT.exists() and not (OUT/'run_seal_v2.json').exists()\n    from v158_nested_runtime import require as old_require\n    old_require(ROOT/'training/v158_nested_base_train.py')\n    assert expected_next()==(0,0,STAGES[1]) and len(list(OUT.rglob('fit.json')))==1")
    s=s.replace('extra={INPUT,TRACE,OFFICIAL,QUAL/',"extra={OUT/'run_seal.json',OUT/'execution_failure.json',INPUT,TRACE,OFFICIAL,QUAL/")
    s=s.replace('seal(__file__,extra)',"extra|=set(folder_for(0,0,STAGES[0]).glob('*'))\n    seal(__file__,extra)")
    s=s.replace('for stage in STAGES:fit(f,j,stage)',"for stage in STAGES:\n                if (folder_for(f,j,stage)/'fit.json').exists():\n                    assert (f,j,stage)==(0,0,STAGES[0]);continue\n                fit(f,j,stage)")
    s=s.replace('new_fits=0,updates=0,','new_fits=0,updates=0,inherited_completed_fits=1,inherited_batch_gradients=6300,inherited_updates=6300,')
    s=s.replace("OUT/'execution_failure.json'","OUT/'execution_failure_v2.json'")
    # The prospective extra must bind the original failure, not the future one.
    s=s.replace("extra={OUT/'run_seal.json',OUT/'execution_failure_v2.json'","extra={OUT/'run_seal.json',OUT/'execution_failure.json'")
    s=s.replace('setup_dummy_optimizer_updates=0,','setup_dummy_optimizer_updates=0,cumulative_setup_dummy_forwards=2,cumulative_setup_dummy_gradients=2,')
    s=s.replace('current_base_phase_fits_max=45,','current_base_phase_fits_max=45,remaining_new_current_base_fits_max=44,remaining_new_total_fits_max=59,')
    target=ROOT/'training/v158_nested_base_train_v2.py';assert not target.exists();target.write_text(s,encoding='utf-8')
    old=ROOT/'training/v158_legacy_nested_train.py';s=old.read_text(encoding='utf-8').replace('from v158_nested_runtime import ','from v158_nested_runtime_v2 import ').replace("OUT/'run_seal.json'","OUT/'run_seal_v2.json'")
    target=ROOT/'training/v158_legacy_nested_train_v2.py';assert not target.exists();target.write_text(s,encoding='utf-8')
    old=ROOT/'training/test_v158_complete_nested_trial.py';s=old.read_text(encoding='utf-8').replace('from v158_nested_runtime import ','from v158_nested_runtime_v2 import ')
    target=ROOT/'training/test_v158_complete_nested_trial_v2.py';assert not target.exists();target.write_text(s,encoding='utf-8')
    plan=read(ROOT/'training/review_policy/v158_complete_nested_trial_plan.json')
    plan['version']='V158-complete-nested-member-fusion-logging-v2'
    plan['execution_entries']=['training/v158_nested_base_train_v2.py','training/v158_legacy_nested_train_v2.py']
    plan['technical_retry']=dict(kind='post_receipt_emit_duplicate_keyword_only',inherited_completed_fits=1,
        inherited_batch_gradients=6300,inherited_updates=6300,additional_current_pipeline_fits_max=44,
        additional_legacy_fits_max=9,additional_fusion_fits_max=6,additional_total_fits_max=59,cumulative_fits_max=60,
        no_completed_fit_repeated=True,no_budget_reset=True,old_seal_preserved=True)
    paths=[ROOT/'training'/n for n in ['v158_nested_runtime_v2.py','v158_nested_base_train_v2.py','v158_legacy_nested_train_v2.py','test_v158_complete_nested_trial_v2.py','v158_create_logging_retry.py']]
    paths.extend([ROOT/'training/review_policy/v158_complete_nested_trial_plan.json',out/'run_seal.json',out/'execution_failure.json',first/'fit.json',ROOT/'docs/V158_COMPLETED_FIT_LOGGING_TECHNICAL_RETRY.md'])
    plan['source_sha256'].update({p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    target=ROOT/'training/review_policy/v158_complete_nested_trial_v2_plan.json';assert not target.exists();target.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    from v158_nested_runtime_v2 import review
    review();print(dict(logging_only_revision_created=True,inherited_fits=1,inherited_updates=6300,total_fits_cap_unchanged=60))

if __name__=='__main__':main()
