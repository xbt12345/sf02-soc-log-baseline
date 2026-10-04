"""Preserve failed sealer and create absolute-runtime-path-aware version."""
from experiment_review import ROOT,sha
import json
def main():
    old=ROOT/'training/v159_seal_evaluation_only_source_sum_repair.py';new=old.with_name('v159_seal_evaluation_only_source_sum_repair_v2.py');assert not new.exists()
    failure=ROOT/'artifacts/v159_evaluation_source_sum_repair_registration_failure_20261002';assert not failure.exists();failure.mkdir()
    (failure/'failure.json').write_text(json.dumps(dict(status='sealer_failed_before_plan_seal_or_any_new_model_call',error_type='ValueError',reason='external runtime DLL physical binding used relative_to(ROOT) unconditionally',failed_source_sha256=sha(old),verbatim_console='artifacts/v159_evaluation_only_repair_registration_original_console_20261002.txt',verbatim_console_sha256=sha(ROOT/'artifacts/v159_evaluation_only_repair_registration_original_console_20261002.txt'),new_model_calls=0,new_features=0,new_gradients=0,new_fits=0,new_updates=0),indent=2)+'\n',encoding='utf-8')
    text=old.read_text(encoding='utf-8')
    text=text.replace('assert not REPAIR.exists() and not PLAN.exists();REPAIR.mkdir()','assert not PLAN.exists();REPAIR.mkdir(exist_ok=True);assert not list(REPAIR.iterdir())')
    text=text.replace('binding={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}','binding={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}')
    text=text.replace("files={Path(__file__).resolve(),evidence,", "files={Path(__file__).resolve(),Path(__file__).with_name('v159_build_evaluation_repair_sealer_v2.py'),ROOT/'training/v159_seal_evaluation_only_source_sum_repair.py',ROOT/'artifacts/v159_evaluation_source_sum_repair_registration_failure_20261002/failure.json',ROOT/'artifacts/v159_evaluation_only_repair_registration_original_console_20261002.txt',ROOT/'artifacts/v159_independent_group_CE_failure_scope_audit_20261002/audit.json',evidence,")
    text=text.replace("fixed_per_row_eps=8,source_sum_rule=","clip_CE_policy='independently require each row clipped probability CE within fixed8eps; propagate its own envelope separately from stable logp CE',fixed_per_row_eps=8,source_sum_rule=")
    new.write_text(text,encoding='utf-8');print('absolute-path-aware sealer v2 created, no model calls')
if __name__=='__main__':main()
