"""Register two new cached continuation probes without changing old seals."""
import ast,importlib.metadata,json,sys
from pathlib import Path
from experiment_review import ROOT,read,sha,check_bindings
import v162_cached_function_restoration_tail as entry

def main():
    assert not entry.OUT.exists() and not entry.PLAN.exists()
    previous=ROOT/'artifacts/v162_fixed_endpoint_finite_restoration_20261002';old=read(previous/'activation_seal_v2.json');check_bindings(old['source_sha256'])
    qualifier=ROOT/'artifacts/v162_cached_function_restoration_tail_qualification_20261002';q=read(qualifier/'qualification.json');assert 'passed' in q['status'] and q['official_heads']==q['official_gradients']==q['official_fits']==0;check_bindings(q['source_sha256'])
    results=ROOT/'artifacts/v162_results_20261002/actual_result_summary.json';r=read(results);assert r['cumulative_heads']==17246 and r['cumulative_all_complete_derivatives']==422 and r['role_finite_pass']==[False,True,False];check_bindings(r['source_sha256'])
    files={ROOT/k for k in old['source_sha256']}|{ROOT/k for k in q['source_sha256']}|{p for p in qualifier.rglob('*') if p.is_file()}|{p for p in previous.rglob('*') if p.is_file()}
    files|={ROOT/k for k in r['source_sha256']}|{results,Path(__file__).resolve(),ROOT/'training/v162_cached_function_restoration_tail.py',previous/'activation_seal_v2.json',previous/'run_seal.json',ROOT/'artifacts/v162_independent_actual_restoration_review_20261002/pre_review_bindings.json',ROOT/'artifacts/v162_independent_actual_restoration_review_20261002/review.json'}
    for p in files:
        if p.suffix=='.py' and p.is_relative_to(ROOT/'training'):ast.parse(p.read_text(encoding='utf-8-sig'))
    bindings={(p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)):sha(p) for p in sorted(files)}
    plan=dict(protocol=entry.PROTOCOL,allowed_entries=['training/v162_cached_function_restoration_tail.py'],new_caps=entry.CAPS,endpoint_parameter_sha256=read(entry.PREVIOUS/'diagnostic.json')['initial_parameter_sha256'],prior_actual_costs=dict(heads=17246,features=17246,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=42,complete_parameter_derivatives_all_types=422),future_cumulative_actual_caps=dict(heads=17334,features=17334,full_original_class_gradients=368,fixed_error_target_gradients=12,margin_gradients=42,complete_parameter_derivatives_all_types=422),preserved_technical_head_cap=82174,preserved_technical_complete_derivative_cap=2426,single_factor='two further restorations from unchanged same-function fourth actual finite displacement and margin',old_four_calls_and_budget_preserved=True,old_run_seal_sha256=sha(previous/'run_seal.json'),old_activation_seal_sha256=sha(previous/'activation_seal_v2.json'),max_new_restorations=2,no_new_function_or_gradient=True,no_fits_or_permanent_updates=True,all_original_classification_and_numeric_gates_unchanged=True,no_short_training_authority=True,source_sha256=bindings,quality_acceptance=False)
    assert 17334<=82174 and 422<=2426
    entry.OUT.mkdir();entry.save(entry.OUT/'pre_registration_bindings.json',dict(source_sha256=bindings,official_calls=0));entry.save(entry.PLAN,plan);bindings[entry.PLAN.relative_to(ROOT).as_posix()]=sha(entry.PLAN)
    entry.save(entry.OUT/'run_seal.json',dict(status='cached_same_function_tail_sealed_before_calls',protocol=entry.PROTOCOL,allowed_entries=plan['allowed_entries'],plan_sha256=sha(entry.PLAN),python_version=sys.version,package_versions={n:importlib.metadata.version(n) for n in ['torch','numpy','scipy','pandas','pyarrow']},source_sha256=bindings))
    entry.require();entry.save(entry.OUT/'registration.json',dict(status='V162_cached_function_tail_registered',physical_sources=len(bindings),new_caps=entry.CAPS,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,run_seal_sha256=sha(entry.OUT/'run_seal.json')))
    print(json.dumps(dict(status='V162_cached_function_tail_sealed',physical_sources=len(bindings),new_caps=entry.CAPS)))

if __name__=='__main__':main()
