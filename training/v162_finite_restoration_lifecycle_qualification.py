"""Actual entry lifecycle using mocked cached forwards, zero official calls."""
import json,traceback
from pathlib import Path
from unittest.mock import patch
import numpy as np
from experiment_review import ROOT,read,sha,check_bindings
import v162_fixed_endpoint_finite_restoration as entry
import v161_fixed_error_endpoint_diagnostic_v2 as primitive
OUT=ROOT/'artifacts/v162_finite_restoration_lifecycle_qualification_20261002'

def main():
    assert not OUT.exists();OUT.mkdir();role=2;root=entry.DIAG/'role2';old=root/'round0/probe4'
    ctx=entry.endpoint_context(role);base={scope:(np.load(root/('baseline_error_class1_repeat0' if scope=='OOF' else 'baseline_deployment')/f'{scope}_q.npy'),np.load(root/('baseline_error_class1_repeat0' if scope=='OOF' else 'baseline_deployment')/f'{scope}_logq.npy')) for scope in ['OOF','deployment']}
    failed={scope:(np.load(old/f'{scope}_q.npy'),np.load(old/f'{scope}_logq.npy')) for scope in ['OOF','deployment']}
    restored={scope:(q.copy(),lp.copy()) for scope,(q,lp) in failed.items()}
    # Two real cached S rows fail at one function; replace just its probabilities
    # with the baseline table to make a coherent mocked recovery response.
    for index in range(2):restored['OOF'][index][21050]=base['OOF'][index][21050]
    baseline_risk={key:np.load(root/f'baseline_error_class1_repeat0/{key}.npy') for key in ['fixed_pure_error_contribution','full_original_class_CE']}
    def rv(table):
        lp=table['OOF'][1][ctx['ids']];return {key:-(counts[ctx['ids']]*lp).sum(0)[1:]/ctx['mass'][1:] for key,counts in [('fixed_pure_error_contribution',ctx['target_counts']),('full_original_class_CE',ctx['counts'])]}
    specs=read(ROOT/'training/review_policy/v161_fixed_error_endpoint_diagnostic_contract.json')['roles'];specs[2]=dict(specs[2],head_cap=178,fresh_margin_gradient_cap=46)
    fakeplan=dict(roles=specs);initial_files={Path(__file__).resolve(),ROOT/'training/v162_fixed_endpoint_finite_restoration.py',ROOT/'training/v161_fixed_error_endpoint_diagnostic_v2.py',ROOT/'training/v162_multi_function_finite_restoration_v2.py'}
    initial_files|={p for p in root.rglob('*') if p.is_file()}|{ROOT/'training/review_policy/v161_fixed_error_endpoint_diagnostic_contract.json',entry.PRIOR/'fold2_B/endpoint.pt'}
    bindings={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(initial_files)};entry.save(OUT/'pre_bindings.json',dict(source_sha256=bindings,official_calls=0));reports=[]
    for case in ['same_function_second_restoration','injected_failure_restore']:
        location=OUT/case;location.mkdir();entry.save(location/'run_seal.json',dict(synthetic_lifecycle_fixture=True,official_calls=0));state=dict(probes=0,current=None)
        def baseline_error(*args,**kwargs):return baseline_risk,*base['OOF'],None
        def baseline_deployment(*args,**kwargs):return base['deployment']
        def candidate_error(*args,**kwargs):
            state['probes']+=1
            if case=='injected_failure_restore':raise RuntimeError('Injected after actual candidate parameter assignment, before mocked forward')
            state['current']=failed if state['probes']==1 else restored
            return rv(state['current']),*state['current']['OOF'],None
        def candidate_deployment(*args,**kwargs):return state['current']['deployment']
        with patch.object(entry,'OUT',location),patch.object(entry,'require',return_value=fakeplan),patch.object(entry,'error_risk',side_effect=baseline_error),patch.object(entry,'probabilities',side_effect=baseline_deployment),patch.object(primitive,'error_risk',side_effect=candidate_error),patch.object(primitive,'probabilities',side_effect=candidate_deployment):
            try:entry.run(2)
            except RuntimeError:
                assert case=='injected_failure_restore'
            else:assert case=='same_function_second_restoration'
        result=read(location/'role2/diagnostic.json');assert result['initial_parameter_sha256']==result['restored_parameter_sha256'] and result['counts']['head_attempts']==result['counts']['feature_attempts']==result['counts']['gradient_attempts']==result['counts']['margin_attempts']==0
        if case=='same_function_second_restoration':
            assert result['finite_proposals']==result['restoration_solves']==2 and result['actual_finite_restoration_pass']
            first=read(location/'role2/restoration0/finite_probe/probe.json');second=read(location/'role2/restoration1/finite_probe/probe.json')
            assert not first['accepted'] and not first['classification_guard'] and first['OOF_stats']['protected_regressions']==2 and second['accepted']
            assert read(location/'role2/restoration0/active_functions.json')==read(location/'role2/restoration1/active_functions.json')
        else:assert result['exception'] and result['finite_proposals']==1 and not result['actual_finite_restoration_pass']
        reports.append(dict(case=case,passed=True,counts=result['counts'],parameter_hash_restored=True,synthetic_cached_forward_tables_only=True))
    check_bindings(bindings);entry.save(OUT/'qualification.json',dict(status='actual_generic_entry_mocked_full_guard_same_function_second_correction_and_injected_exception_restore_passed',reports=reports,real_parameter_assignment_and_hash_restore=True,all_forward_results_mocked_from_cached_tables=True,not_actual_new_model_finite_or_classification_results=True,official_heads=0,official_features=0,official_gradients=0,official_fits=0,permanent_updates=0,source_sha256=bindings))
    print(json.dumps(dict(status='V162_lifecycle_qualification_passed',official_calls=0)))

if __name__=='__main__':
    try:main()
    except Exception as error:
        if OUT.exists():entry.save(OUT/'failure.json',dict(error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),official_calls=0))
        raise
