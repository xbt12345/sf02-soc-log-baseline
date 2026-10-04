"""Replay the real DataFrame wiring failure through revised lifecycle only."""
import json
from pathlib import Path

from experiment_review import ROOT, sha, check_bindings
from v169_root_actual_blocker_type_counterexample import FrameBackend
from v169_root_lifecycle_v2_regression_review import V2Backend
from v169_pair_lifecycle_v3 import run_fit, blocker_count, paired_result

OUT = ROOT/'artifacts/v169_root_real_frame_lifecycle_regression_review_20261002'


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    files = [Path(__file__).resolve(), ROOT/'training/v169_pair_lifecycle_v3.py',
             ROOT/'training/v169_pair_lifecycle.py',
             ROOT/'training/v169_root_actual_blocker_type_counterexample.py',
             ROOT/'training/v169_root_lifecycle_v2_regression_review.py',
             ROOT/'training/v169_root_preparation_adversarial_review.py']
    bindings = {p.relative_to(ROOT).as_posix():sha(p) for p in files}
    cases, results = [], {}
    for empty in [False, True]:
        b = FrameBackend(empty)
        result = run_fit(b)
        assert result['exception'] is None and result['terminal_failure'] is None
        assert b.closed and b.commits == list(range(1, result['accepted_updates']+1))
        assert b.parameter == result['accepted_updates']
        assert b.protection == {0, *b.commits}
        if empty:
            assert result['status'] == 'finite_rejection_without_protected_blocker'
            assert result['accepted_updates'] == 1 and b.normal_calls == b.function_calls == 0
        else:
            assert result['fixed_endpoint_reached'] and result['accepted_updates'] == 20
            assert b.normal_calls == b.function_calls == 19
            assert b.targets_called == [(i,i) for i in range(21)]
        results[str(empty)] = result
        cases.append(dict(actual_blocker_type='pandas.DataFrame', empty=empty,
            status=result['status'], correction_calls=b.normal_calls,
            scripted_commits=result['accepted_updates'], old_and_cumulative_protection_kept=True,
            resources_closed=True))
    assert paired_result(results['False'], results['True'])['supports_candidate_B'] is False
    try:
        blocker_count(set())
    except TypeError:
        unknown_refused = True
    else:
        raise AssertionError('Unknown blocker interface was accepted')
    for mode in ['first_targets','after_second_commit','final_replay','final_receipt']:
        b = V2Backend(mode)
        result = run_fit(b)
        retained = [int(s['state']) for s in result['accepted_states']]
        assert b.closed and b.commits == retained
        assert b.parameter == (retained[-1] if retained else 0)
        assert b.protection == {0, *retained} and not result['fixed_endpoint_reached']
        if mode == 'final_replay':
            assert b.terminal_record is not None and result['terminal_failure'] is not None
        if mode == 'final_receipt':
            assert result['status'] == 'terminal_receipt_failed_inconclusive'
        cases.append(dict(failure_injection=mode, status=result['status'],
            retained_commits=retained, restored=True, resources_closed=True))
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='real_pandas_frame_blocker_branches_and_previous_terminal_failures_replayed',
        all_checks_passed=True, cases=cases, unknown_blocker_type_refused=unknown_refused,
        original_failure_preserved='artifacts/v169_root_actual_blocker_type_counterexample_20261002/review.json',
        official_heads=0, official_features=0, official_derivatives=0,
        fits=0, permanent_updates=0, supports_physical_seal=False,
        source_sha256=bindings,
        scope='Revised lifecycle with the actual blocker type and scripted state/guard oracle only. Does not qualify the real model backend, original-row guard additions, budgets, physical seal or SOC classification.')
    (OUT/'review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','cases']},ensure_ascii=False))


if __name__ == '__main__':
    main()
