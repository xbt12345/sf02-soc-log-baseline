"""Replay real preparation failures through the new synthetic lifecycle only."""
import json
from pathlib import Path

from experiment_review import ROOT, sha, check_bindings
from v169_root_preparation_adversarial_review import Backend
from v169_pair_lifecycle_v2 import run_fit, paired_result

OUT = ROOT / 'artifacts/v169_root_lifecycle_v2_regression_review_20261002'


class V2Backend(Backend):
    def __init__(self, fail_at=None):
        super().__init__(fail_at)
        self.terminal_record = None
        self.final_receipt = None

    def finished(self, stop, accepted, endpoint):
        if self.fail_at == 'final_receipt':
            raise RuntimeError('injected final receipt write failure')
        self.final_receipt = dict(stop=stop, accepted=accepted, endpoint=endpoint)

    def terminal_failure(self, failure): self.terminal_record = failure
    def close_resources(self): self.closed = True


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    paths = [Path(__file__).resolve(),
             ROOT / 'training/v169_root_preparation_adversarial_review.py',
             ROOT / 'training/v169_pair_lifecycle_v2.py',
             ROOT / 'training/v169_pair_lifecycle.py',
             ROOT / 'training/v169_prior_pair_training_entry_v2.py']
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    cases, results = [], {}
    for mode in [None, 'first_targets', 'after_second_commit', 'final_replay', 'final_receipt']:
        backend = V2Backend(mode)
        result = run_fit(backend)
        retained = [int(s['state']) for s in result['accepted_states']]
        assert backend.closed and backend.commits == retained
        assert backend.parameter == (retained[-1] if retained else 0)
        assert backend.protection == {0, *retained}
        if mode is None:
            assert result['fixed_endpoint_reached'] and retained == list(range(1, 21))
            assert backend.targets_called == [(i, i) for i in range(21)]
        else:
            assert not result['fixed_endpoint_reached']
        if mode == 'final_replay':
            assert backend.terminal_record is not None
            assert result['endpoint'] is None and result['terminal_failure'] is not None
        if mode == 'final_receipt':
            assert result['terminal_failure'] is not None
            assert result['status'] == 'terminal_receipt_failed_inconclusive'
        cases.append(dict(failure_injection=mode or 'none', passed=True,
            stop=result['status'], retained_commits=retained,
            parameter_restored=backend.parameter, resources_closed=backend.closed,
            terminal_failure_recorded=backend.terminal_record is not None,
            returned_terminal_failure=result['terminal_failure']))
        results[mode or 'none'] = result
    for mode in ['first_targets', 'after_second_commit', 'final_replay', 'final_receipt']:
        pair = paired_result(results['none'], results[mode])
        assert not pair['supports_candidate_B']
        assert pair['status'] == 'inconclusive_unmatched_registered_fixed_endpoint'
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='V169_new_lifecycle_saved_preparation_failures_synthetic_replay_passed',
        all_checks_passed=True, cases=cases,
        original_counterexample='artifacts/v169_root_preparation_adversarial_review_20261002/review.json',
        original_counterexample_preserved=True,
        official_heads=0, official_features=0, official_derivatives=0, fits=0,
        permanent_updates=0, execution_authority=False, preseal_review=False,
        scope='Synthetic lifecycle and durable-state/resource handling only. Does not certify the actual backend model, gold, cost graph, physical seal or classification quality.',
        source_sha256=bindings)
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__': main()
