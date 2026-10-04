"""Independent synthetic lifecycle pressure review; no official model calls."""
import json
from pathlib import Path

from experiment_review import ROOT, sha, check_bindings
from v169_pair_lifecycle import run_fit, paired_result

OUT = ROOT / 'artifacts/v169_root_preparation_adversarial_review_20261002'


class Backend:
    def __init__(self, fail_at=None):
        self.parameter = 0
        self.protection = {0}
        self.commits = []
        self.fail_at = fail_at
        self.closed = False
        self.targets_called = []
        self.failed = None

    def identity(self): return str(self.parameter)
    def snapshot(self): return self.parameter
    def protection_snapshot(self): return set(self.protection)
    def started(self): pass
    def baseline(self): return {'parameter': self.parameter}
    def accepted_observation(self): return {'parameter': self.parameter}
    def targets(self, observation, state):
        self.targets_called.append((state, self.parameter))
        if self.fail_at == 'first_targets':
            raise RuntimeError('injected first target failure')
        return [None, None]
    def bootstrap(self, observation, gradients):
        return {'accepted': True, 'blockers': [], 'parameter_sha256': '1'}
    def direction_trial(self, observation, gradients, state):
        return {'accepted': True, 'blockers': [],
                'parameter_sha256': str(state + 1)}
    def commit(self, trial, state):
        self.parameter = state
        self.protection.add(state)
        self.commits.append(state)
    def state_review(self, state):
        if self.fail_at == 'after_second_commit' and state == 2:
            raise RuntimeError('injected state review failure after commit receipt')
        return {'state': state, 'parameter_sha256': str(self.parameter)}
    def restore(self, state, ledger):
        self.parameter = state
        self.protection = set(ledger)
    def failure(self, value): self.failed = value
    def final_replay(self):
        if self.fail_at == 'final_replay':
            raise RuntimeError('injected final replay failure')
        return {'parameter_sha256': str(self.parameter),
                'protected_states': sorted(self.protection)}
    def finished(self, stop, accepted, endpoint): self.closed = True


def main():
    if OUT.exists(): raise FileExistsError(OUT)
    paths = [Path(__file__).resolve(), ROOT / 'training/v169_pair_lifecycle.py',
             ROOT / 'training/v169_prior_pair_training_entry.py']
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    cases = []
    normal = Backend()
    result = run_fit(normal)
    cases.append(dict(name='bootstrap_included_in_twenty_fixed_commits',
        passed=result['fixed_endpoint_reached'] and result['accepted_updates'] == 20
            and normal.commits == list(range(1, 21))
            and normal.targets_called == [(i, i) for i in range(21)]
            and normal.closed,
        actual_commits=normal.commits, complete_target_points=normal.targets_called))
    early = Backend('first_targets')
    result_early = run_fit(early)
    cases.append(dict(name='early_failure_restores_initial_parameter_and_guard',
        passed=early.parameter == 0 and early.protection == {0}
            and result_early['accepted_updates'] == 0 and early.closed,
        returned_status=result_early['status']))
    pair = paired_result(result, result_early)
    cases.append(dict(name='unmatched_terminal_cannot_support_candidate',
        passed=pair['status'] == 'inconclusive_unmatched_registered_fixed_endpoint'
            and pair['supports_candidate_B'] is False))
    postcommit = Backend('after_second_commit')
    postresult = run_fit(postcommit)
    cases.append(dict(name='commit_receipt_and_restored_endpoint_consistent_after_exception',
        passed=postcommit.commits == [int(s['state']) for s in postresult['accepted_states']],
        actual_durable_commits=postcommit.commits,
        returned_accepted_states=[int(s['state']) for s in postresult['accepted_states']],
        actual_restored_parameter=postcommit.parameter,
        interpretation='A committed receipt must be finalized as retained or explicitly rolled back; it cannot silently remain while endpoint reports an earlier state.'))
    terminal = Backend('final_replay')
    escaped = None
    try: run_fit(terminal)
    except Exception as error: escaped = dict(type=type(error).__name__, message=str(error))
    cases.append(dict(name='final_replay_exception_closes_counter_and_preserves_failure_receipt',
        passed=terminal.closed and terminal.failed is not None,
        counter_closed=terminal.closed, failure_recorded=terminal.failed is not None,
        escaped_exception=escaped,
        interpretation='Restoration, final replay, and counter closure need separate finally boundaries.'))
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='V169_initial_lifecycle_independent_synthetic_pressure_review',
        all_checks_passed=all(c['passed'] for c in cases), cases=cases,
        official_heads=0, official_features=0, official_derivatives=0, fits=0,
        permanent_updates=0, execution_authority=False, preseal_review=False,
        scope='Synthetic backend lifecycle only; no official classifier instantiated, evaluated or differentiated. Entry must independently enforce full gold/protection and exact cost/resource limits.',
        source_sha256=bindings)
    (OUT / 'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}, ensure_ascii=False))


if __name__ == '__main__': main()
