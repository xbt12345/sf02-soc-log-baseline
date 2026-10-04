"""Scripted lifecycle wired to the actual pandas blocker type, no model calls."""
import json
from pathlib import Path

import pandas as pd

from experiment_review import ROOT, sha, check_bindings
from v169_root_lifecycle_v2_regression_review import V2Backend
from v169_pair_lifecycle_v2 import run_fit

OUT = ROOT / 'artifacts/v169_root_actual_blocker_type_counterexample_20261002'


class FrameBackend(V2Backend):
    def __init__(self, empty):
        super().__init__()
        self.empty = empty
        self.normal_calls = 0
        self.function_calls = 0

    def direction_trial(self, observation, gradients, state):
        records = [] if self.empty else [dict(local=1, truth=2, rival=1, scope='OOF')]
        return dict(accepted=False, blockers=pd.DataFrame.from_records(records),
                    parameter_sha256=str(state+1))

    def blocking_functions(self, trial):
        self.function_calls += 1
        return {'f': dict(input_identity='f', row=1)}

    def trial_normals(self, *args):
        self.normal_calls += 1
        return None

    def correction_trial(self, observation, gradients, trial, active, normals, state, correction):
        return dict(trial, accepted=True)


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    paths = [Path(__file__).resolve(), ROOT/'training/v169_pair_lifecycle_v2.py',
             ROOT/'training/v169_root_lifecycle_v2_regression_review.py',
             ROOT/'training/v169_root_preparation_adversarial_review.py',
             ROOT/'training/v169_prior_pair_training_entry_v7.py']
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    cases = []
    for empty in [False, True]:
        b = FrameBackend(empty)
        result = run_fit(b)
        assert result['exception']['type'] == 'ValueError'
        assert 'truth value of a DataFrame is ambiguous' in result['exception']['message']
        assert result['status'] == 'technical_or_resource_stop_inconclusive'
        assert result['accepted_updates'] == 1 and b.commits == [1]
        assert b.parameter == 1 and b.protection == {0, 1} and b.closed
        assert b.function_calls == b.normal_calls == 0
        cases.append(dict(actual_blocker_type='pandas.DataFrame', empty=empty,
            actual_result=result, correction_stage_never_reached=True,
            retained_commit=1, parameters_and_protection_restored=True,
            resources_closed=True))
    check_bindings(bindings)
    OUT.mkdir()
    report = dict(status='actual_backend_blocker_type_lifecycle_counterexample_reproduced',
        all_checks_passed=False, counterexample_reproduced=True,
        required_action='Replace ambiguous bool(DataFrame) with explicit blocker row emptiness and qualify both real DataFrame branches before any official execution.',
        cases=cases, official_heads=0, official_features=0,
        official_derivatives=0, fits=0, permanent_updates=0,
        supports_physical_seal=False, source_sha256=bindings,
        scope='Scripted lifecycle uses the actual pandas DataFrame interface. Its one scripted commit is not an official model update, and this is not classification evidence.')
    (OUT/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['source_sha256','cases']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
