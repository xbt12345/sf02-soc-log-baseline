"""Read-only failed official preflight accounting, not a model quality test."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v159_class_boundary_trial_20261002'
OUT = ROOT / 'artifacts/v159_independent_preflight_failure_audit_20261002'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    OUT.mkdir()
    failure = RUN / 'execution_failure_preflight_None_None.json'
    log = RUN / 'preflight0_A/calls.jsonl'
    seal = RUN / 'run_seal.json'
    initial = RUN / 'initial.pt'
    source = ROOT / 'training/v159_boundary_train_v3.py'
    paths = [Path(__file__).resolve(), failure, log, seal, initial, source,
             RUN / 'registration.json', RUN / 'preflight_original_console.txt']
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}
    err = json.loads(failure.read_text(encoding='utf-8'))
    sealed = json.loads(seal.read_text(encoding='utf-8'))
    assert sha(source) == err['source_sha256'] == sealed['source_sha256'][source.relative_to(ROOT).as_posix()]
    assert err['error_type'] == 'AssertionError' and 'np.array_equal(gs[0][j],gs[1][j])' in err['traceback']
    events = [json.loads(z) for z in log.read_text(encoding='utf-8').splitlines()]
    counts = {k: {s: sum(e['kind'] == k and e['event'] == s for e in events) for s in ('attempt', 'completed')}
              for k in ('head', 'feature', 'full_class_gradient')}
    assert counts['head'] == counts['feature'] == {'attempt': 84, 'completed': 84}
    assert counts['full_class_gradient'] == {'attempt': 4, 'completed': 4}
    completed = [e for e in events if e['kind'] == 'full_class_gradient' and e['event'] == 'completed']
    assert [e['class_id'] for e in completed] == [1, 2, 1, 2]
    assert all(e['original_class_mass'] == [0, 58840, 33497] for e in completed)
    assert not (RUN / 'preflight.json').exists()
    assert not list(RUN.glob('fold*_*/started.json'))
    assert not list((RUN / 'preflight0_A').glob('*.npy'))
    for k, h in bindings.items():
        assert sha(ROOT / k) == h
    report = dict(status='real_preflight_stopped_before_any_fit_for_nonidentical_repeated_class_gradients',
                  official_attempted_completed_counts=counts,
                  first_role_original_class_mass=[0, 58840, 33497],
                  actual_classifier_fits=0, actual_parameter_updates=0,
                  setup_dummy_forward=1, setup_dummy_gradient=1, setup_dummy_update=0,
                  unconsumed_registered_preflight_head_cap=336 - 84,
                  unconsumed_registered_preflight_class_gradient_cap=12 - 4,
                  actual_difference_magnitude='not saved before assertion; currently unknown',
                  official_gradient_independently_recomputed=False,
                  classification_hypothesis_rejected=False,
                  further_fit_permission=False, quality_acceptance=False, model_promoted=False,
                  source_sha256=bindings,
                  next_action='Bound a limited zero-update diagnostic, save repeated vectors before comparison; account for all failed costs and do not reset budgets.')
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
