"""Independent saved-vector arithmetic; no official forward or gradient."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v159_bound_gradient_repeat_diagnostic_20261002'
OUT = ROOT / 'artifacts/v159_independent_gradient_repeat_diagnostic_audit_20261002'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda: stream.read(2 ** 20), b''):
            h.update(b)
    return h.hexdigest()


def metric(a, b):
    delta = a - b
    same_sign = np.signbit(a) == np.signbit(b)
    assert same_sign.all()
    ulp = np.abs(a.view(np.int64) - b.view(np.int64))
    return dict(bitwise_equal=bool(np.array_equal(a, b)), changed_components=int(np.count_nonzero(delta)),
                maximum_absolute=float(np.max(np.abs(delta))), maximum_ULP=int(ulp.max()),
                relative_L2=float(np.linalg.norm(delta) / max(np.linalg.norm(a), np.linalg.norm(b), np.finfo(float).tiny)),
                sign_changes=int(np.count_nonzero(np.sign(a) != np.sign(b))))


def direction(gm, gs, arm):
    if arm == 'A':
        alpha = 58840 / (58840 + 33497)
    else:
        difference = gm - gs
        denominator = float(difference @ difference)
        alpha = float(np.clip((gs @ gs - gm @ gs) / denominator, 0, 1)) if denominator else 0.5
    raw = -(alpha * gm + (1 - alpha) * gs)
    normalized = raw / np.max(np.abs(raw))
    slopes = [float(g @ normalized) for g in (gm, gs)]
    return alpha, normalized, slopes


def main():
    assert not OUT.exists()
    OUT.mkdir()
    report = json.loads((RUN / 'qualification.json').read_text(encoding='utf-8'))
    seal = json.loads((RUN / 'run_seal.json').read_text(encoding='utf-8'))
    source = ROOT / 'training/v159_bound_gradient_repeat_diagnostic.py'
    assert sha(source) == seal['source_sha256'][source.relative_to(ROOT).as_posix()]
    events = [json.loads(z) for z in (RUN / 'actual_calls.jsonl').read_text(encoding='utf-8').splitlines()]
    counts = {k: {s: sum(e['kind'] == k and e['event'] == s for e in events) for s in ('attempt', 'completed')}
              for k in ('head', 'feature', 'full_class_gradient')}
    assert counts['head'] == counts['feature'] == {'attempt': 40, 'completed': 40}
    assert counts['full_class_gradient'] == {'attempt': 4, 'completed': 4}
    paths = [Path(__file__).resolve(), source, RUN / 'qualification.json', RUN / 'run_seal.json', RUN / 'actual_calls.jsonl']
    vectors = [[], []]
    receipts = []
    for repetition in range(2):
        for cls in (1, 2):
            path = RUN / f'repetition{repetition}_class{cls}_complete_gradient.npy'
            receipt_path = RUN / f'repetition{repetition}_class{cls}_receipt.json'
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            assert sha(path) == receipt['gradient_sha256']
            assert receipt['repetition'] == repetition and receipt['class_id'] == cls
            assert receipt['original_class_mass'] == [0, 58840, 33497]
            g = np.load(path)
            assert g.shape == (1060832,) and g.dtype == np.float64 and np.isfinite(g).all()
            assert not np.count_nonzero(g[:-48]) and np.count_nonzero(g[-48:]) > 0
            vectors[repetition].append(g)
            receipts.append(receipt)
            paths += [path, receipt_path]
    assert len({r['parameters_sha256'] for r in receipts}) == 1
    classes = [dict(class_id=cls, metrics=metric(vectors[0][j], vectors[1][j])) for j, cls in enumerate((1, 2))]
    for actual, claimed in zip(classes, report['class_comparisons']):
        for k in ('bitwise_equal', 'changed_components', 'maximum_absolute', 'relative_L2'):
            assert np.isclose(actual['metrics'][k], claimed['metrics'][k], rtol=1e-14, atol=0)
    arms = []
    for arm in ('A', 'B'):
        first = direction(*vectors[0], arm)
        second = direction(*vectors[1], arm)
        arms.append(dict(arm=arm, first_M_weight=first[0], second_M_weight=second[0],
                         M_weight_difference=abs(first[0] - second[0]), metrics=metric(first[1], second[1]),
                         class_slopes=[first[2], second[2]],
                         descent_signs_identical=bool(np.array_equal(np.sign(first[2]), np.sign(second[2])))))
    output = dict(status='real_saved_repeated_gradient_differences_and_directions_independently_recomputed',
                  classes=classes, arms=arms, diagnostic_actual_counts=counts,
                  cumulative_failed_and_diagnostic_head_calls=124, cumulative_full_class_gradients=8,
                  official_classifier_calls_in_this_audit=0, official_feature_calls_in_this_audit=0,
                  official_gradients_in_this_audit=0, official_fits=0, official_updates=0,
                  source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in paths},
                  scope=['Original failed gradients were not saved and remain unknown.',
                         'Only this new role0 initial state was compared; later states and roles are not certified.',
                         'Worker forward/log-probability identities are asserted by sealed execution, not independently reconstructed from saved arrays.',
                         'Small coordinate errors do not justify an arbitrary tolerance or a classification success claim.'],
                  numerical_policy_changed=False, further_fit_permission=False, quality_acceptance=False)
    (OUT / 'audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in output.items() if k != 'source_sha256'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
