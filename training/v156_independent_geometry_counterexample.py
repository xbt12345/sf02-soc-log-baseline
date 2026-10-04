"""Analytical counterexample: cosine order is not classifier evidence.

Three hand-defined nonnegative vectors, no official data/model/probe fitting.
An invertible coordinate scaling and inverse readout scaling keep logits and
predictions identical while the nearest class reverses.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v156_independent_geometry_counterexample_20261001'


def main():
    assert not OUT.exists(), 'Preserve analytical receipts'
    h = np.array([[2., 1.], [3., .5], [.5, 3.]])
    weight = np.array([[0., -1., 1.], [0., 1., -1.]])
    bias = np.array([-10., 0., 0.])
    # A fixed direct fact contribution is retained in both reparameterizations.
    bypass = np.array([[0., .25, -.25]] * 3)
    logits = h @ weight + bias + bypass
    truth = logits.argmax(1)
    assert truth.tolist() == [2, 2, 1]
    reports = []
    for k in [.125, 8.]:
        transform = np.diag([1., k])
        inverse = np.diag([1., 1/k])
        changed = h @ transform
        adapted = inverse @ weight
        actual = changed @ adapted + bias + bypass
        assert np.array_equal(actual, logits)
        assert np.array_equal(actual.argmax(1), truth)
        normed = changed / np.linalg.norm(changed, axis=1)[:, None]
        similarity = normed[1:] @ normed[0]
        reports.append(dict(second_coordinate_scale=k,
            same_class_cosine=float(similarity[0]), other_class_cosine=float(similarity[1]),
            nearest_class=int(truth[1 + similarity.argmax()]),
            max_logit_change=float(np.abs(actual-logits).max()),
            predictions=actual.argmax(1).tolist(), hidden_values_nonnegative=bool((changed >= 0).all())))
    assert reports[0]['nearest_class'] == 2 and reports[1]['nearest_class'] == 1
    result = dict(status='cosine_neighbor_order_reverses_under_exact_classifier_equivalence',
        cases=reports, classifier_logits=logits.tolist(), complete_predictions_identical=True,
        direct_fact_bypass_retained=True, matrix_algebra_cases=2, official_data_used=False,
        official_model_forwards=0, gradients=0, classifier_fits=0, probe_fits=0, parameter_updates=0,
        implication='Cosine relation can describe a fixed representation, but cannot alone prove information loss, task use, separability or transfer.',
        not_proved=['Actual SOC geometry is arbitrary', 'Any specific representation is useless',
                    'A contrastive objective cannot help', 'Classification quality or any goal is solved'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    OUT.mkdir()
    (OUT/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
