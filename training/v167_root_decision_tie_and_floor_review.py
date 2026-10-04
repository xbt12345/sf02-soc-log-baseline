"""Saved actual tie/curvature and decision-aware floor previews; zero model calls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v160_independent_fixed_diagnostic_review import read, sha
from v160_independent_saved_direction_certificate import dot
from v166_coverage_joint_restoration import _joint_propose

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v167_trial_point_restoration_diagnostic_20261002/role1'
REVIEW = ROOT / 'artifacts/v167_independent_actual_trial_point_review_v2_20261002'
OUT = ROOT / 'artifacts/v167_root_decision_tie_and_floor_review_20261002'
EPS = float(np.finfo(np.float64).eps)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert not OUT.exists()
    previous = read(REVIEW / 'review.json')
    assert previous['official_calls_by_this_review'] == 0 and not previous['supports_new_short_training_registration']
    old_bindings = read(REVIEW / 'pre_review_bindings.json')['source_sha256']
    assert all(sha(ROOT / p) == v for p, v in old_bindings.items())
    plan = read(ROOT / 'training/review_policy/v167_trial_point_restoration_contract.json')
    point = ROOT / 'artifacts/v164_short_supervised_trajectory_20261002/role1' / ('parameter_point'+str(plan['roles'][1]['parameter_point']))
    gs = [np.load(point / f'class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1, 2]]
    paths = {Path(__file__).resolve(), REVIEW / 'review.json', REVIEW / 'pre_review_bindings.json',
             ROOT / 'training/v166_coverage_joint_restoration.py', ROOT / 'training/v160_independent_saved_direction_certificate.py'}
    paths |= {ROOT / p for p in old_bindings}
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(paths)}
    OUT.mkdir()
    save(OUT / 'pre_bindings.json', dict(source_sha256=bindings, official_calls=0))
    stages = []
    for stage in [0, 1]:
        folder = TRIAL / f'correction{stage}'
        probe = TRIAL / f'probe{stage}'
        refs = read(folder / 'active_normal_references.json')
        c = np.load(folder / 'actual_trial_margins.npy')
        e = np.load(folder / 'correction.npy')
        u = np.load(folder / 'current_displacement.npy')
        normals = np.stack([np.load(ROOT / ref['gradient']) for ref in refs.values()])
        actual_logs = {s: np.load(probe / f'{s}_logq.npy') for s in ['OOF', 'deployment']}
        q = {s: np.load(probe / f'{s}_q.npy') for s in actual_logs}
        geometry, floors = [], []
        start_path = ROOT / 'artifacts/v166_coverage_first_diagnostic_20261002/role1/treatment' if stage == 0 else TRIAL / 'probe0'
        start_logs = {s: np.load(start_path / f'{s}_logq.npy') for s in actual_logs}
        for index, (identity, ref) in enumerate(refs.items()):
            meta = ref['metadata']
            scope, local, truth, rival = [meta[k] for k in ['scope', 'local', 'truth', 'rival']]
            linear, arithmetic_error = dot(normals[index], e)
            predicted = float(c[index])+linear
            actual = float(actual_logs[scope][local, truth]-actual_logs[scope][local, rival])
            values = q[scope][local]
            # This is a local constraint target, not a relaxed acceptance
            # tolerance. Smaller class indices win exact argmax ties.
            scale = max(1., abs(float(start_logs[scope][local, truth])), abs(float(start_logs[scope][local, rival])))
            floor = 16*EPS*scale if truth > rival else 0.
            floors.append(floor)
            geometry.append(dict(identity=identity, scope=scope, local=local, truth=truth, rival=rival,
                predicted_margin=predicted, actual_margin=actual, actual_minus_predicted=actual-predicted,
                directional_dot_arithmetic_error=arithmetic_error,
                actual_probability_gap=float(values[truth]-values[rival]),
                actual_argmax=int(values.argmax()), exact_rival_probability_tie=bool(values[truth] == values[rival]),
                correct_class_loses_exact_tie=bool(truth > rival), proposed_local_decision_floor=floor))
        # Both previews are at already observed V167 points. They are NOT a
        # new sequential trajectory: a changed first correction needs fresh
        # official Jacobians at its changed actual parameter point.
        floor_array = np.asarray(floors, np.float64)
        preview = _joint_propose(u, normals, np.zeros_like(c), c-floor_array, *gs)
        for key in ['displacement', 'correction']:
            if key in preview:
                np.save(OUT / f'stage{stage}_local_preview_{key}.npy', preview[key])
        math = {k: v for k, v in preview.items() if k not in ['displacement', 'correction']}
        save(OUT / f'stage{stage}_local_preview_math.json', math)
        np.save(OUT / f'stage{stage}_proposed_floors.npy', floor_array)
        blocked = pd.read_parquet(probe / 'actual_blocking_original_rows.parquet')
        blocking_geometry = [v for v in geometry if v['actual_argmax'] != v['truth']]
        assert len(blocked) == 2 and len(blocking_geometry) == 1 and blocking_geometry[0]['local'] == 21985
        if stage == 1:
            case = blocking_geometry[0]
            assert case['actual_margin'] == case['actual_probability_gap'] == 0.
            assert case['exact_rival_probability_tie'] and case['correct_class_loses_exact_tie']
            assert case['truth'] == 2 and case['actual_argmax'] == case['rival'] == 1
        stages.append(dict(stage=stage, complete_functions=len(refs), actual_protected_blocking_rows=len(blocked),
            blocking_geometry=blocking_geometry, full_geometry=geometry,
            local_decision_floor_preview_status=preview['status'],
            local_preview_CPU_QP_solves=int('optimizer_iterations' in preview),
            actual_finite_model_evaluation_performed=False))
    assert all(sha(ROOT / p) == v for p, v in bindings.items())
    report = dict(status='V167_exact_actual_argmax_tie_and_saved_trial_geometry_independently_confirmed',
        stages=stages, actual_CPU_saved_vector_preview_QPs=sum(s['local_preview_CPU_QP_solves'] for s in stages),
        original_actual_V167_heads=296, original_actual_V167_margin_derivatives=100,
        official_heads=0, official_features=0, official_derivatives=0, fits=0, permanent_updates=0,
        local_floor_rule='16eps times actual trial log probability scale only when truth index loses tie to rival',
        no_new_actual_candidate_or_classification_gain_claim=True,
        supports_registered_finite_diagnostic_design_only=True, training_permission=False, quality_acceptance=False,
        required_action='Do not equate nonnegative log margin with actual correct argmax. Require a representable positive local floor for losing-tie classes and retain all actual original-row guards.',
        scope='Saved supervised development observations and two independent fixed-point CPU previews; no new sequential trajectory or transfer proof.')
    save(OUT / 'review.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'stages'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
