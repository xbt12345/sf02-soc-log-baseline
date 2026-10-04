"""Saved-vector decision-floor counterfactual; no official forward or gradient."""
import json
import traceback
from pathlib import Path

import numpy as np

from v160_independent_fixed_diagnostic_review import read, sha
from v163_one_sided_joint_restoration import propose

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v164_short_supervised_trajectory_20261002'
OUT = ROOT / 'artifacts/v164_independent_decision_floor_counterfactual_20261002'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    assert not OUT.exists()
    locations = sorted(TRIAL.glob('role*/parameter_point*/restoration*/original_unit_restoration_review.json'))
    assert len(locations) == 6 and all((TRIAL / f'role{r}/fit.json').exists() for r in range(3))
    OUT.mkdir()
    sources = {Path(__file__).resolve(), ROOT/'training/v163_one_sided_joint_restoration.py',
               ROOT/'training/v160_independent_saved_direction_certificate.py',
               ROOT/'training/v159_float64_repeat_policy_v2.py'}
    inputs = []
    for location in locations:
        folder = location.parent
        point = folder.parent
        refs = read(folder/'active_normal_references.json')
        paths = [folder/name for name in ['active_normal_references.json', 'current_displacement.npy',
                 'base_margins.npy', 'actual_current_margins.npy', 'displacement.npy', 'correction.npy',
                 'original_unit_restoration_review.json', 'finite_probe/probe.json']]
        normal_paths = [ROOT/ref['gradient'] for ref in refs.values()]
        gradient_paths = [point/f'class{c}_repeat0/complete_fixed_error_target_gradient.npy' for c in [1, 2]]
        sources.update(paths + normal_paths + gradient_paths)
        inputs.append((folder, normal_paths, gradient_paths))
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)}
    save(OUT/'source_bindings.json', dict(source_sha256=bindings, official_calls=0))
    results = []
    qp_count = 0
    for folder, normal_paths, gradient_paths in inputs:
        u = np.load(folder/'current_displacement.npy')
        a = np.stack([np.load(p) for p in normal_paths])
        b = np.load(folder/'base_margins.npy')
        c = np.load(folder/'actual_current_margins.npy')
        gm, gs = [np.load(p) for p in gradient_paths]
        observed = propose(u, a, b, c, gm, gs)
        original = read(folder/'original_unit_restoration_review.json')
        assert {k:v for k,v in observed.items() if k not in ['displacement','correction']} == original
        for key in ['displacement', 'correction']:
            assert np.array_equal(observed[key], np.load(folder/f'{key}.npy'))
        counterfactual = propose(u, a, np.zeros_like(b), c, gm, gs)
        qp_count += int('optimizer_iterations' in observed) + int('optimizer_iterations' in counterfactual)
        target = OUT/f'{folder.parent.parent.name}_{folder.parent.name}_{folder.name}'
        target.mkdir()
        for key in ['displacement', 'correction']:
            if key in counterfactual:
                np.save(target/f'{key}.npy', counterfactual[key])
        cf = {k:v for k,v in counterfactual.items() if k not in ['displacement', 'correction']}
        save(target/'counterfactual_original_unit_review.json', cf)
        actual = read(folder/'finite_probe/probe.json')
        results.append(dict(
            actual_source=folder.relative_to(ROOT).as_posix(), normal_count=len(a),
            original_margin_range=[float(b.min()), float(b.max())],
            original_positive_goals=int(np.count_nonzero(b>0)),
            original_status=observed['status'], counterfactual_status=counterfactual['status'],
            original_correction_norm=observed['correction_norm'],
            counterfactual_correction_norm=counterfactual.get('correction_norm'),
            correction_norm_ratio=counterfactual['correction_norm']/observed['correction_norm']
                if observed['correction_norm'] and 'correction_norm' in counterfactual else None,
            original_class_reviews=observed['class_reviews'], counterfactual_class_reviews=counterfactual.get('class_reviews'),
            actual_original_probe_accepted=actual['accepted'], actual_original_probe_OOF=actual['OOF_stats'],
            counterfactual_forward_executed=False, counterfactual_classification_gain_unknown=True,
            only_local_linear_decision_floor_not_nonlinear_classification_or_optimality_proof=True))
    assert all(sha(ROOT/p)==value for p,value in bindings.items())
    report=dict(status='all_six_saved_actual_restorations_reproduced_and_decision_floor_counterfactuals_recorded',
                results=results, CPU_QP_solves=qp_count, official_calls=0, parameter_updates=0,
                scope='Linear saved-vector comparison only. Zero margin is a mathematical decision floor; '
                      'future actual acceptance still requires original full-population decision, class-count, '
                      'retention and numerical guards. No counterfactual quality claim.')
    save(OUT/'review.json', report)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                   traceback=traceback.format_exc(), official_calls=0))
        raise
