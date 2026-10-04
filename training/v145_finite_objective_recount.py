"""Reconstruct original-frequency finite CE from existing class-risk receipts."""
import json
import numpy as np
from v144_independent_decision_review import ROOT, read, sha

RUN = ROOT/'artifacts/v145_class_direction_qualification_20261001'
OUT = ROOT/'artifacts/v145_finite_objective_recount_20261001'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed recount')
    q = read(RUN/'qualification.json')
    registration = read(RUN/'pre_registered_probe.json')
    step = registration['fixed_L2_step']
    assert step == 0.01 and q['new_optimizer_steps'] == 0
    rows = []
    for role in q['folds']:
        mass = np.asarray(role['original_class_mass_seen'], dtype=np.float64)
        assert mass[0] == 0
        weights = mass[1:] / mass.sum()
        for d in role['directions']:
            before = float(weights @ np.asarray(d['initial_M_S_member_CE']))
            after = float(weights @ np.asarray(d['functional_trial_M_S_member_CE']))
            assert np.isclose(before, role['original_full_member_CE'], rtol=1e-12, atol=1e-15)
            entry = dict(role=role['fold'], arm=d['arm'], class_mass=mass.tolist(),
                         original_frequency_member_CE_before=before,
                         original_frequency_member_CE_after=after,
                         original_frequency_member_CE_change=after-before,
                         original_frequency_member_CE_decreased=after < before,
                         objective_finite_change_known=d['arm'] == 'A',
                         actual_new_TRAIN_classification_errors=0)
            if d['arm'] == 'A':
                derivative = d['objective_descent_derivative']
                assert derivative < 0
                entry.update(objective_initial_directional_derivative=derivative,
                             first_order_predicted_change=step*derivative,
                             observed_change_minus_first_order_prediction=(after-before-step*derivative))
            else:
                entry['objective_limitation'] = 'B includes lambda*auxiliary; finite auxiliary values were not recorded, so this recount cannot assert B total-objective descent/failure.'
            rows.append(entry)
    assert [r['original_frequency_member_CE_decreased'] for r in rows if r['arm'] == 'A'] == [True, False, False]
    report = dict(status='finite_original_frequency_CE_reconstructed_without_model_execution',
                  latest_actual_training='V142', new_fits=0, new_model_forwards=0,
                  new_gradients=0, new_parameter_trials=0, fixed_registered_L2_step=step,
                  rows=rows, conclusion='At fixed 0.01, CE-only control fails to decrease its actual overall objective in roles 1 and 2 despite no classification changes. Keep finite objective decrease and real guards, not merely surrogate class-monotonic gates.',
                  limitations=['Existing risk receipts were algebraically recomposed; no independent logit/function replay.',
                               'Nonlinear residual is observed; one scale does not identify curvature vs activation boundaries.',
                               'No smaller step was selected or evaluated, and no trial was retroactively accepted.',
                               'B finite total objective remains unknown without actual finite auxiliary evaluation.',
                               'No source-held quality or fine support improvement is proved.'],
                  source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in
                                 [ROOT/'training/v145_finite_objective_recount.py', RUN/'qualification.json',
                                  RUN/'pre_registered_probe.json', ROOT/'training/v144_independent_decision_review.py']})
    OUT.mkdir()
    (OUT/'recount.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
