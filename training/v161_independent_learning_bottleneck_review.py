"""Saved inputs/vectors plus a synthetic curvature counterexample; no fit."""
import hashlib
import json
import math
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT/'artifacts/v161_fixed_error_endpoint_diagnostic_20261002'
TAIL = ROOT/'artifacts/v161_cached_direction_backtrack_tail_20261002/role2'
COHORT = ROOT/'artifacts/v161_independent_frozen_error_cohort_review_20261002'
OUT = ROOT/'artifacts/v161_independent_learning_bottleneck_review_20261002'
SEGMENTS = [('observation_weight', 0, 1060592), ('opinion_weight', 1060592, 1060768),
            ('bias', 1060768, 1060784), ('output_weight', 1060784, 1060832)]
EPS = np.finfo(np.float64).eps


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for part in iter(lambda: handle.read(1048576), b''):
            h.update(part)
    return h.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def synthetic_curvature_case():
    m0, curvature = 3.8535841184739184e-13, .27018

    def outputs(point, second_protected=False):
        x, y = point
        margins = [-1+1.4*x-.05*y, -2+2*x+.1*y, m0+y-curvature*x*x]
        truth = [1, 2, 2]
        logits = [[-50., margins[0], 0.], [-50., 0., margins[1]], [-50., 0., margins[2]]]
        if second_protected:
            margins.append(m0-y-curvature*x*x)
            truth.append(1)
            logits.append([-50., margins[-1], 0.])
        z = np.asarray(logits)
        shifted = z-z.max(1, keepdims=True)
        lp = shifted-np.log(np.exp(shifted).sum(1, keepdims=True))
        q = np.exp(lp)
        ce = -lp[np.arange(len(truth)), truth]
        return np.asarray(truth), q.argmax(1), ce, np.asarray(margins)

    point = np.zeros(2)
    truth, initial, before, _ = outputs(point)
    # Protected function normal is [0,1]; tangent step [1,0] passes linear protection.
    base_step = np.array([1., 0.])
    _, raw_pred, _, raw_margins = outputs(point+base_step)
    assert initial.tolist() == [2, 1, 2] and raw_pred[2] != truth[2]
    normal = np.array([0., 1.])
    assert math.fsum(normal*base_step) == 0.
    # Restore the measured nonlinear violation to the original margin.
    correction = (m0-raw_margins[2])/math.fsum(normal*normal)*normal
    displacement = base_step+correction
    _, corrected_pred, after, corrected_margins = outputs(point+displacement)
    assert np.array_equal(corrected_pred, truth)
    sigmoid = lambda v: 1./(1.+math.exp(-v))
    gm = np.array([-1.4, .05])*(1.-sigmoid(-1.))
    gs = np.array([-2., -.1])*(1.-sigmoid(-2.))
    slopes = np.array([math.fsum(g*displacement) for g in [gm, gs]])
    resolution = 16*EPS*np.maximum(1., np.maximum(np.abs(before[:2]), np.abs(after[:2])))
    assert np.all(slopes < 0) and np.all(before[:2]-after[:2] > resolution)
    assert np.all(before[:2]+1e-4*slopes-after[:2] > resolution)
    assert corrected_margins[2] > 0
    # A restoration of one boundary can break another protected classification.
    truth2, initial2, _, _ = outputs(point, True)
    _, corrected2, _, _ = outputs(point+displacement, True)
    assert initial2[3] == truth2[3] and corrected2[3] != truth2[3]
    return dict(linear_tangent_but_finite_protection_failed=True,
                correction=correction.tolist(), both_target_classifications_repaired_in_toy=True,
                protected_classification_preserved_in_first_toy=True,
                original_target_16eps_Armijo_passed=True,
                second_protected_constraint_counterexample_rejected=True,
                risk_before=before[:2].tolist(), risk_after=after[:2].tolist(),
                corrected_displacement_target_slopes=slopes.tolist(),
                actual_SOC_quality_or_model_gain_proven=False)


def main():
    assert not OUT.exists()
    OUT.mkdir()
    files = {Path(__file__).resolve(), ROOT/'training/v159_current_input_boundary_v3.py'}
    roles = []
    for role, j in [(0, 10), (1, 14), (2, 20)]:
        base = DIAG/f'role{role}/baseline_error_class1_repeat0'
        probe = DIAG/f'role{role}/round0/probe{j}' if role != 2 else TAIL/'probe20'
        target_path = COHORT/f'role{role}/fixed_pure_error_targets.parquet'
        bank_path = ROOT/f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/OOF_probabilities.npy'
        direction_path = DIAG/f'role{role}/round0/polished_direction.npy'
        files |= {base/'OOF_original_rows.parquet', probe/'OOF_original_rows.parquet',
                  probe/'probe.json', target_path, bank_path, direction_path}
        targets = pd.read_parquet(target_path)
        b, a = [pd.read_parquet(p/'OOF_original_rows.parquet') for p in [base, probe]]
        assert np.array_equal(a.row_position, b.row_position) and np.array_equal(a.truth, b.truth)
        mask = b.row_position.isin(targets.row_position).to_numpy()
        assert np.array_equal(b.loc[mask, 'row_position'], targets.row_position)
        truth, rival = b.truth.to_numpy(), b.pred.to_numpy()
        ix = np.arange(len(b))
        old, new = [f[['logp0', 'logp1', 'logp2']].to_numpy() for f in [b, a]]
        gap = (old[ix, truth]-old[ix, rival])[mask]
        delta = (new[ix, truth]-new[ix, rival])[mask]-gap
        assert np.all(gap < 0) and a.loc[mask, 'pred'].ne(a.loc[mask, 'truth']).all()
        teacher = np.load(bank_path, mmap_mode='r')[:, :16].mean(1)
        baseline_log = np.log(np.maximum(teacher, 1e-12))
        local, cls = targets.local.to_numpy(), targets.truth.to_numpy()
        teacher_p = teacher[local, cls]
        teacher_gap = baseline_log[local, cls]-baseline_log[local].max(1)
        needed = np.full(len(targets), np.inf)
        np.divide(-gap, delta, out=needed, where=delta > 0)
        ledger = targets[['row_position', 'local', 'root', 'truth']].copy()
        ledger['base_true_probability'] = teacher_p
        ledger['base_gap_vs_top_class'] = teacher_gap
        ledger['current_gap_vs_fixed_wrong_rival'] = gap
        ledger['safe_probe_actual_gap_delta'] = delta
        ledger['constant_local_progress_steps_to_old_rival'] = needed
        ledger.to_parquet(OUT/f'role{role}_original_target_progress_and_base_prior.parquet', index=False)
        classes = []
        for c in [1, 2]:
            m = cls == c
            classes.append(dict(class_id=c, original_target_rows=int(m.sum()),
                                base_true_probability_zero=int(((teacher_p == 0) & m).sum()),
                                base_true_probability_at_or_below_floor=int(((teacher_p <= 1e-12) & m).sum()),
                                median_base_logit_gap=float(np.median(teacher_gap[m])),
                                median_current_wrong_gap=float(np.median(gap[m])),
                                median_safe_step_delta=float(np.median(delta[m])),
                                median_constant_local_progress_steps=float(np.median(needed[m])),
                                within_100_constant_progress_steps=int((needed[m] <= 100).sum()),
                                actual_classification_repairs=0))
        direction = np.load(direction_path)
        step = json.loads((probe/'probe.json').read_text())['step']
        gradient_reports = []
        for c in [1, 2]:
            gp = DIAG/f'role{role}/baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy'
            files.add(gp)
            g = np.load(gp)
            gradient_reports.append(dict(class_id=c, segments=[dict(name=name,
                nonzero_gradient_coordinates=int(np.count_nonzero(g[start:end])),
                gradient_L2=float(np.linalg.norm(g[start:end])),
                nominal_safe_update_L2=float(np.linalg.norm(step*direction[start:end])))
                for name, start, end in SEGMENTS]))
        roles.append(dict(role=role, step=step, classes=classes, gradients=gradient_reports))
    r = DIAG/'role2'
    a = np.load(r/'round0/raw_margin_normals.npy')[0]
    d = np.load(r/'round0/polished_direction.npy')
    gs = [np.load(r/f'baseline_error_class{c}_repeat0/complete_fixed_error_target_gradient.npy') for c in [1, 2]]
    lp0 = np.load(r/'baseline_error_class1_repeat0/OOF_logq.npy')
    m0 = float(lp0[21050, 2]-lp0[21050, 1])
    files |= {r/'round0/raw_margin_normals.npy', r/'baseline_error_class1_repeat0/OOF_logq.npy'}
    witnesses = []
    for j in [4, 8, 12, 16, 19]:
        path = r/f'round0/probe{j}/OOF_logq.npy'
        files.add(path)
        lp = np.load(path)
        margin, alpha = float(lp[21050, 2]-lp[21050, 1]), 2.**(-j)
        correction = (m0-margin)/math.fsum(a*a)*a
        displacement = alpha*d+correction
        slopes = [math.fsum(g*displacement) for g in gs]
        assert all(v < 0 for v in slopes)
        witnesses.append(dict(original_probe=j, step=alpha, measured_failed_margin=margin,
                               least_norm_linear_restoration_target=m0-margin,
                               correction_L2=float(np.linalg.norm(correction)),
                               correction_vs_base_step_norm=float(np.linalg.norm(correction)/np.linalg.norm(alpha*d)),
                               corrected_displacement_target_slopes=slopes,
                               actual_corrected_classifier_not_executed=True,
                               finite_correctness_and_rest_of_population_safety_unproven=True))
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(files)}
    toy = synthetic_curvature_case()
    for path, digest in bindings.items():
        assert sha(ROOT/path) == digest
    result = dict(status='saved_prior_gap_gradient_segments_progress_and_curvature_candidate_review_passed',
                  roles=roles, role2_saved_linear_restoration_witnesses=witnesses,
                  synthetic_curvature_counterexample=toy, source_sha256=bindings,
                  official_calls=0, official_gradients=0, fits=0, permanent_updates=0, quality_acceptance=False,
                  limits=['Linear constant-progress counts are not future training predictions or impossibility proof.',
                          'Floor gaps explain initial required correction magnitude, not uniquely the root cause.',
                          'Zero hidden gradients at a zero output head are expected startup behavior, not permanent inability.',
                          'Toy and saved-vector corrections are not executed SOC classifier or generalization results.'])
    save(OUT/'review.json', result)
    print(json.dumps(dict(status=result['status'], official_calls=0, fits=0,
                          toy_curvature_and_collateral_counterexample_passed=True)))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if OUT.exists():
            save(OUT/'failure.json', dict(error_type=type(error).__name__, error=str(error),
                                         traceback=traceback.format_exc(), official_calls=0))
        raise
