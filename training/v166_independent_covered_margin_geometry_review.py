"""Saved-vector linear/actual margin audit; no new model or derivative call."""
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz

from v160_independent_fixed_diagnostic_review import read, sha
from v160_independent_saved_direction_certificate import dot
from v160_margin_normal import input_identity
from v161_independent_all_finite_results_review import close

ROOT = Path(__file__).resolve().parents[1]
TRIAL = ROOT / 'artifacts/v166_coverage_first_diagnostic_20261002'
AUDIT = ROOT / 'artifacts/v166_independent_actual_coverage_review_20261002'
OUT = ROOT / 'artifacts/v166_independent_covered_margin_geometry_review_20261002'


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    assert not OUT.exists()
    audited = read(AUDIT / 'review.json')
    assert audited['status'] == 'all_three_actual_coverage_math_fresh_gradients_original_gold_costs_and_restoration_verified'
    prior_bindings = read(AUDIT / 'pre_review_bindings.json')['source_sha256']
    assert all(sha(ROOT / p) == value for p, value in prior_bindings.items())
    xfile = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'
    x = load_npz(xfile).tocsr()
    x.sort_indices()
    sources = {Path(__file__).resolve(), AUDIT / 'review.json', AUDIT / 'pre_review_bindings.json', xfile}
    for role in range(3):
        sources.add(AUDIT / f'role{role}_normal_identities.json')
        sources |= {p for p in (TRIAL / f'role{role}').rglob('*') if p.is_file()}
        for scope in ['OOF', 'deployment']:
            sources.add(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{scope}_probabilities.npy')
    bindings = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources)}
    OUT.mkdir()
    save(OUT / 'pre_review_bindings.json', dict(source_sha256=bindings, official_calls=0))
    roles = []
    for role in range(3):
        folder = TRIAL / f'role{role}'
        joint = folder / 'joint_restoration'
        refs = read(joint / 'active_normal_references.json')
        math_review = read(joint / 'original_unit_restoration_review.json')
        assert all(r['passed'] for r in math_review['inequality_reviews'])
        assert all(r['resolved_negative'] for r in math_review['class_reviews'])
        origin_margins = np.load(joint / 'actual_origin_margins.npy')
        control_margins = np.load(joint / 'actual_current_margins.npy')
        correction = np.load(joint / 'correction.npy')
        displacement = np.load(joint / 'displacement.npy')
        logs = {s: np.load(folder / f'treatment/{s}_logq.npy') for s in ['OOF', 'deployment']}
        probabilities = {s: np.load(folder / f'treatment/{s}_q.npy') for s in logs}
        reviews = []
        for index, (identity, ref) in enumerate(refs.items()):
            meta = ref['metadata']
            a = np.load(ROOT / ref['gradient'])
            correction_slope, correction_error = dot(a, correction)
            total_slope, total_error = dot(a, displacement)
            inequality = math_review['inequality_reviews'][index]
            close(correction_slope, inequality['linear_recovery'])
            scope, local, truth, rival = meta['scope'], meta['local'], meta['truth'], meta['rival']
            actual_margin = float(logs[scope][local, truth] - logs[scope][local, rival])
            predicted = float(control_margins[index] + correction_slope)
            origin_linear = float(origin_margins[index] + total_slope)
            limit = float(inequality['residual_limit'])
            reviews.append(dict(identity=identity, scope=scope, local=local, truth=truth, rival=rival,
                origin_margin=float(origin_margins[index]), V165_control_margin=float(control_margins[index]),
                predicted_restoration_margin=predicted, actual_V166_margin=actual_margin,
                actual_minus_restoration_model=actual_margin-predicted,
                origin_linear_margin=origin_linear, actual_minus_origin_linear=actual_margin-origin_linear,
                correction_dot_arithmetic_bound=correction_error, total_dot_arithmetic_bound=total_error,
                registered_linear_residual_limit=limit,
                linear_inequality_passed=inequality['passed'],
                actual_argmax_correct=int(probabilities[scope][local].argmax()) == truth,
                actual_negative_beyond_linear_residual_limit=actual_margin < -limit,
                defect_to_linear_residual_limit=abs(actual_margin-predicted)/limit if limit else None))
        by_id = {r['identity']: r for r in reviews}
        frame = pd.read_parquet(folder / 'baseline/OOF_original_rows.parquet')
        ids_oof = np.sort(frame.local.unique())
        opinions = {s: np.asarray(np.load(ROOT / f'artifacts/v158_legal_fusion_bank_v2_20261001/fold{role}/{s}_probabilities.npy', mmap_mode='r')[:, :16], np.float64) for s in logs}
        blockers = pd.read_parquet(folder / 'treatment/actual_blocking_original_rows.parquet')
        functions = []
        for (scope, local, truth, rival), group in blockers.groupby(['scope', 'local', 'truth', 'rival'], sort=True):
            ids = ids_oof if scope == 'OOF' else np.arange(22546)
            pos = int(np.searchsorted(ids, local))
            assert ids[pos] == local
            chunk = ids[pos//2048*2048:pos//2048*2048+2048]
            identity = input_identity(role, scope, chunk, x[chunk], opinions[scope][chunk], pos % 2048, int(truth), int(rival))
            item = dict(identity=identity, scope=scope, local=int(local), truth=int(truth), rival=int(rival),
                original_rows=len(group), covered=identity in refs)
            if identity in refs:
                item['geometry'] = by_id[identity]
            functions.append(item)
        result = dict(role=role, all_measured_functions=len(refs), actual_blocking_original_rows=len(blockers),
            blocked_functions=len(functions), covered_blocked_functions=sum(f['covered'] for f in functions),
            unmeasured_blocked_functions=sum(not f['covered'] for f in functions),
            all_class_direction_checks_passed=True, all_linear_restoration_inequalities_passed=True,
            measured_argmax_failures=sum(not r['actual_argmax_correct'] for r in reviews),
            functions=functions)
        save(OUT / f'role{role}_all_margin_geometry.json', dict(review=result, measured_functions=reviews))
        roles.append(result)
    assert roles[0]['actual_blocking_original_rows'] == roles[2]['actual_blocking_original_rows'] == 0
    middle = roles[1]
    assert middle['actual_blocking_original_rows'] == 10
    assert middle['covered_blocked_functions'] == 5 and middle['unmeasured_blocked_functions'] == 0
    assert all(f['geometry']['actual_negative_beyond_linear_residual_limit'] for f in middle['functions'])
    assert all(sha(ROOT / p) == value for p, value in bindings.items())
    report = dict(status='V166_covered_S_blockers_have_actual_negative_margins_beyond_original_linear_numeric_limits',
        roles=roles, source_audit_sha256=sha(AUDIT / 'review.json'), official_calls=0, new_derivatives=0,
        new_fits=0, permanent_updates=0,
        next_mechanism='Trial-point Jacobian nonlinear restoration diagnostic, no capacity increase',
        evidence_scope='Observed same-input finite-step linear-versus-actual mismatch; not global model impossibility or certified Hessian/Maratos diagnosis',
        classification_gain_or_training_permission_not_created_by_this_review=True)
    save(OUT / 'review.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'roles'}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        OUT.mkdir(exist_ok=True)
        save(OUT / 'failure.json', dict(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(), official_calls=0))
        raise
