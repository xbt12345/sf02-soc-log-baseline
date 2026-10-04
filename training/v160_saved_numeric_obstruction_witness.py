"""Independent saved-array witnesses, not a generic or official-update solver."""
from decimal import Decimal, localcontext
from pathlib import Path
import hashlib
import json
import time

import numpy as np

from v160_independent_saved_direction_certificate import certificate, dot

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'artifacts/v160_fixed_endpoint_diagnostic_20261002'
OUT = ROOT / 'artifacts/v160_saved_numeric_obstruction_witness_v2_20261002'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load(role, round_number):
    f = IN / f'role{role}'
    paths = [f / f'baseline_class{c}/gradient.npy' for c in [1, 2]]
    paths += [f / f'round{round_number}/raw_margin_normals.npy',
              f / f'round{round_number}/direction.npy']
    arrays = [np.load(p) for p in paths]
    return arrays, {p.relative_to(ROOT).as_posix(): sha(p) for p in paths}


def main():
    assert not OUT.exists()
    OUT.mkdir()
    (gm, gs, a, old), bindings = load(1, 1)
    u, s, vt = np.linalg.svd(a.T, full_matrices=False)
    initial = certificate(gm, gs, a, old)
    target = np.array([16 * p['arithmetic_error'] for p in initial['protection']])
    d = old.copy()
    trace = []
    for j in range(3):
        current = np.array([p['margin_slope'] for p in certificate(gm, gs, a, d)['protection']])
        correction = u @ ((vt @ (target - current)) / s)
        d = d + correction
        trace.append(dict(iteration=j, maximum_direction_change=float(np.abs(d - old).max()),
                          certificate=certificate(gm, gs, a, d)))
    np.save(OUT / 'role1_inward_direction.npy', d)
    role1 = dict(singular_values=s.tolist(), target=target.tolist(), trace=trace,
                 initial_certificate=initial)
    assert not initial['eligible_for_finite_trial_only'] and trace[-1]['certificate']['eligible_for_finite_trial_only']

    (gm, gs, a, old), more = load(2, 0)
    bindings.update(more)
    started = time.perf_counter()
    indices = np.flatnonzero((gm != 0) | (gs != 0) | (a[0] != 0))
    # Removing exact all-zero arithmetic coordinates is lossless. It does not
    # delete, merge, select or reweight any original data record.
    with localcontext() as ctx:
        ctx.prec = 80
        m, s, n = [[Decimal.from_float(float(x[i])) for i in indices] for x in [gm, gs, a[0]]]
        basis = [[x - y for x, y in zip(m, s)], [-x for x in n]]

        def inner(x, y):
            return sum((p * q for p, q in zip(x, y)), Decimal(0))

        h = [[inner(x, y) for y in basis] for x in basis]
        k = [-inner(x, s) for x in basis]
        det = h[0][0] * h[1][1] - h[0][1] * h[1][0]
        alpha = (k[0] * h[1][1] - h[0][1] * k[1]) / det
        mu = (h[0][0] * k[1] - k[0] * h[1][0]) / det
        values = [-(y + b0 * alpha + b1 * mu) for y, b0, b1 in zip(s, *basis)]
        raw = np.zeros_like(gm)
        raw[indices] = [float(x) for x in values]
        d = raw / np.abs(raw).max()
        original_norm, _ = dot(raw, raw)
        slopes = [dot(x, raw)[0] for x in [gm, gs]]
        gap = original_norm + max(slopes)
        gap_limit = 64 * float(np.finfo(float).eps) * 2 * max(original_norm, max(abs(x) for x in slopes))
        original_signs = certificate(gm, gs, a, d)
        role2 = dict(decimal_precision=80, nonzero_union_coordinates=len(indices),
                     elapsed_seconds=time.perf_counter() - started, alpha=str(alpha), mu=str(mu),
                     exact_margin=str(inner(n, values)),
                     exact_gap=str(inner(values, values) + max(inner(m, values), inner(s, values))),
                     float64_fsum_gap=gap, unchanged_gap_limit=gap_limit,
                     gap_pass=bool(abs(gap) <= gap_limit), sign_certificate=original_signs,
                     maximum_direction_change=float(np.abs(d - old).max()))
    assert 0 < alpha < 1 and mu > 0
    assert role2['gap_pass'] and original_signs['eligible_for_finite_trial_only']
    np.save(OUT / 'role2_raw.npy', raw)
    np.save(OUT / 'role2_direction.npy', d)
    bindings[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    bindings['training/v160_independent_saved_direction_certificate.py'] = sha(ROOT / 'training/v160_independent_saved_direction_certificate.py')
    report = dict(status='two_saved_numeric_obstruction_witnesses_reproduced_without_model_calls',
                  role1=role1, role2=role2, source_and_input_sha256=bindings,
                  official_heads=0, official_features=0, official_gradients=0, official_fits=0,
                  permanent_updates=0, finite_step_safety_proven=False, quality_acceptance=False,
                  limitations='Fixed saved active sets only. No generic bound solver qualification, new model call, actual finite SOC protection, classification repair, or permission for a fit.')
    (OUT / 'witness.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=report['status'], official_calls=0, finite_step_safety_proven=False)))


if __name__ == '__main__':
    main()
