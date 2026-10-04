"""Read-only numerical review; no official head, feature, gradient or fit calls."""
from pathlib import Path
import hashlib
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/v159_independent_numeric_policy_saved_state_audit_20261002'
SYN = ROOT / 'artifacts/v159_nonzero_real_dimension_numeric_qualification_v2_20261002'
SEG = [('observation', 0, 1060592), ('opinion', 1060592, 1060768),
       ('bias', 1060768, 1060784), ('output', 1060784, 1060832)]
EPS = np.finfo(np.float64).eps


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dot(a, b):
    # Independent compensated summation of rounded products, not a new gradient.
    mask = (a != 0) & (b != 0)
    return math.fsum((a[mask] * b[mask]).tolist())


def direction(gm, gs, mass, arm):
    if arm == 'A':
        alpha = float(mass[1] / (mass[1] + mass[2]))
    else:
        scale = max(np.abs(gm).max(), np.abs(gs).max())
        m, s = gm / scale, gs / scale
        mm, ms, ss = dot(m, m), dot(m, s), dot(s, s)
        alpha = 1. if ms >= mm else (0. if ms >= ss else (ss-ms)/((mm-ms)+(ss-ms)))
    raw = -(alpha*gm + (1.-alpha)*gs)
    normalizer = np.abs(raw).max()
    d = raw/normalizer if normalizer else raw
    slopes = [dot(gm, d), dot(gs, d)]
    resolution = [16*EPS*math.sqrt(dot(g, g))*math.sqrt(dot(d, d)) for g in [gm, gs]]
    qualified = all(s < -e for s, e in zip(slopes, resolution)) if arm == 'B' else (
        alpha*slopes[0]+(1-alpha)*slopes[1] <
        -16*EPS*math.sqrt(dot(alpha*gm+(1-alpha)*gs, alpha*gm+(1-alpha)*gs))*math.sqrt(dot(d,d)))
    return dict(alpha=alpha, slopes_compensated=slopes, operational_resolution=resolution,
                descent_resolved=bool(qualified), theoretical_roundoff_bound=False)


def main():
    assert not OUT.exists(), 'Do not overwrite an independent review'
    q = json.loads((SYN/'qualification.json').read_text(encoding='utf-8'))
    spec = q['specification']
    assert spec['shape'] == [18543, 66287] and spec['nnz_per_row'] == 177
    assert spec['repeat_eps'] == 8 and spec['finite_step_eps'] == 16
    assert q['counts']['official_fits'] == q['counts']['official_gradients'] == 0
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}
    for rel, expected in q['source_sha256'].items():
        assert sha(ROOT/rel) == expected, rel
        sources[rel] = expected
    for f in SYN.glob('*.npy'):
        sources[f.relative_to(ROOT).as_posix()] = sha(f)
    sources[(SYN/'qualification.json').relative_to(ROOT).as_posix()] = sha(SYN/'qualification.json')
    # Synthetic labels/frequencies are reconstructible without any gold file.
    rng = np.random.default_rng(15902)
    n, width, k = 18543, 66287, 177
    rng.integers(0, width-k, n)
    rng.uniform(-2., 2., n*k)
    rng.uniform(.03, 1., (n,16,3))
    mass = np.array([0, 0, 0], dtype=np.int64)
    c = rng.integers(1,8,n)
    mass[1], mass[2] = c[np.arange(n)%3 != 0].sum(), c[np.arange(n)%3 == 0].sum()
    states = []
    for scale in [0., .125, 8.]:
        vals = [[{key: np.load(SYN/f'scale{scale}_rep{rep}_class{cls}_{key}.npy')
                  for key in ['q','lp','risk','gradient']} for cls in [1,2]] for rep in [0,1]]
        point = vals[0][0]
        value_checks = []
        for rep in vals:
            for v in rep:
                checks = {}
                for key in ['q','lp','risk']:
                    a,b = point[key],v[key]
                    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
                    scaled = np.abs(a-b)/(EPS*np.maximum(1.,np.maximum(np.abs(a),np.abs(b))))
                    assert scaled.max() <= 8
                    checks[key] = dict(max_absolute=float(np.abs(a-b).max()), max_scaled_eps=float(scaled.max()))
                assert np.array_equal(point['q'].argmax(1),v['q'].argmax(1))
                value_checks.append(checks)
        gradients = []
        for cls in range(2):
            a,b = vals[0][cls]['gradient'], vals[1][cls]['gradient']
            assert a.shape == b.shape == (1060832,) and np.isfinite(a).all() and np.isfinite(b).all()
            parts = []
            for name,start,end in SEG:
                x,y = a[start:end], b[start:end]
                own_scale = max(np.abs(x).max(),np.abs(y).max())
                bound = 8*EPS*own_scale
                norm = max(np.linalg.norm(x),np.linalg.norm(y))
                gap = np.abs(x-y).max(); diff_norm = np.linalg.norm(x-y)
                signs = (np.maximum(np.abs(x),np.abs(y)) > bound) & (np.sign(x)!=np.sign(y))
                assert gap <= bound and diff_norm <= 8*EPS*norm and not signs.any()
                parts.append(dict(segment=name,max_absolute=float(gap),own_scale=float(own_scale),
                                  relative_L2=float(diff_norm/norm) if norm else 0., resolved_sign_changes=int(signs.sum())))
            assert np.linalg.norm(a) > 0
            if scale: assert np.linalg.norm(a[:-48]) > 0
            gradients.append(parts)
        dirs = {arm: [direction(*[v['gradient'] for v in rep],mass,arm) for rep in vals] for arm in ['A','B']}
        assert all(d['descent_resolved'] for d in dirs['A'])
        assert all(d['descent_resolved'] == (scale != 8.) for d in dirs['B'])
        states.append(dict(output_scale=scale,value_checks=value_checks,gradient_checks=gradients,
                           directions_using_compensated_sum=dirs,classification_repeats_exact=True))
    # Invoke only pure-array numeric policy, never its official head or runtime.
    sys.path.insert(0,str(ROOT/'training'))
    from v159_float64_repeat_policy_v2 import repeat_gradient, repeat_values, finite_step_review
    fixtures = {}
    g=np.zeros(1060832); g[0]=1e-30; b=g.copy(); b[0]+=1e-20
    fixtures['tiny_gradient_no_absolute_floor'] = not repeat_gradient(g,b)['passed']
    b=g.copy();b[0]=-g[0]
    fixtures['resolved_sign_flip_rejected'] = not repeat_gradient(g,b)['passed']
    fixtures['within_envelope_class_flip_rejected'] = not repeat_values([[.5,.5-EPS,0]],[[.5-EPS,.5,0]],'probability')['passed']
    for arm in ['A','B']:
        fixtures[f'{arm}_zero_progress_rejected'] = not finite_step_review([1,2],[1,2],[-1,-1],2,1,arm,2**-40,True)['accepted']
        fixtures[f'{arm}_unresolved_progress_rejected'] = not finite_step_review([1,2],[1-EPS,2-EPS],[-1,-1],2,1,arm,2**-40,True)['accepted']
        fixtures[f'{arm}_finite_progress_accepted'] = finite_step_review([1,2],[.9,1.9],[-1,-1],2,1,arm,1,True)['accepted']
        fixtures[f'{arm}_classification_regression_rejected'] = not finite_step_review([1,2],[.9,1.9],[-1,-1],2,1,arm,1,False)['accepted']
    fixtures['B_class_sacrifice_rejected'] = not finite_step_review([1,2],[.1,2.001],[-1,-1],99,1,'B',1,True)['accepted']
    assert all(fixtures.values())
    report = dict(status='saved_nonzero_synthetic_states_and_numeric_rejections_independently_recomputed',
                  mass=mass.tolist(),states=states,fixtures=fixtures,source_sha256=sources,
                  official_heads=0,official_features=0,official_gradients=0,official_fits=0,official_updates=0,
                  operational_envelope_not_global_error_theorem=True,official_preflight_still_required=True,
                  full_training_permission=False,first_training_issue_passed=False,quality_acceptance=False)
    OUT.mkdir(); (OUT/'audit.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['status','mass','fixtures','official_gradients','official_fits','quality_acceptance']}))


if __name__ == '__main__':
    main()
