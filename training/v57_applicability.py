"""Applicable observed-value residuals; frozen v56 input/base remain unchanged."""
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import softmax
import run_v54 as v
import run_v55 as n
import run_v56 as t
import v56_pooling as old
import v56_shared_control as prev

def keys(x):
    out = {}
    protocol = x.transport_protocol.to_numpy()
    for field in prev.SHARED:
        values = x[field].to_numpy(dtype=object, copy=True)
        allowed = ~np.isin(values, [v.MISSING, v.UNKNOWN])
        if field.startswith('icmp_'):
            allowed &= protocol == 'icmp'
        if field in ['src_port_range', 'dst_port_range']:
            allowed &= np.isin(protocol, ['tcp', 'udp'])
        values[~allowed] = None
        out['field:' + field] = values
    return {**out, **old.keys(x)}

def fit_design(x, bodies):
    maps, rows, supports, features = {}, [], [], []
    for level, values in keys(x).items():
        z = pd.DataFrame({'value': values, 'body': np.asarray(bodies)}).dropna(subset=['value'])
        g = z.groupby('value', sort=True).agg(rows=('body', 'size'), bodies=('body', 'nunique'))
        maps[level] = {value: len(features) + i for i, value in enumerate(g.index)}
        features.extend([[level, value] for value in g.index])
        rows.extend(g.rows)
        supports.extend(g.bodies)
    return {'maps': maps, 'rows': np.asarray(rows, float), 'bodies': np.asarray(supports, float), 'feature_keys': features, 'fit_rows': len(x)}

def matrix(design, x):
    ri, ci = [], []
    for level, values in keys(x).items():
        lookup = design['maps'][level]
        for row, value in enumerate(values):
            idx = lookup.get(value)
            if idx is not None:
                ri.append(row)
                ci.append(idx)
    return sparse.csr_matrix((np.ones(len(ri)), (ri, ci)), shape=(len(x), len(design['feature_keys'])))

def fit(base, x, y, bodies, strength, options):
    design = fit_design(x, bodies)
    xx, counts, _ = old.compress(x, y)
    A = matrix(design, xx)
    _, probability = n.predict(base, xx)
    offset = np.log(np.maximum(probability.astype(float), 1e-30))
    precision = strength * design['rows'] / len(x) / np.sqrt(design['bodies'])
    scale = np.sqrt(precision)
    scaled = A.multiply(1 / scale).tocsr()
    result = minimize(old.objective, np.zeros(A.shape[1] * 3), args=(scaled, offset, counts, np.ones(len(precision))), jac=True, method='L-BFGS-B', options=options)
    w = result.x.reshape(-1, 3) / scale[:, None]
    w -= w.mean(axis=1, keepdims=True)
    loss, grad = old.objective(w.ravel(), A, offset, counts, precision)
    gmax = float(abs(grad.reshape(-1, 3) / scale[:, None]).max())
    info = {'success': bool(result.success), 'iterations': int(result.nit), 'message': str(result.message), 'objective': loss, 'scaled_gradient': gmax, 'eligible': bool(result.success and gmax <= 5e-6), 'original_class_counts': counts.sum(axis=0).astype(int).tolist(), 'compressed_inputs': len(xx)}
    return {'design': design, 'weights': w, 'strength': strength, 'optimizer': info}

def predictions(bundle, protocol, role, rows, x, mask):
    baseline = pd.read_parquet(t.RUN / protocol / ('base_' + role + '.parquet'))
    parts = []
    for view, fields in v.VIEWS.items():
        xx = x.loc[mask].copy()
        xx[fields] = v.MISSING
        b = baseline[baseline.scenario == view].reset_index(drop=True)
        pd.testing.assert_frame_equal(b[rows.columns], rows.loc[mask].reset_index(drop=True))
        p = softmax(np.log(np.maximum(b[['p_0', 'p_1', 'p_2']].to_numpy().astype(float), 1e-30)) + matrix(bundle['design'], xx) @ bundle['weights'], axis=1)
        b['pred'] = p.argmax(axis=1)
        for k in range(3):
            b['p_' + str(k)] = p[:, k]
        parts.append(b)
    return pd.concat(parts, ignore_index=True)

def compare(base, candidate):
    cells, failures = [], []
    improved = False
    for view in v.VIEWS:
        b = base[base.scenario == view].reset_index(drop=True)
        a = candidate[candidate.scenario == view].reset_index(drop=True)
        np.testing.assert_array_equal(a.row_position, b.row_position)
        np.testing.assert_array_equal(a.label_index, b.label_index)
        np.testing.assert_array_equal(a.eligible_stress, b.eligible_stress)
        masks = {'M': a.route.eq('asa') & a.eligible_stress & a.label_index.eq(1), 'S': a.route.eq('asa') & a.eligible_stress & a.label_index.eq(2), 'normal': a.route.eq('asa_acl'), 'ASA_to_normal': a.route.eq('asa') & a.eligible_stress}
        for name, mask in masks.items():
            bb, aa = b.loc[mask], a.loc[mask]
            errb = int((bb.pred == 0).sum()) if name == 'ASA_to_normal' else int((bb.pred != bb.label_index).sum())
            erra = int((aa.pred == 0).sum()) if name == 'ASA_to_normal' else int((aa.pred != aa.label_index).sum())
            cell = {'view': view, 'class': name, 'support': len(aa), 'base_errors': errb, 'candidate_errors': erra, 'delta': erra - errb}
            cells.append(cell)
            if len(aa) == 0 or erra > errb:
                failures.append(cell)
            if name in ['M', 'S'] and erra < errb:
                improved = True
    return {'cells': cells, 'failures': failures, 'all_cells_nonincrease': not failures, 'any_M_S_improvement': improved, 'quality_passed': not failures and improved}

def design_audit(design, x):
    A = matrix(design, x).tocsc()
    signatures = {}
    for j, (level, value) in enumerate(design['feature_keys']):
        if level.startswith('field:'):
            indices = A.indices[A.indptr[j]:A.indptr[j + 1]]
            signatures.setdefault(indices.tobytes(), []).append([level, value])
    return {'shared_parameters': sum(k.startswith('field:') for k, value in design['feature_keys']), 'remaining_identical_shared_columns': [values for values in signatures.values() if len(values) > 1], 'shared_missing_parameter_count': sum(k.startswith('field:') and value in [v.MISSING, v.UNKNOWN] for k, value in design['feature_keys']), 'scope': 'Only the new observed-value branch is gated. Full frozen base/coarse/fine/exact observations retain availability and all real values. Coincident observed facts are reported, not automatically merged.'}
