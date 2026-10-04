"""Freeze v9.4 execution roles and support before fitting any parameter."""
import hashlib
import json
import time
import numpy as np
import pandas as pd
from v89_common import ROOT, OUT, LAST, read, save, sha, data

DEST = ROOT / 'artifacts/v95_cross_component_training_20260928'
PLAN = ROOT / 'docs/V94_DEEP_RESEARCH_AND_TRAINING_REVISION.md'
V94 = ROOT / 'artifacts/v94_research_mechanism_20260928'
SEED = 9501


def role(component):
    h = hashlib.sha256(('v94:9301:' + str(component)).encode()).hexdigest()
    bucket = int(h[:16], 16) % 10
    return 'A' if bucket < 6 else 'B' if bucket < 8 else 'V'


def main():
    assert not DEST.exists()
    DEST.mkdir()
    r, y, fid, _, _, _, fit = data()
    component_roles = {int(c): ('A' if int(fold) == -1 else role(int(c)))
                       for c, fold in r[['component', 'fold']].drop_duplicates().itertuples(index=False, name=None)}
    assert len(component_roles) == r.component.nunique()
    roles = np.full(len(r), 'L', dtype='<U1')
    roles[fit] = r.loc[fit, 'component'].map(component_roles).to_numpy()
    assert np.array_equal(np.unique(roles[fit]), np.array(['A', 'B', 'V']))
    assert np.all(roles[r.fold.eq(-1).to_numpy()] == 'A')
    for field in ['component', 'body_group', 'source_symbol']:
        temp = pd.DataFrame({'group': r[field], 'role': roles})
        temp = temp[temp.group >= 0]
        assert temp.groupby('group').role.nunique().max() == 1, field
    full = r[['row_position', 'component', 'body_group', 'source_symbol', 'fold', 'route', 'label_index']].copy()
    full['role'] = roles
    full.to_parquet(DEST / 'full_format_roles.parquet', index=False)
    asa_prior = pd.read_parquet(V94 / 'prospective_ASA_roles.parquet')
    isa = fit & r.route.eq('asa').to_numpy()
    now = full.loc[isa, ['row_position', 'component', 'label_index', 'role']]
    prior = asa_prior.set_index('row_position').loc[now.row_position]
    assert np.array_equal(prior.component.to_numpy(), now.component)
    assert np.array_equal(prior.label_index.to_numpy(), now.label_index)
    assert np.array_equal(prior.proposed_role.to_numpy(), now.role)
    eligibility = read(V94 / 'auxiliary_eligibility.json')
    assert eligibility['training_supported_groups'] == 3
    assert eligibility['V_labels_used_to_choose_training_groups'] is False
    # Join the pre-frozen raw-row behavior observation without deriving a label from V.
    support = asa_prior[['row_position', 'behavior']]
    selected = support[support.behavior.isin(eligibility['eligible_behaviors'])]
    meta = selected.merge(now, on='row_position', how='inner', validate='one_to_one')
    assert meta.behavior.nunique() == 3
    meta[['row_position', 'behavior', 'role', 'label_index', 'component']].to_parquet(DEST / 'auxiliary_rows.parquet', index=False)
    registration = {
        'version': 'v95-execute-v94-plan', 'status': 'registered_before_training',
        'registered_unix': time.time(), 'seed': SEED,
        'roles': {'A': 'teacher fit; later A+B branch main and auxiliary fit',
                  'B': 'branch fit and auxiliary cross-component feedback',
                  'V': 'finite checkpoint selection only; never a training gradient',
                  'L': 'original fold0/1/2 locked development; no fit or selection'},
        'role_rule': 'v94:9301:component modulo10; fold-1 forced A',
        'all_fit_original_rows': int(fit.sum()),
        'A_B_V_rows': {x: int((roles == x).sum()) for x in 'ABV'},
        'A_B_V_class_rows': {x: np.bincount(y[roles == x], minlength=3).astype(int).tolist() for x in 'ABV'},
        'ASA_A_B_V_class_rows': {x: np.bincount(y[(roles == x) & r.route.eq('asa').to_numpy()], minlength=3).astype(int).tolist() for x in 'ABV'},
        'auxiliary_behaviors': sorted(eligibility['eligible_behaviors']),
        'auxiliary_training_eligibility': 'Only A and B observed labels; V labels excluded',
        'auxiliary_group_caveat': 'One empty-known-facts group and one 16-row group; report individual strata',
        'teacher': 'Fresh A-only exact raw-frequency OVR BCE, fixed R0, L2 alpha1e-6, SciPy L-BFGS-B max1000 gtol1e-6 ftol1e-12',
        'branch': {'capacity': [66287, 64, 16, 3], 'gate': 'ASA only; non-ASA teacher predictions unchanged',
                   'main': 'Exact A+B raw-row mean softmax CE; non-ASA loss constant',
                   'optimizer': 'Full-batch Adam, lr0.003, 200 steps, checkpoints [0,25,50,100,200]',
                   'initialization': 'same seed and zero output layer for all four arms',
                   'compatibility': 'same soft teacher-correct-label margin penalty for all arms; no hard intermediate clamp; coefficient 0.01',
                   'magnitude': 'output residual squared mean, coefficient 0.01 if enabled',
                   'meta': 'first-order MLDG-style clone: A gradient step eta0.01, B gradient on updated clone; both directions alternated; beta0.01 if enabled',
                   'auxiliary_weighting': 'equal mean per behavior/class inside role; main raw frequency untouched',
                   'arms': ['E00', 'E10', 'E01', 'E11']},
        'selection': 'V classification exact per-class truth; require >0 repairs, 0 negative flips, nondecreasing B/M/S recall precision F1, normal FP no increase. Pick fewer errors then earlier checkpoint then arm name. Locked inner/C/H not used.',
        'continuation': 'No final promotion without old-v85 paired regression and two rotated folds; no automatic hyperparameter search',
        'new_classifier_fit_budget': 5, 'new_calibration_fit_budget': 0,
        'source_sha256': sha(__file__),
        'plan_sha256': sha(PLAN),
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in [ROOT/'data/official/train.parquet', OUT/'rows.parquet',
                            LAST/'row_feature_id.npy', LAST/'X.data', LAST/'X.indices', LAST/'X.indptr',
                            V94/'auxiliary_eligibility.json', V94/'prospective_ASA_roles.parquet']},
        'manifest_sha256': sha(DEST/'full_format_roles.parquet'),
        'auxiliary_rows_sha256': sha(DEST/'auxiliary_rows.parquet')}
    save(DEST / 'registration.json', registration)
    print(json.dumps({'status': registration['status'], 'A_B_V_rows': registration['A_B_V_rows'],
                      'A_B_V_class_rows': registration['A_B_V_class_rows'], 'auxiliary_training_groups': 3}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
