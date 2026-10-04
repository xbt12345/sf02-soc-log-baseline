"""Independent source-label, role, replay, selection, and history verification."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from run_v75 import ROOT, read, save, sha, load_sparse
from v89_common import LAST, data
from v92_train import Branch, csr
from v92_common import DEST as V92
from v95_prepare import DEST, role


def main():
    assert not (DEST / 'verification.json').exists()
    registration = read(DEST / 'registration.json')
    for path, digest in registration['input_sha256'].items(): assert sha(ROOT / path) == digest, path
    assert registration['source_sha256'] == sha(ROOT / 'training/v95_prepare.py')
    assert sha(DEST / 'full_format_roles.parquet') == registration['manifest_sha256']
    assert sha(DEST / 'auxiliary_rows.parquet') == registration['auxiliary_rows_sha256']
    teacher_info = read(DEST / 'teacher_fit.json')
    assert teacher_info['source_sha256'] == sha(ROOT / 'training/v95_teacher.py')
    assert teacher_info['actual_classifier_fits'] == 1 and teacher_info['converged']
    assert teacher_info['model_sha256'] == sha(DEST / 'teacher.joblib')
    r, y, fid, _, _, _, fit = data()
    official = pd.read_parquet(ROOT / 'data/official/train.parquet', columns=['label_binary']).label_binary.map(
        {'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert np.array_equal(official, y)
    manifest = pd.read_parquet(DEST / 'full_format_roles.parquet')
    assert len(manifest) == len(r) and np.array_equal(manifest.label_index, official)
    assert np.array_equal(manifest.row_position, np.arange(len(r)))
    assert np.all(manifest.role[~fit] == 'L')
    assert np.all(manifest.role[r.fold.eq(-1).to_numpy()] == 'A')
    for field in ['component', 'body_group', 'source_symbol']:
        m = manifest[field] >= 0
        assert manifest.loc[m].groupby(field).role.nunique().max() == 1
    for c, f, rr in manifest[['component', 'fold', 'role']].drop_duplicates().itertuples(index=False, name=None):
        assert rr == ('L' if int(f) in [0, 1, 2] else 'A' if int(f) == -1 else role(c))
    for rr in 'ABV':
        take = manifest.role.eq(rr).to_numpy()
        assert np.bincount(y[take], minlength=3).tolist() == registration['A_B_V_class_rows'][rr]
    aux = pd.read_parquet(DEST / 'auxiliary_rows.parquet')
    assert len(aux) == len(aux.row_position.unique())
    assert set(aux.role) == set('ABV')
    assert np.array_equal(official[aux.row_position], aux.label_index)
    for _, subset in aux.groupby('behavior'):
        assert all(((subset.role == rr) & (subset.label_index == cl)).any()
                   for rr in 'AB' for cl in [1, 2])
    assert aux.behavior.nunique() == 3
    model = joblib.load(DEST / 'teacher.joblib')
    x = load_sparse(LAST / 'X')
    scores = np.asarray(x @ model['coef']) + model['intercept']
    assert np.allclose(scores, np.load(DEST / 'teacher_scores.npy', mmap_mode='r'), rtol=0, atol=1e-10)
    assert np.array_equal(scores.argmax(1), np.load(DEST / 'teacher_prediction.npy'))
    ids = np.load(V92 / 'ASA_input_ids.npy'); xx = sparse.load_npz(V92 / 'ASA_R0.npz')
    initial = set()
    branches_replayed = 0
    for arm, step in [('E00', 100), ('E10', 200), ('E01', 50), ('E11', 50)]:
        record = read(DEST / f'{arm}_fit.json')
        assert record['source_sha256'] == sha(ROOT / 'training/v95_four_arm.py')
        assert record['actual_classifier_fits'] == 1 and record['steps_executed'] == 200
        assert record['A_B_original_rows_supervised'] == registration['A_B_V_rows']['A'] + registration['A_B_V_rows']['B']
        assert record['V_gradient_rows_used'] == 0 and record['inner_C_H_gradient_rows_used'] == 0
        assert record['teacher_model_sha256'] == teacher_info['model_sha256']
        initial.add(record['initial_state_sha256'])
        snap = next(s for s in read(DEST / f'{arm}_snapshots.json') if s['step'] == step)
        path = DEST / f'{arm}_step{step:03}_model.pt'
        assert sha(path) == snap['model_sha256']
        state = torch.load(path, map_location='cpu', weights_only=True)
        branch = Branch(xx.shape[1]); branch.load_state_dict(state['state_dict']); branch.eval()
        actual = np.empty((len(ids), 3), float)
        with torch.no_grad():
            for start in range(0, len(ids), 2048):
                end = min(start + 2048, len(ids))
                actual[start:end] = scores[ids[start:end]] + branch(csr(xx[start:end], 'cpu')).double().numpy()
        expected = np.load(DEST / f'{arm}_step{step:03}_prediction.npy')
        predicted = np.load(DEST / 'teacher_prediction.npy').copy()
        predicted[ids] = actual.argmax(1)
        assert np.array_equal(predicted, expected), (arm, step)
        assert sha(DEST / f'{arm}_step{step:03}_prediction.npy') == snap['prediction_sha256']
        branches_replayed += 1
    assert len(initial) == 1
    selection = read(DEST / 'selection.json')
    candidates = []
    for arm in registration['branch']['arms']:
        candidates += [s for s in read(DEST / f'{arm}_snapshots.json') if s['V_eligibility']['eligible']]
    expected_choice = min(candidates, key=lambda s: (s['roles']['V']['errors'], s['step'], s['arm']))
    assert (selection['V_selected']['arm'], selection['V_selected']['step']) == (expected_choice['arm'], expected_choice['step'])
    diagnosis = read(DEST / 'diagnosis.json')
    assert not diagnosis['historic_all_guards_passed'] and not diagnosis['two_rotation_training_triggered']
    assert not diagnosis['model_promoted'] and not diagnosis['quality_acceptance']
    assert read(DEST / 'failure_audit.json')['V_S_top_error_components'][0]['rows'] == 685
    old_paths = list(read(ROOT / 'artifacts/v94_research_mechanism_20260928/verification.json')['receipt_sha256']) + [
        'evidence/2026-09-28/v94_deep_research/delivery.json']
    previous = {}; receipts = {}
    for path in old_paths:
        receipts[path] = sha(ROOT / path)
        for f, h in read(ROOT / path)['artifact_sha256'].items():
            assert f not in previous or previous[f] == h
            previous[f] = h
    changes = [path for path, digest in previous.items() if sha(ROOT / path) != digest]
    assert not changes, changes
    result = {'status': 'passed', 'official_labels_verified': len(y),
              'full_format_roles_verified': len(manifest), 'auxiliary_groups_verified': aux.behavior.nunique(),
              'fresh_A_teacher_scores_replayed': len(scores), 'branch_checkpoints_replayed': branches_replayed,
              'matching_initial_states': True, 'selection_recomputed': True,
              'prior_bound_files_rehashed': len(previous), 'prior_bound_files_changed': changes,
              'prior_delivery_receipts_sha256': receipts,
              'actual_classifier_fits': 5, 'actual_calibration_fits': 0,
              'quality_acceptance': False, 'model_promoted': False,
              'source_sha256': sha(__file__),
              'scope': 'Official labels, role isolation, selected frozen inference and history verified; no external-transfer or official-score claim.'}
    save(DEST / 'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'prior_delivery_receipts_sha256'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=4): main()
