"""Independent evidence/plan check for V130. Never trains or seals a run."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/v130_learning_review_20260930_r2'
PLAN = ROOT/'training/review_policy/v130_learning_qualification_plan.json'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def check_plan(p):
    assert p['status'] == 'designed_not_trained'
    assert p['fit_budget']['this_review_fits'] == p['fit_budget']['this_review_updates'] == 0
    assert not p['runtime_readiness']['trainer_implemented'] and not p['runtime_readiness']['run_seal_created']
    assert p['primary_trial']['fixed_epochs'] == p['primary_trial']['qualify_epoch'] == 100
    assert p['primary_trial']['early_stop'] is False
    assert [(a['arm'], a['hidden'], a['loss']) for a in p['primary_trial']['factorial']] == [
        ('R', 128, 'row_CE'), ('O', 128, 'focused_CE'), ('C', 256, 'row_CE'), ('CO', 256, 'focused_CE')]
    assert p['loss']['focused_CE'] == '0.5 * row_CE + 0.5 * pure_class_CE'
    assert 'no pure-class auxiliary loss' in p['loss']['mixed_safeguard']
    assert p['selection']['fixed_preference'] == ['R', 'C', 'O', 'CO']
    assert p['confirmation']['automatic_model_promotion'] is False
    assert p['learning_gate']['full_train_M_errors_max'] == 0
    assert p['learning_gate']['canonical_nonconflict_S_errors_max'] == 26
    assert p['learning_gate']['full_train_S_errors_max'] == 32
    assert p['learning_gate']['matched_344_M_errors_max'] == 0
    assert p['learning_gate']['new_errors_on_B124_correct_train_S_max'] == 0
    assert p['preserved_full_quality_gate']['official_original_rows'] == 2056871
    for rel, expected in p['evidence_sha256'].items():
        assert sha(ROOT/rel) == expected, rel


def main():
    audit = read(OUT/'audit.json'); p = read(PLAN); check_plan(p)
    for rel, expected in audit['source_sha256'].items():
        assert sha(ROOT/rel) == expected, rel
    for name, expected in read(OUT/'output_receipt.json')['sha256'].items():
        assert sha(OUT/name) == expected, name
    d = pd.read_parquet(OUT/'training_role_error_ledger.parquet')
    trace = pd.read_parquet(ROOT/'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet')
    official = pd.read_parquet(ROOT/'data/official/train.parquet', columns=['label_binary'])
    labels = official.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(official) == 2056871 and len(d) == 225614
    assert np.array_equal(labels[d.row_position], d.truth)
    assert d.groupby('row_position').outer_fit_role.nunique().eq(2).all()
    assert d[['row_position', 'outer_fit_role']].duplicated().sum() == 0
    roles = pd.read_parquet(ROOT/'artifacts/v128_mechanism_review_20260929/nested_score_roles.parquet')
    recounted = []
    for fold in range(3):
        r = d[d.outer_fit_role == fold]
        expected = trace[trace.fold != fold]
        assert set(r.row_position) == set(expected.row_position)
        assert set(r.root).isdisjoint(set(trace.loc[trace.fold == fold, 'root']))
        ix = r.local.to_numpy(np.int64)
        for key in ['original_key', 'canonical_key', 'body_key']:
            counts = r.groupby([key, 'truth']).size().unstack(fill_value=0)
            mixed = set(counts[(counts[1] > 0) & (counts[2] > 0)].index)
            assert np.array_equal(r[key+'_mixed'], r[key].isin(mixed))
        for column, folder in [('A125_pred', 'v125_order_trial_20260929'), ('B124_pred', 'v124_header_trial_20260929')]:
            arm = 'A' if column == 'A125_pred' else 'B'
            prob = np.load(ROOT/f'artifacts/{folder}/fold{fold}_{arm}/epoch25_prob.npy')[ix]
            assert np.array_equal(prob.argmax(1), r[column])
        for arm in ['P_IS', 'P_CF']:
            rf = roles[roles.outer_fold == fold][['local', 'inner_fold']].drop_duplicates()
            z = np.zeros((22546, 16, 3), np.float32)
            for j in range(3):
                ids = rf.loc[rf.inner_fold == j, 'local'].to_numpy(np.int64)
                teacher = j if arm == 'P_CF' else (j+1) % 3
                z[ids] = np.load(ROOT/f'artifacts/v128_nested_score_trial_20260929_r3/teacher{fold}_{teacher}/canonical_member_logits.npy')[ids]
            delta = np.load(ROOT/f'artifacts/v128_nested_score_trial_20260929_r3/fold{fold}_{arm}/epoch50_residual.npy')
            pred = softmax(z[ix]+delta[ix, None, :], axis=-1).mean(1).argmax(1)
            assert np.array_equal(pred, r[arm+'_pred'])
        for c in (1, 2):
            n = int(((r.truth == c) & (r.A125_pred != c) & ~r.original_key_mixed).sum())
            claim = next(a for a in audit['results'] if a['fold'] == fold and
                a['kind'] == 'direct_fit_endpoint' and a['class'] == c)
            assert n == claim['actual_input_nonconflict_errors']
            recounted.append({'fold': fold, 'class': c, 'A125_nonconflict_errors': n})
    f1 = d[d.outer_fit_role == 1]
    hard = f1[(f1.truth == 2) & (f1.B124_pred != 2) & ~f1.canonical_key_mixed]
    panel = pd.read_parquet(OUT/'fold1_train_only_panel.parquet')
    assert len(hard) == 527 and hard.root.nunique() == 145
    assert set(panel.row_position) == set(f1.loc[f1.body_key.isin(set(hard.body_key)), 'row_position'])
    assert panel.truth.value_counts().to_dict() == {2: 527, 1: 344}
    probe = pd.read_parquet(OUT/'tiny64_train_only_inputs.parquet')
    assert len(probe) == 64 and not probe.canonical_key.duplicated().any()
    assert probe.truth.value_counts().to_dict() == {1: 32, 2: 32}
    assert set(probe.canonical_key).issubset(set(panel.canonical_key))
    assert not f1[f1.canonical_key.isin(probe.canonical_key)].canonical_key_mixed.any()
    for row in read(OUT/'loss_collision_counterexamples.json'):
        r = d[(d.outer_fit_role == row['fold']) & (d.canonical_key == row['canonical_key'])]
        m, s = int((r.truth == 1).sum()), int((r.truth == 2).sum())
        assert (m, s) == (row['M_rows'], row['S_rows'])
        total = d[d.outer_fit_role == row['fold']].truth.value_counts()
        ps = (s/total[2])/(m/total[1]+s/total[2])
        assert abs(ps-row['class_CE_optimal_S_probability']) < 1e-12
    delivery = read(ROOT/p['latest_actual_delivery'])
    assert delivery['status'] == 'v128_21_fit_trial_verified_quality_failed'
    # Counterexamples: an early endpoint or a blind mixed-label weighting arm
    # must be rejected by the focused design checker. A valid plan passes above.
    rejected = []
    for name in ['early_endpoint', 'blind_class_weighting']:
        altered = json.loads(json.dumps(p))
        if name == 'early_endpoint': altered['primary_trial']['qualify_epoch'] = 25
        else: altered['primary_trial']['factorial'][1]['loss'] = 'class_CE'
        try:
            check_plan(altered)
        except AssertionError:
            rejected.append(name)
        else:
            raise AssertionError('Counterexample accepted: ' + name)
    result = {'status': 'evidence_and_design_verified_not_trained', 'all_checks_passed': True,
        'source_files_verified': len(audit['source_sha256']), 'train_role_rows': len(d),
        'official_truth_rows': len(official), 'classifier_fits': 0, 'optimizer_steps': 0,
        'recounted': recounted, 'rejected_counterexamples': rejected,
        'plan_sha256': sha(PLAN), 'verifier_sha256': sha(Path(__file__)),
        'scope': 'Hashes, original truth, saved probability decisions and design constraints; no fresh member-model inference, gradients, optimizer run or transfer validation.'}
    (OUT/'verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
