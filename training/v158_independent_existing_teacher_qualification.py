"""Audit saved historical teachers and their scope; never call a model."""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import pandas as pd
from scipy import sparse
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v128_nested_score_trial_20260929_r3'
OUT = ROOT / 'artifacts/v158_independent_existing_teacher_qualification_20261001'
TRACE = ROOT / 'artifacts/v108_root_evidence_review_20260928/ASA_raw_fact_prediction_trace.parquet'
ROLES = ROOT / 'artifacts/v128_mechanism_review_20260929/nested_score_roles.parquet'
ORIGINAL = ROOT / 'artifacts/v101_full_input_group_n1_20260928/N1_ASA.npz'
CANON = ROOT / 'artifacts/v124_header_trial_20260929/B_header_ASA.npz'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def split(outer, root):
    key = f'V128|split=12801|outer={outer}|root={root}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'big') % 3


def tensor_hash(state):
    h = hashlib.sha256()
    for key, tensor in state.items():
        h.update(key.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    assert not OUT.exists(), 'Preserve completed audits'
    seal = read(RUN / 'run_seal.json')
    assert seal['trainer'] == 'training/v128_train_r3.py'
    for name, identity in seal['source_sha256'].items():
        assert sha(ROOT / name) == identity, name
    d = pd.read_parquet(TRACE, columns=['row_position', 'local', 'root', 'fold', 'truth'])
    official = pd.read_parquet(ROOT / 'data/official/train.parquet', columns=['label_binary'])
    gold = official.label_binary.map({'benign': 0, 'malicious': 1, 'suspicious': 2}).to_numpy()
    assert len(gold) == 2056871 and len(d) == 112807 and d.row_position.is_unique
    assert np.array_equal(gold[d.row_position], d.truth)
    roles = pd.read_parquet(ROLES)
    assert len(roles) == 225614
    x = sparse.load_npz(ORIGINAL).tocsr()
    canon = sparse.load_npz(CANON).tocsr()
    assert x.shape == canon.shape == (22546, 66287)
    delta = (x - canon).tocsr()
    delta.eliminate_zeros()
    fact_delta = delta[:, 65792:]
    assert fact_delta.nnz == 0
    changed_locals = np.flatnonzero(np.diff(delta.indptr))
    teacher_reports, class_reports, inputs = [], [], [TRACE, ROLES, ORIGINAL, CANON,
        RUN / 'run_seal.json', Path(__file__)]
    for fold in range(3):
        r = roles.loc[roles.outer_fold.eq(fold)].copy()
        legal = d.loc[d.fold.ne(fold)]
        held_roots = set(d.loc[d.fold.eq(fold), 'root'])
        assert len(r) == len(legal) and r.row_position.is_unique
        joined = r.merge(legal, on='row_position', suffixes=('', '_reference'), validate='one_to_one')
        for col in ['local', 'root', 'fold', 'truth']:
            assert np.array_equal(joined[col], joined[col + '_reference'])
        assert not set(r.root) & held_roots
        assert np.array_equal(r.inner_fold, r.root.map(lambda root: split(fold, root)))
        assert np.array_equal(r.crossfit_teacher, r.inner_fold)
        assert np.array_equal(r.matched_infit_teacher, (r.inner_fold + 1) % 3)
        assert r.groupby('local').inner_fold.nunique().max() == 1
        oof_pred = np.empty(len(r), np.int8)
        oof_member_count = np.empty(len(r), np.int8)
        for inner in range(3):
            folder = RUN / f'teacher{fold}_{inner}'
            receipt = read(folder / 'fit.json')
            fit = r.loc[r.inner_fold.ne(inner)]
            query = r.loc[r.inner_fold.eq(inner)]
            assert not set(fit.root) & set(query.root)
            masses = [int(fit.truth.eq(cl).sum()) for cl in range(3)]
            nb = math.ceil(fit.local.nunique() / 256)
            assert (receipt['status'], receipt['completed_epochs'], receipt['prediction_epoch']) == ('fit_executed', 25, 25)
            assert (receipt['fold'], receipt['excluded_inner'], receipt['fit_rows']) == (fold, inner, len(fit))
            assert [receipt['fit_M'], receipt['fit_S']] == masses[1:]
            assert receipt['fit_root_count'] == fit.root.nunique()
            assert receipt['excluded_root_count'] == query.root.nunique()
            assert receipt['optimizer_steps'] == 25 * nb
            for file, key in [('epoch25_model.pt', 'model_sha256'), ('canonical_member_logits.npy', 'logits_sha256'),
                              ('steps.jsonl', 'steps_sha256'), ('progress.json', 'progress_sha256')]:
                assert sha(folder / file) == receipt[key], str(folder / file)
                inputs.append(folder / file)
            assert receipt['seal_sha256'] == sha(RUN / 'run_seal.json')
            checkpoint = torch.load(folder / 'epoch25_model.pt', map_location='cpu', weights_only=True)
            assert checkpoint['epoch'] == 25 and checkpoint['optimizer_steps'] == 25 * nb
            assert checkpoint['fold'] == fold and checkpoint['excluded_inner'] == inner
            assert checkpoint['seal_sha256'] == receipt['seal_sha256']
            assert tensor_hash(checkpoint['base']) == receipt['base_state_sha256']
            steps = pd.DataFrame([json.loads(line) for line in (folder / 'steps.jsonl').read_text().splitlines()])
            assert np.array_equal(steps.step, np.arange(1, 25 * nb + 1)) and steps.finite.all()
            for epoch, group in steps.groupby('epoch'):
                assert len(group) == nb
                assert [int(group.M_original_rows.sum()), int(group.S_original_rows.sum())] == masses[1:]
            progress = read(folder / 'progress.json')
            assert len(progress) == 25
            for epoch, item in enumerate(progress, 1):
                assert (item['epoch'], item['steps'], item['fit_M'], item['fit_S']) == (epoch, epoch * nb, *masses[1:])
            z = np.load(folder / 'canonical_member_logits.npy').astype(np.float64)
            assert z.shape == (22546, 16, 3) and np.isfinite(z).all()
            exponential = np.exp(z - z.max(2, keepdims=True))
            p = exponential / exponential.sum(2, keepdims=True)
            prediction = p.mean(1).argmax(1)
            for name, frame in [('fit', fit), ('crossfit', query)]:
                for cl in [1, 2]:
                    g = frame.loc[frame.truth.eq(cl)]
                    expected = receipt['role_diagnostic'][name][str(cl)]
                    assert expected['support'] == len(g)
                    assert expected['errors'] == int((prediction[g.local] != cl).sum())
            mask = r.inner_fold.eq(inner).to_numpy()
            oof_pred[mask] = prediction[query.local]
            member_right = (z[query.local].argmax(2) == query.truth.to_numpy()[:, None]).sum(1)
            oof_member_count[mask] = member_right
            teacher_reports.append(dict(outer=fold, excluded_inner=inner, fit_rows=len(fit),
                fit_M=masses[1], fit_S=masses[2], query_M=int(query.truth.eq(1).sum()),
                query_S=int(query.truth.eq(2).sum()), original_mass_each_epoch_verified=True,
                saved_model_and_logits_hash_verified=True, legal_all_supervised_fit_roots=True,
                actual_historical_steps=25 * nb, new_fits=0))
            inputs.extend([folder / 'fit.json', folder / 'started.json'])
        for cl in [1, 2]:
            mask = r.truth.eq(cl).to_numpy()
            wrong = oof_pred != r.truth.to_numpy()
            class_reports.append(dict(outer=fold, truth=cl, original_rows=int(mask.sum()),
                OOF_errors=int((mask & wrong).sum()), OOF_errors_with_correct_member=int((mask & wrong & (oof_member_count > 0)).sum()),
                OOF_all_members_wrong=int((mask & (oof_member_count == 0)).sum()),
                OOF_all_members_correct=int((mask & (oof_member_count == 16)).sum())))
    report = dict(status='historical_saved_teacher_roles_scores_and_mass_independently_verified',
        old_teachers=9, original_rows=112807, nested_role_rows=225614,
        original_and_current_input_width=66287, fact_width=495, fact_coordinates_changed=0,
        changed_text_locals=len(changed_locals), changed_text_original_rows=int(d.local.isin(changed_locals).sum()),
        teacher_reports=teacher_reports, OOF_profiles=class_reports,
        current_V146_OOF_qualification=False,
        reason='Legal V128 source exclusions do not reproduce V146: old N1 fit/header prediction, 25-epoch SparseTabM, versus canonical 100-epoch deterministic network and four later supervised stages.',
        own_official_classifier_calls=0, own_feature_calls=0, own_gradients=0, own_fits=0, own_updates=0,
        limitation='Saved checkpoint identity, source/row receipts and arithmetic only; no independent model replay and no claim that old diversity transfers to current deployment.',
        source_sha256={p.relative_to(ROOT).as_posix(): sha(p) for p in inputs})
    OUT.mkdir()
    (OUT / 'audit.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['status', 'fact_coordinates_changed', 'changed_text_locals',
        'changed_text_original_rows', 'current_V146_OOF_qualification', 'OOF_profiles']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
