"""Read-only saved component/role audit; no official model or feature calls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'artifacts/v157_complete_function_logit_audit_v2_20261001'
OUT = ROOT / 'artifacts/v157_independent_saved_function_audit_20261001'
LAST = ROOT / 'artifacts/v155_guarded_full_gradient_sam_20261001'
REFERENCE = ROOT / 'artifacts/v153_independent_training_transfer_gap_20261001/all_original_classifier_gap_and_control_ledger.parquet'
ARITHMETIC_BOUND = 2e-12


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def verify(bindings):
    for name, expected in bindings.items():
        p = Path(name)
        if not p.is_absolute():
            p = ROOT / p
        assert sha(p) == expected, name


def main():
    assert not OUT.exists()
    audit = read(RUN / 'audit.json')
    seal = read(RUN / 'run_seal.json')
    verify(audit['source_sha256'])
    verify(seal['source_sha256'])
    assert audit['classifier_forward_calls'] == 36
    assert audit['cumulative_classifier_forward_calls'] == 48
    assert audit['gradients'] == audit['classifier_fits'] == audit['updates'] == 0
    failed = read(ROOT / 'artifacts/v157_complete_function_logit_audit_20261001/failure.json')
    assert failed['progress']['classifier_completed'] == 12
    assert failed['entry_sha256'] == sha(ROOT / 'training/v157_functional_logit_audit.py')
    partition = read(RUN / 'field_partition.json')
    assert sorted(i for indices in partition.values() for i in indices) == list(range(495))
    rows = pd.read_parquet(RUN / 'all_original_role_function_summary.parquet')
    original = pd.read_parquet(RUN / 'original_reference.parquet')
    reference = pd.read_parquet(REFERENCE)
    assert len(rows) == 338421 and len(original) == len(reference) == 112807
    assert original.row_position.is_unique
    assert not rows.duplicated(['row_position', 'training_role']).any()
    for col in ['row_position', 'local', 'root', 'fold', 'truth', 'canonical_key']:
        assert np.array_equal(original[col], reference[col]), col
    reports = []
    paths = [RUN / 'audit.json', RUN / 'run_seal.json', REFERENCE,
             RUN / 'original_reference.parquet', RUN / 'field_partition.json',
             RUN / 'all_original_role_function_summary.parquet', Path(__file__)]
    for fold in range(3):
        folder = RUN / f'fold{fold}'
        names = ['logits', 'body', 'facts', 'bias', 'probabilities']
        values = {name: np.load(folder / f'{name}.npy') for name in names}
        z, body, facts, bias, probability = [values[name] for name in names]
        assert z.shape == body.shape == (22546, 16, 3)
        assert facts.shape == (22546, 3) and bias.shape == (16, 3)
        assert np.array_equal(z, body + bias + facts[:, None, :])
        exponential = np.exp(z - z.max(axis=2, keepdims=True))
        members = exponential / exponential.sum(axis=2, keepdims=True)
        reconstructed = members.mean(axis=1)
        gap = float(np.max(np.abs(reconstructed - probability)))
        assert gap <= ARITHMETIC_BOUND
        assert np.array_equal(reconstructed.argmax(1), probability.argmax(1))
        assert np.array_equal(probability, np.load(LAST / f'fold{fold}_zero_probability.npy'))
        fields = np.load(folder / 'field_logits.npz')
        assert set(fields.files) == set(partition)
        field_sum = sum(fields[name] for name in partition)
        field_gap = float(np.max(np.abs(field_sum - facts)))
        assert field_gap <= ARITHMETIC_BOUND
        local = pd.read_parquet(folder / 'local_function_summary.parquet')
        assert np.array_equal(local.local, np.arange(22546))
        assert np.array_equal(local.probability_mean_pred, probability.argmax(1))
        assert np.array_equal(local.mean_logit_pred, z.mean(axis=1).argmax(1))
        member_predictions = z.argmax(2)
        for cl, label in enumerate(['N', 'M', 'S']):
            assert np.array_equal(local[f'member_{label}_predictions'], (member_predictions == cl).sum(1))
        role = rows[rows.training_role.eq(fold)].reset_index(drop=True)
        assert np.array_equal(role.row_position, original.row_position)
        for col in ['local', 'root', 'fold', 'truth', 'canonical_key']:
            assert np.array_equal(role[col], original[col]), col
        assert np.array_equal(role.query_role, np.where(role.fold.eq(fold), 'outer_HELD', 'legal_TRAIN'))
        assert np.array_equal(role.pred, probability[role.local].argmax(1))
        for cl in range(3):
            assert np.array_equal(role[f'p{cl}'], probability[role.local, cl])
        reports.append(dict(fold=fold,original_role_rows=len(role),
                            saved_logits_recomposition_exact=True,
                            saved_probability_identity_exact=True,
                            CPU_softmax_probability_max_gap=gap,
                            saved_field_sum_max_gap=field_gap))
        paths.extend(folder / f'{name}.npy' for name in names)
        paths.extend([folder / 'field_logits.npz', folder / 'local_function_summary.parquet',
                      LAST / f'fold{fold}_zero_probability.npy'])
    matched = rows.merge(reference[['row_position', 'known_578_cohort',
                                   'same_family_and_outer_fold_control_S']],
                         on='row_position', how='left', validate='many_to_one')
    profiles = []
    for cohort, mask in [('all', np.ones(len(matched), dtype=bool)),
                         ('hard578', matched.known_578_cohort),
                         ('correct51', matched.same_family_and_outer_fold_control_S)]:
        for (role, truth), g in matched.loc[mask].groupby(['query_role', 'truth']):
            errors = g.pred.ne(g.truth)
            correct_member_count = g[f'member_{"NMS"[int(truth)]}_predictions']
            profiles.append(dict(cohort=cohort,role=role,truth=int(truth),
                original_role_rows=len(g),classification_errors=int(errors.sum()),
                no_correct_member_rows=int(correct_member_count.eq(0).sum()),
                errors_with_at_least_one_correct_member=int((errors & correct_member_count.gt(0)).sum()),
                all_16_correct_member_rows=int(correct_member_count.eq(16).sum()),
                mean_logit_diagnostic_errors=int(g.mean_logit_pred.ne(g.truth).sum()),
                mean_logit_and_actual_probability_disagreements=int(g.mean_logit_pred.ne(g.pred).sum()),
                body_margin_mean=float(g.body_S_minus_M_mean.mean()),
                direct_fact_margin_mean=float(g.facts_S_minus_M.mean()),
                bias_margin_mean=float(g.bias_S_minus_M_mean.mean())))
    result = dict(status='saved_full_function_algebra_probabilities_and_original_roles_independently_verified',
        original_rows=112807,original_role_rows=338421,physical_sealed_files=len(seal['source_sha256']),
        actual_executor_classifier_calls_successful_retry=36,actual_executor_classifier_calls_including_failure=48,
        own_official_classifier_calls=0,own_feature_calls=0,own_gradients=0,own_fits=0,own_updates=0,
        arithmetic_bound=ARITHMETIC_BOUND,reports=reports,profiles=profiles,
        limits=['Saved logits/body/facts/bias algebra is checked; H2 and its projection were not independently rerun.',
                'CPU saved-logit softmax recomputation is numerical verification, not a new classifier or adopted prediction.',
                'Component magnitudes and readout disagreements do not prove security semantics, causal feature importance or transfer.'],
        source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in paths})
    OUT.mkdir()
    (OUT / 'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['status','original_rows','original_role_rows','reports','profiles']},ensure_ascii=False))


if __name__ == '__main__':
    main()
