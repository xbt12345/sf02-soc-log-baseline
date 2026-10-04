"""Same-row base exposure and input-context audit; descriptive only, no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import softmax
from v128_no_fit_review import ROOT, OUT, PARENT, TRACE, OFFICIAL, sha, write, counts, floor_report
from v75_metadata import parse_port


def main():
    target = OUT / 'score_role_audit.json'
    if target.exists():
        raise FileExistsError(target)
    d = pd.read_parquet(OUT / 'body_support_and_errors.parquet')
    trace = pd.read_parquet(TRACE)
    port = pd.read_parquet(OFFICIAL, columns=['src_port']).iloc[d.row_position].reset_index(drop=True).src_port
    full = d.copy()
    full['facts'] = trace.facts_json
    full['raw_record_port'] = port
    parsed = [parse_port(v) for v in port]
    # Normalize only observed valid numbers. Opaque strings are not port values.
    full['observed_record_port'] = ['n:' + str(n) if n is not None else '<UNOBSERVED>' for n, _ in parsed]
    projections = {}
    for name, cols in [('body', ['body_id']), ('body_facts', ['body_id', 'facts']),
                       ('body_observed_record_port', ['body_id', 'observed_record_port']),
                       ('body_raw_record_port_DIAGNOSTIC_ONLY', ['body_id', 'raw_record_port'])]:
        cc = full.groupby(cols, dropna=False).truth.value_counts().unstack(fill_value=0)
        projections[name] = {'groups': len(cc), 'mixed_groups': int((cc.gt(0).sum(1) > 1).sum()),
                            'empirical_floor': int((cc.sum(1) - cc.max(1)).sum())}
    local = d.local.to_numpy(); f = d.fold.to_numpy(); y = d.truth.to_numpy()
    # Two models trained on each record and one model whose fit excluded its fold.
    probabilities = np.stack([softmax(np.load(PARENT / f'fold{k}_base_member_logits.npy'), axis=-1).mean(1)[local] for k in range(3)])
    predictions = probabilities.argmax(2)
    seen_correct = ((predictions == y[None, :]) & (np.arange(3)[:, None] != f[None, :])).sum(0)
    held_pred = predictions[f, np.arange(len(d))]
    held_prob = probabilities[f, np.arange(len(d))]
    assert np.array_equal(held_pred, d.pred_AH)
    seen_true_prob = sum(np.where(f != k, probabilities[k, np.arange(len(d)), y], 0.) for k in range(3)) / 2
    held_true_prob = held_prob[np.arange(len(d)), y]
    d['in_fit_base_models_correct_count_of_2'] = seen_correct
    d['in_fit_base_mean_true_probability'] = seen_true_prob
    d['out_of_fit_base_true_probability'] = held_true_prob
    records = []
    for name, mask in [('all', np.ones(len(d), bool)), ('P_errors', d.pred_P.to_numpy() != y),
                       ('AH_errors', held_pred != y), ('P_new_errors_vs_A0', (d.pred_A0.to_numpy() == y) & (d.pred_P.to_numpy() != y))]:
        for c in (1, 2):
            m = mask & (y == c)
            records.append({'slice': name, 'truth': c, 'rows': int(m.sum()),
                'both_in_fit_models_correct': int((m & (seen_correct == 2)).sum()),
                'one_in_fit_model_correct': int((m & (seen_correct == 1)).sum()),
                'neither_in_fit_model_correct': int((m & (seen_correct == 0)).sum()),
                'in_fit_mean_true_probability': float(seen_true_prob[m].mean()) if m.any() else None,
                'out_of_fit_mean_true_probability': float(held_true_prob[m].mean()) if m.any() else None})
    new_m = d[(y == 1) & (d.pred_A0 == 1) & (d.pred_P != 1)]
    cases = []
    for row in new_m.drop_duplicates('local').itertuples():
        k = int(row.fold); z = np.load(PARENT / f'fold{k}_base_member_logits.npy')[row.local]
        rr = np.load(PARENT / f'fold{k}_P/epoch50_residual.npy')[row.local]
        bias = json.loads((PARENT / f'fold{k}_K/fit.json').read_text())['bias']
        cases.append({'local': int(row.local), 'root': int(row.root), 'fold': k,
            'row_count': int((new_m.local == row.local).sum()),
            'facts': json.loads(trace.loc[trace.local == row.local, 'facts_json'].iloc[0]),
            'AH_probability': softmax(z, -1).mean(0).tolist(),
            'K_probability': softmax(z + bias, -1).mean(0).tolist(),
            'P_probability': softmax(z + rr, -1).mean(0).tolist(),
            'both_in_fit_base_correct': bool(row.in_fit_base_models_correct_count_of_2 == 2)})
    d.to_parquet(OUT / 'score_role_and_error_ledger.parquet', index=False)
    result = {'status': 'posthoc_same_row_exposure_audit_no_fit', 'classifier_fits': 0, 'optimizer_steps': 0,
        'projections': projections, 'score_exposure': records, 'new_M_cases': cases,
        'role_limitations': [
            'Different base models also see different training populations; this is not a randomized causal ablation of in-sample fitting.',
            'The two in-fit base models include the current outer-held fold elsewhere. These probabilities MUST NOT be used to train a corrector for that outer fold.',
            'Raw opaque port strings can distinguish labels without encoding real ports. Zero observed collision does not establish useful transferable information.'
        ],
        'source_sha256': sha(__file__),
        'inputs_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in (OUT/'audit.json', TRACE, OFFICIAL, PARENT/'delivery.json')},
        'scope': 'No updated model or selected threshold. V128 only diagnoses development data; latest actual training V127.'}
    write(target, result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
