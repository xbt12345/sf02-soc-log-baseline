"""Zero-fit factorial interface projection audit; never relabel events."""
import json
import re
import numpy as np
import pandas as pd
from scipy import sparse
from run_v75 import ROOT, save, sha
from v75_views import BYTE_FEATURES, byte_matrix, matrix_hashes
from v112_fine_control_preflight import DEST, N1, floor


def main():
    target = DEST / 'factor_audit.json'
    assert not target.exists()
    pre = json.loads((DEST / 'preflight.json').read_text(encoding='utf-8'))
    assert all(sha(DEST / p) == h for p, h in pre['output_hashes'].items())
    views = pd.read_parquet(DEST / 'interface_views.parquet').sort_values('local')
    rows = pd.read_parquet(DEST / 'interface_row_audit.parquet')
    x = sparse.load_npz(N1)
    assert np.array_equal(views.local, np.arange(x.shape[0]))
    output = rows[['row_position', 'local', 'root', 'fold', 'truth']].copy()
    results = []
    variants = {
        'lowercase_only_keep_suffix': lambda s: s.lower(),
        'remove_numeric_suffix_keep_case': lambda s: re.sub(r'[-_]\d+$', '', s),
        'lowercase_and_remove_suffix': lambda s: re.sub(r'[-_]\d+$', '', s.lower()),
    }
    for name, transform in variants.items():
        texts = []
        for v in views.itertuples(index=False):
            s = v.before
            for a, b, _ in reversed(json.loads(v.name_spans)):
                s = s[:a] + transform(s[a:b]) + s[b:]
            texts.append(s)
        nx = sparse.hstack([byte_matrix(texts), x[:, BYTE_FEATURES:]], format='csr')
        assert (nx[:, BYTE_FEATURES:] != x[:, BYTE_FEATURES:]).nnz == 0
        h = matrix_hashes(nx)
        output[name] = np.asarray(h, dtype=object)[rows.local.to_numpy()]
        z = rows.copy()
        z['variant_hash'] = output[name]
        collapsed = z.groupby('variant_hash').original_full_input_hash.nunique()
        cross = z.groupby('variant_hash').fold.nunique()
        changed = np.asarray(texts) != views.before.to_numpy()
        results.append({
            'variant': name, 'observed_floor': floor(z, 'variant_hash'),
            'changed_unique_inputs': int(changed.sum()),
            'changed_event_rows': int(changed[rows.local.to_numpy()].sum()),
            'new_collapsed_groups': int((collapsed > 1).sum()),
            'cross_current_fold_groups': int((cross > 1).sum()),
            'rows_in_cross_fold_groups': int(z.variant_hash.isin(cross[cross > 1].index).sum()),
        })
    output.to_parquet(DEST / 'factor_hashes.parquet', index=False)
    # How often does plain text literally agree but the full encoding differs?
    # This does not grant the transformed input a label-preserving interpretation.
    mixed = rows.groupby('isolated_full_input_hash').truth.nunique()
    detail = rows[rows.isolated_full_input_hash.isin(mixed[mixed > 1].index)].copy()
    detail['isolated_group'] = detail.isolated_full_input_hash.map(lambda h: h.hex())
    groups = detail.groupby(['isolated_group', 'interface_literal_pair', 'truth']).agg(
        rows=('row_position', 'size'), roots=('root', 'nunique'),
        inputs=('local', 'nunique')).reset_index()
    groups.to_csv(DEST / 'isolated_conflict_name_composition.csv', index=False, encoding='utf-8')
    report = {
        'status': 'no_fit_factorial_information_audit', 'classifier_fits': 0, 'calibration_fits': 0,
        'source_sha256': sha(__file__),
        'input_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in
            [N1, DEST / 'preflight.json', DEST / 'interface_views.parquet', DEST / 'interface_row_audit.parquet']},
        'original_floor': pre['original_floor'], 'variants': results,
        'output_hashes': {p.name: sha(p) for p in [DEST / 'factor_hashes.parquet', DEST / 'isolated_conflict_name_composition.csv']},
        'limits': ['No-fit feature collision counts; no prediction or quality improvement measured.',
            'No added collision does not establish security-semantic invariance.',
            'Interface spans are in existing N1-normalized text, not offsets into original raw logs.',
            'Subzone suffixes may carry context; changes are diagnostic, not approved augmentations.'],
    }
    save(target, report)
    print(json.dumps({'original_floor': pre['original_floor'], 'variants': results}, indent=2))


if __name__ == '__main__':
    main()
