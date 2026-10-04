"""Refresh train-only parser/support evidence. No fits, no validation answers."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from v73_inference import load_bound, sha

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / 'artifacts/v74_train_support_audit_20260921'
    if out.exists():
        raise FileExistsError(out)
    out.mkdir()
    _, adapter, _, bindings = load_bound()
    train = ROOT / 'data/official/train.parquet'
    prepared = ROOT / 'artifacts/v39_local_r2_20260913/prepared/rows.parquet'
    manifest = ROOT / 'artifacts/v51_fact_residual_20260914/pressure/training_manifest.parquet'
    old = pd.read_parquet(prepared, columns=['row_position', 'event_id', 'route', 'label_index'])
    fit = set(pd.read_parquet(manifest, columns=['row_position']).row_position)
    parts = []
    offset = 0
    start = time.monotonic()
    names = ['benign', 'malicious', 'suspicious']
    for batch in pq.ParquetFile(train).iter_batches(batch_size=16384,
            columns=['event_id', 'message_sanitized', 'src_ip', 'label_binary'], use_threads=False):
        df = batch.to_pandas()
        inverse, unique = pd.factorize(df.message_sanitized.fillna('').astype(str), sort=False)
        parsed = [adapter.prepare_record({'message_sanitized': s}) for s in unique]
        routes = np.array([p['route'] for p in parsed])[inverse]
        z = old.iloc[offset:offset + len(df)]
        np.testing.assert_array_equal(z.event_id, df.event_id)
        np.testing.assert_array_equal(z.label_index, df.label_binary.map(dict(zip(names, range(3)))))
        positions = np.arange(offset, offset + len(df))
        parts.append(pd.DataFrame({
            'route': routes, 'old_route': z.route.to_numpy(),
            'label': df.label_binary.to_numpy(), 'source': df.src_ip.to_numpy(),
            'actual_fit': [int(i) in fit for i in positions],
            'raw_nonempty': df.message_sanitized.fillna('').str.strip().ne('').to_numpy(),
            'text_view_nonempty': np.array([bool(p['text'].strip()) for p in parsed])[inverse],
        }))
        offset += len(df)
        if offset % 262144 == 0:
            print(json.dumps({'rows': offset, 'seconds': round(time.monotonic()-start, 1)}), flush=True)
    d = pd.concat(parts, ignore_index=True)
    assert len(d) == len(old) == pq.ParquetFile(train).metadata.num_rows
    support = []
    for (route, label), z in d.groupby(['route', 'label'], sort=True):
        support.append({'route': route, 'label': label, 'official_train_rows': len(z),
            'actual_fit_rows': int(z.actual_fit.sum()),
            'source_symbols': int(z.source.replace('', np.nan).nunique()),
            'nonempty_messages': int(z.raw_nonempty.sum()),
            'nonempty_text_views': int(z.text_view_nonempty.sum())})
    result = {'new_fits': 0, 'answers_read': False, 'rows': len(d),
        'scope': 'All official train rows freshly routed by frozen v48 input adapter; source symbols are not independent incidents; empty text view does not imply empty structured input.',
        'official_label_counts': d.label.value_counts().to_dict(),
        'route_mismatches_vs_prepared': int(d.route.ne(d.old_route).sum()),
        'support': support, 'seconds': time.monotonic()-start,
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in (train, prepared, manifest)},
        'runtime_bindings': bindings, 'source_sha256': sha(__file__)}
    for p, h in bindings.items():
        assert sha(ROOT / p) == h
    (out/'support.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['rows', 'new_fits', 'answers_read', 'route_mismatches_vs_prepared', 'official_label_counts', 'seconds']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
