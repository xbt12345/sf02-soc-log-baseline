"""Export only A+B supervision and prove raw-to-inference N1 equivalence."""
import hashlib
import json
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_v75 import ROOT, OUT, read, save, sha
from v75_views import view
from v75_corrective import stable
from v99_normalization_feasibility import normalize
from v101_prepare import DEST, V95, LAST


def main():
    target = DEST/'source_rows.parquet'
    if target.exists():
        raise FileExistsError(target)
    registration = read(DEST/'registration.json')
    assert registration['status'] == 'registered_before_fits'
    for name, digest in registration['output_sha256'].items():
        assert sha(DEST/name) == digest, name
    r = pd.read_parquet(OUT/'rows.parquet', columns=[
        'row_position', 'new_text_id', 'component', 'route', 'label_index'])
    role = pd.read_parquet(V95/'full_format_roles.parquet', columns=['role']).role.to_numpy()
    fold = np.load(DEST/'source_fold.npy')
    fid = np.load(LAST/'row_feature_id.npy', mmap_mode='r')
    source = np.isin(role, ['A', 'B'])
    assert np.array_equal(source, fold >= 0)
    src = pd.DataFrame({'row_position': r.row_position.to_numpy()[source],
        'fid': fid[source], 'label_index': r.label_index.to_numpy()[source],
        'component': r.component.to_numpy()[source], 'fold': fold[source],
        'is_ASA': r.route.eq('asa').to_numpy()[source]})
    assert len(src) == registration['source_rows']
    assert src.groupby('component').fold.nunique().max() == 1

    # Check the production transformation from actual sanitized raw messages,
    # not merely from an already transformed dictionary. One representative
    # per ASA text/input pair is sufficient for this functional-equivalence
    # test; all original rows remain in the source population.
    provenance = pd.read_parquet(DEST/'span_provenance.parquet')
    positions = dict(zip(provenance.representative_row_position.to_numpy(),
                         provenance.itertuples(index=False)))
    dictionary = pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    raw = ROOT/'data/official/train.parquet'
    checked = 0
    offset = 0
    start = time.monotonic()
    for batch in pq.ParquetFile(raw).iter_batches(batch_size=8192,
            columns=['message_sanitized'], use_threads=False):
        values = batch.column(0).to_pylist()
        for local, value in enumerate(values):
            info = positions.get(offset+local)
            if info is None:
                continue
            text, ledger = view(value)
            assert ledger['sha256'] == info.raw_sha256
            assert text == dictionary.loc[int(info.text_id)]
            before = stable(text)
            actual = normalize(before, 'placeholder_cluster')
            assert hashlib.sha256(before.encode()).hexdigest() == info.before_sha256
            assert hashlib.sha256(actual.encode()).hexdigest() == info.after_sha256
            assert normalize(actual, 'placeholder_cluster') == actual
            checked += 1
        offset += len(values)
    assert offset == len(r) and checked == len(provenance)
    # N1 must canonically remove alternative wrapper spellings around an
    # existing placeholder while leaving the real numeric/protocol text.
    variants = 0
    for item in provenance.itertuples(index=False):
        parent = stable(dictionary.loc[int(item.text_id)])
        canonical = normalize(parent, 'placeholder_cluster')
        if '<IDENTITY>' not in canonical:
            continue
        for candidate in [' CRED-<IDENTITY> ', '\tHOST-  <IDENTITY>\t', ' USER-CRED-<IDENTITY> ']:
            artificial = canonical.replace(' <IDENTITY> ', candidate, 1)
            assert normalize(artificial, 'placeholder_cluster') == canonical
            variants += 1
    src.to_parquet(target, index=False)
    receipt = {'status': 'source_export_and_inference_equivalence_passed',
        'source_rows': len(src), 'official_rows_read_for_check': offset,
        'ASA_distinct_text_input_pairs_checked_from_raw': checked,
        'canonical_wrapper_variants_checked': variants,
        'other_role_labels_exported': 0,
        'source_sha256': sha(__file__), 'source_rows_sha256': sha(target),
        'seconds': time.monotonic()-start}
    save(DEST/'source_export.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
