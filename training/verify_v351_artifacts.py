"""Independent partition verification of v3.5.1's sparse patch and union."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/2026-09-12/v351_input_review/verified'


def assert_same_partition(a, b):
    for source, target in ((a, b), (b, a)):
        _, inverse = np.unique(source, return_inverse=True)
        lo = np.full(int(inverse.max())+1, np.iinfo(np.int64).max, dtype=np.int64)
        hi = np.full(len(lo), -1, dtype=np.int64)
        np.minimum.at(lo, inverse, target)
        np.maximum.at(hi, inverse, target)
        assert np.array_equal(lo, hi)


def run():
    old = pq.read_table(ROOT/'artifacts/v331_ready_20260912_r2/group_manifest.parquet')
    new = pq.read_table(OUT/'candidate_group_manifest.parquet')
    patch = pq.read_table(OUT/'candidate_text_patch.parquet')
    pos = patch['row_position'].to_numpy()
    assert len(np.unique(pos)) == len(pos) == 47044
    for col in ('row_position', 'label_index', 'informative'):
        assert np.array_equal(old[col].to_numpy(), new[col].to_numpy())
    text = pq.read_table(ROOT/'artifacts/v331_ready_20260912_r2/prepared_corpus.parquet', columns=['text'])['text'].combine_chunks()
    before = pc.take(text, pa.array(pos)).to_pylist()
    assert all(hashlib.sha256(s.encode()).hexdigest() == h for s, h in zip(before, patch['old_text_sha256'].to_pylist()))
    enc = pc.dictionary_encode(text)
    codes = enc.indices.to_numpy().astype(np.int64)
    # Re-encode strings independently, without the audit's matching/DSU logic.
    replacement = pc.index_in(patch['text'], value_set=enc.dictionary).to_pylist()
    novel, next_code = {}, len(enc.dictionary)
    for i, value in enumerate(replacement):
        if value is None:
            s = patch['text'][i].as_py()
            if s not in novel:
                novel[s] = next_code
                next_code += 1
            value = novel[s]
        codes[pos[i]] = value
    informative = old['informative'].to_numpy()
    assert_same_partition(codes[informative], new['candidate_text_group'].to_numpy()[informative])
    # Independent graph connected-components check proves both compatibility and
    # minimality of the union; no unsupported extra group merges are accepted.
    a = old['group_id'].to_numpy().astype(np.int64)
    _, b = np.unique(new['candidate_text_group'].to_numpy(), return_inverse=True)
    n_a, n_b = int(a.max())+1, int(b.max())+1
    graph = coo_matrix((np.ones(len(a), dtype=np.int32), (a, n_a+b)), shape=(n_a+n_b, n_a+n_b)).tocsr()
    _, labels = connected_components(graph, directed=False)
    assert_same_partition(labels[a], new['old_new_union_group'].to_numpy())
    result = {'all_checks_passed': True, 'rows': len(old), 'patched_rows': len(pos),
              'labels_and_positions_unchanged': True, 'old_text_hashes_match': True,
              'new_nonempty_text_partition_matches_independent_dictionary_encoding': True,
              'union_partition_matches_independent_graph_components': True,
              'scope': 'Artifact equivalence and repeat constraints only, not semantic completeness or model quality'}
    (OUT/'independent_verification.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    run()
