"""Register the corrected v9.9 full-input-isolated source experiment.

No classifier is fitted here. V and historic held-out labels are not exported
to fitters. The original message and span ledger remain in the frozen inputs.
"""
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse

from run_v75 import ROOT, OUT, read, save, sha, load_sparse
from v75_views import BYTE_FEATURES, byte_matrix
from v75_corrective import stable
from v89_common import LAST
from v99_normalization_feasibility import CLUSTER, normalize


DEST = ROOT / 'artifacts/v101_full_input_group_n1_20260928'
SEED = 9901
V92 = ROOT / 'artifacts/v92_evidence_training_20260928'
V95 = ROOT / 'artifacts/v95_cross_component_training_20260928'
V99 = ROOT / 'artifacts/v99_evidence_training_plan_20260928'


def emit(**facts):
    print(json.dumps(facts, ensure_ascii=False), flush=True)


def _hash_input(x):
    x = x.tocsr(copy=True)
    x.eliminate_zeros()
    x.sort_indices()
    result = []
    for i in range(x.shape[0]):
        a, b = x.indptr[i:i+2]
        h = hashlib.sha256(x.indices[a:b].astype('<i4').tobytes())
        h.update(x.data[a:b].astype('<f4').tobytes())
        result.append(h.digest())
    return result


def main():
    if DEST.exists():
        raise FileExistsError(DEST)
    plan = read(V99/'next_training_contract.json')
    assert plan['status'] == 'revised_plan_not_trained'
    started = time.monotonic()
    r = pd.read_parquet(OUT/'rows.parquet', columns=[
        'row_position', 'new_text_id', 'component', 'route', 'label_index'])
    assert np.array_equal(r.row_position.to_numpy(), np.arange(len(r)))
    roles = pd.read_parquet(V95/'full_format_roles.parquet', columns=['role']).role.to_numpy()
    fid = np.load(LAST/'row_feature_id.npy', mmap_mode='r')
    assert len(r) == len(roles) == len(fid) == 2056871
    source = np.isin(roles, ['A', 'B'])
    asa = r.route.eq('asa').to_numpy()
    asa_rows = np.flatnonzero(asa)
    ids = np.load(V92/'ASA_input_ids.npy')
    old_asa = sparse.load_npz(V92/'ASA_R0.npz')
    assert old_asa.shape == (len(ids), 66287)
    lookup = np.full(int(fid.max())+1, -1, np.int32)
    lookup[ids] = np.arange(len(ids))
    assert (lookup[fid[asa_rows]] >= 0).all()

    # Every observed ASA text/input pair is audited. Byte 1/2 grams can give
    # the same old R0 to different text, so no one-text-per-R0 assumption.
    pairs = pd.DataFrame({'text_id': r.new_text_id.iloc[asa_rows].to_numpy(),
                          'fid': fid[asa_rows]}).drop_duplicates().reset_index(drop=True)
    dictionary = pd.read_parquet(OUT/'text_dictionary.parquet').set_index('text_id').text
    before = [stable(dictionary.loc[int(t)]) for t in pairs.text_id]
    after = [normalize(t, 'placeholder_cluster') for t in before]
    assert all(normalize(t, 'placeholder_cluster') == t for t in after)
    original = old_asa[lookup[pairs.fid.to_numpy()]]
    present_bytes = byte_matrix(before)
    mismatch = present_bytes-original[:, :BYTE_FEATURES]
    assert mismatch.nnz == 0 or abs(mismatch.data).max() < 1e-7
    proposed = sparse.hstack([byte_matrix(after), original[:, BYTE_FEATURES:]], format='csr')
    assert (proposed[:, BYTE_FEATURES:]-original[:, BYTE_FEATURES:]).nnz == 0
    hashes = _hash_input(proposed)
    group, _ = pd.factorize(hashes, sort=False)
    order = np.argsort(group, kind='stable')
    for a, b in zip(order[:-1], order[1:]):
        if group[a] == group[b]:
            assert (proposed[a]-proposed[b]).nnz == 0
    # If the old representation merged two strings that N1 separates, the
    # feature ID needs splitting. Do not silently pick one representative.
    paired = pd.DataFrame({'fid': pairs.fid, 'group': group})
    assert paired.groupby('fid').group.nunique().max() == 1
    rep = paired.drop_duplicates('fid').index.to_numpy()
    normalized_asa = old_asa.copy().tolil()
    for index in rep:
        local = int(lookup[int(pairs.fid.iat[index])])
        normalized_asa[local, :BYTE_FEATURES] = proposed[index, :BYTE_FEATURES]
    normalized_asa = normalized_asa.tocsr()
    normalized_asa.eliminate_zeros()
    normalized_asa.sort_indices()
    assert (normalized_asa[:, BYTE_FEATURES:]-old_asa[:, BYTE_FEATURES:]).nnz == 0

    pair_id = {(int(t), int(f)): i for i, (t, f) in enumerate(zip(pairs.text_id, pairs.fid))}
    asa_pair = np.fromiter((pair_id[(int(t), int(f))] for t, f in
        zip(r.new_text_id.iloc[asa_rows], fid[asa_rows])), dtype=np.int32, count=len(asa_rows))
    asa_group = group[asa_pair]
    old_map = pd.read_parquet(V99/'placeholder_cluster_group_map.parquet')
    assert np.array_equal(old_map.row_position.to_numpy(), asa_rows)
    assert np.array_equal(old_map.new_group.to_numpy(), asa_group)

    # Span provenance: source text ID links to the raw immutable row ledger;
    # each rewritten span is recorded in the stable-text coordinate system.
    row_first = pd.Series(asa_rows).groupby(asa_pair, sort=False).first()
    raw_ledger = pd.read_parquet(OUT/'raw_ledger.parquet', columns=['row_position', 'raw_sha256'])
    assert np.array_equal(raw_ledger.row_position.to_numpy(), np.arange(len(r)))
    provenance = []
    for i, (text_id, f) in enumerate(zip(pairs.text_id, pairs.fid)):
        pos = int(row_first.loc[i])
        spans = [[m.start(), m.end(), m.group()] for m in CLUSTER.finditer(before[i])]
        provenance.append({'pair_id': i, 'representative_row_position': pos,
            'raw_sha256': raw_ledger.raw_sha256.iat[pos], 'text_id': int(text_id),
            'old_R0': int(f), 'before_sha256': hashlib.sha256(before[i].encode()).hexdigest(),
            'after_sha256': hashlib.sha256(after[i].encode()).hexdigest(),
            'rewritten_stable_text_spans_json': json.dumps(spans, ensure_ascii=False),
            'before_length': len(before[i]), 'after_length': len(after[i])})

    # Group solely by source component and identical observed inputs across
    # ALL formats. The first execution isolated ASA inputs but allowed 4,876
    # repeated non-ASA R0 IDs to cross folds. That execution is retained as
    # an invalid pilot; this correction binds the full teacher exposure.
    comps = r.component.to_numpy()
    source_components = np.unique(comps[source])
    component_to_index = {int(c): i for i, c in enumerate(source_components)}
    parent = np.arange(len(source_components), dtype=np.int32)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    first = {}
    for c, g in zip(comps[asa_rows[source[asa_rows]]], asa_group[source[asa_rows]]):
        ci = component_to_index[int(c)]
        if int(g) not in first:
            first[int(g)] = ci
        else:
            a, b = find(ci), find(first[int(g)])
            if a != b: parent[max(a, b)] = min(a, b)
    # The unchanged R0 representation of non-ASA rows can repeat across
    # independent old components. Union those too, without inspecting labels.
    input_components = pd.DataFrame({'fid': fid[source], 'component': comps[source]}).drop_duplicates()
    shared = input_components.groupby('fid', sort=False).component.agg(list)
    for cells in shared[shared.map(len) > 1]:
        lead = component_to_index[int(cells[0])]
        for component in cells[1:]:
            a, b = find(lead), find(component_to_index[int(component)])
            if a != b: parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(parent))], np.int32)
    folds = np.array([int(hashlib.sha256(f'{SEED}:{int(source_components[root])}'.encode()).hexdigest()[:8], 16) % 3
                      for root in roots], np.int8)
    source_fold = np.full(len(r), -1, np.int8)
    source_fold[source] = folds[np.searchsorted(source_components, comps[source])]
    assert np.all(source_fold[~source] == -1)
    assert pd.DataFrame({'component': comps[source], 'fold': source_fold[source]}).groupby('component').fold.nunique().max() == 1
    assert pd.DataFrame({'fid': fid[source], 'fold': source_fold[source]}).groupby('fid').fold.nunique().max() == 1
    assert pd.DataFrame({'group': asa_group[source[asa_rows]],
                         'fold': source_fold[asa_rows[source[asa_rows]]]}).groupby('group').fold.nunique().max() == 1
    # All three folds must have enough independent M and S support to make
    # a per-class metric meaningful. Do not reroll a bad seed.
    class_rows = {}
    for k in range(3):
        mask = source & (source_fold == k)
        class_rows[str(k)] = np.bincount(r.label_index.to_numpy()[mask], minlength=3).astype(int).tolist()
        assert class_rows[str(k)][1] > 0 and class_rows[str(k)][2] > 0

    DEST.mkdir(parents=True)
    sparse.save_npz(DEST/'N1_ASA.npz', normalized_asa, compressed=True)
    difference = normalized_asa-old_asa
    aidx = np.repeat(ids, np.diff(difference.indptr))
    delta = sparse.csr_matrix((difference.data, (aidx, difference.indices)),
                              shape=(int(fid.max())+1, old_asa.shape[1]))
    sparse.save_npz(DEST/'N1_delta.npz', delta, compressed=True)
    np.save(DEST/'source_fold.npy', source_fold)
    pd.DataFrame(provenance).to_parquet(DEST/'span_provenance.parquet', index=False)
    pd.DataFrame({'row_position': asa_rows, 'old_R0': fid[asa_rows],
                  'N1_group': asa_group}).to_parquet(DEST/'N1_ASA_group_map.parquet', index=False)
    inputs = [ROOT/'data/official/train.parquet', OUT/'rows.parquet', OUT/'text_dictionary.parquet',
              OUT/'raw_ledger.parquet', LAST/'row_feature_id.npy', V92/'ASA_input_ids.npy',
              V92/'ASA_R0.npz', V95/'full_format_roles.parquet',
              V99/'next_training_contract.json', V99/'placeholder_cluster_group_map.parquet']
    receipt = {'status': 'registered_before_fits', 'seed': SEED, 'plan': plan,
        'source_rows': int(source.sum()), 'ASA_source_rows': int((source & asa).sum()),
        'official_rows_preserved': len(r), 'source_class_rows_by_fold': class_rows,
        'source_component_count': len(source_components),
        'union_component_count': int(len(np.unique(roots))),
        'all_input_cross_fold_groups': 0,
        'normalized_input_group_count': int(len(np.unique(asa_group))),
        'text_input_pair_count': len(pairs), 'mapped_spans': int(sum(len(CLUSTER.findall(t)) for t in before)),
        'facts_metadata_difference_nnz': 0, 'new_delta_nnz': int(delta.nnz),
        'split_old_R0_by_N1': 0, 'V_and_locked_rows_in_fit': 0,
        'source_sha256': sha(__file__),
        'input_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in inputs},
        'output_sha256': {p.name: sha(p) for p in DEST.iterdir() if p.is_file()},
        'seconds': time.monotonic()-started}
    save(DEST/'registration.json', receipt)
    emit(stage='prepared', source_rows=receipt['source_rows'], class_rows_by_fold=class_rows,
         normalized_groups=receipt['normalized_input_group_count'],
         union_components=receipt['union_component_count'], seconds=round(receipt['seconds'], 1))


if __name__ == '__main__':
    main()
