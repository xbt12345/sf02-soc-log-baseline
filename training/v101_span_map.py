"""Trace each N1 replacement back to exact sanitized-raw intervals."""
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from run_v75 import ROOT, OUT, save, sha
from v75_views import view
from v75_corrective import EMBEDDED, stable
from v99_normalization_feasibility import CLUSTER, normalize
from v101_prepare import DEST


def view_with_map(raw):
    text, ledger = view(raw)
    result = []
    mapping = []
    for a, b, kind in ledger['spans']:
        if kind == 'behavior_or_unresolved':
            piece = raw[a:b]
            mapping.extend((i, i+1) for i in range(a, b))
        else:
            piece = ' <'+kind.upper()+'> '
            mapping.extend([(a, b)]*len(piece))
        result.append(piece)
    assert ''.join(result) == text and len(mapping) == len(text)
    return text, ledger, mapping


def replace_with_map(text, mapping, pattern, replacement):
    chunks = []
    newmap = []
    traces = []
    cursor = 0
    for match in pattern.finditer(text):
        a, b = match.span()
        chunks.append(text[cursor:a]); newmap.extend(mapping[cursor:a])
        intervals = sorted(set(mapping[a:b]))
        merged = []
        for left, right in intervals:
            if merged and left <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], right)
            else:
                merged.append([left, right])
        assert merged
        chunks.append(replacement)
        newmap.extend([(merged[0][0], merged[-1][1])]*len(replacement))
        traces.append((a, b, merged))
        cursor = b
    chunks.append(text[cursor:]); newmap.extend(mapping[cursor:])
    output = ''.join(chunks)
    assert len(output) == len(newmap)
    return output, newmap, traces


def main():
    target = DEST/'N1_raw_span_map.parquet'
    if target.exists():
        raise FileExistsError(target)
    provenance = pd.read_parquet(DEST/'span_provenance.parquet')
    by_position = dict(zip(provenance.representative_row_position.to_numpy(),
                           provenance.itertuples(index=False)))
    records = []
    checked = 0
    offset = 0
    for batch in pq.ParquetFile(ROOT/'data/official/train.parquet').iter_batches(
            batch_size=8192, columns=['message_sanitized'], use_threads=False):
        for local, raw in enumerate(batch.column(0).to_pylist()):
            info = by_position.get(offset+local)
            if info is None:
                continue
            text, ledger, mapping = view_with_map(raw)
            assert ledger['sha256'] == info.raw_sha256
            before, mapping, _ = replace_with_map(text, mapping, EMBEDDED, ' <IDENTITY> ')
            assert before == stable(text)
            assert hashlib.sha256(before.encode()).hexdigest() == info.before_sha256
            after, mapping, traces = replace_with_map(before, mapping, CLUSTER, ' <IDENTITY> ')
            assert after == normalize(before, 'placeholder_cluster')
            assert hashlib.sha256(after.encode()).hexdigest() == info.after_sha256
            for a, b, intervals in traces:
                assert 0 <= intervals[0][0] < intervals[-1][1] <= len(raw)
                records.append({'pair_id': int(info.pair_id),
                    'representative_row_position': int(info.representative_row_position),
                    'raw_sha256': info.raw_sha256, 'stable_text_start': a,
                    'stable_text_end': b, 'raw_intervals_json': json.dumps(intervals),
                    'raw_covering_start': intervals[0][0],
                    'raw_covering_end': intervals[-1][1],
                    'raw_covering_sha256': hashlib.sha256(
                        raw[intervals[0][0]:intervals[-1][1]].encode()).hexdigest()})
            checked += 1
        offset += len(batch)
    assert checked == len(provenance) and offset == 2056871
    assert len(records) > checked
    pd.DataFrame(records).to_parquet(target, index=False)
    receipt = {'status': 'raw_to_stable_to_N1_span_chain_verified',
        'representative_text_input_pairs': checked, 'N1_replacements_mapped': len(records),
        'raw_training_rows_scanned': offset, 'raw_span_map_sha256': sha(target),
        'source_sha256': sha(__file__), 'facts_not_modified': True,
        'scope': 'Exact sanitized-raw source intervals for each N1 regex replacement; pre-redaction values do not exist in available input.'}
    save(DEST/'span_map_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
