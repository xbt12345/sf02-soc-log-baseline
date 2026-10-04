"""All affected real-clock records, plus exact re-use preconditions."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import v39_core as core
from run_v39_prepare import sha, save


def main(a):
    root = Path(a.root); prepared = Path(a.prepared); previous = Path(a.previous)
    out = Path(a.output)
    if out.exists():
        raise FileExistsError('Preserve audit receipt')
    out.parent.mkdir(parents=True, exist_ok=True)
    old_rows = pq.read_table(previous / 'rows.parquet')
    new_rows = pq.read_table(prepared / 'rows.parquet')
    old = pq.read_table(previous / 'projections.parquet').to_pandas()
    new = pq.read_table(prepared / 'projections.parquet').to_pandas()
    checks = {'all_row_values_identical': old_rows.equals(new_rows),
              'baseline_inputs_identical': old[['baseline_text', 'baseline_facts']].equals(new[['baseline_text', 'baseline_facts']]),
              'semantic_fact_values_identical': old.facts.equals(new.facts)}
    r = new_rows.select(['row_position', 'projection_id']).to_pandas()
    changed_ids = set(new.loc[old.text != new.text, 'projection_id'])
    targeted = set(new.loc[new.audit.str.contains('absolute_lexical_clock|absolute_boundary'), 'projection_id'])
    positions = set(r.loc[r.projection_id.isin(targeted), 'row_position'])
    failures = []; audited = 0; mutations = 0; offset = 0
    for b in pq.ParquetFile(root / 'data/official/train.parquet').iter_batches(batch_size=8192, columns=['message_sanitized'], use_threads=False):
        for pos in sorted(positions.intersection(range(offset, offset + len(b)))):
            raw = b.column(0)[pos - offset].as_py() or ''
            original = core.prepare_message(raw)
            pid = int(r.projection_id.iloc[pos])
            if original['text'] != new.text.iloc[pid] or core.canonical(original['facts']) != new.facts.iloc[pid]:
                failures.append(['cached', pos])
            for clock in ('2099-01-01T00:00:00Z', '2011-12-31T23:59:59Z'):
                modified = core.ISO_LITERAL.sub(clock, raw)
                modified = core.TASK_BOUNDARY.sub(lambda m: m[1] + clock + m[3], modified)
                p = core.prepare_message(modified)
                if (p['text'], p['facts']) != (original['text'], original['facts']):
                    failures.append(['clock', pos, clock])
                mutations += 1
            audited += 1
        offset += len(b)
    checks['all_targeted_clock_records_checked'] = audited == len(positions)
    checks['clock_invariance_passed'] = not failures
    result = {'checks': checks, 'all_checks_passed': all(checks.values()), 'real_records': audited,
        'clock_transformations': mutations, 'failures': failures,
        'changed_semantic_projection_keys': len(changed_ids),
        'changed_semantic_rows_all': int(r.projection_id.isin(changed_ids).sum()),
        'previous_failed_probe': {'runtime': 'v39-controlled-development-1.0', 'real_records': 210, 'failures': 1,
                                  'row_position': 811245, 'cause': 'partially redacted startboundary retained minutes and seconds'},
        'implementation_sha256': sha(Path(core.__file__)), 'new_prepared_receipt_sha256': sha(prepared / 'complete.json'),
        'reusable_views': ['BASELINE_C', 'AVAILABILITY'] if all(checks.values()) else [],
        'scope': 'Input and fold identity plus specified absolute-time invariance; no quality inference'}
    save(out, result); print(json.dumps(result, ensure_ascii=False))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ['root', 'prepared', 'previous', 'output']:
        p.add_argument('--' + name, required=True)
    main(p.parse_args())
