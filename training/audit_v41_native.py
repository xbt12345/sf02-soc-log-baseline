"""Label-free full allowed Windows coverage audit before any v41 fitting."""
import argparse
import collections
import json
import time
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import v41_native_auth as native
from run_v39_prepare import sha, save


def main(a):
    start = time.perf_counter(); root = Path(a.root); out = Path(a.out)
    assert not out.exists(); out.mkdir(parents=True)
    prep = root/'artifacts/v39_local_r2_20260913/prepared'
    rows = pq.read_table(prep/'rows.parquet', columns=['row_position','projection_id','route','fold']).to_pandas()
    pr = pq.read_table(prep/'projections.parquet', columns=['text','facts']).to_pandas()
    facts = [json.loads(v) for v in pr.facts]
    allowed = (rows.fold.to_numpy()>=0) & rows.route.isin(['windows_message','windows_rendered']).to_numpy()
    known = np.array([f.get('auth_result') in ('success','failure') for f in facts])[rows.projection_id.to_numpy()]
    counts = collections.Counter(); reasons = collections.Counter(); updates = []; observations = []; examples = {}
    old_to_new = collections.defaultdict(set); native_scope_rows = []
    offset = 0; parsed = 0
    for batch in pq.ParquetFile(root/'data/official/train.parquet').iter_batches(batch_size=8192, columns=['message_sanitized'], use_threads=False):
        idx = np.flatnonzero(allowed[offset:offset+len(batch)])
        for k in idx:
            pos = int(offset+k); pid = int(rows.projection_id.iloc[pos]); raw = batch.column(0)[int(k)].as_py() or ''
            counts['allowed_windows_rows_scanned'] += 1
            candidate = known[pos] or any(title in raw.casefold() for title in native.old.AUTH_TITLES)
            if not candidate:
                counts['no_known_auth_or_native_title'] += 1
                continue
            parsed += 1; p = native.extract(raw); counts['state:'+p['state']] += 1
            counts['legacy_recognized_auth'] += int(known[pos]); reasons[str(p['reason'])] += 1
            if p['native_title']:
                native_scope_rows.append(pos)
                counts['native_title:'+p['native_title']] += 1
            if p['state'] == 'observed':
                for f in p['facts']: counts['observed:'+f] += 1
                counts['multiple_bounded_carriers'] += int(len(p['carriers'])>len(p['facts']))
                counts['unbounded_duplicate_candidates_present'] += int(p['unbounded_candidate_fields']>0)
                observations.append({'row_position': pos, 'old_projection_id': pid, 'facts': native.prior.canonical(p['facts']),
                                     'evidence': native.prior.canonical(p)})
            base = {'text': pr.text.iloc[pid], 'facts': facts[pid]}; changed = native.repair(base, p)
            key = native.prior.canonical([changed['text'], changed['facts']]); old_to_new[pid].add(key)
            if changed['text'] != base['text'] or changed['facts'] != base['facts']:
                updates.append({'row_position': pos, 'old_projection_id': pid, 'text': changed['text'], 'facts': native.prior.canonical(changed['facts'])})
            example_key = p['state']+':'+str(p['reason'])
            if example_key not in examples:
                examples[example_key] = {'row_position': pos, 'raw': raw, 'observation': p, 'baseline': base, 'repaired': changed}
        offset += len(batch)
        if offset % (8192*32) == 0:
            print(json.dumps({'raw_rows_scanned': offset, 'native_candidates_parsed': parsed, 'changed_rows': len(updates)}), flush=True)
    assert offset == len(rows) == 2056871
    assert counts['allowed_windows_rows_scanned'] == int(allowed.sum()) == 962244
    assert counts['legacy_recognized_auth'] == int((allowed&known).sum()) == 95278
    for name, data, columns in [('native_updates.parquet',updates,['row_position','old_projection_id','text','facts']),
                              ('native_observations.parquet',observations,['row_position','old_projection_id','facts','evidence'])]:
        schema = pa.schema([(c, pa.int64() if c in ('row_position','old_projection_id') else pa.string()) for c in columns])
        pq.write_table(pa.Table.from_pylist(data,schema=schema),out/name,compression='zstd')
    pq.write_table(pa.table({'row_position':pa.array(native_scope_rows,type=pa.int64())}),out/'native_scope.parquet',compression='zstd')
    save(out/'examples.json',examples)
    diagnostic=json.loads((root/'evidence/2026-09-13/v41_planning/native_status_probe.json').read_text(encoding='utf-8'))
    target={r['row_position'] for r in diagnostic['records']}
    observed={r['row_position'] for r in observations}
    checks={'all_18_previous_regressions_have_bounded_observations':target<=observed,
            'all_recognized_windows_auth_rows_audited':counts['legacy_recognized_auth']==95278,
            'all_allowed_windows_rows_scanned':counts['allowed_windows_rows_scanned']==962244,
            'no_label_used_for_extraction':True}
    save(out/'audit.json',{'counts':dict(counts),'reasons':dict(reasons),'changed_rows':len(updates),'native_scope_rows':len(native_scope_rows),
        'old_projection_ids_splitting_within_audited_candidates':sum(len(v)>1 for v in old_to_new.values()),
        'checks':checks,'A_boundary_ready':all(checks.values()),'seconds':time.perf_counter()-start,
        'scope':'All allowed Windows rows screened; all legacy-recognized or literal-native-title candidates parsed. No labels used. Unknown/conflict uses unchanged B fallback; no claim of recovering damaged native parents.',
        'files':{p.name:sha(p) for p in out.iterdir() if p.is_file()},
        'source_sha256':{n:sha(Path(__file__).parent/n) for n in ['audit_v41_native.py','v41_native_auth.py']}})
    print(json.dumps({'counts':dict(counts),'changed_rows':len(updates),'checks':checks,'seconds':round(time.perf_counter()-start)}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);main(p.parse_args())
