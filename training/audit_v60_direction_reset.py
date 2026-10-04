"""Read-only source audit for a new training design; no fitting or label edits."""
import hashlib
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/2026-09-14/v60_direction_reset'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def collision(frame, key):
    table = pd.crosstab(key, frame.label_index)
    mixed = table.gt(0).sum(axis=1).gt(1)
    return {'keys': len(table), 'mixed_keys': int(mixed.sum()),
            'mixed_rows': int(table.loc[mixed].sum().sum()),
            'finite_sample_minimum_errors': int((table.sum(axis=1) - table.max(axis=1)).sum())}


def main():
    paths = [ROOT / 'artifacts/v55_risk_validation_20260914/rows.parquet',
             ROOT / 'artifacts/v55_risk_validation_20260914/observations.parquet',
             ROOT / 'evidence/2026-09-14/v53_raw_audit/raw_fields.parquet',
             ROOT / 'evidence/2026-09-14/v53_raw_audit/summary.json']
    assert not OUT.exists(), 'Use a new output path; never overwrite an audit.'
    rows, obs, raw = [pd.read_parquet(p) for p in paths[:3]]
    assert len(rows) == len(obs) == 99398
    use = rows.route.eq('asa')
    rows = rows.loc[use].reset_index(drop=True)
    obs = obs.loc[use].reset_index(drop=True)
    raw = raw.set_index('row_position').loc[rows.row_position].reset_index()
    assert len(raw) == 99382
    assert (raw.label_index.to_numpy() == rows.label_index.to_numpy()).all()
    assert raw.row_position.is_unique
    assert rows.inner_role.ne(2).all()
    source_identity = collision(raw, raw.src_address)
    dst_identity = collision(raw, raw.dst_address)
    current = obs.astype(str).agg('|'.join, axis=1)
    # Additional observations: no literal identities, dates, or inferred hidden numbers.
    extra = pd.DataFrame(index=raw.index)
    extra['same_zone'] = raw.src_zone.eq(raw.dst_zone)
    extra['same_address'] = raw.src_address.eq(raw.dst_address)
    for side in ['src', 'dst']:
        tok = raw[side + '_port_token'].fillna('')
        extra[side + '_port_kind'] = tok.map(lambda t: 'absent' if not t else
            'literal_number' if re.fullmatch(r'\d+', t) else 'opaque')
    extra['same_nonempty_port_token'] = (raw.src_port_token.fillna('').ne('') &
                                         raw.src_port_token.eq(raw.dst_port_token))
    pri = pd.to_numeric(raw.header.str.extract(r'^<(\d+)>', expand=False), errors='coerce')
    extra['syslog_severity'] = pri.mod(8).fillna(-1).astype(int)
    extra['acl_syntax'] = raw.aclword.fillna('').str.lower()
    # This is a *candidate* information control, not proof these features are causal.
    extra_key = extra.astype(str).agg('|'.join, axis=1)
    by_source = raw.groupby('src_address').agg(rows=('row_position', 'size'),
        distinct_raw=('raw_sha256', 'nunique'), distinct_body=('body_sha256', 'nunique'),
        distinct_destinations=('dst_address', 'nunique'), labels=('label_index', 'nunique'))
    by_source['duplicate_rows'] = by_source.rows - by_source.distinct_raw
    # Only clearly readable, literal dates. No replacement of USER/CRED tokens with guesses.
    exact = raw.header.str.extract(r'^<\d+>([A-Za-z]{3} \d{1,2} \d{4} \d{2}:\d{2}:\d{2}):', expand=False)
    partial = raw.header.str.extract(r'^<\d+>([A-Za-z]{3} \d{1,2}) (?:\d{4}|USER-\d+) (\d{2}:\d{2}:\d{2}):')
    label_stats = []
    for label, g in raw.groupby('label_index'):
        label_stats.append({'label_index': int(label), 'rows': len(g),
            'unique_raw_messages': int(g.raw_sha256.nunique()),
            'unique_literal_bodies': int(g.body_sha256.nunique()),
            'literal_full_datetime_rows': int(exact.loc[g.index].notna().sum()),
            'readable_month_day_clock_rows': int(partial.loc[g.index].notna().all(axis=1).sum())})
    summary = {
        'scope': 'ASA subset of existing v55 fit-side development rows only. No model fits, source edits, external training data, new test claim, or raw-data rehash.',
        'rows': len(raw), 'original_observation_columns': list(obs.columns),
        'constant_observation_fields': {k: str(obs[k].iloc[0]) for k in obs if obs[k].nunique() == 1},
        'current_collisions': collision(raw, current),
        'current_plus_identity_free_candidates': collision(raw, current + '|' + extra_key),
        'current_plus_literal_zone_diagnostic': collision(raw, current + '|' + raw.src_zone + '|' + raw.dst_zone),
        'literal_body_diagnostic': collision(raw, raw.body_sha256),
        'literal_raw_diagnostic': collision(raw, raw.raw_sha256),
        'source_address_label_association': source_identity,
        'destination_address_label_association': dst_identity,
        'label_statistics': label_stats,
        'sources_with_multiple_distinct_raw_events': int(by_source.distinct_raw.gt(1).sum()),
        'sources_with_multiple_distinct_bodies': int(by_source.distinct_body.gt(1).sum()),
        'sources_with_multiple_distinct_destinations': int(by_source.distinct_destinations.gt(1).sum()),
        'candidate_observation_cardinalities': {k: int(extra[k].nunique()) for k in extra},
        'limits': [
            'Distinct records can be separate samples from a corpus; they do not establish real sessions or incident continuity.',
            'Per-address single-label association is a finite-corpus shortcut warning, not proof every address-based relationship is invalid.',
            'A zero collision floor for literal messages/identities is not learned generalization or an achievable blind-test score.',
            'Full datetime coverage uses an explicit conservative regex; missing values can include alternative or redacted syntax. It does not prove all time information was destroyed.',
            'Unfolded finite-sample floors here must not be directly compared with historical fold-conditional floors.',
            'No audit feature is promoted to a classifier by this script.'
        ],
        'source_bindings': {p.relative_to(ROOT).as_posix(): sha(p) for p in paths + [Path(__file__)]}
    }
    OUT.mkdir(parents=True)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    pd.crosstab(raw.src_zone, raw.label_index).to_csv(OUT / 'source_zone_label_counts.csv')
    pd.crosstab(raw.dst_zone, raw.label_index).to_csv(OUT / 'destination_zone_label_counts.csv')
    # Only aggregates are exposed; individual identities remain in existing protected project data.
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
