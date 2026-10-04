"""Inspect whether natural fine negatives differ beyond source-port facts."""
import json
import pandas as pd
from run_v75 import ROOT, save, sha
from v112_fine_control_preflight import DEST
from v110_layer_probes import LEDGER


def main():
    target = DEST / 'negative_evidence_audit.json'
    assert not target.exists()
    rows = pd.read_parquet(DEST / 'interface_row_audit.parquet')
    ledger = pd.read_parquet(LEDGER, columns=['local', 'projection_id']).drop_duplicates('local').set_index('local')
    fp = ROOT / 'artifacts/v75_four_arm_20260921_r2/projections.parquet'
    facts = pd.read_parquet(fp, columns=['facts'])
    mapping = {}
    for local, record in ledger.iterrows():
        f = json.loads(facts.facts.iat[int(record.projection_id)])
        mapping[local] = json.dumps({k: v for k, v in f.items() if not k.startswith('src_port')}, sort_keys=True)
    rows['facts_without_src_port'] = rows.local.map(mapping)
    out = []; summary = []
    for fold in range(3):
        train = rows[(rows.fold != fold) & rows.interface_parse_verified & rows.behavior.notna()].copy()
        cache = {}
        for k, g in train.groupby(['behavior', 'interface_literal_pair', 'truth']):
            cache[k] = set(zip(g.root, g.facts_without_src_port))
        neg = []; beyond = []
        for r in train.itertuples(index=False):
            partners = cache.get((r.behavior, r.interface_literal_pair, 3-int(r.truth)), set())
            neg.append(any(root != r.root for root, f in partners))
            beyond.append(any(root != r.root and f != r.facts_without_src_port for root, f in partners))
        train['has_cross_source_negative'] = neg
        train['has_negative_with_non_source_port_fact_difference'] = beyond
        train['audit_fold'] = fold
        for c in [1, 2]:
            a = train[train.truth == c]
            summary.append({'fold': fold, 'class': 'M' if c == 1 else 'S',
                'candidate_negative_anchor_rows': int(a.has_cross_source_negative.sum()),
                'has_non_source_port_fact_difference_rows': int(a.has_negative_with_non_source_port_fact_difference.sum()),
                'has_non_source_port_fact_difference_roots': int(a.loc[a.has_negative_with_non_source_port_fact_difference, 'root'].nunique())})
        out.append(train[['row_position', 'local', 'root', 'truth', 'audit_fold', 'has_cross_source_negative',
            'has_negative_with_non_source_port_fact_difference']])
    pd.concat(out).to_parquet(DEST / 'negative_evidence_eligibility.parquet', index=False)
    report = {'status': 'no_fit_observed_negative_evidence_audit', 'classifier_fits': 0, 'calibration_fits': 0,
        'source_sha256': sha(__file__), 'summary': summary,
        'input_hashes': {p.relative_to(ROOT).as_posix(): sha(p) for p in [fp, LEDGER, DEST / 'interface_row_audit.parquet']},
        'output_hashes': {'negative_evidence_eligibility.parquet': sha(DEST / 'negative_evidence_eligibility.parquet')},
        'scope': 'Existing parsed facts only; excluding source port is a diagnostic projection, not deleting model information.',
        'limits': ['Parsed facts are incomplete; no observed difference here does not prove raw inputs or true incidents indistinguishable.',
            'A source-port difference may be meaningful in some contexts; it is not automatically certified causal evidence for M/S.',
            'Counting anchors with some partner is not the number of independent pairs or security-labeled counterfactuals.']}
    save(target, report)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
