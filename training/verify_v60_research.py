"""Verify load-bearing research counts, prior evidence and local report links."""
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/2026-09-14/v60_research_review.json'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    assert not OUT.exists()
    report_path = ROOT / 'docs/V60_FUNDAMENTAL_RESET_RESEARCH.md'
    report = report_path.read_text(encoding='utf-8')
    plan = json.loads((ROOT / 'training/v60_research_plan.json').read_text())
    audit = json.loads((ROOT / 'evidence/2026-09-14/v60_direction_reset/summary.json').read_text(encoding='utf-8'))
    rows = pd.read_parquet(ROOT / 'artifacts/v55_risk_validation_20260914/rows.parquet')
    x = pd.read_parquet(ROOT / 'artifacts/v55_risk_validation_20260914/observations.parquet')
    mask = rows.route.eq('asa')
    counts = defaultdict(Counter)
    for key, label in zip(x.loc[mask].itertuples(index=False, name=None), rows.loc[mask, 'label_index']):
        counts[key][int(label)] += 1
    empirical_floor = sum(sum(v.values()) - max(v.values()) for v in counts.values())
    assert empirical_floor == audit['current_collisions']['finite_sample_minimum_errors'] == 2306
    assert len(counts) == audit['current_collisions']['keys'] == 15658
    raw = pd.read_parquet(ROOT / 'evidence/2026-09-14/v53_raw_audit/raw_fields.parquet')
    raw = raw[raw.row_position.isin(rows.loc[mask, 'row_position'])]
    for col, expected in [('src_address', 7933), ('dst_address', 142)]:
        groups = defaultdict(set)
        for value, label in zip(raw[col], raw.label_index):
            groups[value].add(int(label))
        assert len(groups) == expected and all(len(v) == 1 for v in groups.values())
    assert raw.raw_sha256.nunique() == 29031 and len(raw) == 99382
    assert plan['new_models_trained_this_round'] == 0
    assert plan['supervised_two_stage_fit_upper_bound'] == plan['core_fit_upper_bound'] + plan['conditional_destination_holdout_fit_upper_bound']
    local_links = []
    # Markdown links to local artifacts, including a literal balanced '(1)' filename.
    for link in re.findall(r'\]\(([^\n]+?)\)(?=[。；，、\s]|$)', report):
        if link.startswith(('https://', 'http://')):
            continue
        target = (report_path.parent / link).resolve()
        assert target.exists(), str(target)
        local_links.append(link)
    prior = json.loads((ROOT / 'evidence/2026-09-14/v57_delivery/delivery.json').read_text())
    prior_failures = [p for p, digest in prior['bindings'].items() if sha(ROOT / p) != digest]
    assert not prior_failures, prior_failures
    for p, digest in audit['source_bindings'].items():
        assert sha(ROOT / p) == digest, p
    files = [report_path, ROOT / 'training/v60_research_plan.json',
        ROOT / 'training/audit_v60_direction_reset.py', ROOT / 'training/capture_v60_research_metadata.py',
        ROOT / 'training/verify_v60_research.py']
    for folder in ['v60_direction_reset', 'v60_research_metadata']:
        files += list((ROOT / 'evidence/2026-09-14' / folder).glob('*'))
    result = {'scope': 'Research delivery and finite-corpus calculations only; no quality or external-validity claim',
        'all_checks_passed': True, 'counter_based_floor': empirical_floor,
        'prior_v57_bound_files_unchanged': len(prior['bindings']),
        'local_report_links_checked': len(local_links),
        'new_model_fits': 0,
        'bindings': {p.relative_to(ROOT).as_posix(): sha(p) for p in files}}
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'bindings'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
