from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common import ensure_dir, save_json


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_text_lines(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        return fh.read().splitlines()


def source_from_label_path(rel_path: Path) -> str:
    parts = [p.lower() for p in rel_path.parts]
    candidates = ('dnsmasq', 'openvpn', 'audit', 'apache2', 'auth', 'suricata', 'logstash', 'syslog')
    for c in candidates:
        if c in parts or c in rel_path.name.lower():
            return c
    return parts[2] if len(parts) >= 3 else 'unknown'


def raw_path_from_label_path(root: Path, label_rel: Path) -> Path:
    if label_rel.parts[0] != 'labels':
        raise ValueError(f'Unexpected label path: {label_rel}')
    return root / 'gather' / Path(*label_rel.parts[1:])


def audit_labels(root: Path, sample_per_file: int = 5) -> dict:
    label_root = root / 'labels'
    if not label_root.exists():
        raise FileNotFoundError(f'labels directory not found: {label_root}')

    label_files = sorted(p for p in label_root.rglob('*') if p.is_file())
    global_label_counts = Counter()
    global_source_counts = Counter()
    per_file = []
    missing_raw_files = []

    for label_file in label_files:
        rel = label_file.relative_to(root)
        source = source_from_label_path(rel)
        rows = read_jsonl(label_file)
        raw_path = raw_path_from_label_path(root, rel)
        raw_exists = raw_path.exists()
        raw_lines = read_text_lines(raw_path) if raw_exists else []
        if not raw_exists:
            missing_raw_files.append(str(raw_path.relative_to(root)))

        label_counter = Counter()
        rule_counter = Counter()
        samples = []

        for row in rows:
            labels = row.get('labels', [])
            rules = row.get('rules', {})
            line_no = int(row['line'])
            for label in labels:
                label_counter[label] += 1
                global_label_counts[label] += 1
            for label_name, rule_names in rules.items():
                for rule_name in rule_names:
                    rule_counter[rule_name] += 1
            if len(samples) < sample_per_file:
                raw_line = ''
                idx = line_no - 1
                if 0 <= idx < len(raw_lines):
                    raw_line = raw_lines[idx]
                samples.append(
                    {
                        'line': line_no,
                        'labels': labels,
                        'raw_line': raw_line,
                    }
                )

        global_source_counts[source] += len(rows)
        per_file.append(
            {
                'label_path': str(rel),
                'raw_path': str(raw_path.relative_to(root)) if raw_exists else None,
                'source_guess': source,
                'labeled_events': len(rows),
                'unique_labels': sorted(label_counter.keys()),
                'label_counts': dict(label_counter.most_common()),
                'top_rules': dict(rule_counter.most_common(20)),
                'samples': samples,
            }
        )

    by_source = defaultdict(lambda: {'files': 0, 'labeled_events': 0, 'label_counts': Counter()})
    for item in per_file:
        bucket = by_source[item['source_guess']]
        bucket['files'] += 1
        bucket['labeled_events'] += item['labeled_events']
        for label, count in item['label_counts'].items():
            bucket['label_counts'][label] += count

    by_source_summary = []
    for source, bucket in by_source.items():
        by_source_summary.append(
            {
                'source': source,
                'files': bucket['files'],
                'labeled_events': bucket['labeled_events'],
                'label_counts': dict(bucket['label_counts'].most_common()),
            }
        )
    by_source_summary.sort(key=lambda x: x['labeled_events'], reverse=True)

    return {
        'root': str(root),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'label_file_count': len(label_files),
        'missing_raw_files': missing_raw_files,
        'global_label_counts': dict(global_label_counts.most_common()),
        'global_source_counts': dict(global_source_counts.most_common()),
        'by_source': by_source_summary,
        'per_file': per_file,
    }


def to_markdown(summary: dict) -> str:
    lines = [
        '# AIT Label Audit',
        '',
        f"- root: `{summary['root']}`",
        f"- generated_at: `{summary['generated_at']}`",
        f"- label_file_count: `{summary['label_file_count']}`",
        f"- missing_raw_files: `{len(summary['missing_raw_files'])}`",
        '',
        '## Global Label Counts',
    ]
    for label, count in list(summary['global_label_counts'].items())[:50]:
        lines.append(f"- `{label}`: {count}")

    lines.extend(['', '## By Source'])
    for item in summary['by_source']:
        lines.append(f"- `{item['source']}`: files={item['files']}, labeled_events={item['labeled_events']}")
        top_labels = list(item['label_counts'].items())[:10]
        if top_labels:
            lines.append("  " + ", ".join(f"{k}={v}" for k, v in top_labels))

    lines.extend(['', '## Per File Samples'])
    for item in summary['per_file'][:20]:
        lines.append(f"- `{item['label_path']}` | source={item['source_guess']} | labeled_events={item['labeled_events']}")
        for sample in item['samples'][:3]:
            preview = sample['raw_line'][:200]
            lines.append(f"  line={sample['line']} labels={sample['labels']} raw={preview}")
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Extracted AIT dataset root directory')
    parser.add_argument('--artifacts-dir', default='./artifacts')
    parser.add_argument('--sample-per-file', type=int, default=5)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    artifacts_root = ensure_dir(Path(args.artifacts_dir).resolve())
    reports_dir = ensure_dir(artifacts_root / 'reports')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    summary = audit_labels(root, sample_per_file=args.sample_per_file)
    json_path = reports_dir / f'ait_label_audit_{timestamp}.json'
    md_path = reports_dir / f'ait_label_audit_{timestamp}.md'
    save_json(json_path, summary)
    md_path.write_text(to_markdown(summary), encoding='utf-8')

    print(f'Saved JSON audit to {json_path}')
    print(f'Saved Markdown audit to {md_path}')
    print(f"label_file_count={summary['label_file_count']}")
    print('Top labels:')
    for label, count in list(summary['global_label_counts'].items())[:15]:
        print(f'- {label}: {count}')


if __name__ == '__main__':
    main()
