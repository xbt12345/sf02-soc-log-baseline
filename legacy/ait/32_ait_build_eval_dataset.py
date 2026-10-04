from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common import ensure_dir, save_json


IP_RE = re.compile(r'(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)')
PORT_RE = re.compile(r'(?:(?:port|src_port|sport)[ =:]+)(\d{1,5})', re.IGNORECASE)
AUDIT_TS_RE = re.compile(r'audit\((\d+(?:\.\d+)?):')
APACHE_TS_RE = re.compile(r'\[(\d{2}/[A-Za-z]{3}/\d{4}:\d{2}:\d{2}:\d{2}) [+\-]\d{4}\]')
SYSLOG_TS_RE = re.compile(r'^([A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})')
OPENVPN_TS_RE = re.compile(r'^([A-Z][a-z]{2}\s+[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\d{4})')
USERNAME_PATTERNS = [
    re.compile(r'\bfor user (\w[\w.\-@]*)', re.IGNORECASE),
    re.compile(r'\buser(?:name)?[ =:]+([A-Za-z0-9_.\-@]+)', re.IGNORECASE),
    re.compile(r'\bfrom ([A-Za-z0-9_.\-@]+)@', re.IGNORECASE),
]

MALICIOUS_LABELS = {
    'dnsteal',
    'dnsteal-received',
    'exfiltration-service',
    'escalate',
    'escalated_command',
    'escalated_sudo_command',
    'escalated_sudo_session',
    'crack_passwords',
    'webshell_cmd',
}
SUSPICIOUS_LABELS = {
    'attacker',
    'attacker_vpn',
    'attacker_http',
    'foothold',
    'service_scan',
    'dns_scan',
    'network_scan',
    'dirb',
    'wpscan',
    'attacker_change_user',
    'dnsteal-dropped',
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def read_lines(path: Path) -> list[str]:
    with path.open('r', encoding='utf-8', errors='replace') as fh:
        return fh.read().splitlines()


def source_from_relpath(rel_path: Path) -> str:
    lowered_parts = [p.lower() for p in rel_path.parts]
    candidates = ('dnsmasq', 'openvpn', 'audit', 'apache2', 'auth', 'suricata', 'logstash', 'syslog')
    for c in candidates:
        if c in lowered_parts or c in rel_path.name.lower():
            return c
    return rel_path.parts[2] if len(rel_path.parts) >= 3 else 'unknown'


def raw_path_from_label_path(root: Path, label_rel: Path) -> Path:
    return root / 'gather' / Path(*label_rel.parts[1:])


def collapse_labels(raw_labels: list[str]) -> str:
    labels = set(raw_labels)
    if labels & MALICIOUS_LABELS:
        return 'malicious'
    if labels & SUSPICIOUS_LABELS:
        return 'suspicious'
    if labels:
        return 'suspicious'
    return 'benign'


def parse_timestamp(line: str, dataset_start: str | None = None) -> float | None:
    match = AUDIT_TS_RE.search(line)
    if match:
        return float(match.group(1))

    match = APACHE_TS_RE.search(line)
    if match:
        try:
            dt = datetime.strptime(match.group(1), '%d/%b/%Y:%H:%M:%S').replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass

    match = OPENVPN_TS_RE.search(line)
    if match:
        try:
            dt = datetime.strptime(match.group(1), '%a %b %d %H:%M:%S %Y').replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass

    match = SYSLOG_TS_RE.search(line)
    if match and dataset_start:
        try:
            year = datetime.fromisoformat(dataset_start.replace('Z', '+00:00')).year
            dt = datetime.strptime(f'{year} {match.group(1)}', '%Y %b %d %H:%M:%S').replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            pass
    return None


def extract_username(line: str) -> str | None:
    for pattern in USERNAME_PATTERNS:
        match = pattern.search(line)
        if match:
            return match.group(1)
    return None


def extract_ips(line: str) -> tuple[str | None, str | None]:
    ips = IP_RE.findall(line)
    if not ips:
        return None, None
    src_ip = ips[0]
    dst_ip = ips[1] if len(ips) > 1 else None
    return src_ip, dst_ip


def extract_port(line: str) -> str | None:
    match = PORT_RE.search(line)
    if match:
        return match.group(1)
    return None


def build_record(
    host_name: str,
    source: str,
    rel_raw_path: Path,
    line_no: int,
    raw_line: str,
    label_value: str,
    raw_labels: list[str],
    dataset_start: str | None,
) -> dict:
    timestamp = parse_timestamp(raw_line, dataset_start=dataset_start)
    src_ip, dst_ip = extract_ips(raw_line)
    username = extract_username(raw_line)
    src_port = extract_port(raw_line)
    event_id = f'ait::{host_name}::{rel_raw_path.as_posix()}::{line_no}'
    return {
        'event_id': event_id,
        'timestamp': timestamp,
        'pipeline': source,
        'src_ip': src_ip,
        'dst_ip': dst_ip,
        'src_port': src_port,
        'src_host': host_name,
        'dst_host': None,
        'username': username,
        'message_sanitized': raw_line,
        'product_name': source,
        'vendor_name': 'AIT-LDS-V2',
        'label_binary': label_value,
        'ait_raw_labels': '|'.join(raw_labels),
        'ait_source_path': rel_raw_path.as_posix(),
        'ait_line_number': line_no,
    }


def load_dataset_meta(root: Path) -> dict:
    dataset_yaml = root / 'dataset.yaml'
    if not dataset_yaml.exists():
        return {}
    try:
        import yaml

        with dataset_yaml.open('r', encoding='utf-8') as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def build_dataset(root: Path, benign_multiplier: float, random_seed: int) -> tuple[pd.DataFrame, dict]:
    meta = load_dataset_meta(root)
    dataset_start = meta.get('start')
    rng = random.Random(random_seed)

    label_root = root / 'labels'
    label_files = sorted(p for p in label_root.rglob('*') if p.is_file())

    records = []
    per_file_summary = []
    label_counts = Counter()
    raw_label_counts = Counter()

    for label_file in label_files:
        rel_label = label_file.relative_to(root)
        rel_raw = Path('gather') / Path(*rel_label.parts[1:])
        raw_file = root / rel_raw
        if not raw_file.exists():
            continue

        host_name = rel_raw.parts[1] if len(rel_raw.parts) > 1 else 'unknown_host'
        source = source_from_relpath(rel_label)
        raw_lines = read_lines(raw_file)
        labeled_rows = read_jsonl(label_file)
        labeled_map = {int(row['line']): row for row in labeled_rows}

        for row in labeled_rows:
            line_no = int(row['line'])
            if line_no < 1 or line_no > len(raw_lines):
                continue
            raw_labels = row.get('labels', [])
            for raw_label in raw_labels:
                raw_label_counts[raw_label] += 1
            final_label = collapse_labels(raw_labels)
            label_counts[final_label] += 1
            records.append(
                build_record(
                    host_name=host_name,
                    source=source,
                    rel_raw_path=rel_raw,
                    line_no=line_no,
                    raw_line=raw_lines[line_no - 1],
                    label_value=final_label,
                    raw_labels=raw_labels,
                    dataset_start=dataset_start,
                )
            )

        unlabeled_line_numbers = [i for i in range(1, len(raw_lines) + 1) if i not in labeled_map]
        target_benign = min(len(unlabeled_line_numbers), int(round(len(labeled_rows) * benign_multiplier)))
        benign_sample = rng.sample(unlabeled_line_numbers, target_benign) if target_benign > 0 else []
        for line_no in benign_sample:
            label_counts['benign'] += 1
            records.append(
                build_record(
                    host_name=host_name,
                    source=source,
                    rel_raw_path=rel_raw,
                    line_no=line_no,
                    raw_line=raw_lines[line_no - 1],
                    label_value='benign',
                    raw_labels=[],
                    dataset_start=dataset_start,
                )
            )

        per_file_summary.append(
            {
                'label_path': rel_label.as_posix(),
                'raw_path': rel_raw.as_posix(),
                'source': source,
                'raw_lines': len(raw_lines),
                'labeled_lines': len(labeled_rows),
                'sampled_benign_lines': len(benign_sample),
            }
        )

    df = pd.DataFrame(records)
    summary = {
        'dataset_meta': meta,
        'rows': int(len(df)),
        'label_counts': dict(label_counts),
        'raw_label_counts': dict(raw_label_counts.most_common()),
        'per_file_summary': per_file_summary,
        'benign_multiplier': benign_multiplier,
        'random_seed': random_seed,
    }
    return df, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Extracted AIT dataset root directory')
    parser.add_argument('--output', required=True, help='Output parquet path')
    parser.add_argument('--artifacts-dir', default='./artifacts')
    parser.add_argument('--benign-multiplier', type=float, default=3.0)
    parser.add_argument('--random-seed', type=int, default=42)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    output_path = Path(args.output).resolve()
    artifacts_root = ensure_dir(Path(args.artifacts_dir).resolve())
    reports_dir = ensure_dir(artifacts_root / 'reports')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df, summary = build_dataset(root, benign_multiplier=args.benign_multiplier, random_seed=args.random_seed)
    if df.empty:
        raise RuntimeError('No records were built from AIT labels')

    df.to_parquet(output_path, index=False)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = reports_dir / f'ait_build_dataset_{timestamp}.json'
    save_json(report_path, summary)

    print(f'Saved eval parquet to {output_path}')
    print(f'Saved build summary to {report_path}')
    print(f'rows={len(df)}')
    print('label_counts=')
    print(df['label_binary'].value_counts().to_string())


if __name__ == '__main__':
    main()
