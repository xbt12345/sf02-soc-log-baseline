from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.common import ensure_dir, save_json


TEXT_SUFFIXES = {'.txt', '.log', '.csv', '.json', '.yaml', '.yml', '.md', '.xml'}
METADATA_KEYWORDS = ('readme', 'label', 'ground', 'truth', 'manifest', 'meta', 'mapping', 'schema')
SOURCE_KEYWORDS = (
    'suricata',
    'syslog',
    'vpn',
    'dns',
    'audit',
    'auth',
    'firewall',
    'proxy',
    'zeek',
    'bro',
    'windows',
    'linux',
)


def safe_read_head(path: Path, max_lines: int = 5, max_bytes: int = 8192) -> list[str]:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return []
    try:
        with path.open('r', encoding='utf-8', errors='replace') as fh:
            lines = []
            total = 0
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                total += len(line.encode('utf-8', errors='ignore'))
                if total > max_bytes:
                    break
                lines.append(line.rstrip('\n'))
            return lines
    except OSError:
        return []


def likely_source(path: Path) -> str:
    lowered = str(path).lower()
    for keyword in SOURCE_KEYWORDS:
        if keyword in lowered:
            return keyword
    return 'unknown'


def scan_dataset(root: Path, max_preview_files: int) -> dict:
    files = [p for p in root.rglob('*') if p.is_file()]
    suffix_counter = Counter(p.suffix.lower() or '<no_suffix>' for p in files)
    source_counter = Counter(likely_source(p) for p in files)
    top_level_counter = Counter()
    top_level_bytes = defaultdict(int)
    candidate_metadata = []
    previews_taken = 0

    for path in files:
        rel = path.relative_to(root)
        top = rel.parts[0] if rel.parts else '<root>'
        top_level_counter[top] += 1
        top_level_bytes[top] += path.stat().st_size

        lowered = path.name.lower()
        if any(keyword in lowered for keyword in METADATA_KEYWORDS):
            preview = []
            if previews_taken < max_preview_files:
                preview = safe_read_head(path)
                previews_taken += 1
            candidate_metadata.append(
                {
                    'path': str(rel),
                    'size_bytes': path.stat().st_size,
                    'preview': preview,
                }
            )

    largest_files = sorted(files, key=lambda p: p.stat().st_size, reverse=True)[:20]
    file_manifest = []
    for path in largest_files:
        rel = path.relative_to(root)
        file_manifest.append(
            {
                'path': str(rel),
                'size_bytes': path.stat().st_size,
                'suffix': path.suffix.lower() or '<no_suffix>',
                'source_guess': likely_source(path),
            }
        )

    return {
        'root': str(root),
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'total_files': len(files),
        'total_bytes': sum(p.stat().st_size for p in files),
        'suffix_counts': dict(suffix_counter.most_common()),
        'source_guess_counts': dict(source_counter.most_common()),
        'top_level_dirs': [
            {
                'name': name,
                'file_count': top_level_counter[name],
                'size_bytes': top_level_bytes[name],
            }
            for name, _ in top_level_counter.most_common()
        ],
        'candidate_metadata_files': candidate_metadata[:50],
        'largest_files': file_manifest,
    }


def build_markdown(summary: dict) -> str:
    lines = [
        '# AIT Dataset Inspect Summary',
        '',
        f"- root: `{summary['root']}`",
        f"- generated_at: `{summary['generated_at']}`",
        f"- total_files: `{summary['total_files']}`",
        f"- total_size_gb: `{summary['total_bytes'] / 1_073_741_824:.2f}`",
        '',
        '## Top-Level Dirs',
    ]
    for item in summary['top_level_dirs'][:20]:
        lines.append(
            f"- `{item['name']}`: files={item['file_count']}, size_gb={item['size_bytes'] / 1_073_741_824:.2f}"
        )

    lines.extend(['', '## Suffix Counts'])
    for suffix, count in list(summary['suffix_counts'].items())[:20]:
        lines.append(f"- `{suffix}`: {count}")

    lines.extend(['', '## Source Guess Counts'])
    for source, count in list(summary['source_guess_counts'].items())[:20]:
        lines.append(f"- `{source}`: {count}")

    lines.extend(['', '## Candidate Metadata Files'])
    for item in summary['candidate_metadata_files'][:15]:
        lines.append(f"- `{item['path']}` ({item['size_bytes']} bytes)")
        for preview_line in item['preview'][:3]:
            lines.append(f"  {preview_line}")

    lines.extend(['', '## Largest Files'])
    for item in summary['largest_files'][:20]:
        lines.append(
            f"- `{item['path']}` | suffix={item['suffix']} | source_guess={item['source_guess']} | size_gb={item['size_bytes'] / 1_073_741_824:.2f}"
        )
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Extracted AIT dataset root directory')
    parser.add_argument('--artifacts-dir', default='./artifacts', help='Where to write reports')
    parser.add_argument('--max-preview-files', type=int, default=10)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f'Root not found: {root}')

    artifacts_root = ensure_dir(Path(args.artifacts_dir).resolve())
    reports_dir = ensure_dir(artifacts_root / 'reports')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary = scan_dataset(root, max_preview_files=args.max_preview_files)

    json_path = reports_dir / f'ait_inspect_{timestamp}.json'
    md_path = reports_dir / f'ait_inspect_{timestamp}.md'
    save_json(json_path, summary)
    md_path.write_text(build_markdown(summary), encoding='utf-8')

    print(f'Saved JSON summary to {json_path}')
    print(f'Saved Markdown summary to {md_path}')
    print(f"total_files={summary['total_files']}")
    print(f"total_size_gb={summary['total_bytes'] / 1_073_741_824:.2f}")
    print('Top source guesses:')
    for source, count in list(summary['source_guess_counts'].items())[:10]:
        print(f'- {source}: {count}')


if __name__ == '__main__':
    main()
