"""Reversible NTFS compression of sealed historical gradient files only."""
import ctypes
import json
import shutil
import stat
import subprocess
from pathlib import Path

from experiment_review import ROOT, read, sha

OUT = ROOT/'artifacts/v169_root_disk_headroom_compression_20261002'
BUNDLE = ROOT/'artifacts/v169_full_preseal_bundle_v5_20261002/bundle.json'


def allocated(path):
    high = ctypes.c_ulong(0)
    fn = ctypes.windll.kernel32.GetCompressedFileSizeW
    fn.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_ulong)]
    fn.restype = ctypes.c_ulong
    ctypes.windll.kernel32.SetLastError(0)
    low = fn(str(path), ctypes.byref(high))
    if low == 0xffffffff and ctypes.windll.kernel32.GetLastError():
        raise OSError('Cannot query actual compressed allocation')
    return (high.value << 32) | low


def main():
    if OUT.exists():
        raise FileExistsError(OUT)
    bundle = read(BUNDLE)
    required = bundle['resources']['minimum_free_disk_start_bytes']
    candidates = []
    for name, expected in bundle['source_sha256'].items():
        if not name.startswith('artifacts/') or not name.endswith('gradient.npy'):
            continue
        path = (ROOT/name).resolve(strict=True)
        if not path.is_relative_to(ROOT/'artifacts'):
            raise ValueError('Historical gradient path escaped workspace')
        info = path.stat()
        if info.st_size >= 8*1024**2 and not info.st_file_attributes & stat.FILE_ATTRIBUTE_COMPRESSED:
            candidates.append((path, expected))
    # One GiB headroom over the complete-run prerequisite; no broad directory
    # operation or deletion, and no active run/source/model file is modified.
    candidates.sort(key=lambda v: v[0].as_posix())
    before = shutil.disk_usage(ROOT).free
    OUT.mkdir()
    records = []
    for path, expected in candidates[:256]:
        if shutil.disk_usage(ROOT).free-required >= 1024**3:
            break
        old_hash = sha(path)
        if old_hash != expected:
            raise ValueError('Historical source changed before compression: '+str(path))
        old_allocation = allocated(path)
        proc = subprocess.run(['compact.exe', '/C', '/I', str(path)],
                              capture_output=True, check=False)
        new_hash = sha(path)
        if new_hash != old_hash:
            subprocess.run(['compact.exe', '/U', '/I', str(path)], capture_output=True, check=False)
            raise RuntimeError('Logical bytes changed; compression aborted: '+str(path))
        compressed = bool(path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_COMPRESSED)
        item = dict(path=path.relative_to(ROOT).as_posix(), logical_bytes=path.stat().st_size,
            sha256_before=old_hash, sha256_after=new_hash, logical_bytes_exact=True,
            allocation_before=old_allocation, allocation_after=allocated(path),
            NTFS_compressed_attribute=compressed, compact_exit_code=proc.returncode)
        records.append(item)
        with (OUT/'file_receipts.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(item, ensure_ascii=False)+'\n')
        if proc.returncode or not compressed:
            raise RuntimeError('Compression failed; verified original bytes retained')
    after = shutil.disk_usage(ROOT).free
    report = dict(status='historical_full_gradients_losslessly_NTFS_compressed_with_byte_receipts',
        scope='Only filesystem compression attributes changed on bound historical gradient.npy files',
        source_bundle=BUNDLE.relative_to(ROOT).as_posix(), source_bundle_sha256=sha(BUNDLE),
        files=len(records), all_logical_SHA256_preserved=all(r['logical_bytes_exact'] for r in records),
        allocation_reduction_bytes=sum(r['allocation_before']-r['allocation_after'] for r in records),
        free_disk_before=before, free_disk_after=after, complete_run_start_minimum=required,
        headroom_above_complete_run_start_bytes=after-required,
        free_RAM_not_fixed_by_this_action=True, no_files_deleted=True,
        official_heads=0, official_features=0, official_derivatives=0, fits=0, permanent_updates=0,
        execution_authority=False, source_sha256={Path(__file__).resolve().relative_to(ROOT).as_posix():sha(Path(__file__))})
    (OUT/'review.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
