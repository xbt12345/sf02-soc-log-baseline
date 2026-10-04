"""Restore byte-exact official data from upload parts using only Python stdlib.

Run in the cloud with: python3 /root/work/sf02_data/restore_and_verify.py
Use --verify-only to check transfer files without creating train.parquet.
Existing output is verified and never overwritten. Upload parts are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


PARTS = [{'name': 'train.parquet.part01', 'size': 67108864, 'sha256': '291706558c8f9286495bcf54769cd31268c22f98506ababf0ffcc9f474f0b587'}, {'name': 'train.parquet.part02', 'size': 60009782, 'sha256': 'fff182acb975b9a0b0c7b83c1375157ab1e224aab8283a0d6087f7b9f5592fa1'}]
TRAIN = {"size": 127118646, "sha256": "6b6d5e23caebfd1c4f6b70c9e58c27f437bca7f0cd26497eefa3e4908f2cb742"}
VALID = {"size": 69944975, "sha256": "731af597711aae0b870912c36fc1a0f592fd6e53d911c7fec6fe1405ec1a97b0"}
BLOCK = 4 * 1024 * 1024


def blocks(path):
    with path.open("rb") as source:
        while True:
            data = source.read(BLOCK)
            if not data:
                break
            yield data


def verify_file(path, expected):
    if not path.is_file():
        raise RuntimeError("Missing file: " + str(path))
    if path.stat().st_size != expected["size"]:
        raise RuntimeError("File size mismatch: " + path.name)
    digest = hashlib.sha256()
    for data in blocks(path):
        digest.update(data)
    if digest.hexdigest() != expected["sha256"]:
        raise RuntimeError("SHA-256 mismatch: " + path.name)


def restore(base, verify_only=False):
    if not PARTS:
        raise RuntimeError("Upload part specifications are missing")
    combined = hashlib.sha256()
    total = 0
    for part in PARTS:
        path = base / part["name"]
        verify_file(path, part)
        for data in blocks(path):
            combined.update(data)
            total += len(data)
    if total != TRAIN["size"] or combined.hexdigest() != TRAIN["sha256"]:
        raise RuntimeError("Combined training data does not match the official original")
    verify_file(base / "valid_input.parquet", VALID)

    output = base / "train.parquet"
    status = "transfer_files_verified"
    if not verify_only:
        if output.exists():
            verify_file(output, TRAIN)
            status = "existing_original_verified"
        else:
            # Exclusive creation protects an existing file, including a race.
            stream = output.open("xb")
            try:
                with stream:
                    for part in PARTS:
                        for data in blocks(base / part["name"]):
                            stream.write(data)
                verify_file(output, TRAIN)
            except BaseException:
                # Remove only the incomplete output created by this invocation.
                output.unlink()
                raise
            status = "restored_and_verified"
    return {
        "status": status,
        "verified": True,
        "directory": str(base),
        "train_sha256": TRAIN["sha256"],
        "valid_input_sha256": VALID["sha256"],
        "note": "File identity verified; no training or detection-quality evaluation performed",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        print(json.dumps(restore(Path(__file__).resolve().parent, args.verify_only), indent=2))
        return 0
    except Exception as exc:
        print("ERROR: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
