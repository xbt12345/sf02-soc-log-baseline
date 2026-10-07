"""Download and verify the published P_IS weights without overwriting code."""
from __future__ import annotations

import argparse
import hashlib
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

from sf02_model.model import WEIGHT_SHA256

URL = ("https://github.com/xbt12345/sf02-soc-log-baseline/releases/download/"
       "sf02-pis-competition-20261007/SF02_PIS_weights.zip")
ZIP_SHA256 = "fa003dd4234fa159bec95e35c0641471a750cde79c3720ce02617469e64a416d"
API_URL = "https://api.github.com/repos/xbt12345/sf02-soc-log-baseline/releases/assets/617543741"


def download(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    needed = {f"weights/fold{fold}/{name}": digest
              for fold, files in WEIGHT_SHA256.items() for name, digest in files.items()}
    existing = 0
    for member, expected in needed.items():
        target = destination / Path(member).relative_to("weights")
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise FileExistsError(f"Refusing to replace a different weight: {target}")
            existing += 1
    if existing == len(needed):
        print(f"Verified 9 existing published weights in {destination}")
        return
    with tempfile.TemporaryDirectory(prefix="sf02-download-") as temp:
        archive = Path(temp) / "weights.zip"
        failure = None
        for address in (URL, API_URL):
            request = Request(address, headers={"User-Agent": "SF02-PIS-downloader",
                                               "Accept": "application/octet-stream"})
            try:
                digest = hashlib.sha256()
                with urlopen(request, timeout=30) as response, archive.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        digest.update(block)
                        output.write(block)
                if digest.hexdigest() != ZIP_SHA256:
                    raise ValueError("Release archive SHA-256 mismatch")
                break
            except (OSError, ValueError) as error:
                failure = error
        else:
            raise RuntimeError("Both public GitHub download endpoints failed; "
                               "download SF02_PIS_weights.zip from the release page") from failure
        with ZipFile(archive) as source:
            for member, expected in needed.items():
                data = source.read(member)
                if hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError(f"Weight SHA-256 mismatch: {member}")
                target = destination / Path(member).relative_to("weights")
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with target.open("xb") as output:
                        output.write(data)
    print(f"Verified 9 published weights in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("weights"))
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
