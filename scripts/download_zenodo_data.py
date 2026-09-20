#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import sys
import time
import requests
import yaml

from typing import Any
from pathlib import Path
from tqdm import tqdm


CHUNK_SIZE = 1024 * 1024  # 1 MiB
MAX_RETRIES = 5
TIMEOUT_S = 60


def compute_checksum(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, dest: Path) -> None:
    """Stream url to dest, resuming a partial .part file if one exists."""
    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    resume_from = tmp_dest.stat().st_size if tmp_dest.exists() else 0
    headers = {"Range": f"bytes={resume_from}-"} if resume_from else {}

    with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT_S) as r:
        if resume_from and r.status_code == 200:
            # Server ignored the Range request; fall back to a full
            # re-download instead of corrupting the file by appending.
            resume_from = 0
        r.raise_for_status()
        served_partial = bool(resume_from) and r.status_code == 206
        total = int(r.headers.get("content-length", 0)) + (resume_from if served_partial else 0)
        mode = "ab" if served_partial else "wb"
        with (
            tmp_dest.open(mode) as f,
            tqdm(
                total=total or None,
                initial=resume_from if served_partial else 0,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=dest.name,
            ) as bar,
        ):
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
    tmp_dest.rename(dest)


def ensure_file(entry: dict[str, Any], base_url: str, raw_dir: Path) -> None:
    name = entry["name"]
    algo = entry.get("checksum_algo", "md5")
    expected = entry["checksum"]
    dest = raw_dir / name
    url = f"{base_url}/{name}?download=1"

    if dest.exists():
        actual = compute_checksum(dest, algo)
        if actual == expected:
            print(f"[skip] {name} already present and verified ({algo}={actual[:12]}...)")
            return
        print(f"[warn] {name} exists but checksum mismatch, re-downloading")
        dest.unlink()

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[download] {name} ({entry.get('size_human', '?')}) attempt {attempt}/{MAX_RETRIES}")
            download_with_resume(url, dest)
            last_exc = None
            break
        except (requests.RequestException, OSError) as exc:
            last_exc = exc
            print(f"[retry] {name}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(2**attempt)
    if last_exc is not None:
        raise SystemExit(f"Giving up on {name} after {MAX_RETRIES} attempts: {last_exc}")

    actual = compute_checksum(dest, algo)
    if actual != expected:
        dest.unlink(missing_ok=True)
        raise SystemExit(
            f"Checksum mismatch for {name}: expected {expected}, got {actual}. "
            f"Deleted the corrupted file — re-run this script to retry."
        )
    print(f"[ok] {name} verified ({algo}={actual[:12]}...)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    base_url = manifest["zenodo"]["base_url"]
    raw_dir = args.repo_root / manifest["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    for entry in manifest["files"]:
        ensure_file(entry, base_url, raw_dir)

    print(f"All files downloaded and verified into {raw_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
