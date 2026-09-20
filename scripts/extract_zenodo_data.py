#!/usr/bin/env python3


from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import yaml

from pathlib import Path


def safe_extract(tar_path: Path, dest_dir: Path) -> None:
    """Extract a tar.gz, refusing any member that would land outside dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_resolved = dest_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            member_path = (dest_dir / member.name).resolve()
            if dest_resolved not in member_path.parents and member_path != dest_resolved:
                raise SystemExit(f"Refusing to extract unsafe path from {tar_path.name}: {member.name}")
        tar.extractall(dest_dir)


def report_top_level_coverage(resources_output_dir: Path) -> None:
    """Print what actually landed in resources/output/, so you can sanity
    check it against the Included/Excluded lists in manifests.txt."""
    if not resources_output_dir.is_dir():
        return
    entries = sorted(p.name for p in resources_output_dir.iterdir() if p.is_dir())
    print(f"\n[coverage] {resources_output_dir}/ now contains {len(entries)} top-level run directories:")
    for e in entries:
        print(f"  - {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    args = ap.parse_args()

    manifest = yaml.safe_load(args.manifest.read_text())
    raw_dir = args.repo_root / manifest["raw_dir"]

    for entry in manifest["files"]:
        src = raw_dir / entry["name"]
        extract_to = entry.get("extract_to")

        if not src.exists():
            raise SystemExit(f"Missing {src}; run './run.sh download' first.")

        if extract_to is None:
            continue  # file's final home *is* raw_dir; nothing more to do

        target_dir = args.repo_root / extract_to
        if entry.get("extract"):
            print(f"[extract] {entry['name']} -> {target_dir}/")
            safe_extract(src, target_dir)
        else:
            target_dir.mkdir(parents=True, exist_ok=True)
            dst = target_dir / entry["name"]
            print(f"[copy] {entry['name']} -> {dst}")
            shutil.copy2(src, dst)

    report_top_level_coverage(args.repo_root / "resources" / "output")
    print("Extraction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
