#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys

from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    pip_freeze = subprocess.run(
        ["uv", "pip", "freeze", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()

    report = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "pip_freeze": pip_freeze,
    }
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Wrote environment report to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
