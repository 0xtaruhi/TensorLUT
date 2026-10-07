#!/usr/bin/env python3
"""Print local large-RTL checkout and smoke-test status."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "benchmark_manifest.json"


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "missing"


def main() -> None:
    data = json.loads(MANIFEST.read_text())

    print("External checkouts")
    for repo in data["external_checkouts"]:
        path = ROOT / repo["path"]
        actual = git_commit(path)
        marker = "ok" if actual == repo["commit"] else f"manifest={repo['commit']}"
        print(f"- {repo['name']}: {actual} ({marker})")
        print(f"  {repo['path']} | rtl_files={repo['rtl_files']} | {repo['status']}")

    print("\nLarge-RTL smoke tests")
    for test in data["large_rtl_smoke_tests"]:
        result = ROOT / test["result_json"]
        exists = "present" if result.exists() else "missing"
        summary = test["result"]
        fields = ", ".join(
            f"{k}={v}" for k, v in summary.items() if k.endswith("bits") or k == "cells"
        )
        print(f"- {test['name']}: {exists} | {fields}")
        print(f"  {test['result_json']}")

    print("\nTensorLUT large-RTL compile rows")
    for group in data.get("large_rtl_tensorlut_compile", []):
        print(f"- {group['name']} | {group['script']}")
        for row in group["rows"]:
            print(
                f"  {row['module']}: dff={row['dff']} lut={row['lut']} "
                f"layers={row['layers']} mono={row['mono_total']} "
                f"chunks={row['chunks_tau512']}"
            )


if __name__ == "__main__":
    main()
