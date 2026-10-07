#!/usr/bin/env python3
"""Print the benchmark tiers recorded in benchmarks/benchmark_manifest.json."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "benchmarks" / "benchmark_manifest.json"


def main() -> None:
    data = json.loads(MANIFEST.read_text())
    print("Measured tiers")
    for tier in data["measured_in_artifact"]:
        print(f"- {tier['tier']}: {tier['purpose']}")
        designs = tier["designs"]
        if designs and isinstance(designs[0], dict):
            names = ", ".join(item["name"] for item in designs)
        else:
            names = ", ".join(designs)
        print(f"  designs: {names}")

    print("\nLarge-RTL targets from prior work")
    for source in data["large_rtl_targets_from_prior_work"]:
        print(f"- {source['source']}")
        for design in source["designs"]:
            print(f"  {design['name']}: {design['workload']}")

    print("\nLocal external checkouts")
    for repo in data["external_checkouts"]:
        print(f"- {repo['name']}: {repo['commit']} at {repo['path']}")


if __name__ == "__main__":
    main()
