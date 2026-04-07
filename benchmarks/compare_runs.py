#!/usr/bin/env python3
"""Compare structured benchmark result files honestly.

This script refuses to present apples-to-oranges numbers as if they were
comparable. If workload keys differ, it prints the mismatches and exits non-zero.
"""

from __future__ import annotations

import argparse
import sys

from benchmarks.benchmarking import comparison_report, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Compare two benchmark result JSON files")
    parser.add_argument("left")
    parser.add_argument("right")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    left = load_json(args.left)
    right = load_json(args.right)

    report = comparison_report(right, left)
    if not report["comparable"]:
        print("Runs are not comparable. Mismatched workload axes:")
        for mismatch in report["mismatches"]:
            print(f"- {mismatch}")
        return 2

    left_name = left.get("system", {}).get("name", args.left)
    right_name = right.get("system", {}).get("name", args.right)
    print(f"Comparable workload confirmed: {left.get('workload', {})}")
    print(f"{'metric':<20} {'left':>12} {'right':>12} {'delta(right-left)':>18}")
    for name, values in report["metrics"].items():
        lval = values["baseline"]
        rval = values["current"]
        delta = values["delta"]
        print(f"{name:<20} {str(lval):>12} {str(rval):>12} {str(round(delta, 3)):>18}")
    print(f"left={left_name}  right={right_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
