"""Reproducible evaluation runner CLI."""
from __future__ import annotations

import argparse
import json

from app import config
from app.evaluation.runner import run_evaluation, run_graph_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-only", action="store_true")
    args = parser.parse_args()
    config.ensure_dirs()
    if args.graph_only:
        report = run_graph_analysis()
        print(json.dumps(report, indent=2))
    else:
        report = run_evaluation()
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
