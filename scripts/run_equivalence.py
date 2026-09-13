"""CLI for comparing a notebook metric with a generated pipeline metric."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.equivalence_runner import run_equivalence  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description="Compare a printed notebook metric with a pipeline metric."
    )
    parser.add_argument(
        "--notebook",
        required=True,
        help="Path to the source notebook.",
    )
    parser.add_argument(
        "--pipeline-command",
        required=True,
        help='Command to run the generated pipeline, e.g. "python run.py".',
    )
    parser.add_argument(
        "--pipeline-dir",
        default=".",
        help="Working directory for the generated pipeline.",
    )
    parser.add_argument("--metric", default="final_mse")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Absolute metric tolerance (default: 0.05).",
    )
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main() -> int:
    """Run the equivalence comparison and print JSON evidence."""
    args = build_parser().parse_args()
    result = run_equivalence(
        notebook_path=args.notebook,
        pipeline_command=shlex.split(args.pipeline_command),
        pipeline_dir=args.pipeline_dir,
        metric_name=args.metric,
        tolerance=args.tolerance,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    return 0 if result.status.value == "equivalente" else 1


if __name__ == "__main__":
    raise SystemExit(main())
