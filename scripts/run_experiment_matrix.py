"""Run the notebook/model experiment matrix and write CSV evidence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.experiment_harness import (  # noqa: E402
    DEFAULT_EXPERIMENT_NOTEBOOKS,
    InvalidExperimentNotebookError,
    run_experiment_matrix,
)


def main() -> int:
    """Parse options, run the matrix, and report the CSV path."""
    parser = argparse.ArgumentParser(
        description="Run a headless notebook x model experiment matrix."
    )
    parser.add_argument(
        "--notebooks",
        nargs="+",
        default=list(DEFAULT_EXPERIMENT_NOTEBOOKS),
        help=(
            "Experiment notebooks. Defaults to the Etapa 9.0 junior and senior "
            "notebooks. Parser fixtures under tests/fixtures/ are rejected."
        ),
    )
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--output-csv", default="output/experiments/results.csv")
    parser.add_argument("--output-root", default="output/experiments/runs")
    parser.add_argument(
        "--pipeline-command",
        help="Command template with optional {output_dir} and {notebook}.",
    )
    parser.add_argument("--tolerance", type=float, default=0.05)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--llm-timeout", type=float, default=300.0)
    parser.add_argument("--llm-retries", type=int, default=3)
    parser.add_argument("--llm-retry-backoff", type=float, default=5.0)
    parser.add_argument(
        "--enable-review",
        action="store_true",
        help="Enable the optional LLM Reviewer (disabled by default).",
    )
    parser.add_argument("--review-max-calls", type=int, default=1)
    parser.add_argument("--review-max-seconds", type=float, default=120.0)
    parser.add_argument(
        "--max-run-seconds",
        type=int,
        default=180,
        help="Maximum wall-clock time per notebook/model run.",
    )
    parser.add_argument(
        "--skip-mutation",
        action="store_true",
        help="Skip the expensive anti-empty-test mutation pass.",
    )
    args = parser.parse_args()

    try:
        rows = run_experiment_matrix(
            notebooks=args.notebooks,
            models=args.models,
            repetitions=args.repetitions,
            output_csv=args.output_csv,
            output_root=args.output_root,
            pipeline_command=args.pipeline_command,
            tolerance=args.tolerance,
            timeout_seconds=args.timeout,
            llm_timeout_seconds=args.llm_timeout,
            llm_max_retries=args.llm_retries,
            llm_retry_backoff_seconds=args.llm_retry_backoff,
            enable_review=args.enable_review,
            review_max_llm_calls=args.review_max_calls,
            review_max_seconds=args.review_max_seconds,
            max_run_seconds=args.max_run_seconds,
            run_mutation=not args.skip_mutation,
        )
    except InvalidExperimentNotebookError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"completed_runs={len(rows)}")
    print(f"csv={args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
