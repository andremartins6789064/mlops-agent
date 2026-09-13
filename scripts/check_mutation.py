"""CLI for checking whether generated tests detect function mutations."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.application.mutation_checker import run_mutation_check  # noqa: E402


def main() -> int:
    """Run the mutation check and print a JSON summary."""
    parser = argparse.ArgumentParser(
        description="Check whether generated tests detect sabotaged functions."
    )
    parser.add_argument("project_dir", help="Generated project directory.")
    parser.add_argument("--timeout", type=int, default=60)
    args = parser.parse_args()

    result = run_mutation_check(args.project_dir, timeout_seconds=args.timeout)
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    print(f"mutation_score={result.mutation_score:.3f}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
