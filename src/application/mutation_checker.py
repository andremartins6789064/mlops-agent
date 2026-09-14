"""Mutation-based validation for generated test suites."""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class FunctionTarget:
    """Identify one top-level function in a generated module."""

    path: Path
    name: str
    lineno: int


@dataclass(slots=True)
class MutationResult:
    """Summary of mutation detection by the generated test suite."""

    total_mutations: int
    killed_mutations: int
    survived_mutations: int
    timed_out_mutations: int = 0
    errors: list[str] | None = None
    baseline_passed: bool = True
    baseline_error: str | None = None

    @property
    def mutation_score(self) -> float:
        """Return the fraction of mutations detected by the test suite."""
        if not self.baseline_passed or self.total_mutations == 0:
            return 0.0
        return self.killed_mutations / self.total_mutations

    @property
    def passed(self) -> bool:
        """Return whether every mutation was detected."""
        return (
            self.baseline_passed
            and self.total_mutations > 0
            and self.survived_mutations == 0
        )

    @property
    def status(self) -> str:
        """Return a reportable outcome for the mutation check."""
        if not self.baseline_passed:
            return "suite_invalida"
        return "aprovada" if self.passed else "reprovada"


def run_mutation_check(
    project_dir: str,
    *,
    timeout_seconds: int = 60,
) -> MutationResult:
    """Mutate each top-level generated function and run its test suite."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    root = Path(project_dir).resolve()
    baseline = _run_tests(root, timeout_seconds)
    if baseline[0] is False:
        error = baseline[1] or "baseline test suite failed"
        return MutationResult(
            total_mutations=0,
            killed_mutations=0,
            survived_mutations=0,
            errors=[error],
            baseline_passed=False,
            baseline_error=error,
        )

    targets = _find_function_targets(root / "src")
    errors: list[str] = []
    killed = 0
    survived = 0
    timed_out = 0

    for target in targets:
        with tempfile.TemporaryDirectory(prefix="mlops-mutation-") as directory:
            mutated_root = Path(directory) / "project"
            shutil.copytree(root, mutated_root)
            mutation_path = mutated_root / target.path.relative_to(root)
            try:
                _mutate_function(mutation_path, target)
                completed = subprocess.run(
                    [sys.executable, "-m", "pytest", "tests", "-q"],
                    cwd=mutated_root,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                survived += 1
                timed_out += 1
                continue

            if completed.returncode == 0:
                survived += 1
            else:
                killed += 1
                if completed.returncode < 0:
                    errors.append(
                        f"{target.path}:{target.lineno}:{target.name} "
                        f"terminated by signal {-completed.returncode}"
                    )

    return MutationResult(
        total_mutations=len(targets),
        killed_mutations=killed,
        survived_mutations=survived,
        timed_out_mutations=timed_out,
        errors=errors or None,
    )


def _run_tests(root: Path, timeout_seconds: int) -> tuple[bool, str | None]:
    """Run the unmodified suite before attempting any mutation."""
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, f"baseline test suite timed out after {timeout_seconds}s"

    if completed.returncode == 0:
        return True, None
    output = (completed.stdout + "\n" + completed.stderr).strip()
    return (
        False,
        f"baseline test suite failed with status {completed.returncode}: {output}",
    )


def _find_function_targets(source_root: Path) -> list[FunctionTarget]:
    targets: list[FunctionTarget] = []
    if not source_root.exists():
        return targets
    for path in sorted(source_root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        targets.extend(
            FunctionTarget(path=path, name=node.name, lineno=node.lineno)
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        )
    return targets


def _mutate_function(path: Path, target: FunctionTarget) -> None:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == target.name
            and node.lineno == target.lineno
        ):
            node.body = [
                ast.Raise(
                    exc=ast.Call(
                        func=ast.Name(id="AssertionError", ctx=ast.Load()),
                        args=[ast.Constant(value="mutation")],
                        keywords=[],
                    ),
                    cause=None,
                )
            ]
            break
    path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")
