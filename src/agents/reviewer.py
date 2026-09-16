from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.application.validate_output import ValidationResult, validate_output
from src.domain.interfaces import ILLMClient
from src.domain.pipeline_contract import entrypoint_source
from src.domain.value_objects import QualityMetrics
from src.shared.llm_parsing import parse_json_object, parse_python_block


@dataclass(slots=True)
class ReviewerResult:
    """Result from reviewer loop execution."""

    validation_result: ValidationResult
    quality_metrics: QualityMetrics
    iterations: int
    generated_modules: dict[str, str]
    review_incomplete: bool = False
    review_error: str | None = None


class ReviewBudgetExceededError(RuntimeError):
    """Raised when the optional Reviewer reaches its configured budget."""


class ReviewerAgent:
    """Run quality checks and trigger limited auto-correction loop."""

    def __init__(
        self,
        *,
        validator: Callable[[str], ValidationResult] | None = None,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/reviewer.txt",
        max_iterations: int = 1,
        context_budget_tokens: int = 2_000,
        inter_call_delay_seconds: float = 0.0,
        max_llm_calls: int = 1,
        max_review_seconds: float = 120.0,
    ) -> None:
        if context_budget_tokens <= 0:
            raise ValueError("context_budget_tokens must be positive")
        if inter_call_delay_seconds < 0:
            raise ValueError("inter_call_delay_seconds cannot be negative")
        if max_llm_calls <= 0:
            raise ValueError("max_llm_calls must be positive")
        if max_review_seconds <= 0:
            raise ValueError("max_review_seconds must be positive")
        self._validator = validator or validate_output
        self._llm_client = llm_client
        self._prompt_path = prompt_path
        self._max_iterations = max_iterations
        self._context_budget_tokens = context_budget_tokens
        self._inter_call_delay_seconds = inter_call_delay_seconds
        self._max_llm_calls = max_llm_calls
        self._max_review_seconds = max_review_seconds
        self._current_tests: dict[str, str] = {}
        self._last_review_incomplete = False
        self._last_review_error: str | None = None
        self._review_started = 0.0
        self._llm_calls = 0

    def review(
        self,
        *,
        project_dir: str,
        generated_modules: dict[str, str],
        generated_tests: dict[str, str],
        fixer: Callable[[dict[str, str], ValidationResult], dict[str, str]]
        | None = None,
    ) -> ReviewerResult:
        """Validate generated project and optionally fix code up to max iterations."""
        project_path = Path(project_dir)
        iterations = 0
        current_modules = dict(generated_modules)
        self._current_tests = dict(generated_tests)
        self._last_review_incomplete = False
        self._last_review_error = None
        self._review_started = time.monotonic()
        self._llm_calls = 0
        active_fixer = fixer
        if active_fixer is None and self._llm_client is not None:
            active_fixer = self._fix_with_llm

        self._write_entrypoint(project_dir=str(project_path))
        validation = self._validator(str(project_path))

        while (
            validation.has_errors
            and active_fixer is not None
            and iterations < self._max_iterations
        ):
            iterations += 1
            current_modules = active_fixer(current_modules, validation)
            self._write_modules(
                project_dir=str(project_path), generated_modules=current_modules
            )
            self._write_tests(
                project_dir=str(project_path), generated_tests=generated_tests
            )
            validation = self._validator(str(project_path))

        quality = QualityMetrics(
            lint_errors=validation.lint_errors,
            type_errors=validation.type_errors,
            test_coverage=validation.test_coverage,
            review_iterations=iterations,
        )
        return ReviewerResult(
            validation_result=validation,
            quality_metrics=quality,
            iterations=iterations,
            generated_modules=current_modules,
            review_incomplete=self._last_review_incomplete,
            review_error=self._last_review_error,
        )

    def _write_entrypoint(self, *, project_dir: str) -> None:
        """Write `src/main.py` before validation so coverage matches the ZIP."""
        root = Path(project_dir) / "src"
        root.mkdir(parents=True, exist_ok=True)
        (root / "main.py").write_text(entrypoint_source(), encoding="utf-8")

    def _write_modules(
        self, *, project_dir: str, generated_modules: dict[str, str]
    ) -> None:
        root = Path(project_dir) / "src"
        root.mkdir(parents=True, exist_ok=True)
        for stage_name, source in generated_modules.items():
            (root / f"{stage_name}.py").write_text(source, encoding="utf-8")

    def _write_tests(
        self, *, project_dir: str, generated_tests: dict[str, str]
    ) -> None:
        tests_root = Path(project_dir) / "tests"
        tests_root.mkdir(parents=True, exist_ok=True)
        for stage_name, source in generated_tests.items():
            (tests_root / f"test_{stage_name}.py").write_text(source, encoding="utf-8")

    def _fix_with_llm(
        self, modules: dict[str, str], validation: ValidationResult
    ) -> dict[str, str]:
        if self._llm_client is None:
            return modules
        prompt_template = Path(self._prompt_path).read_text(encoding="utf-8")
        diagnostics = {
            "lint_output": validation.lint_output,
            "type_output": validation.type_output,
            "test_output": validation.test_output,
            "lint_errors": validation.lint_errors,
            "type_errors": validation.type_errors,
            "test_coverage": validation.test_coverage,
            "minimum_coverage": validation.minimum_coverage,
        }
        fixed_modules = dict(modules)
        findings: list[dict[str, str]] = []
        affected_stages = self._affected_stages(modules, validation)
        for stage_name in affected_stages:
            if self._llm_calls >= self._max_llm_calls:
                self._mark_incomplete("Reviewer call budget exceeded")
                break
            payload = {
                "stage_name": stage_name,
                "module_code": modules[stage_name],
                "tests": self._read_stage_tests(stage_name),
                "diagnostics": diagnostics,
            }
            prompt, truncated = self._build_prompt(prompt_template, payload)
            try:
                self._check_budget()
                raw_response = self._generate(prompt)
            except Exception as exc:  # noqa: BLE001
                self._mark_incomplete(f"{stage_name}: {exc}")
                findings.append(
                    {
                        "stage": stage_name,
                        "status": "error",
                        "truncated": str(truncated),
                    }
                )
                continue
            parsed = parse_json_object(raw_response).value
            if isinstance(parsed, dict):
                module_code = parsed.get("module_code")
                if isinstance(module_code, str) and "def " in module_code:
                    fixed_modules[stage_name] = module_code.strip()
                    findings.append(
                        {
                            "stage": stage_name,
                            "status": "fixed",
                            "truncated": str(truncated).lower(),
                        }
                    )
                    continue
            python_code = parse_python_block(raw_response).value
            if isinstance(python_code, str):
                fixed_modules[stage_name] = python_code
                findings.append(
                    {
                        "stage": stage_name,
                        "status": "fixed",
                        "truncated": str(truncated).lower(),
                    }
                )
            else:
                self._mark_incomplete(
                    f"{stage_name}: no valid module code in reviewer response"
                )
                findings.append(
                    {
                        "stage": stage_name,
                        "status": "inconclusive",
                        "truncated": str(truncated).lower(),
                    }
                )
        if len(findings) < len(affected_stages) and self._last_review_error is None:
            self._mark_incomplete("Reviewer did not process every affected stage")
        return fixed_modules

    def _affected_stages(
        self, modules: dict[str, str], validation: ValidationResult
    ) -> list[str]:
        """Select stages named in errors, or all when attribution is unknown."""
        diagnostics = " ".join(
            (
                validation.lint_output,
                validation.type_output,
                _without_coverage_report(validation.test_output),
            )
        ).lower()
        affected = [
            stage
            for stage in modules
            if _stage_named_in_diagnostics(stage, diagnostics)
        ]
        return affected or list(modules)

    def _check_budget(self) -> None:
        """Stop optional review work when its time budget is exhausted."""
        elapsed = time.monotonic() - self._review_started
        if elapsed >= self._max_review_seconds:
            raise ReviewBudgetExceededError(
                f"Reviewer time budget exceeded ({self._max_review_seconds:.1f}s)"
            )

    def _mark_incomplete(self, error: str) -> None:
        """Record an inconclusive review without turning it into approval."""
        self._last_review_incomplete = True
        self._last_review_error = error

    def _read_stage_tests(self, stage_name: str) -> str:
        """Read the relevant generated test when it is available."""
        return self._current_tests.get(stage_name, "")

    def _build_prompt(
        self, prompt_template: str, payload: dict[str, Any]
    ) -> tuple[str, bool]:
        """Build a bounded prompt and report whether content was truncated."""
        prefix = (
            f"{prompt_template}\n\nReturn JSON with field 'module_code'.\n\n"
            "Context JSON:\n"
        )
        budget_chars = self._context_budget_tokens * 4
        serialized = json.dumps(payload, ensure_ascii=True)
        truncated = len(prefix) + len(serialized) > budget_chars
        if truncated:
            compacted = dict(payload)
            available = max(0, budget_chars - len(prefix))
            for field in ("module_code", "tests"):
                value = compacted.get(field)
                if isinstance(value, str):
                    compacted[field] = value[: max(0, available // 2)]
            serialized = json.dumps(compacted, ensure_ascii=True)
            if len(prefix) + len(serialized) > budget_chars:
                compacted["module_code"] = ""
                compacted["tests"] = ""
                serialized = json.dumps(compacted, ensure_ascii=True)
        return f"{prefix}{serialized}", truncated

    def _generate(self, prompt: str) -> str:
        """Call the LLM with configurable spacing between requests."""
        if self._llm_client is None:
            return ""
        self._check_budget()
        if self._inter_call_delay_seconds:
            time.sleep(self._inter_call_delay_seconds)
        self._check_budget()
        self._llm_calls += 1
        response = self._llm_client.generate(prompt=prompt)
        self._check_budget()
        return str(response)

    def _consolidate_findings(self, findings: list[dict[str, str]]) -> None:
        """Ask for a short consolidation without resending source code."""
        if self._llm_client is None or not findings:
            return
        payload = {"partial_findings": findings}
        prefix = (
            "You are consolidating partial code-review findings.\n"
            "Return JSON with fields 'issues' and 'fix_plan'.\n"
            "Consolidate only these partial findings; do not request source code.\n"
            "Findings JSON:\n"
        )
        budget_chars = self._context_budget_tokens * 2
        serialized = json.dumps(payload, ensure_ascii=True)[:budget_chars]
        response = self._generate(f"{prefix}{serialized}")
        parsed = parse_json_object(response).value
        if not isinstance(parsed, dict) or not {
            "issues",
            "fix_plan",
        }.issubset(parsed):
            self._last_review_incomplete = True
            self._last_review_error = "consolidation: invalid reviewer response"


def _without_coverage_report(test_output: str) -> str:
    """Drop pytest-cov tables so they do not mark every stage as affected."""
    lower = test_output.lower()
    cut = len(test_output)
    for marker in ("tests coverage", "coverage: platform", "----- coverage"):
        idx = lower.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return test_output[:cut]


def _stage_named_in_diagnostics(stage: str, diagnostics: str) -> bool:
    """True when a diagnostic path names the stage module or its tests."""
    stage_l = stage.lower()
    return (
        f"test_{stage_l}.py" in diagnostics
        or f"/{stage_l}.py" in diagnostics
        or f" {stage_l}.py" in diagnostics
        or diagnostics.startswith(f"{stage_l}.py")
    )
