from __future__ import annotations

from typing import Any

from src.agents.reviewer import ReviewerAgent
from src.application.validate_output import ValidationResult
from src.domain.interfaces import ILLMClient


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.calls += 1
        return self._response


def test_reviewer_stops_when_validation_is_clean() -> None:
    def _validator(project_dir: str) -> ValidationResult:
        return ValidationResult(
            lint_errors=0,
            type_errors=0,
            test_coverage=91.0,
            lint_output="",
            type_output="",
            test_output="",
            lint_exit_code=0,
            type_exit_code=0,
            test_exit_code=0,
        )

    reviewer = ReviewerAgent(validator=_validator, max_iterations=2)
    result = reviewer.review(
        project_dir=".",
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
    )

    assert result.iterations == 0
    assert result.quality_metrics.test_coverage == 91.0


def test_reviewer_runs_fixer_until_validation_passes(tmp_path: Any) -> None:
    calls = {"count": 0}

    def _validator(project_dir: str) -> ValidationResult:
        if calls["count"] == 0:
            return ValidationResult(
                lint_errors=1,
                type_errors=0,
                test_coverage=70.0,
                lint_output="E501",
                type_output="",
                test_output="",
                lint_exit_code=1,
                type_exit_code=0,
                test_exit_code=0,
            )
        return ValidationResult(
            lint_errors=0,
            type_errors=0,
            test_coverage=88.0,
            lint_output="",
            type_output="",
            test_output="",
            lint_exit_code=0,
            type_exit_code=0,
            test_exit_code=0,
        )

    def _fixer(modules: dict[str, str], validation: ValidationResult) -> dict[str, str]:
        calls["count"] += 1
        assert validation.lint_errors == 1
        return modules

    reviewer = ReviewerAgent(validator=_validator, max_iterations=2)
    result = reviewer.review(
        project_dir=str(tmp_path),
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
        fixer=_fixer,
    )

    assert calls["count"] == 1
    assert result.iterations == 1
    assert result.quality_metrics.lint_errors == 0


def test_reviewer_uses_llm_fixer_when_no_manual_fixer(tmp_path: Any) -> None:
    calls = {"count": 0}
    llm = _StubLLMClient(
        '{"module_code":"def train_model() -> None:\\n    return None\\n"}'
    )

    def _validator(project_dir: str) -> ValidationResult:
        if calls["count"] == 0:
            calls["count"] += 1
            return ValidationResult(
                lint_errors=1,
                type_errors=0,
                test_coverage=90.0,
                lint_output="E501",
                type_output="",
                test_output="",
                lint_exit_code=1,
                type_exit_code=0,
                test_exit_code=0,
            )
        return ValidationResult(
            lint_errors=0,
            type_errors=0,
            test_coverage=90.0,
            lint_output="",
            type_output="",
            test_output="",
            lint_exit_code=0,
            type_exit_code=0,
            test_exit_code=0,
        )

    reviewer = ReviewerAgent(validator=_validator, llm_client=llm, max_iterations=2)
    result = reviewer.review(
        project_dir=str(tmp_path),
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
    )

    assert llm.calls >= 1
    assert result.iterations == 1
