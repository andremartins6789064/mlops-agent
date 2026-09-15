from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.reviewer import ReviewerAgent
from src.application.validate_output import ValidationResult
from src.domain.interfaces import ILLMClient
from src.domain.pipeline_contract import entrypoint_source


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response
        self.calls = 0
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self._response


def test_reviewer_stops_when_validation_is_clean(tmp_path: Any) -> None:
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
        project_dir=str(tmp_path),
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
    )

    assert result.iterations == 0
    assert result.quality_metrics.test_coverage == 91.0


def test_reviewer_writes_entrypoint_before_validation(tmp_path: Any) -> None:
    seen_before_validation: dict[str, object] = {}

    def _validator(project_dir: str) -> ValidationResult:
        entrypoint = Path(project_dir) / "src" / "main.py"
        seen_before_validation["exists"] = entrypoint.is_file()
        seen_before_validation["source"] = (
            entrypoint.read_text(encoding="utf-8") if entrypoint.is_file() else ""
        )
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

    reviewer = ReviewerAgent(validator=_validator)
    reviewer.review(
        project_dir=str(tmp_path),
        generated_modules={"training": "def train_model() -> None:\n    return None\n"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
    )

    assert seen_before_validation["exists"] is True
    assert seen_before_validation["source"] == entrypoint_source()
    assert (tmp_path / "src" / "main.py").read_text(
        encoding="utf-8"
    ) == entrypoint_source()


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


def test_reviewer_bounds_context_and_limits_calls(
    tmp_path: Any,
) -> None:
    llm = _StubLLMClient(
        '{"module_code":"def train_model() -> None:\\n    return None\\n"}'
    )

    def _validator(project_dir: str) -> ValidationResult:
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

    reviewer = ReviewerAgent(
        validator=_validator,
        llm_client=llm,
        max_iterations=1,
        context_budget_tokens=20,
    )
    reviewer.review(
        project_dir=str(tmp_path),
        generated_modules={"training": "def train_model() -> None:\n" + "x" * 500},
        generated_tests={"training": "def test_training() -> None:\n" + "y" * 500},
    )

    assert len(llm.prompts) == 1
    assert len(llm.prompts[0]) < 800


def test_reviewer_marks_unprocessed_stages_inconclusive(tmp_path: Any) -> None:
    llm = _StubLLMClient(
        '{"module_code":"def train_model() -> None:\\n    return None\\n"}'
    )

    def _validator(project_dir: str) -> ValidationResult:
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

    reviewer = ReviewerAgent(
        validator=_validator,
        llm_client=llm,
        max_llm_calls=1,
    )
    result = reviewer.review(
        project_dir=str(tmp_path),
        generated_modules={
            "training": "def train_model() -> None:\n    pass\n",
            "evaluation": "def evaluate() -> None:\n    pass\n",
        },
        generated_tests={
            "training": "def test_training() -> None:\n    assert True\n",
            "evaluation": "def test_evaluation() -> None:\n    assert True\n",
        },
    )

    assert llm.calls == 1
    assert result.review_incomplete is True


def test_reviewer_marks_invalid_partial_response_incomplete(tmp_path: Any) -> None:
    llm = _StubLLMClient('{"issues":["could not fix"]}')

    def _validator(project_dir: str) -> ValidationResult:
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

    reviewer = ReviewerAgent(
        validator=_validator,
        llm_client=llm,
        max_iterations=1,
    )
    result = reviewer.review(
        project_dir=str(tmp_path),
        generated_modules={"training": "def train_model() -> None:\n    pass\n"},
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
    )

    assert result.review_incomplete is True
    assert result.review_error is not None
