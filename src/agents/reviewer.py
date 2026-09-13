from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.application.validate_output import ValidationResult, validate_output
from src.domain.interfaces import ILLMClient
from src.domain.value_objects import QualityMetrics
from src.shared.llm_parsing import parse_json_object, parse_python_block


@dataclass(slots=True)
class ReviewerResult:
    """Result from reviewer loop execution."""

    validation_result: ValidationResult
    quality_metrics: QualityMetrics
    iterations: int


class ReviewerAgent:
    """Run quality checks and trigger limited auto-correction loop."""

    def __init__(
        self,
        *,
        validator: Callable[[str], ValidationResult] | None = None,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/reviewer.txt",
        max_iterations: int = 2,
    ) -> None:
        self._validator = validator or validate_output
        self._llm_client = llm_client
        self._prompt_path = prompt_path
        self._max_iterations = max_iterations

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
        active_fixer = fixer
        if active_fixer is None and self._llm_client is not None:
            active_fixer = self._fix_with_llm

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
        )

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
        for stage_name, source_code in modules.items():
            payload = {
                "stage_name": stage_name,
                "module_code": source_code,
                "diagnostics": diagnostics,
            }
            prompt = (
                f"{prompt_template}\n\nReturn JSON with field 'module_code'.\n\n"
                f"Context JSON:\n{json.dumps(payload, ensure_ascii=True)}"
            )
            raw_response = self._llm_client.generate(prompt=prompt)
            parsed = parse_json_object(raw_response).value
            if isinstance(parsed, dict):
                module_code = parsed.get("module_code")
                if isinstance(module_code, str) and "def " in module_code:
                    fixed_modules[stage_name] = module_code.strip()
                    continue
            python_code = parse_python_block(raw_response).value
            if isinstance(python_code, str):
                fixed_modules[stage_name] = python_code
        return fixed_modules
