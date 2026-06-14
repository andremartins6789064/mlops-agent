from __future__ import annotations

from src.application.analyze_quality import analyze_quality
from src.application.validate_output import ValidationResult


def test_analyze_quality_maps_validation_to_domain_metrics() -> None:
    validation = ValidationResult(
        lint_errors=2,
        type_errors=1,
        test_coverage=86.0,
        lint_output="",
        type_output="",
        test_output="",
        lint_exit_code=1,
        type_exit_code=1,
        test_exit_code=0,
    )

    metrics = analyze_quality(validation, review_iterations=2)

    assert metrics.lint_errors == 2
    assert metrics.type_errors == 1
    assert metrics.test_coverage == 86.0
    assert metrics.review_iterations == 2
