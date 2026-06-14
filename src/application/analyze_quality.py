from __future__ import annotations

from src.application.validate_output import ValidationResult
from src.domain.value_objects import QualityMetrics


def analyze_quality(
    validation_result: ValidationResult,
    *,
    review_iterations: int,
) -> QualityMetrics:
    """Convert validator output into domain quality metrics."""
    return QualityMetrics(
        lint_errors=validation_result.lint_errors,
        type_errors=validation_result.type_errors,
        test_coverage=validation_result.test_coverage,
        review_iterations=review_iterations,
    )
