from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ConversionStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(slots=True)
class QualityMetrics:
    lint_errors: int = 0
    type_errors: int = 0
    test_coverage: float = 0.0
    review_iterations: int = 0

    def __post_init__(self) -> None:
        if self.test_coverage < 0 or self.test_coverage > 100:
            msg = "test_coverage must be between 0 and 100."
            raise ValueError(msg)

    def meets_minimum_coverage(self, minimum_coverage: float = 80.0) -> bool:
        return self.test_coverage >= minimum_coverage
