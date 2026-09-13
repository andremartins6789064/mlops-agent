"""Progress events emitted by the conversion workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Describe one completed conversion phase."""

    phase: str
    message: str
    completed: int
    total: int

    @property
    def fraction(self) -> float:
        """Return completion as a value between zero and one."""
        return self.completed / self.total if self.total else 0.0


ProgressCallback = Callable[[ProgressEvent], None]
