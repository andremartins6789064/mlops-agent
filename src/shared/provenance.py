"""Provenance data for generated pipeline stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(slots=True)
class StageProvenance:
    """Describe how one generated stage was produced."""

    origin: str
    fallback_reason: str | None = None
    model: str | None = None
    duration_seconds: float = 0.0
    context_truncated: bool = False
    parse_method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return asdict(self)
