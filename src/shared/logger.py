"""Structured execution logging utilities."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class StructuredExecutionLogger:
    """Append structured JSON events to one execution log."""

    def __init__(self, log_path: str | Path) -> None:
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def log_path(self) -> Path:
        """Return the path of the execution log."""
        return self._log_path

    def log(self, event: str, **fields: Any) -> None:
        """Append one timestamped event to the execution log."""
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
