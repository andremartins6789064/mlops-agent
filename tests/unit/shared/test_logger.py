from __future__ import annotations

import json
from pathlib import Path

from src.shared.logger import StructuredExecutionLogger


def test_structured_execution_logger_appends_json_events(tmp_path: Path) -> None:
    log_path = tmp_path / "run" / "execution.log"
    logger = StructuredExecutionLogger(log_path)

    logger.log("stage_generated", stage="training", origin="template")
    logger.log("review_completed", iterations=1)

    records = [
        json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["event"] == "stage_generated"
    assert records[0]["stage"] == "training"
    assert records[1]["event"] == "review_completed"
    assert records[1]["iterations"] == 1
