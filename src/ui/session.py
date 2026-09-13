from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.agents.orchestrator import OrchestrationResult

CONFIG_KEY = "mlops_config"
RESULT_KEY = "mlops_result"
ERROR_KEY = "mlops_error"
NOTEBOOK_NAME_KEY = "mlops_notebook_name"
NOTEBOOK_BYTES_KEY = "mlops_notebook_bytes"


@dataclass(slots=True)
class UIConfig:
    """Configuration captured from the Streamlit form."""

    use_llm: bool = False
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: str = "ollama"
    llm_model: str = "smollm2:1.7b"
    output_dir: str = "output/ui_runs"


def default_config() -> UIConfig:
    """Return default values for UI configuration."""
    return UIConfig()


def get_result(state: Mapping[str, Any]) -> OrchestrationResult | None:
    """Safely read the orchestration result from session state."""
    value = state.get(RESULT_KEY)
    if isinstance(value, OrchestrationResult):
        return value
    return None
