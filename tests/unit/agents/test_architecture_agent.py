from __future__ import annotations

import json

from src.agents.architecture_agent import ArchitectureAgent
from src.domain.interfaces import ILLMClient


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return self._response


def test_architecture_agent_uses_llm_result_when_valid() -> None:
    payload = {
        "modules": {
            "feature_engineering": {
                "functions": ["load_data", "split_data"],
                "inputs": "dataset",
                "outputs": "splits",
            },
            "training": {
                "functions": ["train_model"],
                "inputs": "splits",
                "outputs": "model",
            },
            "inference": {
                "functions": ["predict"],
                "inputs": "model and X_new",
                "outputs": "predictions",
            },
            "evaluation": {
                "functions": ["evaluate_model"],
                "inputs": "y_true and y_pred",
                "outputs": "metrics",
            },
        }
    }
    agent = ArchitectureAgent(llm_client=_StubLLMClient(json.dumps(payload)))

    result = agent.plan({"libraries": []})

    assert result["modules"]["training"]["functions"] == ["train_model"]


def test_architecture_agent_falls_back_when_llm_response_is_invalid() -> None:
    agent = ArchitectureAgent(llm_client=_StubLLMClient("not-json"))

    result = agent.plan({"libraries": ["sklearn"]})

    assert "scale_features" in result["modules"]["feature_engineering"]["functions"]


def test_architecture_agent_falls_back_when_llm_schema_is_invalid() -> None:
    agent = ArchitectureAgent(llm_client=_StubLLMClient(json.dumps({"modules": {}})))

    result = agent.plan({"libraries": []})

    assert set(result["modules"].keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
