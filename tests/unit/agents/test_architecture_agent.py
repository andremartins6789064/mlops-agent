from __future__ import annotations

import json

from src.agents.architecture_agent import ArchitectureAgent
from src.domain.interfaces import ILLMClient


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt = ""

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.last_prompt = prompt
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
    assert result["entrypoint"]["path"] == "src/main.py"
    assert result["entrypoint"]["metric_name"] == "final_mse"


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


def test_architecture_agent_includes_user_feedback_in_prompt() -> None:
    llm = _StubLLMClient("not-json")
    agent = ArchitectureAgent(llm_client=llm, user_feedback="Prefer smaller functions")
    agent.plan({"libraries": []})
    assert "User feedback" in llm.last_prompt
    assert "Prefer smaller functions" in llm.last_prompt
