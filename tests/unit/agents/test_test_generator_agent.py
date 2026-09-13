from __future__ import annotations

import json
from pathlib import Path

from src.agents.test_generator import PipelineTestGeneratorAgent
from src.domain.interfaces import ILLMClient


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        return self._response


def test_test_generator_creates_tests_for_all_modules() -> None:
    generator = PipelineTestGeneratorAgent()
    tests = generator.generate_tests(
        generated_modules={
            "feature_engineering": "def load_data() -> None:\n    return None\n",
            "training": "def train_model() -> None:\n    return None\n",
            "inference": "def predict() -> None:\n    return None\n",
            "evaluation": "def evaluate_model() -> None:\n    return None\n",
        }
    )

    assert set(tests.keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert "def test_feature_eng_load_and_split" in tests["feature_engineering"]
    assert "parents[1] / 'src'" in tests["feature_engineering"]


def test_test_generator_uses_llm_test_code_when_available() -> None:
    llm_response = json.dumps(
        {"test_code": "def test_custom() -> None:\n    assert True\n"}
    )
    generator = PipelineTestGeneratorAgent(llm_client=_StubLLMClient(llm_response))

    tests = generator.generate_tests(
        generated_modules={
            "feature_engineering": "def load_data() -> None:\n    return None\n"
        }
    )

    assert tests["feature_engineering"].startswith("def test_custom")


def test_test_generator_writes_tests_to_disk(tmp_path: Path) -> None:
    generator = PipelineTestGeneratorAgent()
    file_map = generator.write_tests(
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
        output_dir=str(tmp_path),
    )

    assert Path(file_map["training"]).exists()
