from __future__ import annotations

import json
from pathlib import Path

from src.agents.test_generator import PipelineTestGeneratorAgent
from src.domain.interfaces import ILLMClient
from src.shared.python_source import GENERATED_TEST_SRC_BOOTSTRAP


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
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["feature_engineering"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["training"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["inference"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["evaluation"]


def _llm_test_with_src_bootstrap(*, stage_name: str) -> str:
    return (
        "from __future__ import annotations\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"{GENERATED_TEST_SRC_BOOTSTRAP}\n"
        f"import {stage_name} as stage_module\n\n"
        "def test_custom() -> None:\n"
        "    assert True\n"
    )


def test_test_generator_uses_llm_test_code_when_available() -> None:
    llm_response = json.dumps(
        {"test_code": _llm_test_with_src_bootstrap(stage_name="feature_engineering")}
    )
    generator = PipelineTestGeneratorAgent(llm_client=_StubLLMClient(llm_response))

    tests = generator.generate_tests(
        generated_modules={
            "feature_engineering": "def load_data() -> None:\n    return None\n"
        }
    )

    assert "def test_custom" in tests["feature_engineering"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["feature_engineering"]


def test_test_generator_accepts_fenced_python_with_surrounding_prose() -> None:
    response = (
        "Here is the test file:\n"
        f"```python\n{_llm_test_with_src_bootstrap(stage_name='training')}```\n"
        "Done."
    )
    generator = PipelineTestGeneratorAgent(llm_client=_StubLLMClient(response))

    tests = generator.generate_tests(
        generated_modules={"training": "def train_model() -> None:\n    pass\n"}
    )

    assert "def test_custom" in tests["training"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["training"]


def test_test_generator_prompt_requires_exact_module_signatures() -> None:
    generator = PipelineTestGeneratorAgent()

    prompt = generator._build_prompt(
        stage_name="training",
        module_code=(
            "def train_model(features: list[int]) -> object:\n    return features\n"
        ),
    )

    assert "exact signature" in prompt
    assert "Do not invent functions" in prompt
    assert "Do not call a function with arguments" in prompt
    assert 'parents[1] / "src"' in prompt
    assert "without extra PYTHONPATH" in prompt


def test_test_generator_accepts_raw_python_response() -> None:
    generator = PipelineTestGeneratorAgent(
        llm_client=_StubLLMClient(_llm_test_with_src_bootstrap(stage_name="training"))
    )

    tests = generator.generate_tests(
        generated_modules={"training": "def train_model() -> None:\n    pass\n"}
    )

    assert "def test_custom" in tests["training"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["training"]


def test_test_generator_falls_back_when_llm_omits_src_bootstrap() -> None:
    generator = PipelineTestGeneratorAgent(
        llm_client=_StubLLMClient("def test_custom() -> None:\n    assert True\n")
    )

    tests = generator.generate_tests(
        generated_modules={"training": "def train_model() -> None:\n    pass\n"}
    )

    assert "def test_training_train_and_save_model" in tests["training"]
    assert GENERATED_TEST_SRC_BOOTSTRAP in tests["training"]


def test_test_generator_falls_back_when_llm_test_is_invalid() -> None:
    generator = PipelineTestGeneratorAgent(
        llm_client=_StubLLMClient(
            "```python\ndef test_custom() -> None:\n    return X = value\n```"
        )
    )

    tests = generator.generate_tests(
        generated_modules={"training": "def train_model() -> None:\n    pass\n"}
    )

    assert "def test_training_train_and_save_model" in tests["training"]


def test_test_generator_writes_tests_to_disk(tmp_path: Path) -> None:
    generator = PipelineTestGeneratorAgent()
    file_map = generator.write_tests(
        generated_tests={"training": "def test_training() -> None:\n    assert True\n"},
        output_dir=str(tmp_path),
    )

    assert Path(file_map["training"]).exists()
