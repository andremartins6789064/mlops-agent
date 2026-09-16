from __future__ import annotations

import json
import subprocess
import sys
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


def test_training_template_uses_2d_features_and_allows_none_save_return() -> None:
    generator = PipelineTestGeneratorAgent()
    tests = generator.generate_tests(
        generated_modules={"training": "def train_model() -> None:\n    return None\n"}
    )
    training_test = tests["training"]

    assert "train_model([1], [1])" not in training_test
    assert "features = [[1.0], [2.0]]" in training_test
    assert "labels = [1.0, 2.0]" in training_test
    assert "saved is None" in training_test
    assert GENERATED_TEST_SRC_BOOTSTRAP in training_test


def test_training_template_passes_against_linear_regression_like_fit(
    tmp_path: Path,
) -> None:
    src_root = tmp_path / "src"
    tests_root = tmp_path / "tests"
    src_root.mkdir()
    tests_root.mkdir()
    (src_root / "training.py").write_text(
        "from pathlib import Path\n"
        "from typing import Any\n\n"
        "def train_model(train_features: Any, train_labels: Any) -> dict[str, int]:\n"
        "    first = train_features[0]\n"
        "    if not hasattr(first, '__len__') or isinstance(first, (str, bytes)):\n"
        "        raise ValueError('Expected 2D array, got 1D array instead')\n"
        "    return {'coef': 1}\n\n"
        "def save_model(model: object, path: str) -> None:\n"
        "    Path(path).write_text('ok', encoding='utf-8')\n",
        encoding="utf-8",
    )
    generator = PipelineTestGeneratorAgent()
    generated = generator.generate_tests(
        generated_modules={
            "training": (src_root / "training.py").read_text(encoding="utf-8")
        }
    )
    (tests_root / "test_training.py").write_text(
        generated["training"], encoding="utf-8"
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(tests_root / "test_training.py"),
            "-q",
            "-o",
            "addopts=",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
