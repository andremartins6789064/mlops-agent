from __future__ import annotations

import json
from pathlib import Path

from src.agents.code_generator import CodeGeneratorAgent
from src.domain.interfaces import ILLMClient
from src.infrastructure.parsers.notebook_parser import NotebookParser


class _StubLLMClient(ILLMClient):
    def __init__(self, response: str) -> None:
        self._response = response
        self.last_prompt = ""
        self.prompts: list[str] = []

    def generate(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self.last_prompt = prompt
        self.prompts.append(prompt)
        return self._response


def _sample_plan() -> dict[str, object]:
    return {
        "modules": {
            "feature_engineering": {"functions": ["load_data", "split_data"]},
            "training": {"functions": ["train_model", "save_model"]},
            "inference": {"functions": ["load_model", "predict"]},
            "evaluation": {"functions": ["evaluate_model"]},
        }
    }


def test_code_generator_creates_four_modules_with_template() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    agent = CodeGeneratorAgent()

    generated = agent.generate_modules(
        notebook=notebook,
        notebook_analysis={"libraries": ["pandas", "sklearn"]},
        architecture_plan=_sample_plan(),
    )

    assert set(generated.keys()) == {
        "feature_engineering",
        "training",
        "inference",
        "evaluation",
    }
    assert "def load_data" in generated["feature_engineering"]
    assert "def train_model" in generated["training"]
    assert "def predict" in generated["inference"]


def test_code_generator_uses_llm_module_code_when_valid_json() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    llm_response = json.dumps(
        {"module_code": "def custom_stage() -> int:\n    return 1\n"}
    )
    agent = CodeGeneratorAgent(llm_client=_StubLLMClient(llm_response))

    generated = agent.generate_modules(
        notebook=notebook,
        notebook_analysis={"libraries": []},
        architecture_plan=_sample_plan(),
    )

    assert "def custom_stage" in generated["feature_engineering"]


def test_code_generator_writes_modules_to_output_dir(tmp_path: Path) -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    agent = CodeGeneratorAgent()

    generated = agent.generate_modules(
        notebook=notebook,
        notebook_analysis={"libraries": []},
        architecture_plan=_sample_plan(),
    )
    file_map = agent.write_modules(
        generated_modules=generated, output_dir=str(tmp_path)
    )

    for module_name in ("feature_engineering", "training", "inference", "evaluation"):
        assert module_name in file_map
        assert Path(file_map[module_name]).exists()


def test_code_generator_applies_stage_feedback_to_template_header() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    agent = CodeGeneratorAgent(
        stage_feedback={"training": "Use train validation split"}
    )
    generated = agent.generate_modules(
        notebook=notebook,
        notebook_analysis={"libraries": []},
        architecture_plan=_sample_plan(),
    )
    assert "User feedback: Use train validation split" in generated["training"]


def test_code_generator_includes_stage_feedback_in_llm_prompt() -> None:
    notebook = NotebookParser().parse("tests/fixtures/simple_regression.ipynb")
    llm_response = json.dumps(
        {"module_code": "def custom_stage() -> int:\n    return 1\n"}
    )
    llm = _StubLLMClient(llm_response)
    agent = CodeGeneratorAgent(
        llm_client=llm, stage_feedback={"feature_engineering": "Add logging"}
    )
    agent.generate_modules(
        notebook=notebook,
        notebook_analysis={"libraries": []},
        architecture_plan=_sample_plan(),
    )
    assert any("stage_feedback" in prompt for prompt in llm.prompts)
    assert any("Add logging" in prompt for prompt in llm.prompts)
