from __future__ import annotations

import json
from pathlib import Path

from src.domain.interfaces import ILLMClient
from src.shared.llm_parsing import parse_json_object, parse_python_block
from src.shared.python_source import (
    generated_test_matches_module,
    python_syntax_error,
)


class PipelineTestGeneratorAgent:
    """Generate pytest files for each pipeline module."""

    def __init__(
        self,
        *,
        llm_client: ILLMClient | None = None,
        prompt_path: str = "src/infrastructure/prompts/test_generator.txt",
    ) -> None:
        self._llm_client = llm_client
        self._prompt_path = prompt_path

    def generate_tests(
        self,
        *,
        generated_modules: dict[str, str],
    ) -> dict[str, str]:
        """Return pytest files for generated stage modules."""
        tests: dict[str, str] = {}
        for stage_name, module_code in generated_modules.items():
            llm_test = self._generate_test_with_llm(
                stage_name=stage_name, module_code=module_code
            )
            if llm_test is not None:
                tests[stage_name] = llm_test
                continue
            tests[stage_name] = self._generate_test_with_templates(
                stage_name=stage_name
            )
        return tests

    def write_tests(
        self,
        *,
        generated_tests: dict[str, str],
        output_dir: str = "output/tests",
    ) -> dict[str, str]:
        """Persist generated test files and return path map."""
        tests_root = Path(output_dir)
        tests_root.mkdir(parents=True, exist_ok=True)
        file_map: dict[str, str] = {}
        for stage_name, test_code in generated_tests.items():
            test_path = tests_root / f"test_{stage_name}.py"
            test_path.write_text(test_code, encoding="utf-8")
            file_map[stage_name] = str(test_path)
        return file_map

    def _generate_test_with_llm(
        self, *, stage_name: str, module_code: str
    ) -> str | None:
        if self._llm_client is None:
            return None
        prompt = self._build_prompt(stage_name=stage_name, module_code=module_code)
        raw_response = self._llm_client.generate(prompt=prompt)
        python_result = parse_python_block(raw_response)
        if python_result.method == "fenced" and isinstance(python_result.value, str):
            if (
                "def test_" in python_result.value
                and python_syntax_error(python_result.value) is None
                and generated_test_matches_module(
                    python_result.value, module_code, stage_name
                )
            ):
                return python_result.value
            return None
        parsed = parse_json_object(raw_response).value
        if isinstance(parsed, dict):
            test_code = parsed.get("test_code")
            if (
                isinstance(test_code, str)
                and "def test_" in test_code
                and python_syntax_error(test_code) is None
                and generated_test_matches_module(test_code, module_code, stage_name)
            ):
                return test_code.strip()
        python_code = python_result.value
        if (
            isinstance(python_code, str)
            and "def test_" in python_code
            and python_syntax_error(python_code) is None
            and generated_test_matches_module(python_code, module_code, stage_name)
        ):
            return python_code
        return None

    def _generate_test_with_templates(self, *, stage_name: str) -> str:
        if stage_name == "feature_engineering":
            return self._feature_engineering_tests()
        if stage_name == "training":
            return self._training_tests()
        if stage_name == "inference":
            return self._inference_tests()
        if stage_name == "evaluation":
            return self._evaluation_tests()
        return self._generic_module_tests(stage_name=stage_name)

    def _build_prompt(self, *, stage_name: str, module_code: str) -> str:
        prompt_template = Path(self._prompt_path).read_text(encoding="utf-8")
        payload = {"stage_name": stage_name, "module_code": module_code}
        return f"{prompt_template}\n\nContext JSON:\n{json.dumps(payload)}"

    def _feature_engineering_tests(self) -> str:
        return (
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "import feature_engineering as stage_module\n\n\n"
            "def test_feature_eng_load_and_split(tmp_path: Path) -> None:\n"
            '    """Exercise feature engineering helper functions."""\n'
            "    csv_path = tmp_path / 'data.csv'\n"
            "    csv_path.write_text('x,y\\n1,2\\n', encoding='utf-8')\n"
            "    if hasattr(stage_module, 'load_data'):\n"
            "        loaded = stage_module.load_data()\n"
            "        assert loaded is not None\n"
            "    if hasattr(stage_module, 'split_data'):\n"
            "        split = stage_module.split_data([1, 2, 3], [1, 2, 3])\n"
            "        assert len(split) == 4\n"
        )

    def _training_tests(self) -> str:
        return (
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "import training as stage_module\n\n\n"
            "def test_training_train_and_save_model(tmp_path: Path) -> None:\n"
            '    """Exercise training functions and serialization."""\n'
            "    model = {'ok': True}\n"
            "    if hasattr(stage_module, 'train_model'):\n"
            "        model = stage_module.train_model([1], [1])\n"
            "    if hasattr(stage_module, 'save_model'):\n"
            "        model_path = tmp_path / 'model.pkl'\n"
            "        saved_path = stage_module.save_model(model, str(model_path))\n"
            "        assert Path(saved_path).exists()\n"
        )

    def _inference_tests(self) -> str:
        return (
            "from __future__ import annotations\n\n"
            "import pickle\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "import inference as stage_module\n\n\n"
            "class _DummyModel:\n"
            "    def predict(self, features: object) -> list[int]:\n"
            "        return [1]\n\n\n"
            "def test_inference_load_and_predict(tmp_path: Path) -> None:\n"
            '    """Exercise model loading and prediction helpers."""\n'
            "    model_path = tmp_path / 'model.pkl'\n"
            "    with model_path.open('wb') as file_obj:\n"
            "        pickle.dump(_DummyModel(), file_obj)\n"
            "    model = _DummyModel()\n"
            "    if hasattr(stage_module, 'load_model'):\n"
            "        model = stage_module.load_model(str(model_path))\n"
            "    if hasattr(stage_module, 'predict'):\n"
            "        predictions = stage_module.predict(model, [1, 2, 3])\n"
            "        assert predictions is not None\n"
        )

    def _evaluation_tests(self) -> str:
        return (
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            "import evaluation as stage_module\n\n\n"
            "def test_evaluation_functions_return_metrics() -> None:\n"
            '    """Exercise evaluation helpers and verify dictionary output."""\n'
            "    y_true = [1, 0, 1]\n"
            "    y_pred = [1, 1, 1]\n"
            "    if hasattr(stage_module, 'evaluate_model'):\n"
            "        metrics = stage_module.evaluate_model(y_true, y_pred)\n"
            "        assert isinstance(metrics, dict)\n"
            "    if hasattr(stage_module, 'build_metrics_report'):\n"
            "        report = stage_module.build_metrics_report(y_true, y_pred)\n"
            "        assert isinstance(report, dict)\n"
        )

    def _generic_module_tests(self, *, stage_name: str) -> str:
        return (
            "from __future__ import annotations\n\n"
            "import sys\n"
            "from pathlib import Path\n\n"
            "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))\n"
            f"import {stage_name} as stage_module\n\n\n"
            f"def test_{stage_name}_module_has_functions() -> None:\n"
            "    callables = [\n"
            "        name\n"
            "        for name in dir(stage_module)\n"
            "        if callable(getattr(stage_module, name))\n"
            "        and not name.startswith('_')\n"
            "    ]\n"
            "    assert callables\n"
        )
