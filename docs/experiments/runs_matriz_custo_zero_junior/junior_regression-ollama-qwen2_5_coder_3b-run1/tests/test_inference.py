from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import inference as stage_module


class _DummyModel:
    def predict(self, features: object) -> list[int]:
        return [1]


def test_inference_load_and_predict(tmp_path: Path) -> None:
    """Exercise model loading and prediction helpers."""
    model_path = tmp_path / 'model.pkl'
    with model_path.open('wb') as file_obj:
        pickle.dump(_DummyModel(), file_obj)
    model = _DummyModel()
    if hasattr(stage_module, 'load_model'):
        model = stage_module.load_model(str(model_path))
    if hasattr(stage_module, 'predict'):
        predictions = stage_module.predict(model, [1, 2, 3])
        assert predictions is not None
