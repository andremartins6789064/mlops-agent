from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import feature_engineering as stage_module


def test_feature_eng_load_and_split(tmp_path: Path) -> None:
    """Exercise feature engineering helper functions."""
    csv_path = tmp_path / 'data.csv'
    csv_path.write_text('x,y\n1,2\n', encoding='utf-8')
    if hasattr(stage_module, 'load_data'):
        loaded = stage_module.load_data()
        assert loaded is not None
    if hasattr(stage_module, 'split_data'):
        split = stage_module.split_data([1, 2, 3], [1, 2, 3])
        assert len(split) == 4
