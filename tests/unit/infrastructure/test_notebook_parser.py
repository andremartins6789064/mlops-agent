from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from src.domain.entities import CellType, Notebook
from src.infrastructure.parsers.notebook_parser import NotebookParser
from src.shared.exceptions import NotebookFileNotFoundError


def _write_notebook(path: Path, cells: list[dict[str, Any]]) -> None:
    normalized_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        normalized_cell = {**cell}
        normalized_cell.setdefault("id", f"cell-{index}")
        normalized_cells.append(normalized_cell)

    notebook_payload = {
        "cells": normalized_cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook_payload), encoding="utf-8")


def test_parse_empty_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "empty.ipynb"
    _write_notebook(notebook_path, cells=[])

    parser = NotebookParser()
    notebook = parser.parse(str(notebook_path))

    assert isinstance(notebook, Notebook)
    assert notebook.path == str(notebook_path)
    assert notebook.cells == []


def test_parse_markdown_only_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "markdown_only.ipynb"
    _write_notebook(
        notebook_path,
        cells=[
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": "## Notebook with markdown only",
            }
        ],
    )

    parser = NotebookParser()
    notebook = parser.parse(str(notebook_path))

    assert len(notebook.cells) == 1
    assert notebook.cells[0].cell_type == CellType.MARKDOWN
    assert notebook.cells[0].outputs == []


def test_parse_imports_only_notebook(tmp_path: Path) -> None:
    notebook_path = tmp_path / "imports_only.ipynb"
    _write_notebook(
        notebook_path,
        cells=[
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "outputs": [],
                "source": "import pandas as pd\nimport numpy as np",
            }
        ],
    )

    parser = NotebookParser()
    notebook = parser.parse(str(notebook_path))

    assert len(notebook.cells) == 1
    assert notebook.cells[0].cell_type == CellType.CODE
    assert "import pandas as pd" in notebook.cells[0].source


def test_parse_complete_notebook_fixture() -> None:
    notebook_path = Path("tests/fixtures/simple_regression.ipynb")
    parser = NotebookParser()

    notebook = parser.parse(str(notebook_path))

    assert isinstance(notebook, Notebook)
    assert len(notebook.cells) > 0
    assert all(cell.index >= 0 for cell in notebook.cells)
    assert all(isinstance(cell.source, str) for cell in notebook.cells)
    assert notebook.cells[0].cell_type == CellType.MARKDOWN
    assert any(cell.cell_type == CellType.CODE for cell in notebook.cells)
    assert any(
        "score=1.0" in output for cell in notebook.cells for output in cell.outputs
    )


def test_parse_raises_custom_exception_for_missing_file() -> None:
    parser = NotebookParser()

    with pytest.raises(NotebookFileNotFoundError):
        parser.parse("tests/fixtures/does_not_exist.ipynb")


def test_parse_raw_cells_and_multiple_output_shapes(tmp_path: Path) -> None:
    notebook_path = tmp_path / "raw_and_outputs.ipynb"
    _write_notebook(
        notebook_path,
        cells=[
            {
                "cell_type": "raw",
                "metadata": {},
                "source": "raw context",
            },
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": 1,
                "source": "print('hello')",
                "outputs": [
                    {
                        "output_type": "stream",
                        "name": "stdout",
                        "text": ["hello", "\n"],
                    },
                    {
                        "output_type": "display_data",
                        "data": {"text/plain": ["42"]},
                        "metadata": {},
                    },
                    {
                        "output_type": "display_data",
                        "data": {"application/json": {"ok": True}},
                        "metadata": {},
                    },
                ],
            },
        ],
    )

    parser = NotebookParser()
    notebook = parser.parse(str(notebook_path))

    assert notebook.cells[0].cell_type == CellType.RAW
    assert notebook.cells[1].outputs[0] == "hello\n"
    assert notebook.cells[1].outputs[1] == "42"
    assert '{"application/json": {"ok": true}}' == notebook.cells[1].outputs[2]


def test_metadata_conversion_returns_empty_dict_for_non_mapping() -> None:
    parser = NotebookParser()

    invalid_metadata = cast(Any, ["invalid", "metadata"])
    metadata = parser._to_json_compatible_dict(invalid_metadata)

    assert metadata == {}
