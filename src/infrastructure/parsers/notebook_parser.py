from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import nbformat
from nbformat.notebooknode import NotebookNode

from src.domain.entities import CellType, Notebook, NotebookCell
from src.domain.interfaces import INotebookParser
from src.shared.exceptions import NotebookFileNotFoundError, NotebookParsingError


class NotebookParser(INotebookParser):
    """Read and transform .ipynb files into domain entities."""

    def parse(self, notebook_path: str) -> Notebook:
        path = Path(notebook_path)
        if not path.exists():
            msg = f"Notebook file not found: {notebook_path}"
            raise NotebookFileNotFoundError(msg)

        try:
            typed_reader = cast(Callable[[Path, int], NotebookNode], nbformat.read)
            notebook_data = typed_reader(path, 4)
        except Exception as exc:  # pragma: no cover - defensive branch
            msg = f"Failed to parse notebook at {notebook_path}"
            raise NotebookParsingError(msg) from exc

        cells = [
            self._to_notebook_cell(index, cell)
            for index, cell in enumerate(notebook_data.cells)
        ]
        metadata = self._to_json_compatible_dict(notebook_data.metadata)

        return Notebook(path=str(path), cells=cells, metadata=metadata)

    def _to_notebook_cell(self, index: int, cell: NotebookNode) -> NotebookCell:
        cell_type = self._map_cell_type(str(cell.get("cell_type", CellType.RAW.value)))
        source = str(cell.get("source", ""))
        outputs = self._extract_outputs(cell) if cell_type == CellType.CODE else []

        return NotebookCell(
            index=index,
            cell_type=cell_type,
            source=source,
            outputs=outputs,
        )

    def _map_cell_type(self, raw_cell_type: str) -> CellType:
        if raw_cell_type == CellType.CODE.value:
            return CellType.CODE
        if raw_cell_type == CellType.MARKDOWN.value:
            return CellType.MARKDOWN
        return CellType.RAW

    def _extract_outputs(self, cell: NotebookNode) -> list[str]:
        raw_outputs = cell.get("outputs", [])
        outputs: list[str] = []

        for output in raw_outputs:
            text = output.get("text")
            if isinstance(text, list):
                outputs.append("".join(str(part) for part in text))
                continue
            if text is not None:
                outputs.append(str(text))
                continue

            data = output.get("data", {})
            if isinstance(data, dict):
                plain_text = data.get("text/plain")
                if isinstance(plain_text, list):
                    outputs.append("".join(str(part) for part in plain_text))
                elif plain_text is not None:
                    outputs.append(str(plain_text))
                else:
                    outputs.append(json.dumps(data, ensure_ascii=True))

        return outputs

    def _to_json_compatible_dict(self, metadata: Any) -> dict[str, Any]:
        # Ensure NotebookNode metadata is plain JSON-compatible dict.
        serialized = json.loads(json.dumps(metadata))
        if not isinstance(serialized, dict):
            return {}
        return serialized
