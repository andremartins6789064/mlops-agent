from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import streamlit as st

from src.domain.entities import Notebook

_SNIPPET_LIMIT = 120


def build_cell_stage_rows(
    *, notebook: Notebook, analysis: Mapping[str, Any]
) -> list[dict[str, int | str]]:
    """Build display rows mapping every notebook cell to its pipeline stage."""
    stage_by_index: dict[int, str] = {}
    cells_by_pipeline = analysis.get("cells_by_pipeline", {})
    if isinstance(cells_by_pipeline, Mapping):
        for stage_name, cell_indices in cells_by_pipeline.items():
            if not isinstance(stage_name, str) or not isinstance(cell_indices, list):
                continue
            for cell_index in cell_indices:
                if isinstance(cell_index, int):
                    stage_by_index[cell_index] = stage_name

    return [
        {
            "Índice": cell.index,
            "Código": _build_snippet(cell.source),
            "Estágio": stage_by_index.get(cell.index, "não atribuído"),
        }
        for cell in notebook.cells
    ]


def render_cell_stage_table(*, notebook: Notebook, analysis: Mapping[str, Any]) -> None:
    """Render the cell-to-stage table and the raw analysis for debugging."""
    rows = build_cell_stage_rows(notebook=notebook, analysis=analysis)
    if rows:
        st.dataframe(rows, hide_index=True, use_container_width=True)
    else:
        st.info("O notebook não possui células para exibir.")

    with st.expander("Ver JSON bruto da análise"):
        st.json(dict(analysis))


def _build_snippet(source: str) -> str:
    normalized = " ".join(source.split())
    if not normalized:
        return "(célula vazia)"
    if len(normalized) <= _SNIPPET_LIMIT:
        return normalized
    return f"{normalized[: _SNIPPET_LIMIT - 1]}…"
