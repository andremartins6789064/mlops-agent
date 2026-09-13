from __future__ import annotations

import streamlit as st

from src.shared.provenance import StageProvenance


def render_stage_provenance(
    *, stage_name: str, provenance: StageProvenance | None
) -> None:
    """Render the origin and details of a generated pipeline stage."""
    label = stage_name.replace("_", " ").title()
    if provenance is None:
        st.info(f"Proveniência de {label}: não disponível.")
        return

    if provenance.origin == "llm":
        st.success(f"{label}: código gerado pela LLM.")
    else:
        st.warning(
            f"{label}: código gerado por template. "
            "Não confundir com uma saída produzida pela LLM."
        )

    details = [
        f"Modelo: {provenance.model or 'não informado'}",
        f"Duração: {provenance.duration_seconds:.2f}s",
        f"Parser: {provenance.parse_method or 'não informado'}",
    ]
    if provenance.context_truncated:
        details.append("Contexto truncado: sim")
    if provenance.fallback_reason:
        details.append(f"Motivo do fallback: {provenance.fallback_reason}")
    st.caption(" | ".join(details))
