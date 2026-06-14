from __future__ import annotations

import streamlit as st


def render_code_viewer(*, title: str, source_code: str | None) -> None:
    """Render generated code with fallback text."""
    st.subheader(title)
    if not source_code:
        st.info("Ainda não há conteúdo gerado para esta seção.")
        return
    st.code(source_code, language="python")
