from __future__ import annotations

import streamlit as st

from src.ui.pages.common import (
    read_result,
    regenerate_from_feedback,
    render_analysis_blocks,
    render_missing_result_message,
)

st.set_page_config(page_title="MLOps Agent - Analysis", page_icon=":mag:")
st.title("Análise")
st.write("Revise os resultados do Notebook Analyzer e do Architecture Agent.")

result = read_result()
if result is None:
    render_missing_result_message()
else:
    render_analysis_blocks(result)
    st.text_area(
        "Instruções para ajustar o plano de arquitetura",
        placeholder="Descreva mudanças desejadas antes de gerar novamente.",
        key="chat_analysis",
    )
    if st.button("Regenerar com feedback de arquitetura", key="regenerate_analysis"):
        regenerate_from_feedback()
