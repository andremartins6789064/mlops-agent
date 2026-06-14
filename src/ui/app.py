from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is importable when Streamlit runs this file directly.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ui.session import ERROR_KEY, NOTEBOOK_NAME_KEY, RESULT_KEY
from src.ui.workflow import summarize_review_status

st.set_page_config(
    page_title="MLOps Agent",
    page_icon=":robot_face:",
    layout="wide",
)

st.title("MLOps Agent")
st.write(
    "Faça upload do notebook na página Configuração, execute a conversão, "
    "revise cada estágio do pipeline e baixe o projeto final em .zip."
)

result = st.session_state.get(RESULT_KEY)
notebook_name = st.session_state.get(NOTEBOOK_NAME_KEY)
error_message = st.session_state.get(ERROR_KEY)

if isinstance(error_message, str) and error_message:
    st.error(error_message)
elif result is None:
    st.info(
        "Nenhuma conversão executada ainda. Abra a página Configuração para começar."
    )
else:
    st.success("Resultado mais recente disponível em todas as páginas.")
    if isinstance(notebook_name, str) and notebook_name:
        st.write(f"Notebook atual: `{notebook_name}`")
    quality_metrics = getattr(result, "quality_metrics", None)
    st.write(summarize_review_status(quality_metrics))
