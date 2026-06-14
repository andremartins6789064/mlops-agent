from __future__ import annotations

import streamlit as st

from src.ui.pages.common import reset_result_state, set_result_state
from src.ui.session import CONFIG_KEY, ERROR_KEY, UIConfig, default_config
from src.ui.workflow import run_conversion

st.set_page_config(page_title="MLOps Agent - Config", page_icon=":gear:")
st.title("Configuração")
st.write("Defina o provedor LLM, envie um notebook e execute a conversão completa.")

stored = st.session_state.get(CONFIG_KEY)
config = stored if isinstance(stored, UIConfig) else default_config()

with st.form("conversion_form"):
    use_llm = st.checkbox("Usar provedor LLM", value=config.use_llm)
    llm_base_url = st.text_input("URL base do LLM", value=config.llm_base_url)
    llm_api_key = st.text_input(
        "Chave de API do LLM", value=config.llm_api_key, type="password"
    )
    llm_model = st.text_input("Modelo LLM", value=config.llm_model)
    output_dir = st.text_input("Diretório de saída", value=config.output_dir)
    uploaded_file = st.file_uploader("Enviar notebook (.ipynb)", type=["ipynb"])
    submit = st.form_submit_button("Executar conversão")

if submit:
    if uploaded_file is None:
        st.error("Selecione um arquivo .ipynb antes de executar a conversão.")
    else:
        reset_result_state()
        active_config = UIConfig(
            use_llm=use_llm,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            output_dir=output_dir,
        )
        st.session_state[CONFIG_KEY] = active_config
        with st.spinner("Executando pipeline de orquestração..."):
            try:
                result = run_conversion(
                    notebook_bytes=uploaded_file.getvalue(),
                    notebook_name=uploaded_file.name,
                    config=active_config,
                )
                set_result_state(
                    result=result,
                    error=None,
                    notebook_name=uploaded_file.name,
                    notebook_bytes=uploaded_file.getvalue(),
                )
                st.success("Conversão concluída com sucesso.")
            except Exception as exc:  # pragma: no cover - streamlit runtime branch
                set_result_state(
                    result=None,
                    error=str(exc),
                    notebook_name=uploaded_file.name,
                    notebook_bytes=uploaded_file.getvalue(),
                )
                st.error(f"Falha na conversão: {exc}")

error_message = st.session_state.get(ERROR_KEY)
if isinstance(error_message, str) and error_message:
    st.warning(error_message)
