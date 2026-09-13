from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, cast

import streamlit as st

from src.agents.orchestrator import OrchestrationResult
from src.ui.components import (
    render_cell_stage_table,
    render_code_viewer,
    render_stage_provenance,
)
from src.ui.session import (
    CONFIG_KEY,
    ERROR_KEY,
    NOTEBOOK_BYTES_KEY,
    NOTEBOOK_NAME_KEY,
    RESULT_KEY,
    UIConfig,
    get_result,
)
from src.ui.workflow import (
    STAGE_NAMES,
    get_stage_code,
    get_stage_tests,
    run_conversion,
    summarize_review_status,
)


def render_missing_result_message() -> None:
    """Render a consistent warning for missing conversion results."""
    st.warning("Execute a conversão na página Configuração antes de abrir esta aba.")


def read_result() -> OrchestrationResult | None:
    """Read result object from Streamlit session state."""
    state = cast(Mapping[str, Any], st.session_state)
    return get_result(state)


def render_stage_page(*, stage_name: str, title: str) -> None:
    """Render one pipeline page with code and test output."""
    st.title(title)
    result = read_result()
    if result is None:
        render_missing_result_message()
        return

    render_stage_provenance(
        stage_name=stage_name,
        provenance=(result.stage_provenance or {}).get(stage_name),
    )
    render_code_viewer(
        title="Módulo gerado",
        source_code=get_stage_code(result, stage_name=stage_name),
    )
    render_code_viewer(
        title="Testes gerados",
        source_code=get_stage_tests(result, stage_name=stage_name),
    )

    st.text_area(
        "Ajustes para este estágio",
        placeholder=(
            "Descreva os ajustes desejados para este estágio. "
            "Depois clique em regenerar para aplicar."
        ),
        key=f"chat_{stage_name}",
    )
    if st.button("Regenerar estágio com feedback", key=f"regenerate_{stage_name}"):
        regenerate_from_feedback()


def render_analysis_blocks(result: OrchestrationResult) -> None:
    """Render the cell mapping, analysis details, and architecture plan."""
    st.subheader("Saída do Notebook Analyzer")
    render_cell_stage_table(
        notebook=result.notebook,
        analysis=result.notebook_analysis,
    )
    st.subheader("Plano de arquitetura")
    st.json(result.architecture_plan)
    st.subheader("Proveniência por estágio")
    for stage_name in STAGE_NAMES:
        render_stage_provenance(
            stage_name=stage_name,
            provenance=(result.stage_provenance or {}).get(stage_name),
        )
    st.subheader("Resumo rápido")
    st.write(summarize_review_status(result.quality_metrics))


def render_download_section(result: OrchestrationResult) -> None:
    """Render download action and execution metadata."""
    notebook_name = str(st.session_state.get(NOTEBOOK_NAME_KEY, "notebook"))
    if result.exported_zip_path:
        with open(result.exported_zip_path, "rb") as file_obj:
            st.download_button(
                label="Baixar projeto gerado (.zip)",
                data=file_obj.read(),
                file_name=f"{notebook_name}.zip",
                mime="application/zip",
            )
    else:
        st.warning("O exportador ZIP não gerou arquivo nesta execução.")

    if result.generated_file_paths:
        st.subheader("Arquivos de módulos gerados")
        st.write(json.dumps(result.generated_file_paths, indent=2))
    if result.generated_test_file_paths:
        st.subheader("Arquivos de testes gerados")
        st.write(json.dumps(result.generated_test_file_paths, indent=2))


def reset_result_state() -> None:
    """Clear result and error keys before a new run."""
    for key in (RESULT_KEY, ERROR_KEY):
        if key in st.session_state:
            del st.session_state[key]


def set_result_state(
    *,
    result: OrchestrationResult | None,
    error: str | None,
    notebook_name: str,
    notebook_bytes: bytes | None = None,
) -> None:
    """Persist conversion output and metadata in session state."""
    if result is not None:
        st.session_state[RESULT_KEY] = result
    if error is None:
        st.session_state.pop(ERROR_KEY, None)
    else:
        st.session_state[ERROR_KEY] = error
    st.session_state[NOTEBOOK_NAME_KEY] = notebook_name
    if notebook_bytes is not None:
        st.session_state[NOTEBOOK_BYTES_KEY] = notebook_bytes


def regenerate_from_feedback() -> None:
    """Re-run conversion using current chat feedback from the UI."""
    config_value = st.session_state.get(CONFIG_KEY)
    notebook_name = st.session_state.get(NOTEBOOK_NAME_KEY)
    notebook_bytes = st.session_state.get(NOTEBOOK_BYTES_KEY)
    if not isinstance(config_value, UIConfig):
        st.warning(
            "Configuração não encontrada. Execute a conversão em Configuração primeiro."
        )
        return
    if not isinstance(notebook_name, str) or not isinstance(notebook_bytes, bytes):
        st.warning(
            "Notebook não encontrado em sessão. Reenvie o arquivo em Configuração."
        )
        return

    architecture_feedback = _feedback_text(st.session_state.get("chat_analysis"))
    stage_feedback = _collect_stage_feedback()
    with st.spinner("Aplicando feedback e executando novamente..."):
        try:
            result = run_conversion(
                notebook_bytes=notebook_bytes,
                notebook_name=notebook_name,
                config=config_value,
                architecture_feedback=architecture_feedback,
                stage_feedback=stage_feedback,
            )
        except Exception as exc:  # pragma: no cover - streamlit runtime branch
            set_result_state(result=None, error=str(exc), notebook_name=notebook_name)
            st.error(f"Falha na regeneração: {exc}")
            return

    set_result_state(
        result=result,
        error=None,
        notebook_name=notebook_name,
        notebook_bytes=notebook_bytes,
    )
    st.success("Regeneração concluída com o feedback mais recente.")


def _collect_stage_feedback() -> dict[str, str]:
    feedback: dict[str, str] = {}
    for stage_name in STAGE_NAMES:
        value = _feedback_text(st.session_state.get(f"chat_{stage_name}"))
        if value:
            feedback[stage_name] = value
    return feedback


def _feedback_text(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None
