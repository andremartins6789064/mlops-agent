from __future__ import annotations

from collections.abc import Mapping

import streamlit as st

from src.application.validate_output import ValidationResult


def render_validation_details(
    *,
    validation: ValidationResult | None,
    validated_output_dir: str | None,
    generated_file_paths: Mapping[str, str] | None,
) -> None:
    """Render the raw quality-command outputs and validation context."""
    st.subheader("Validação real do artefato")
    if validation is None:
        st.info("A validação do artefato ainda não está disponível.")
        return

    st.write(
        "Estrutura validada: projeto gerado com módulos em `src/` e testes em "
        f"`tests/` ({validated_output_dir or 'diretório não informado'})."
    )
    st.write(
        "Ambiente: validação isolada com `uv run --no-project`, usando as "
        "dependências declaradas pelo artefato."
    )
    if generated_file_paths:
        st.caption(f"Módulos validados: {', '.join(generated_file_paths.values())}")

    _render_output("Saída do ruff", validation.lint_output)
    _render_output("Saída do mypy", validation.type_output)
    _render_output("Saída do pytest", validation.test_output)


def _render_output(title: str, output: str) -> None:
    with st.expander(title):
        st.code(output or "Nenhuma saída registrada.", language="text")
