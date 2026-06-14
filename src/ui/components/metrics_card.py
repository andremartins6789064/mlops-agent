from __future__ import annotations

import streamlit as st

from src.domain.value_objects import QualityMetrics


def render_metrics_card(metrics: QualityMetrics | None) -> None:
    """Render quality metrics in a compact 4-column layout."""
    st.subheader("Métricas de qualidade")
    if metrics is None:
        st.info("As métricas aparecem após a execução da conversão.")
        return
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Erros de lint", str(metrics.lint_errors))
    col2.metric("Erros de tipo", str(metrics.type_errors))
    col3.metric("Cobertura", f"{metrics.test_coverage:.1f}%")
    col4.metric("Iterações de revisão", str(metrics.review_iterations))
