from __future__ import annotations

import streamlit as st

from src.ui.components import render_metrics_card, render_validation_details
from src.ui.pages.common import (
    read_result,
    render_download_section,
    render_missing_result_message,
)
from src.ui.workflow import summarize_review_status

st.set_page_config(page_title="MLOps Agent - Download", page_icon=":inbox_tray:")
st.title("Download")
st.write("Confira as métricas finais e baixe o pacote .zip gerado.")

result = read_result()
if result is None:
    render_missing_result_message()
else:
    render_metrics_card(result.quality_metrics)
    st.write(summarize_review_status(result.quality_metrics))
    render_validation_details(
        validation=result.validation_result,
        validated_output_dir=result.validated_output_dir,
        generated_file_paths=result.generated_file_paths,
    )
    render_download_section(result)
