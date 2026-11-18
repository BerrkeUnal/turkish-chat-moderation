from __future__ import annotations

import gradio as gr

from chat_moderation.ui.presenters import (
    benchmark_rows,
    benchmark_table_html,
    quantization_summary,
    results_summary_cards,
    tab_heading,
)


def build_results_tab() -> None:
    """Build the held-out benchmark results tab."""

    with gr.Tab("Results"):
        with gr.Column(elem_classes=["tab-shell"]):
            gr.HTML(
                tab_heading(
                    "Held-out test results",
                    "All five models use the same untouched 3,528-example test split.",
                )
            )
            gr.HTML(results_summary_cards())
            gr.HTML(benchmark_table_html())

            with gr.Accordion(
                "Quantization and detailed metrics",
                open=False,
            ):
                gr.Markdown(quantization_summary())
                gr.Dataframe(
                    value=benchmark_rows(),
                    headers=[
                        "Model",
                        "Macro-F1",
                        "Accuracy",
                        "Offensive Recall",
                        "Size (MB)",
                        "Mean Latency (ms)",
                        "P95 Latency (ms)",
                    ],
                    datatype=[
                        "str",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                        "number",
                    ],
                    type="array",
                    interactive=False,
                    buttons=[],
                    label="Complete test metrics",
                    elem_id="detailed-results-table",
                )
