from __future__ import annotations

import gradio as gr

from chat_moderation.ui.handlers import compare_all_models
from chat_moderation.ui.presenters import (
    comparison_empty_state,
    tab_heading,
)


def build_compare_tab() -> None:
    """Build the all-model comparison tab."""

    with gr.Tab("Compare"):
        with gr.Column(elem_classes=["tab-shell"]):
            gr.HTML(
                tab_heading(
                    "Compare all models",
                    "Run one message through all five moderation models.",
                )
            )

            with gr.Row(elem_classes=["compare-controls"]):
                compare_message = gr.Textbox(
                    label="Message",
                    placeholder="Enter one Turkish message...",
                    lines=2,
                    max_lines=3,
                )
                compare_button = gr.Button(
                    "Compare models",
                    variant="primary",
                    elem_id="compare-action",
                )

            comparison_summary = gr.HTML(
                comparison_empty_state(),
                elem_classes=["comparison-summary"],
            )
            comparison_table = gr.Dataframe(
                value=[],
                headers=[
                    "Model",
                    "Prediction",
                    "Confidence (%)",
                    "Latency (ms)",
                    "Action",
                ],
                datatype=["str", "str", "number", "number", "str"],
                type="array",
                interactive=False,
                buttons=[],
                label="Model predictions",
                elem_id="comparison-table",
            )

            compare_outputs = [
                compare_message,
                comparison_table,
                comparison_summary,
            ]
            compare_button.click(
                fn=compare_all_models,
                inputs=compare_message,
                outputs=compare_outputs,
            )
            compare_message.submit(
                fn=compare_all_models,
                inputs=compare_message,
                outputs=compare_outputs,
            )
