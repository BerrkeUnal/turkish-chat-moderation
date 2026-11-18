from __future__ import annotations

import gradio as gr

from chat_moderation.ui.constants import DEFAULT_MODEL_KEY
from chat_moderation.ui.handlers import (
    moderate_message,
    reset_live_moderation,
)
from chat_moderation.ui.presenters import (
    empty_result_card,
    initial_history,
    loaded_models_markdown,
    model_information,
    tab_heading,
)
from chat_moderation.ui.runtime import MODEL_CHOICES


def build_moderate_tab() -> None:
    """Build the live single-model moderation tab."""

    with gr.Tab("Moderate"):
        with gr.Column(elem_classes=["tab-shell"]):
            gr.HTML(
                tab_heading(
                    "Live moderation",
                    "Choose a model and analyze one Turkish message.",
                )
            )

            with gr.Row(elem_classes=["moderation-controls"]):
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICES,
                    value=DEFAULT_MODEL_KEY,
                    label="Model",
                    scale=2,
                    min_width=300,
                )

                live_message = gr.Textbox(
                    label="Message",
                    placeholder="Type a Turkish message...",
                    lines=2,
                    max_lines=3,
                    scale=5,
                    min_width=0,
                )

            with gr.Row(elem_classes=["compact-actions"]):
                send_button = gr.Button(
                    "Analyze message",
                    variant="primary",
                    scale=4,
                    elem_id="primary-action",
                )
                clear_button = gr.Button(
                    "Clear",
                    variant="secondary",
                    scale=1,
                )

            with gr.Row(elem_classes=["moderation-workspace"]):
                with gr.Column(
                    min_width=0,
                    elem_classes=["workspace-panel"],
                ):
                    gr.HTML('<div class="section-label">Conversation</div>')
                    chatbot = gr.Chatbot(
                        value=initial_history(),
                        height=220,
                        show_label=False,
                        buttons=[],
                        feedback_options=None,
                        elem_id="chat-window",
                    )

                with gr.Column(
                    min_width=0,
                    elem_classes=["workspace-panel"],
                ):
                    gr.HTML('<div class="section-label">Prediction</div>')
                    live_result = gr.HTML(empty_result_card())

                    with gr.Accordion(
                        "Model & runtime details",
                        open=False,
                    ):
                        live_loaded_models = gr.Markdown(
                            loaded_models_markdown(),
                            elem_classes=["loaded-models"],
                        )
                        model_description = gr.Markdown(model_information(DEFAULT_MODEL_KEY))

            live_inputs = [live_message, chatbot, model_dropdown]
            live_outputs = [
                live_message,
                chatbot,
                live_result,
                live_loaded_models,
            ]

            send_button.click(
                fn=moderate_message,
                inputs=live_inputs,
                outputs=live_outputs,
            )
            live_message.submit(
                fn=moderate_message,
                inputs=live_inputs,
                outputs=live_outputs,
            )
            clear_button.click(
                fn=reset_live_moderation,
                inputs=None,
                outputs=live_outputs,
            )
            model_dropdown.change(
                fn=model_information,
                inputs=model_dropdown,
                outputs=model_description,
            )
