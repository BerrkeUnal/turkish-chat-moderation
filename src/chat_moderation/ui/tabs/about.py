from __future__ import annotations

import gradio as gr

from chat_moderation.ui.presenters import about_content, tab_heading


def build_about_tab() -> None:
    """Build project and responsible-use information."""

    with gr.Tab("About"):
        with gr.Column(elem_classes=["tab-shell"]):
            gr.HTML(
                tab_heading(
                    "About this project",
                    "Architecture, evaluation focus and responsible use.",
                )
            )
            gr.HTML(about_content())
