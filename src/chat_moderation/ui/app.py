from __future__ import annotations

import gradio as gr

from chat_moderation.ui.styles import APP_CSS
from chat_moderation.ui.tabs.about import build_about_tab
from chat_moderation.ui.tabs.compare import build_compare_tab
from chat_moderation.ui.tabs.moderate import build_moderate_tab
from chat_moderation.ui.tabs.results import build_results_tab


def create_app() -> gr.Blocks:
    """Create the Gradio moderation dashboard."""

    with gr.Blocks(title="Turkish Chat Moderation") as app:
        gr.HTML(
            """
            <header class="page-header">
                <div class="eyebrow">SWE015 · NLP PROJECT</div>
                <div class="header-row">
                    <h1>Turkish Chat Moderation</h1>
                    <p>
                        Analyze Turkish chat messages with five NLP
                        approaches and compare their predictions.
                    </p>
                </div>
            </header>
            """
        )

        with gr.Tabs():
            build_moderate_tab()
            build_compare_tab()
            build_results_tab()
            build_about_tab()

        gr.HTML(
            """
            <footer class="footer-note">
                Turkish Chat Moderation · SWE015
            </footer>
            """
        )

    return app


def main() -> None:
    """Launch the local Gradio application."""

    theme = gr.themes.Base(
        primary_hue="blue",
        neutral_hue="slate",
    )

    app = create_app()
    app.queue().launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        theme=theme,
        css=APP_CSS,
    )


if __name__ == "__main__":
    main()
