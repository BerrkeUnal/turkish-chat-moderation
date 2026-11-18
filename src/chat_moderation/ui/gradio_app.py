"""Backward-compatible entry point for the Gradio application."""

from chat_moderation.ui.app import create_app, main

__all__ = ["create_app", "main"]


if __name__ == "__main__":
    main()
