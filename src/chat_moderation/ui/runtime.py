from __future__ import annotations

import json
from typing import Any

from chat_moderation.config import load_config, resolve_project_path
from chat_moderation.model_registry import ModelRegistry


def load_comparison_report() -> dict[str, Any]:
    """Load the final held-out test comparison report."""

    config = load_config()
    report_path = (
        resolve_project_path(config["paths"]["reports_dir"])
        / "metrics"
        / "model_comparison_test.json"
    )

    if not report_path.exists():
        raise FileNotFoundError(
            "The final comparison report was not found: "
            f"{report_path}. Run scripts/06_evaluate_models.py first."
        )

    with report_path.open(encoding="utf-8") as report_file:
        return json.load(report_file)


REGISTRY = ModelRegistry()
MODEL_CHOICES = REGISTRY.model_choices()
MODEL_NAMES = {model_key: display_name for display_name, model_key in MODEL_CHOICES}
COMPARISON_REPORT = load_comparison_report()


def loaded_model_names() -> list[str]:
    """Return display names for models currently loaded in memory."""

    return [MODEL_NAMES.get(model_key, model_key) for model_key in REGISTRY.loaded_model_keys()]
