from __future__ import annotations

import html
from typing import Any

from chat_moderation.ui.constants import (
    DEFAULT_MODEL_KEY,
    MAX_MESSAGE_LENGTH,
)
from chat_moderation.ui.presenters import (
    comparison_result_summary,
    empty_result_card,
    initial_history,
    loaded_models_markdown,
    prediction_result_card,
    warning_result_card,
)
from chat_moderation.ui.runtime import (
    MODEL_CHOICES,
    MODEL_NAMES,
    REGISTRY,
)


def moderate_message(
    message: str,
    history: list[dict[str, Any]] | None,
    model_key: str | None,
) -> tuple[str, list[dict[str, Any]], str, str]:
    """Moderate one message and update the live conversation."""

    text = (message or "").strip()
    updated_history = list(history or initial_history())

    if not text:
        return (
            message,
            updated_history,
            warning_result_card("Please enter a Turkish message."),
            loaded_models_markdown(),
        )

    if len(text) > MAX_MESSAGE_LENGTH:
        return (
            message,
            updated_history,
            warning_result_card(f"Messages must be at most {MAX_MESSAGE_LENGTH} characters."),
            loaded_models_markdown(),
        )

    selected_key = model_key or DEFAULT_MODEL_KEY
    prediction = REGISTRY.predict(selected_key, text)
    is_offensive = prediction.label_name.lower() == "offensive"
    model_name = MODEL_NAMES[selected_key]

    if is_offensive:
        visible_message = "🚫 **Blocked message hidden for a safer demonstration.**"
        moderation_response = (
            f"🛡️ **Blocked** by {model_name} · {prediction.confidence * 100:.2f}% confidence"
        )
    else:
        visible_message = html.escape(text)
        moderation_response = (
            f"✅ **Allowed** by {model_name} · {prediction.confidence * 100:.2f}% confidence"
        )

    updated_history.extend(
        [
            {"role": "user", "content": visible_message},
            {"role": "assistant", "content": moderation_response},
        ]
    )

    return (
        "",
        updated_history,
        prediction_result_card(
            label_name=prediction.label_name,
            confidence=prediction.confidence,
            latency_ms=prediction.latency_ms,
            model_name=model_name,
        ),
        loaded_models_markdown(),
    )


def reset_live_moderation() -> tuple[
    str,
    list[dict[str, str]],
    str,
    str,
]:
    """Reset all live moderation components."""

    return (
        "",
        initial_history(),
        empty_result_card(),
        loaded_models_markdown(),
    )


def compare_all_models(
    message: str,
) -> tuple[str, list[list[Any]], str]:
    """Run the same message through every available model."""

    text = (message or "").strip()

    if not text:
        return (
            message,
            [],
            warning_result_card("Please enter a message to compare."),
        )

    if len(text) > MAX_MESSAGE_LENGTH:
        return (
            message,
            [],
            warning_result_card(f"Messages must be at most {MAX_MESSAGE_LENGTH} characters."),
        )

    predictions = REGISTRY.predict_all(text)
    rows: list[list[Any]] = []

    for model_name, model_key in MODEL_CHOICES:
        prediction = predictions[model_key]
        is_offensive = prediction.label_name.lower() == "offensive"
        rows.append(
            [
                model_name,
                prediction.label_name.title(),
                round(prediction.confidence * 100, 2),
                round(prediction.latency_ms, 3),
                "Block" if is_offensive else "Allow",
            ]
        )

    offensive_count = sum(
        prediction.label_name.lower() == "offensive" for prediction in predictions.values()
    )
    fastest_key = min(
        predictions,
        key=lambda key: predictions[key].latency_ms,
    )
    fastest_prediction = predictions[fastest_key]

    return (
        "",
        rows,
        comparison_result_summary(
            offensive_count=offensive_count,
            total_count=len(predictions),
            fastest_model_name=MODEL_NAMES[fastest_key],
            fastest_latency_ms=fastest_prediction.latency_ms,
        ),
    )
