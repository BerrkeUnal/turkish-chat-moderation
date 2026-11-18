from __future__ import annotations

import html
from typing import Any

from chat_moderation.ui.constants import (
    DEFAULT_MODEL_KEY,
    MODEL_DESCRIPTIONS,
)
from chat_moderation.ui.runtime import (
    COMPARISON_REPORT,
    MODEL_NAMES,
    loaded_model_names,
)


def tab_heading(title: str, subtitle: str) -> str:
    """Render a compact heading shared by all tabs."""

    return f"""
    <div class="tab-heading">
        <h2>{html.escape(title)}</h2>
        <p>{html.escape(subtitle)}</p>
    </div>
    """


def initial_history() -> list[dict[str, str]]:
    """Return a fresh initial chat history."""

    return [
        {
            "role": "assistant",
            "content": "Enter a message below to start moderation.",
        }
    ]


def loaded_models_markdown() -> str:
    """Describe which models are currently loaded in memory."""

    loaded_names = loaded_model_names()

    if not loaded_names:
        return "**Loaded models:** None. Models load only when needed."

    return "**Loaded models:** " + ", ".join(loaded_names)


def empty_result_card() -> str:
    """Render the initial moderation result card."""

    return """
    <div class="result-card">
        <div class="result-title">Waiting for a message</div>
        <div class="result-model">
            Prediction, confidence and latency will appear here.
        </div>
    </div>
    """


def warning_result_card(message: str) -> str:
    """Render an input warning card."""

    return f"""
    <div class="result-card warning">
        <div class="result-title">Input required</div>
        <div class="result-model">{html.escape(message)}</div>
    </div>
    """


def prediction_result_card(
    *,
    label_name: str,
    confidence: float,
    latency_ms: float,
    model_name: str,
) -> str:
    """Render one prediction without exposing blocked text."""

    is_offensive = label_name.lower() == "offensive"

    if is_offensive:
        css_class = "blocked"
        status = "Blocked"
        title = "Message blocked"
        explanation = "The selected model classified it as offensive."
    else:
        css_class = "allowed"
        status = "Allowed"
        title = "Message allowed"
        explanation = "The selected model classified it as clean."

    return f"""
    <div class="result-card {css_class}">
        <div class="result-card-header">
            <span class="status-badge {css_class}">{status}</span>
            <span class="result-title">{title}</span>
        </div>
        <p>{explanation}</p>
        <div class="prediction-metrics">
            <span><strong>{confidence * 100:.2f}%</strong> confidence</span>
            <span><strong>{latency_ms:.3f} ms</strong> latency</span>
        </div>
        <div class="result-model">{html.escape(model_name)}</div>
    </div>
    """


def model_information(model_key: str | None) -> str:
    """Return the selected model's test metrics and description."""

    selected_key = model_key or DEFAULT_MODEL_KEY
    metrics = COMPARISON_REPORT["models"][selected_key]
    description = MODEL_DESCRIPTIONS[selected_key]

    return (
        f"### {MODEL_NAMES[selected_key]}\n\n"
        f"{description}\n\n"
        f"**Held-out test:** Macro-F1 "
        f"`{metrics['macro_f1']:.4f}` · "
        f"Offensive recall `{metrics['offensive_recall']:.4f}` · "
        f"Size `{metrics['model_size_mb']:.3f} MB` · "
        f"Mean CPU latency `{metrics['mean_latency_ms']:.3f} ms`"
    )


def comparison_empty_state() -> str:
    """Render a compact empty state for model comparison."""

    return """
    <div class="comparison-band empty">
        <strong>No comparison yet</strong>
        <span>Enter one message to run all five models.</span>
    </div>
    """


def comparison_result_summary(
    *,
    offensive_count: int,
    total_count: int,
    fastest_model_name: str,
    fastest_latency_ms: float,
) -> str:
    """Render the compact result band shown above the comparison table."""

    clean_count = total_count - offensive_count

    if offensive_count == total_count:
        agreement = f"{total_count}/{total_count} models: Offensive"
    elif clean_count == total_count:
        agreement = f"{total_count}/{total_count} models: Clean"
    else:
        agreement = f"{offensive_count} offensive · {clean_count} clean"

    return f"""
    <div class="comparison-band complete">
        <strong>{agreement}</strong>
        <span>
            Fastest: {html.escape(fastest_model_name)} ·
            {fastest_latency_ms:.3f} ms
        </span>
    </div>
    """


def benchmark_rows() -> list[list[Any]]:
    """Build detailed display rows from the official test report."""

    rows: list[list[Any]] = []

    for model_key in COMPARISON_REPORT["ranking_by_macro_f1"]:
        metrics = COMPARISON_REPORT["models"][model_key]
        rows.append(
            [
                metrics["model_name"],
                round(metrics["macro_f1"], 4),
                round(metrics["accuracy"], 4),
                round(metrics["offensive_recall"], 4),
                round(metrics["model_size_mb"], 3),
                round(metrics["mean_latency_ms"], 3),
                round(metrics["p95_latency_ms"], 3),
            ]
        )

    return rows


def quantization_summary() -> str:
    """Summarize the measured Transformer quantization impact."""

    impact = COMPARISON_REPORT["quantization_impact"]
    fp32 = COMPARISON_REPORT["models"]["berturk"]
    int8 = COMPARISON_REPORT["models"]["berturk_int8"]

    return (
        "### Quantization impact\n\n"
        f"- **Macro-F1:** `{fp32['macro_f1']:.4f}` → "
        f"`{int8['macro_f1']:.4f}` "
        f"(delta `{impact['macro_f1_delta']:+.4f}`)\n"
        f"- **Model size:** `{fp32['model_size_mb']:.2f} MB` → "
        f"`{int8['model_size_mb']:.2f} MB` "
        f"(`{impact['size_reduction_percent']:.2f}%` reduction)\n"
        f"- **Mean CPU latency:** "
        f"`{fp32['mean_latency_ms']:.2f} ms` → "
        f"`{int8['mean_latency_ms']:.2f} ms` "
        f"(`{impact['latency_speedup']:.2f}×` speed-up)\n\n"
        '<div class="metric-note">'
        "<strong>Deployment decision:</strong> The INT8 ONNX model "
        "retains most Transformer quality while substantially reducing "
        "model size and CPU inference latency."
        "</div>"
    )


def results_summary_cards() -> str:
    """Render the primary held-out test findings."""

    models = COMPARISON_REPORT["models"]
    impact = COMPARISON_REPORT["quantization_impact"]
    best_key = COMPARISON_REPORT["ranking_by_macro_f1"][0]
    best_model = models[best_key]
    deployment_model = models["berturk_int8"]

    return f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Best test Macro-F1</div>
            <div class="kpi-value">{best_model["macro_f1"]:.4f}</div>
            <div class="kpi-caption">
                {html.escape(best_model["model_name"])}
            </div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">Model size reduction</div>
            <div class="kpi-value">
                {impact["size_reduction_percent"]:.2f}%
            </div>
            <div class="kpi-caption">Dynamic INT8 quantization</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">CPU latency speed-up</div>
            <div class="kpi-value">{impact["latency_speedup"]:.2f}×</div>
            <div class="kpi-caption">FP32 compared with INT8</div>
        </div>
    </div>
    <div class="deployment-strip">
        <span class="deployment-label">Deployment model</span>
        <strong>{html.escape(deployment_model["model_name"])}</strong>
        <span>Selected for its quality, size and latency balance.</span>
    </div>
    """


def benchmark_table_html() -> str:
    """Render a compact benchmark table with model badges."""

    rows: list[str] = []
    ranking = COMPARISON_REPORT["ranking_by_macro_f1"]
    models = COMPARISON_REPORT["models"]

    for model_key in ranking:
        metrics = models[model_key]
        badges: list[str] = []
        row_class = ""

        if model_key == ranking[0]:
            badges.append('<span class="model-badge best-badge">Best quality</span>')

        if model_key == "berturk_int8":
            badges.append('<span class="model-badge deployment-badge">Deployment</span>')
            row_class = ' class="deployment-row"'

        rows.append(
            f"""
            <tr{row_class}>
                <td>
                    <div class="model-name">
                        {html.escape(metrics["model_name"])}
                    </div>
                    <div class="model-badges">{" ".join(badges)}</div>
                </td>
                <td>{metrics["macro_f1"]:.4f}</td>
                <td>{metrics["offensive_recall"]:.4f}</td>
                <td>{metrics["model_size_mb"]:.2f} MB</td>
                <td>{metrics["mean_latency_ms"]:.3f} ms</td>
            </tr>
            """
        )

    return f"""
    <div class="benchmark-table-wrapper">
        <table class="benchmark-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Macro-F1</th>
                    <th>Offensive recall</th>
                    <th>Size</th>
                    <th>Mean latency</th>
                </tr>
            </thead>
            <tbody>{"".join(rows)}</tbody>
        </table>
    </div>
    """


def about_content() -> str:
    """Render project facts and responsible-use information."""

    return """
    <div class="about-layout">
        <section class="about-card about-summary">
            <div class="about-label">Project purpose</div>
            <h3>Local Turkish offensive-language detection</h3>
            <p>
                An educational prototype comparing rule-based, classical
                machine-learning, neural and Transformer approaches.
            </p>
        </section>
        <section class="fact-grid">
            <div class="fact-card">
                <span>Dataset</span><strong>Turkish OffensEval 2020</strong>
            </div>
            <div class="fact-card">
                <span>Primary metric</span><strong>Macro-F1</strong>
            </div>
            <div class="fact-card">
                <span>Deployment model</span>
                <strong>BERTurk Dynamic INT8 ONNX</strong>
            </div>
            <div class="fact-card">
                <span>Processing</span><strong>Local inference</strong>
            </div>
        </section>
    </div>
    <section class="responsible-use">
        <div>
            <div class="about-label">Limitations & responsible use</div>
            <h3>Human review remains important</h3>
        </div>
        <ul>
            <li>
                Sarcasm, quotations, spelling variation and conversation
                context may cause incorrect predictions.
            </li>
            <li>
                Identity-related language requires particular care and
                should not be judged automatically.
            </li>
            <li>
                Offensive messages are hidden in the live demo; the system
                is intended to assist, not replace, human moderation.
            </li>
        </ul>
    </section>
    """
