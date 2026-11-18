from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from chat_moderation.models.base import Prediction


def calculate_path_size_mb(path: str | Path) -> float:
    """Calculate the total size of a model file or directory."""

    model_path = Path(path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    files = (
        [model_path]
        if model_path.is_file()
        else [file for file in model_path.rglob("*") if file.is_file()]
    )

    total_bytes = sum(file.stat().st_size for file in files)

    return round(total_bytes / (1024**2), 6)


def calculate_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Calculate binary classification metrics."""

    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal lengths.")

    if not y_true:
        raise ValueError("At least one prediction is required.")

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 6),
        "balanced_accuracy": round(
            float(balanced_accuracy_score(y_true, y_pred)),
            6,
        ),
        "macro_f1": round(
            float(f1_score(y_true, y_pred, average="macro")),
            6,
        ),
        "offensive_precision": round(
            float(
                precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            6,
        ),
        "offensive_recall": round(
            float(
                recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            6,
        ),
        "offensive_f1": round(
            float(
                f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                )
            ),
            6,
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
    }


def summarize_latencies(
    latencies_ms: Sequence[float],
) -> dict[str, float | int]:
    """Summarize per-message inference latency."""

    if not latencies_ms:
        raise ValueError("At least one latency measurement is required.")

    values = np.asarray(latencies_ms, dtype=np.float64)

    return {
        "measured_runs": int(len(values)),
        "mean_ms": round(float(values.mean()), 6),
        "median_ms": round(float(np.median(values)), 6),
        "p95_ms": round(float(np.percentile(values, 95)), 6),
        "p99_ms": round(float(np.percentile(values, 99)), 6),
        "minimum_ms": round(float(values.min()), 6),
        "maximum_ms": round(float(values.max()), 6),
    }


def build_evaluation_report(
    *,
    model_name: str,
    split_name: str,
    y_true: Sequence[int],
    predictions: Sequence[Prediction],
    model_artifact: str | Path,
    hardware: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a reproducible evaluation report."""

    if len(y_true) != len(predictions):
        raise ValueError("Labels and predictions must have equal lengths.")

    y_pred = [prediction.label for prediction in predictions]
    latencies = [prediction.latency_ms for prediction in predictions]
    confidences = [prediction.confidence for prediction in predictions]

    return {
        "model_name": model_name,
        "split": split_name,
        "sample_count": len(predictions),
        "model_size_mb": calculate_path_size_mb(model_artifact),
        "metrics": calculate_classification_metrics(y_true, y_pred),
        "latency": summarize_latencies(latencies),
        "average_confidence": round(
            float(np.mean(confidences)),
            6,
        ),
        "hardware": dict(hardware),
    }


def save_evaluation_report(
    report: Mapping[str, Any],
    output_path: str | Path,
) -> Path:
    """Save an evaluation report as formatted JSON."""

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as report_file:
        json.dump(
            dict(report),
            report_file,
            ensure_ascii=False,
            indent=2,
        )

    return destination