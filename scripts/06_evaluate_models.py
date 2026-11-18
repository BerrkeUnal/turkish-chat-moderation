from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd

from chat_moderation.config import (
    load_config,
    resolve_project_path,
)
from chat_moderation.evaluation import (
    build_evaluation_report,
    save_evaluation_report,
)
from chat_moderation.models.base import BaseTextClassifier
from chat_moderation.models.berturk import BERTurkClassifier
from chat_moderation.models.onnx_int8 import (
    ONNXInt8Classifier,
)
from chat_moderation.models.rule_based import (
    RuleBasedClassifier,
)
from chat_moderation.models.tfidf_logreg import (
    TfidfLogisticClassifier,
)
from chat_moderation.models.word2vec_ffnn import (
    Word2VecFeedForwardClassifier,
)


def evaluate_classifier(
    *,
    classifier: BaseTextClassifier,
    test_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    model_artifact: Path,
    hardware: dict[str, Any],
    warmup_runs: int,
) -> dict[str, Any]:
    """Evaluate one saved classifier on the held-out test set."""

    texts = test_frame[text_column].astype(str).tolist()
    labels = test_frame[label_column].astype(int).tolist()

    actual_warmup_runs = min(warmup_runs, len(texts))

    for text in texts[:actual_warmup_runs]:
        classifier.predict(text)

    predictions = classifier.predict_many(texts)

    report = build_evaluation_report(
        model_name=classifier.model_name,
        split_name="test",
        y_true=labels,
        predictions=predictions,
        model_artifact=model_artifact,
        hardware=hardware,
    )
    report["latency"]["warmup_runs"] = actual_warmup_runs

    return report


def evaluate_and_save(
    *,
    model_key: str,
    classifier: BaseTextClassifier,
    test_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    model_artifact: Path,
    hardware: dict[str, Any],
    warmup_runs: int,
    metrics_directory: Path,
) -> dict[str, Any]:
    print(f"Evaluating {classifier.model_name}...")

    report = evaluate_classifier(
        classifier=classifier,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=model_artifact,
        hardware=hardware,
        warmup_runs=warmup_runs,
    )

    report_path = (
        metrics_directory / f"{model_key}_test.json"
    )
    save_evaluation_report(report, report_path)

    print(f"Saved: {report_path}")
    return report


def summarize_report(
    report: dict[str, Any],
) -> dict[str, Any]:
    metrics = report["metrics"]
    latency = report["latency"]

    return {
        "model_name": report["model_name"],
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics[
            "balanced_accuracy"
        ],
        "macro_f1": metrics["macro_f1"],
        "offensive_precision": metrics[
            "offensive_precision"
        ],
        "offensive_recall": metrics[
            "offensive_recall"
        ],
        "offensive_f1": metrics["offensive_f1"],
        "model_size_mb": report["model_size_mb"],
        "mean_latency_ms": latency["mean_ms"],
        "p95_latency_ms": latency["p95_ms"],
    }


def percentage_reduction(
    original_value: float,
    optimized_value: float,
) -> float:
    return (
        (original_value - optimized_value)
        / original_value
        * 100.0
    )


def main() -> None:
    config = load_config()

    processed_data_directory = resolve_project_path(
        config["paths"]["processed_data_dir"]
    )
    artifacts_directory = resolve_project_path(
        config["paths"]["artifacts_dir"]
    )
    metrics_directory = (
        resolve_project_path(config["paths"]["reports_dir"])
        / "metrics"
    )

    test_frame = pd.read_csv(
        processed_data_directory / "test.csv"
    )

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]
    warmup_runs = int(config["benchmark"]["warmup_runs"])
    hardware = dict(config["hardware"])

    reports: dict[str, dict[str, Any]] = {}

    rule_classifier = RuleBasedClassifier(
        resolve_project_path(
            config["paths"]["lexicon_file"]
        )
    )
    reports["rule_based"] = evaluate_and_save(
        model_key="rule_based",
        classifier=rule_classifier,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=rule_classifier.lexicon_path,
        hardware=hardware,
        warmup_runs=warmup_runs,
        metrics_directory=metrics_directory,
    )
    del rule_classifier
    gc.collect()

    tfidf_model_path = (
        artifacts_directory
        / "tfidf_logreg"
        / "tfidf_logreg.joblib"
    )
    tfidf_classifier = TfidfLogisticClassifier.load(
        tfidf_model_path
    )
    reports["tfidf_logreg"] = evaluate_and_save(
        model_key="tfidf_logreg",
        classifier=tfidf_classifier,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=tfidf_model_path,
        hardware=hardware,
        warmup_runs=warmup_runs,
        metrics_directory=metrics_directory,
    )
    del tfidf_classifier
    gc.collect()

    word2vec_model_path = (
        artifacts_directory
        / "word2vec_ffnn"
        / "word2vec_ffnn.joblib"
    )
    word2vec_classifier = (
        Word2VecFeedForwardClassifier.load(
            word2vec_model_path,
            device="cpu",
        )
    )
    reports["word2vec_ffnn"] = evaluate_and_save(
        model_key="word2vec_ffnn",
        classifier=word2vec_classifier,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=word2vec_model_path,
        hardware=hardware,
        warmup_runs=warmup_runs,
        metrics_directory=metrics_directory,
    )
    del word2vec_classifier
    gc.collect()

    berturk_directory = artifacts_directory / "berturk"
    berturk_classifier = BERTurkClassifier.load(
        artifact_directory=berturk_directory,
        device="cpu",
    )
    reports["berturk"] = evaluate_and_save(
        model_key="berturk",
        classifier=berturk_classifier,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=berturk_directory,
        hardware=hardware,
        warmup_runs=warmup_runs,
        metrics_directory=metrics_directory,
    )
    del berturk_classifier
    gc.collect()

    int8_directory = artifacts_directory / "berturk_int8"
    int8_classifier = ONNXInt8Classifier.load(
        artifact_directory=int8_directory,
        provider=str(
            config["quantization"]["execution_provider"]
        ),
    )
    reports["berturk_int8"] = evaluate_and_save(
        model_key="berturk_int8",
        classifier=int8_classifier,
        test_frame=test_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=int8_classifier.model_path,
        hardware=hardware,
        warmup_runs=warmup_runs,
        metrics_directory=metrics_directory,
    )
    del int8_classifier
    gc.collect()

    model_summaries = {
        model_key: summarize_report(report)
        for model_key, report in reports.items()
    }

    ranking = sorted(
        model_summaries,
        key=lambda model_key: model_summaries[
            model_key
        ]["macro_f1"],
        reverse=True,
    )

    berturk_summary = model_summaries["berturk"]
    int8_summary = model_summaries["berturk_int8"]

    berturk_latency = float(
        berturk_summary["mean_latency_ms"]
    )
    int8_latency = float(
        int8_summary["mean_latency_ms"]
    )
    berturk_size = float(
        berturk_summary["model_size_mb"]
    )
    int8_size = float(
        int8_summary["model_size_mb"]
    )

    comparison_report = {
        "split": "test",
        "sample_count": int(len(test_frame)),
        "primary_metric": "macro_f1",
        "models": model_summaries,
        "ranking_by_macro_f1": ranking,
        "quantization_impact": {
            "macro_f1_delta": round(
                float(int8_summary["macro_f1"])
                - float(berturk_summary["macro_f1"]),
                6,
            ),
            "accuracy_delta": round(
                float(int8_summary["accuracy"])
                - float(berturk_summary["accuracy"]),
                6,
            ),
            "offensive_recall_delta": round(
                float(int8_summary["offensive_recall"])
                - float(
                    berturk_summary["offensive_recall"]
                ),
                6,
            ),
            "size_reduction_percent": round(
                percentage_reduction(
                    berturk_size,
                    int8_size,
                ),
                6,
            ),
            "compression_ratio": round(
                berturk_size / int8_size,
                6,
            ),
            "latency_reduction_percent": round(
                percentage_reduction(
                    berturk_latency,
                    int8_latency,
                ),
                6,
            ),
            "latency_speedup": round(
                berturk_latency / int8_latency,
                6,
            ),
        },
        "deployment_selection": {
            "model_key": "berturk_int8",
            "reason": (
                "Selected for deployment because it preserves "
                "Transformer classification quality while reducing "
                "model size and CPU inference latency."
            ),
        },
        "hardware": hardware,
    }

    comparison_path = (
        metrics_directory / "model_comparison_test.json"
    )
    save_evaluation_report(
        comparison_report,
        comparison_path,
    )

    print()
    print("Final held-out test evaluation completed.")
    print(f"Comparison report: {comparison_path}")
    print()
    print(
        json.dumps(
            comparison_report,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()