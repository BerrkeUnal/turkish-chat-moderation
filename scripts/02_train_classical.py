from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from chat_moderation.config import load_config, resolve_project_path
from chat_moderation.evaluation import (
    build_evaluation_report,
    save_evaluation_report,
)
from chat_moderation.models.base import BaseTextClassifier
from chat_moderation.models.rule_based import RuleBasedClassifier
from chat_moderation.models.tfidf_logreg import (
    TfidfLogisticClassifier,
)


def evaluate_classifier(
    *,
    classifier: BaseTextClassifier,
    validation_frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    model_artifact: Path,
    hardware: dict[str, Any],
    warmup_runs: int,
) -> dict[str, Any]:
    """Evaluate one classifier on the validation split."""

    texts = validation_frame[text_column].astype(str).tolist()
    labels = validation_frame[label_column].astype(int).tolist()

    actual_warmup_runs = min(warmup_runs, len(texts))

    for text in texts[:actual_warmup_runs]:
        classifier.predict(text)

    predictions = classifier.predict_many(texts)

    report = build_evaluation_report(
        model_name=classifier.model_name,
        split_name="validation",
        y_true=labels,
        predictions=predictions,
        model_artifact=model_artifact,
        hardware=hardware,
    )
    report["latency"]["warmup_runs"] = actual_warmup_runs

    return report


def main() -> None:
    config = load_config()

    processed_data_dir = resolve_project_path(
        config["paths"]["processed_data_dir"]
    )
    artifacts_dir = resolve_project_path(
        config["paths"]["artifacts_dir"]
    )
    metrics_dir = (
        resolve_project_path(config["paths"]["reports_dir"])
        / "metrics"
    )

    train_frame = pd.read_csv(processed_data_dir / "train.csv")
    validation_frame = pd.read_csv(
        processed_data_dir / "validation.csv"
    )

    text_column = config["data"]["text_column"]
    label_column = config["data"]["label_column"]
    warmup_runs = int(config["benchmark"]["warmup_runs"])
    hardware = dict(config["hardware"])

    rule_classifier = RuleBasedClassifier(
        resolve_project_path(config["paths"]["lexicon_file"])
    )

    rule_report = evaluate_classifier(
        classifier=rule_classifier,
        validation_frame=validation_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=rule_classifier.lexicon_path,
        hardware=hardware,
        warmup_runs=warmup_runs,
    )

    rule_report_path = metrics_dir / "rule_based_validation.json"
    save_evaluation_report(rule_report, rule_report_path)

    tfidf_classifier = TfidfLogisticClassifier(
        classical_config=config["classical"],
        preprocessing_config=config["preprocessing"],
        random_state=int(config["project"]["seed"]),
    )

    training_start = perf_counter()

    tfidf_classifier.fit(
        texts=train_frame[text_column].astype(str).tolist(),
        labels=train_frame[label_column].astype(int).tolist(),
    )

    training_seconds = perf_counter() - training_start

    tfidf_model_path = (
        artifacts_dir
        / "tfidf_logreg"
        / "tfidf_logreg.joblib"
    )
    tfidf_classifier.save(tfidf_model_path)

    tfidf_report = evaluate_classifier(
        classifier=tfidf_classifier,
        validation_frame=validation_frame,
        text_column=text_column,
        label_column=label_column,
        model_artifact=tfidf_model_path,
        hardware=hardware,
        warmup_runs=warmup_runs,
    )
    tfidf_report["training"] = {
        "sample_count": int(len(train_frame)),
        "training_seconds": round(training_seconds, 6),
    }

    tfidf_report_path = metrics_dir / "tfidf_logreg_validation.json"
    save_evaluation_report(tfidf_report, tfidf_report_path)

    combined_results = {
        "rule_based": rule_report,
        "tfidf_logreg": tfidf_report,
    }

    print("Classical model training and evaluation completed.")
    print(f"Rule report: {rule_report_path}")
    print(f"TF-IDF model: {tfidf_model_path}")
    print(f"TF-IDF report: {tfidf_report_path}")
    print()
    print(json.dumps(combined_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()