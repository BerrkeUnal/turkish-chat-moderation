from __future__ import annotations

import json
from time import perf_counter

import pandas as pd

from chat_moderation.config import load_config, resolve_project_path
from chat_moderation.evaluation import (
    build_evaluation_report,
    save_evaluation_report,
)
from chat_moderation.models.berturk import BERTurkClassifier


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

    train_texts = train_frame[text_column].astype(str).tolist()
    train_labels = train_frame[label_column].astype(int).tolist()
    validation_texts = (
        validation_frame[text_column].astype(str).tolist()
    )
    validation_labels = (
        validation_frame[label_column].astype(int).tolist()
    )

    transformer_config = config["transformer"]

    classifier = BERTurkClassifier(
        model_name_or_path=str(
            transformer_config["model_name"]
        ),
        max_length=int(transformer_config["max_length"]),
        seed=int(config["project"]["seed"]),
        device=str(transformer_config["training_device"]),
    )

    training_start = perf_counter()

    training_summary = classifier.fit(
        training_texts=train_texts,
        training_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
        batch_size=int(transformer_config["batch_size"]),
        learning_rate=float(
            transformer_config["learning_rate"]
        ),
        max_epochs=int(transformer_config["epochs"]),
        weight_decay=float(
            transformer_config["weight_decay"]
        ),
        gradient_accumulation_steps=int(
            transformer_config[
                "gradient_accumulation_steps"
            ]
        ),
        warmup_ratio=float(
            transformer_config["warmup_ratio"]
        ),
        gradient_clip_norm=float(
            transformer_config["gradient_clip_norm"]
        ),
        patience=int(transformer_config["patience"]),
    )

    training_seconds = perf_counter() - training_start

    # All latency measurements use CPU for a fair comparison.
    classifier.set_inference_device("cpu")

    model_directory = artifacts_dir / "berturk"
    classifier.save(model_directory)

    warmup_runs = min(
        int(config["benchmark"]["warmup_runs"]),
        len(validation_texts),
    )

    for text in validation_texts[:warmup_runs]:
        classifier.predict(text)

    predictions = classifier.predict_many(validation_texts)

    report = build_evaluation_report(
        model_name=classifier.model_name,
        split_name="validation",
        y_true=validation_labels,
        predictions=predictions,
        model_artifact=model_directory,
        hardware=config["hardware"],
    )

    report["latency"]["warmup_runs"] = warmup_runs
    report["training"] = {
        **training_summary,
        "sample_count": len(train_texts),
        "training_seconds": round(training_seconds, 6),
    }

    report_path = (
        metrics_dir / "berturk_validation.json"
    )
    save_evaluation_report(report, report_path)

    print("BERTurk fine-tuning and evaluation completed.")
    print(f"Model: {model_directory}")
    print(f"Report: {report_path}")
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()