from __future__ import annotations

import json
from time import perf_counter

import pandas as pd

from chat_moderation.config import load_config, resolve_project_path
from chat_moderation.evaluation import (
    build_evaluation_report,
    save_evaluation_report,
)
from chat_moderation.models.word2vec_ffnn import (
    Word2VecFeedForwardClassifier,
)


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

    classifier = Word2VecFeedForwardClassifier(
        word2vec_config=config["word2vec"],
        ffnn_config=config["ffnn"],
        random_state=int(config["project"]["seed"]),
    )

    training_start = perf_counter()

    training_summary = classifier.fit(
        train_texts=train_texts,
        train_labels=train_labels,
        validation_texts=validation_texts,
        validation_labels=validation_labels,
    )

    training_seconds = perf_counter() - training_start

    # All latency measurements use CPU for a fair comparison.
    classifier.set_inference_device("cpu")

    model_path = (
        artifacts_dir
        / "word2vec_ffnn"
        / "word2vec_ffnn.joblib"
    )
    classifier.save(model_path)

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
        model_artifact=model_path,
        hardware=config["hardware"],
    )

    report["latency"]["warmup_runs"] = warmup_runs
    report["training"] = {
        "sample_count": len(train_texts),
        "training_seconds": round(training_seconds, 6),
        **training_summary,
    }

    report_path = (
        metrics_dir / "word2vec_ffnn_validation.json"
    )
    save_evaluation_report(report, report_path)

    print("Word2Vec and feedforward training completed.")
    print(f"Model: {model_path}")
    print(f"Report: {report_path}")
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()