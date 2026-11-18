from chat_moderation.evaluation import (
    build_evaluation_report,
    calculate_classification_metrics,
)
from chat_moderation.models.base import Prediction
from chat_moderation.models.tfidf_logreg import (
    TfidfLogisticClassifier,
)
from chat_moderation.models.word2vec_ffnn import (
    Word2VecFeedForwardClassifier,
)


def create_prediction(label: int) -> Prediction:
    return Prediction(
        label=label,
        label_name="offensive" if label == 1 else "clean",
        confidence=0.9,
        latency_ms=1.0,
    )


def test_classification_metrics_for_perfect_predictions() -> None:
    metrics = calculate_classification_metrics(
        y_true=[0, 0, 1, 1],
        y_pred=[0, 0, 1, 1],
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["offensive_recall"] == 1.0


def test_evaluation_report_contains_required_fields(tmp_path) -> None:
    artifact = tmp_path / "model.txt"
    artifact.write_text("model", encoding="utf-8")

    report = build_evaluation_report(
        model_name="Test Model",
        split_name="validation",
        y_true=[0, 1],
        predictions=[
            create_prediction(0),
            create_prediction(1),
        ],
        model_artifact=artifact,
        hardware={"processor": "Test CPU"},
    )

    assert report["model_name"] == "Test Model"
    assert report["sample_count"] == 2
    assert report["model_size_mb"] > 0
    assert report["latency"]["measured_runs"] == 2

def test_tfidf_classifier_can_fit_save_and_load(tmp_path) -> None:
    classical_config = {
        "word_ngram_range": [1, 2],
        "char_ngram_range": [3, 4],
        "word_max_features": 100,
        "char_max_features": 100,
        "min_document_frequency": 1,
        "logistic_regression_c": 1.0,
        "max_iter": 200,
        "class_weight": "balanced",
        "solver": "liblinear",
    }
    preprocessing_config = {
        "lowercase_for_classical": True,
        "replace_urls": True,
        "replace_user_mentions": True,
        "max_repeated_characters": 2,
    }

    classifier = TfidfLogisticClassifier(
        classical_config=classical_config,
        preprocessing_config=preprocessing_config,
        random_state=42,
    )

    classifier.fit(
        texts=[
            "Bugün hava çok güzel",
            "Harika bir gün",
            "Toplantı yarın yapılacak",
            "Sen aptalsın",
            "Tam bir salak",
            "Bu mesaj çok kötü",
        ],
        labels=[0, 0, 0, 1, 1, 1],
    )

    prediction = classifier.predict("Sen aptalsın")

    assert prediction.label in {0, 1}
    assert 0.0 <= prediction.confidence <= 1.0
    assert prediction.latency_ms >= 0.0

    model_path = tmp_path / "tfidf_model.joblib"
    classifier.save(model_path)

    loaded_classifier = TfidfLogisticClassifier.load(model_path)
    loaded_prediction = loaded_classifier.predict("Harika bir gün")

    assert loaded_prediction.label in {0, 1}


def test_word2vec_ffnn_can_fit_save_and_load(tmp_path) -> None:
    word2vec_config = {
        "vector_size": 8,
        "window": 2,
        "min_count": 1,
        "workers": 1,
        "epochs": 5,
        "skip_gram": 1,
        "negative_samples": 3,
    }
    ffnn_config = {
        "hidden_sizes": [8],
        "dropout": 0.1,
        "batch_size": 2,
        "learning_rate": 0.01,
        "max_epochs": 2,
        "patience": 2,
        "weight_decay": 0.0001,
        "validation_batch_size": 4,
        "training_device": "cpu",
    }

    classifier = Word2VecFeedForwardClassifier(
        word2vec_config=word2vec_config,
        ffnn_config=ffnn_config,
        random_state=42,
    )

    training_summary = classifier.fit(
        train_texts=[
            "hava bugün çok güzel",
            "harika bir gün",
            "toplantı yarın",
            "sen aptalsın",
            "tam bir salak",
            "çok kötü mesaj",
        ],
        train_labels=[0, 0, 0, 1, 1, 1],
        validation_texts=[
            "güzel bir gün",
            "sen salaksın",
        ],
        validation_labels=[0, 1],
    )

    assert training_summary["epochs_completed"] >= 1

    prediction = classifier.predict("harika bir gün")

    assert prediction.label in {0, 1}
    assert 0.0 <= prediction.confidence <= 1.0

    model_path = tmp_path / "word2vec_ffnn.joblib"
    classifier.save(model_path)

    loaded_classifier = Word2VecFeedForwardClassifier.load(
        model_path,
        device="cpu",
    )
    loaded_prediction = loaded_classifier.predict("sen aptalsın")

    assert loaded_prediction.label in {0, 1}