from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from chat_moderation.models.base import BaseTextClassifier, Prediction
from chat_moderation.preprocessing import normalize_turkish_text


class TfidfLogisticClassifier(BaseTextClassifier):
    """TF-IDF and logistic-regression text classifier."""

    def __init__(
        self,
        classical_config: Mapping[str, Any],
        preprocessing_config: Mapping[str, Any],
        random_state: int,
    ) -> None:
        self.classical_config = dict(classical_config)
        self.preprocessing_config = dict(preprocessing_config)
        self.random_state = random_state
        self.pipeline = self._build_pipeline()

    @property
    def model_name(self) -> str:
        return "TF-IDF Logistic Regression"

    def _build_pipeline(self) -> Pipeline:
        normalizer = partial(
            normalize_turkish_text,
            lowercase=bool(
                self.preprocessing_config["lowercase_for_classical"]
            ),
            replace_urls=bool(
                self.preprocessing_config["replace_urls"]
            ),
            replace_user_mentions=bool(
                self.preprocessing_config["replace_user_mentions"]
            ),
            max_repeated_characters=int(
                self.preprocessing_config[
                    "max_repeated_characters"
                ]
            ),
        )

        minimum_document_frequency = int(
            self.classical_config["min_document_frequency"]
        )

        word_vectorizer = TfidfVectorizer(
            preprocessor=normalizer,
            lowercase=False,
            analyzer="word",
            ngram_range=tuple(
                self.classical_config["word_ngram_range"]
            ),
            max_features=int(
                self.classical_config["word_max_features"]
            ),
            min_df=minimum_document_frequency,
            sublinear_tf=True,
            dtype=np.float32,
        )

        character_vectorizer = TfidfVectorizer(
            preprocessor=normalizer,
            lowercase=False,
            analyzer="char_wb",
            ngram_range=tuple(
                self.classical_config["char_ngram_range"]
            ),
            max_features=int(
                self.classical_config["char_max_features"]
            ),
            min_df=minimum_document_frequency,
            sublinear_tf=True,
            dtype=np.float32,
        )

        features = FeatureUnion(
            transformer_list=[
                ("word_tfidf", word_vectorizer),
                ("character_tfidf", character_vectorizer),
            ],
            n_jobs=1,
        )

        classifier = LogisticRegression(
            C=float(
                self.classical_config["logistic_regression_c"]
            ),
            max_iter=int(self.classical_config["max_iter"]),
            class_weight=self.classical_config["class_weight"],
            solver=str(self.classical_config["solver"]),
            random_state=self.random_state,
        )

        return Pipeline(
            steps=[
                ("features", features),
                ("classifier", classifier),
            ]
        )

    def fit(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
    ) -> TfidfLogisticClassifier:
        """Fit the vectorizers and logistic-regression classifier."""

        if len(texts) != len(labels):
            raise ValueError("Texts and labels must have equal lengths.")

        if not texts:
            raise ValueError("Training data cannot be empty.")

        self.pipeline.fit(list(texts), list(labels))

        return self

    def predict(self, text: str) -> Prediction:
        """Predict one message and measure online latency."""

        start_time = perf_counter()
        probabilities = self.pipeline.predict_proba([text])[0]
        latency_ms = (perf_counter() - start_time) * 1000

        classifier = self.pipeline.named_steps["classifier"]
        classes = classifier.classes_

        predicted_index = int(np.argmax(probabilities))
        predicted_label = int(classes[predicted_index])

        offensive_indexes = np.flatnonzero(classes == 1)
        offensive_probability = (
            float(probabilities[offensive_indexes[0]])
            if len(offensive_indexes) == 1
            else 0.0
        )

        return Prediction(
            label=predicted_label,
            label_name=(
                "offensive" if predicted_label == 1 else "clean"
            ),
            confidence=round(
                float(probabilities[predicted_index]),
                6,
            ),
            latency_ms=round(latency_ms, 6),
            metadata={
                "offensive_probability": round(
                    offensive_probability,
                    6,
                ),
                "confidence_type": "probability",
            },
        )

    def save(self, output_path: str | Path) -> Path:
        """Save the complete fitted model pipeline."""

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination, compress=3)

        return destination

    @classmethod
    def load(
        cls,
        model_path: str | Path,
    ) -> TfidfLogisticClassifier:
        """Load a previously saved model pipeline."""

        loaded_model = joblib.load(model_path)

        if not isinstance(loaded_model, cls):
            raise TypeError(
                f"Unexpected model type: {type(loaded_model).__name__}"
            )

        return loaded_model