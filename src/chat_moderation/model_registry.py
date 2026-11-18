from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from chat_moderation.config import (
    load_config,
    resolve_project_path,
)
from chat_moderation.models.base import (
    BaseTextClassifier,
    Prediction,
)
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


@dataclass(frozen=True, slots=True)
class ModelDescriptor:
    key: str
    display_name: str
    description: str
    deployment_default: bool = False


MODEL_DESCRIPTORS = (
    ModelDescriptor(
        key="rule_based",
        display_name="Rule-Based Edit Distance",
        description=(
            "Fuzzy offensive lexicon baseline using "
            "Levenshtein distance."
        ),
    ),
    ModelDescriptor(
        key="tfidf_logreg",
        display_name="TF-IDF Logistic Regression",
        description=(
            "Word and character TF-IDF features with "
            "logistic regression."
        ),
    ),
    ModelDescriptor(
        key="word2vec_ffnn",
        display_name="Word2Vec Feedforward Network",
        description=(
            "Custom neural model using averaged Word2Vec "
            "embeddings and an FFNN."
        ),
    ),
    ModelDescriptor(
        key="berturk",
        display_name="BERTurk Fine-Tuned",
        description=(
            "Full-precision fine-tuned Turkish BERT "
            "Transformer."
        ),
    ),
    ModelDescriptor(
        key="berturk_int8",
        display_name="BERTurk Dynamic INT8 ONNX",
        description=(
            "Dynamically quantized deployment model using "
            "ONNX Runtime."
        ),
        deployment_default=True,
    ),
)

DESCRIPTORS_BY_KEY = {
    descriptor.key: descriptor
    for descriptor in MODEL_DESCRIPTORS
}


class ModelRegistry:
    """Lazily load and cache all moderation models."""

    default_model_key = "berturk_int8"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
    ) -> None:
        self.config = config or load_config()
        self._models: dict[str, BaseTextClassifier] = {}
        self._lock = Lock()

    @property
    def descriptors(self) -> tuple[ModelDescriptor, ...]:
        return MODEL_DESCRIPTORS

    def model_choices(self) -> list[tuple[str, str]]:
        return [
            (descriptor.display_name, descriptor.key)
            for descriptor in self.descriptors
        ]

    def get_descriptor(
        self,
        model_key: str,
    ) -> ModelDescriptor:
        try:
            return DESCRIPTORS_BY_KEY[model_key]
        except KeyError as error:
            available_keys = sorted(DESCRIPTORS_BY_KEY)
            raise ValueError(
                f"Unknown model key {model_key!r}. "
                f"Available models: {available_keys}"
            ) from error

    def loaded_model_keys(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                descriptor.key
                for descriptor in self.descriptors
                if descriptor.key in self._models
            )

    def get_model(
        self,
        model_key: str,
    ) -> BaseTextClassifier:
        self.get_descriptor(model_key)

        with self._lock:
            cached_model = self._models.get(model_key)

            if cached_model is not None:
                return cached_model

            classifier = self._load_model(model_key)
            self._models[model_key] = classifier
            return classifier

    def predict(
        self,
        model_key: str,
        text: str,
    ) -> Prediction:
        classifier = self.get_model(model_key)
        return classifier.predict(text)

    def predict_all(
        self,
        text: str,
    ) -> dict[str, Prediction]:
        return {
            descriptor.key: self.predict(
                descriptor.key,
                text,
            )
            for descriptor in self.descriptors
        }

    def _load_model(
        self,
        model_key: str,
    ) -> BaseTextClassifier:
        artifacts_directory = resolve_project_path(
            self.config["paths"]["artifacts_dir"]
        )

        if model_key == "rule_based":
            lexicon_path = resolve_project_path(
                self.config["paths"]["lexicon_file"]
            )
            return RuleBasedClassifier(lexicon_path)

        if model_key == "tfidf_logreg":
            model_path = (
                artifacts_directory
                / "tfidf_logreg"
                / "tfidf_logreg.joblib"
            )
            return TfidfLogisticClassifier.load(model_path)

        if model_key == "word2vec_ffnn":
            model_path = (
                artifacts_directory
                / "word2vec_ffnn"
                / "word2vec_ffnn.joblib"
            )
            return Word2VecFeedForwardClassifier.load(
                model_path,
                device="cpu",
            )

        if model_key == "berturk":
            model_directory = artifacts_directory / "berturk"
            return BERTurkClassifier.load(
                artifact_directory=model_directory,
                device="cpu",
            )

        if model_key == "berturk_int8":
            model_directory = (
                artifacts_directory / "berturk_int8"
            )
            provider = str(
                self.config["quantization"][
                    "execution_provider"
                ]
            )
            return ONNXInt8Classifier.load(
                artifact_directory=model_directory,
                provider=provider,
            )

        raise ValueError(f"Unsupported model: {model_key}")