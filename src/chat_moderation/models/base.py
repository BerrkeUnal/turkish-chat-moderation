from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Prediction:
    """Standard prediction returned by every model."""

    label: int
    label_name: str
    confidence: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseTextClassifier(ABC):
    """Common interface for all text classifiers."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the display name of the model."""

    @abstractmethod
    def predict(self, text: str) -> Prediction:
        """Predict a label for one text."""

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Predict labels for multiple texts."""

        return [self.predict(text) for text in texts]