from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from .base import BaseTextClassifier, Prediction


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted_logits = logits - np.max(logits)
    exponentials = np.exp(shifted_logits)
    return exponentials / np.sum(exponentials)


class ONNXInt8Classifier(BaseTextClassifier):
    """Dynamically quantized BERTurk classifier using ONNX Runtime."""

    def __init__(
        self,
        artifact_directory: Path,
        provider: str | None = None,
    ) -> None:
        metadata_path = artifact_directory / "metadata.json"

        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"INT8 metadata not found: {metadata_path}"
            )

        self.artifact_directory = artifact_directory
        self.metadata: dict[str, Any] = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        self.max_length = int(self.metadata["max_length"])
        self.output_name = str(self.metadata["output_name"])
        self.label_names = {
            int(label): str(name)
            for label, name in self.metadata["labels"].items()
        }

        configured_provider = str(
            self.metadata["execution_provider"]
        )
        self.provider = provider or configured_provider

        available_providers = ort.get_available_providers()

        if self.provider not in available_providers:
            raise ValueError(
                f"ONNX provider {self.provider!r} is unavailable. "
                f"Available providers: {available_providers}"
            )

        onnx_file = str(self.metadata["onnx_file"])
        self.model_path = artifact_directory / onnx_file

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"INT8 ONNX model not found: {self.model_path}"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(
            artifact_directory
        )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=session_options,
            providers=[self.provider],
        )
        self.input_names = tuple(
            input_info.name
            for input_info in self.session.get_inputs()
        )

    @property
    def model_name(self) -> str:
        return str(self.metadata["model_name"])

    def predict(self, text: str) -> Prediction:
        started_at = perf_counter()

        encoded_inputs = self.tokenizer(
            text,
            return_tensors="np",
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )

        if (
            "token_type_ids" in self.input_names
            and "token_type_ids" not in encoded_inputs
        ):
            encoded_inputs["token_type_ids"] = np.zeros_like(
                encoded_inputs["input_ids"]
            )

        missing_inputs = set(self.input_names).difference(
            encoded_inputs
        )

        if missing_inputs:
            raise ValueError(
                "Tokenizer did not produce required inputs: "
                f"{sorted(missing_inputs)}"
            )

        runtime_inputs = {
            name: np.asarray(
                encoded_inputs[name],
                dtype=np.int64,
            )
            for name in self.input_names
        }

        logits = np.asarray(
            self.session.run(
                [self.output_name],
                runtime_inputs,
            )[0]
        )[0]
        probabilities = _softmax(logits)

        label = int(np.argmax(probabilities))
        confidence = float(probabilities[label])
        latency_ms = (perf_counter() - started_at) * 1000.0

        return Prediction(
            label=label,
            label_name=self.label_names[label],
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "provider": self.provider,
                "quantization": "dynamic_int8",
                "max_length": self.max_length,
            },
        )

    @classmethod
    def load(
        cls,
        artifact_directory: Path,
        provider: str | None = None,
    ) -> ONNXInt8Classifier:
        return cls(
            artifact_directory=artifact_directory,
            provider=provider,
        )