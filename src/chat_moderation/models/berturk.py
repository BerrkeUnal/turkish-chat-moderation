from __future__ import annotations

import json
import math
import random
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from .base import BaseTextClassifier, Prediction


class _EncodedTextDataset(Dataset[dict[str, torch.Tensor]]):
    """Tokenized text dataset used during fine-tuning."""

    def __init__(
        self,
        texts: Sequence[str],
        labels: Sequence[int],
        tokenizer: Any,
        max_length: int,
    ) -> None:
        self.encodings = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {
            name: tensor[index]
            for name, tensor in self.encodings.items()
        }
        item["labels"] = self.labels[index]
        return item


def _resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    return torch.device(requested_device)


def _set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class BERTurkClassifier(BaseTextClassifier):
    """Fine-tuned BERTurk classifier for Turkish chat moderation."""

    def __init__(
        self,
        model_name_or_path: str,
        max_length: int = 128,
        seed: int = 42,
        device: str = "auto",
    ) -> None:
        self.source_model_name = model_name_or_path
        self.max_length = max_length
        self.seed = seed
        self.device = _resolve_device(device)
        self.training_summary: dict[str, Any] = {}

        _set_random_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path
        )
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=2,
                id2label={0: "clean", 1: "offensive"},
                label2id={"clean": 0, "offensive": 1},
            )
        )
        self.model.to(self.device)

    @property
    def model_name(self) -> str:
        return "BERTurk Fine-Tuned"

    def fit(
        self,
        training_texts: Sequence[str],
        training_labels: Sequence[int],
        validation_texts: Sequence[str],
        validation_labels: Sequence[int],
        batch_size: int,
        learning_rate: float,
        max_epochs: int,
        weight_decay: float,
        gradient_accumulation_steps: int,
        warmup_ratio: float,
        gradient_clip_norm: float,
        patience: int,
    ) -> dict[str, Any]:
        if len(training_texts) != len(training_labels):
            raise ValueError(
                "Training texts and labels must have equal lengths."
            )

        if len(validation_texts) != len(validation_labels):
            raise ValueError(
                "Validation texts and labels must have equal lengths."
            )

        _set_random_seed(self.seed)

        training_dataset = _EncodedTextDataset(
            texts=training_texts,
            labels=training_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )
        validation_dataset = _EncodedTextDataset(
            texts=validation_texts,
            labels=validation_labels,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
        )

        generator = torch.Generator()
        generator.manual_seed(self.seed)

        training_loader = DataLoader(
            training_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
            pin_memory=False,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        label_counts = torch.bincount(
            torch.tensor(training_labels, dtype=torch.long),
            minlength=2,
        ).float()

        if torch.any(label_counts == 0):
            raise ValueError(
                "Both clean and offensive classes are required."
            )

        class_weights = len(training_labels) / (
            2.0 * label_counts
        )
        loss_function = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device)
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        optimizer_steps_per_epoch = math.ceil(
            len(training_loader) / gradient_accumulation_steps
        )
        total_optimizer_steps = (
            optimizer_steps_per_epoch * max_epochs
        )
        warmup_steps = int(total_optimizer_steps * warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )

        best_macro_f1 = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        history: list[dict[str, float | int]] = []

        training_started_at = perf_counter()

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0

            for batch_index, batch in enumerate(
                training_loader,
                start=1,
            ):
                labels = batch.pop("labels").to(self.device)
                model_inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in batch.items()
                }

                outputs = self.model(**model_inputs)
                loss = loss_function(outputs.logits, labels)
                total_loss += float(loss.item())

                scaled_loss = (
                    loss / gradient_accumulation_steps
                )
                scaled_loss.backward()

                should_update = (
                    batch_index % gradient_accumulation_steps == 0
                    or batch_index == len(training_loader)
                )

                if should_update:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=gradient_clip_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            validation_macro_f1 = self._evaluate_macro_f1(
                validation_loader
            )
            average_training_loss = total_loss / len(
                training_loader
            )

            history.append(
                {
                    "epoch": epoch,
                    "training_loss": round(
                        average_training_loss,
                        6,
                    ),
                    "validation_macro_f1": round(
                        validation_macro_f1,
                        6,
                    ),
                }
            )

            print(
                f"Epoch {epoch:02d} | "
                f"loss={average_training_loss:.6f} | "
                f"validation_macro_f1="
                f"{validation_macro_f1:.6f}"
            )

            if validation_macro_f1 > best_macro_f1:
                best_macro_f1 = validation_macro_f1
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    name: parameter.detach().cpu().clone()
                    for name, parameter
                    in self.model.state_dict().items()
                }
            else:
                epochs_without_improvement += 1

                if (
                    patience > 0
                    and epochs_without_improvement >= patience
                ):
                    print("Early stopping activated.")
                    break

        if best_state is None:
            raise RuntimeError(
                "Transformer training did not produce a checkpoint."
            )

        self.model.load_state_dict(best_state)
        self.model.to(self.device)
        self.model.eval()

        training_seconds = perf_counter() - training_started_at

        self.training_summary = {
            "sample_count": len(training_labels),
            "training_seconds": round(training_seconds, 6),
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_validation_macro_f1": round(
                best_macro_f1,
                6,
            ),
            "training_device": str(self.device),
            "batch_size": batch_size,
            "gradient_accumulation_steps": (
                gradient_accumulation_steps
            ),
            "effective_batch_size": (
                batch_size * gradient_accumulation_steps
            ),
            "learning_rate": learning_rate,
            "warmup_steps": warmup_steps,
            "class_weights": [
                round(float(weight), 6)
                for weight in class_weights
            ],
            "history": history,
        }

        return self.training_summary

    def _evaluate_macro_f1(
        self,
        data_loader: DataLoader[
            dict[str, torch.Tensor]
        ],
    ) -> float:
        self.model.eval()

        true_labels: list[int] = []
        predicted_labels: list[int] = []

        with torch.inference_mode():
            for batch in data_loader:
                labels = batch.pop("labels")
                model_inputs = {
                    name: tensor.to(self.device)
                    for name, tensor in batch.items()
                }

                logits = self.model(**model_inputs).logits
                predictions = torch.argmax(
                    logits,
                    dim=-1,
                ).cpu()

                true_labels.extend(labels.tolist())
                predicted_labels.extend(predictions.tolist())

        return float(
            f1_score(
                true_labels,
                predicted_labels,
                average="macro",
                zero_division=0,
            )
        )

    def predict(self, text: str) -> Prediction:
        started_at = perf_counter()

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        model_inputs = {
            name: tensor.to(self.device)
            for name, tensor in encoded.items()
        }

        self.model.eval()

        with torch.inference_mode():
            logits = self.model(**model_inputs).logits
            probabilities = torch.softmax(
                logits,
                dim=-1,
            )[0].cpu()
            label = int(torch.argmax(probabilities).item())

        latency_ms = (perf_counter() - started_at) * 1000.0

        return Prediction(
            label=label,
            label_name=(
                "offensive" if label == 1 else "clean"
            ),
            confidence=float(probabilities[label].item()),
            latency_ms=latency_ms,
            metadata={
                "device": str(self.device),
                "max_length": self.max_length,
            },
        )

    def set_inference_device(self, device: str) -> None:
        self.device = _resolve_device(device)
        self.model.to(self.device)
        self.model.eval()

    def save(self, artifact_directory: Path) -> None:
        artifact_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            artifact_directory,
            safe_serialization=True,
        )
        self.tokenizer.save_pretrained(artifact_directory)

        metadata = {
            "model_name": self.model_name,
            "source_model_name": self.source_model_name,
            "max_length": self.max_length,
            "seed": self.seed,
            "training_summary": self.training_summary,
        }

        metadata_path = artifact_directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(
        cls,
        artifact_directory: Path,
        device: str = "cpu",
    ) -> BERTurkClassifier:
        metadata_path = artifact_directory / "metadata.json"

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        classifier = cls(
            model_name_or_path=str(artifact_directory),
            max_length=int(metadata["max_length"]),
            seed=int(metadata["seed"]),
            device=device,
        )
        classifier.source_model_name = str(
            metadata["source_model_name"]
        )
        classifier.training_summary = dict(
            metadata.get("training_summary", {})
        )

        return classifier