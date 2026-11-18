from __future__ import annotations

import copy
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import torch
from gensim.models import Word2Vec
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from chat_moderation.models.base import BaseTextClassifier, Prediction
from chat_moderation.preprocessing import tokenize_turkish_text


class FeedForwardNetwork(nn.Module):
    """Feedforward network operating on averaged Word2Vec vectors."""

    def __init__(
        self,
        input_size: int,
        hidden_sizes: Sequence[int],
        dropout: float,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        previous_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend(
                [
                    nn.Linear(previous_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous_size = hidden_size

        layers.append(nn.Linear(previous_size, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def select_device(preference: str) -> torch.device:
    """Select an available PyTorch device."""

    if preference == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")

        if torch.cuda.is_available():
            return torch.device("cuda")

        return torch.device("cpu")

    device = torch.device(preference)

    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    return device


class Word2VecFeedForwardClassifier(BaseTextClassifier):
    """Classifier using self-trained Word2Vec and a feedforward network."""

    def __init__(
        self,
        word2vec_config: Mapping[str, Any],
        ffnn_config: Mapping[str, Any],
        random_state: int,
    ) -> None:
        self.word2vec_config = dict(word2vec_config)
        self.ffnn_config = dict(ffnn_config)
        self.random_state = random_state

        self.device = select_device(
            str(self.ffnn_config["training_device"])
        )
        self.word2vec_model: Word2Vec | None = None
        self.network: FeedForwardNetwork | None = None
        self.training_history: list[dict[str, float | int]] = []

    @property
    def model_name(self) -> str:
        return "Word2Vec Feedforward Network"

    def _set_random_seeds(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def _document_vector(self, text: str) -> np.ndarray:
        if self.word2vec_model is None:
            raise RuntimeError("Word2Vec model has not been trained.")

        tokens = tokenize_turkish_text(text)
        vectors = [
            self.word2vec_model.wv[token]
            for token in tokens
            if token in self.word2vec_model.wv
        ]

        vector_size = int(self.word2vec_config["vector_size"])

        if not vectors:
            return np.zeros(vector_size, dtype=np.float32)

        return np.mean(
            np.asarray(vectors, dtype=np.float32),
            axis=0,
            dtype=np.float32,
        )

    def _create_document_matrix(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        return np.vstack(
            [self._document_vector(text) for text in texts]
        ).astype(np.float32)

    def _predict_feature_labels(
        self,
        features: np.ndarray,
    ) -> list[int]:
        if self.network is None:
            raise RuntimeError("Feedforward network has not been trained.")

        dataset = TensorDataset(torch.from_numpy(features))
        data_loader = DataLoader(
            dataset,
            batch_size=int(
                self.ffnn_config["validation_batch_size"]
            ),
            shuffle=False,
        )

        predictions: list[int] = []
        self.network.eval()

        with torch.no_grad():
            for (feature_batch,) in data_loader:
                logits = self.network(
                    feature_batch.to(self.device)
                )
                batch_predictions = logits.argmax(dim=1)
                predictions.extend(
                    batch_predictions.cpu().tolist()
                )

        return predictions

    def fit(
        self,
        train_texts: Sequence[str],
        train_labels: Sequence[int],
        validation_texts: Sequence[str],
        validation_labels: Sequence[int],
    ) -> dict[str, Any]:
        """Train Word2Vec and the feedforward classifier."""

        if len(train_texts) != len(train_labels):
            raise ValueError(
                "Training texts and labels must have equal lengths."
            )

        if len(validation_texts) != len(validation_labels):
            raise ValueError(
                "Validation texts and labels must have equal lengths."
            )

        if not train_texts or not validation_texts:
            raise ValueError("Training and validation data are required.")

        self._set_random_seeds()

        tokenized_sentences = [
            tokens
            for text in train_texts
            if (tokens := tokenize_turkish_text(text))
        ]

        self.word2vec_model = Word2Vec(
            sentences=tokenized_sentences,
            vector_size=int(
                self.word2vec_config["vector_size"]
            ),
            window=int(self.word2vec_config["window"]),
            min_count=int(self.word2vec_config["min_count"]),
            workers=int(self.word2vec_config["workers"]),
            epochs=int(self.word2vec_config["epochs"]),
            sg=int(self.word2vec_config["skip_gram"]),
            negative=int(
                self.word2vec_config["negative_samples"]
            ),
            seed=self.random_state,
        )

        train_features = self._create_document_matrix(train_texts)
        validation_features = self._create_document_matrix(
            validation_texts
        )

        train_targets = np.asarray(train_labels, dtype=np.int64)
        validation_targets = np.asarray(
            validation_labels,
            dtype=np.int64,
        )

        self.network = FeedForwardNetwork(
            input_size=int(self.word2vec_config["vector_size"]),
            hidden_sizes=self.ffnn_config["hidden_sizes"],
            dropout=float(self.ffnn_config["dropout"]),
        ).to(self.device)

        class_counts = np.bincount(train_targets, minlength=2)

        if np.any(class_counts == 0):
            raise ValueError("Both classes are required for training.")

        class_weights = len(train_targets) / (
            2.0 * class_counts
        )

        loss_function = nn.CrossEntropyLoss(
            weight=torch.tensor(
                class_weights,
                dtype=torch.float32,
                device=self.device,
            )
        )

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=float(self.ffnn_config["learning_rate"]),
            weight_decay=float(
                self.ffnn_config["weight_decay"]
            ),
        )

        train_dataset = TensorDataset(
            torch.from_numpy(train_features),
            torch.from_numpy(train_targets),
        )

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        train_loader = DataLoader(
            train_dataset,
            batch_size=int(self.ffnn_config["batch_size"]),
            shuffle=True,
            generator=generator,
        )

        best_score = -1.0
        best_epoch = 0
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0

        maximum_epochs = int(self.ffnn_config["max_epochs"])
        patience = int(self.ffnn_config["patience"])

        for epoch in range(1, maximum_epochs + 1):
            self.network.train()
            total_loss = 0.0

            for feature_batch, target_batch in train_loader:
                feature_batch = feature_batch.to(self.device)
                target_batch = target_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.network(feature_batch)
                loss = loss_function(logits, target_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(target_batch)

            validation_predictions = (
                self._predict_feature_labels(validation_features)
            )
            validation_macro_f1 = float(
                f1_score(
                    validation_targets,
                    validation_predictions,
                    average="macro",
                )
            )

            average_loss = total_loss / len(train_dataset)

            self.training_history.append(
                {
                    "epoch": epoch,
                    "training_loss": round(average_loss, 6),
                    "validation_macro_f1": round(
                        validation_macro_f1,
                        6,
                    ),
                }
            )

            print(
                f"Epoch {epoch:02d} | "
                f"loss={average_loss:.6f} | "
                f"validation_macro_f1="
                f"{validation_macro_f1:.6f}"
            )

            if validation_macro_f1 > best_score + 1e-6:
                best_score = validation_macro_f1
                best_epoch = epoch
                best_state = copy.deepcopy(
                    self.network.state_dict()
                )
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= patience:
                break

        if best_state is None:
            raise RuntimeError("No valid neural-network state was produced.")

        self.network.load_state_dict(best_state)
        self.network.eval()

        return {
            "best_epoch": best_epoch,
            "epochs_completed": len(self.training_history),
            "best_validation_macro_f1": round(best_score, 6),
            "word2vec_vocabulary_size": len(
                self.word2vec_model.wv
            ),
            "training_device": str(self.device),
            "history": self.training_history,
        }

    def set_inference_device(self, device: str) -> None:
        """Move the neural network to an inference device."""

        if self.network is None:
            raise RuntimeError("Feedforward network has not been trained.")

        self.device = select_device(device)
        self.network = self.network.to(self.device)
        self.network.eval()

    def predict(self, text: str) -> Prediction:
        """Predict one text and measure online latency."""

        if self.network is None or self.word2vec_model is None:
            raise RuntimeError("The classifier has not been trained.")

        start_time = perf_counter()
        vector = self._document_vector(text)
        feature_tensor = torch.from_numpy(vector).unsqueeze(0)
        feature_tensor = feature_tensor.to(self.device)

        self.network.eval()

        with torch.no_grad():
            logits = self.network(feature_tensor)
            probabilities = torch.softmax(logits, dim=1)

        if self.device.type == "mps":
            torch.mps.synchronize()

        probability_values = probabilities.cpu().numpy()[0]
        predicted_label = int(np.argmax(probability_values))
        latency_ms = (perf_counter() - start_time) * 1000

        return Prediction(
            label=predicted_label,
            label_name=(
                "offensive" if predicted_label == 1 else "clean"
            ),
            confidence=round(
                float(probability_values[predicted_label]),
                6,
            ),
            latency_ms=round(latency_ms, 6),
            metadata={
                "offensive_probability": round(
                    float(probability_values[1]),
                    6,
                ),
                "confidence_type": "probability",
            },
        )

    def save(self, output_path: str | Path) -> Path:
        """Save the complete classifier as one artifact."""

        if self.network is None or self.word2vec_model is None:
            raise RuntimeError("The classifier has not been trained.")

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        original_device = self.device

        try:
            self.network = self.network.to("cpu")
            self.device = torch.device("cpu")
            joblib.dump(self, destination, compress=3)
        finally:
            self.device = original_device
            self.network = self.network.to(original_device)

        return destination

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        device: str = "cpu",
    ) -> Word2VecFeedForwardClassifier:
        """Load a saved classifier."""

        loaded_model = joblib.load(model_path)

        if not isinstance(loaded_model, cls):
            raise TypeError(
                f"Unexpected model type: "
                f"{type(loaded_model).__name__}"
            )

        loaded_model.set_inference_device(device)

        return loaded_model