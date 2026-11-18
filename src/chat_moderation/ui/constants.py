from __future__ import annotations

DEFAULT_MODEL_KEY = "berturk_int8"
MAX_MESSAGE_LENGTH = 500

MODEL_DESCRIPTIONS = {
    "rule_based": (
        "Deterministic edit-distance baseline using an offensive-term "
        "lexicon. Extremely fast and small, but it misses contextual "
        "offensive language."
    ),
    "tfidf_logreg": (
        "Classical machine-learning baseline using TF-IDF features and "
        "logistic regression. It provides a strong quality-efficiency "
        "trade-off."
    ),
    "word2vec_ffnn": (
        "Custom neural model using averaged Word2Vec embeddings and a "
        "feedforward network. It represents the neural approach trained "
        "from scratch for this project."
    ),
    "berturk": (
        "Fine-tuned Turkish BERT Transformer. It achieves the highest "
        "held-out test Macro-F1, but has the largest model size and "
        "highest CPU latency."
    ),
    "berturk_int8": (
        "Dynamically quantized ONNX version of the fine-tuned BERTurk "
        "model. It is the deployment model because it preserves most of "
        "the Transformer quality while reducing size and CPU latency."
    ),
}
