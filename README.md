# Turkish Chat Moderation

A real-time Turkish offensive-language detection application that compares rule-based, classical machine-learning, neural-network and Transformer approaches.

The application provides live message moderation, confidence scores, CPU inference latency, model comparison and held-out test results through a Gradio interface.

## Features

- Real-time Turkish message classification
- Clean/offensive prediction with confidence
- Offensive-message masking
- Per-message inference latency
- Runtime model selection
- Side-by-side comparison of five models
- Quantized Transformer deployment with ONNX Runtime
- Held-out benchmark results
- Lazy model loading for efficient memory usage

## Model Ladder

| Area            | Implementation                                                                       |
| --------------- | ------------------------------------------------------------------------------------ |
| Classical NLP   | Text normalization, edit-distance lexicon matcher and TF-IDF logistic regression     |
| Neural network  | Word2Vec embeddings trained on the project dataset with a feedforward neural network |
| Transformer     | Fine-tuned `dbmdz/bert-base-turkish-cased` model                                     |
| Efficient model | Dynamic INT8 ONNX quantization and ONNX Runtime CPU inference                        |

Five models are included in the comparison because two classical approaches are evaluated:

1. Rule-Based Edit Distance
2. TF-IDF Logistic Regression
3. Word2Vec Feedforward Network
4. BERTurk Fine-Tuned
5. BERTurk Dynamic INT8 ONNX

## Held-Out Test Results

All models were evaluated on the same untouched test split containing 3,528 examples.

Latency was measured on CPU using an Apple M4 system with 24 GB memory.

| Model                        | Macro-F1 | Accuracy | Offensive Recall | Size (MB) | Mean Latency (ms) |
| ---------------------------- | -------: | -------: | ---------------: | --------: | ----------------: |
| BERTurk Fine-Tuned           |   0.8160 |   0.8824 |           0.6941 |    422.72 |            14.740 |
| BERTurk Dynamic INT8 ONNX    |   0.8115 |   0.8798 |           0.6844 |    178.82 |             5.768 |
| TF-IDF Logistic Regression   |   0.7665 |   0.8492 |           0.6257 |      2.07 |             0.599 |
| Word2Vec Feedforward Network |   0.6789 |   0.7650 |           0.6089 |     27.11 |             0.059 |
| Rule-Based Edit Distance     |   0.6132 |   0.8265 |           0.2067 |    <0.001 |             0.100 |

Dynamic INT8 quantization reduced the Transformer model size by approximately **57.70%** and produced a **2.56× CPU latency speed-up**, with a Macro-F1 decrease of only **0.0045**.

The INT8 ONNX model is therefore used as the default deployment model.

## Architecture

```text
User message
    │
    ▼
Gradio interface
    │
    ▼
Text validation and preprocessing
    │
    ├── Rule-Based Edit Distance
    ├── TF-IDF Logistic Regression
    ├── Word2Vec Feedforward Network
    ├── BERTurk Fine-Tuned
    └── BERTurk Dynamic INT8 ONNX
    │
    ▼
Prediction, confidence and latency
    │
    ▼
Allow message or mask offensive content
```

The application loads models only when they are requested. This prevents all model artifacts from occupying memory during application startup.

## Requirements

- Python 3.11
- Approximately 2 GB of free disk space
- CPU inference is supported
- Apple Silicon MPS is optional and only used during training

Python 3.12 or newer is not currently supported by this project.

## Installation

Clone the repository:

```bash
git clone https://github.com/BerrkeUnal/turkish-chat-moderation.git
cd turkish-chat-moderation
```

Create and activate a Python 3.11 virtual environment:

### macOS or Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Model Artifacts

Large trained model files are not stored in the repository.

Download `artifacts.zip` from:

```text
<MODEL_DOWNLOAD_URL>
```

Extract it into the repository root. The final structure must be:

```text
artifacts/
├── berturk/
├── berturk_int8/
├── berturk_onnx/
├── tfidf_logreg/
└── word2vec_ffnn/
```

The rule-based model uses the lexicon stored in `resources/`.

## Run the Application

From the repository root, run:

```bash
python app.py
```

The application opens locally at:

```text
http://127.0.0.1:7860
```

## Run Tests

```bash
pytest -q
```

Run code-quality checks with:

```bash
ruff check src scripts tests app.py
```

## Reproducing the Pipeline

The dataset and generated model artifacts are excluded from the repository.

Run the scripts in the following order:

```bash
python scripts/01_prepare_data.py
python scripts/02_train_classical.py
python scripts/03_train_neural.py
python scripts/04_train_transformer.py
python scripts/05_quantize_transformer.py
python scripts/06_evaluate_models.py
```

Transformer fine-tuning is the most computationally expensive step.

## Dataset

The project uses the Turkish OffensEval 2020 dataset:

- Source: https://coltekin.github.io/offensive-turkish/
- Task: Binary clean/offensive message classification
- Training pool after cleaning: 31,742 examples
- Held-out test set: 3,528 examples
- Primary metric: Macro-F1

Duplicate and overlapping messages were removed before creating the final train and validation splits.

## Project Structure

```text
turkish-chat-moderation/
├── app.py
├── configs/
│   └── config.yaml
├── reports/
│   └── metrics/
├── resources/
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_classical.py
│   ├── 03_train_neural.py
│   ├── 04_train_transformer.py
│   ├── 05_quantize_transformer.py
│   └── 06_evaluate_models.py
├── src/
│   └── chat_moderation/
│       ├── models/
│       └── ui/
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Limitations

- Sarcasm and quotations may be classified incorrectly.
- Spelling variations and previously unseen slang may affect predictions.
- Identity-related expressions require careful human review.
- Individual messages are classified without full conversation context.
- The system should assist human moderation rather than make high-impact decisions automatically.

## Responsible Use

Messages classified as offensive are hidden from the live conversation display. Predictions should be reviewed by a human when used in a real moderation workflow.

## AI Usage

AI coding assistants were used for code drafting, debugging, interface refactoring and documentation support. All dataset statistics, evaluation metrics and latency results included in this repository were produced by local project runs.
