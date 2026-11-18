from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from chat_moderation.config import (
    load_config,
    resolve_project_path,
)
from chat_moderation.models.berturk import BERTurkClassifier
from chat_moderation.quantization import (
    calculate_file_size_mb,
    export_sequence_classifier_to_onnx,
    quantize_onnx_dynamic_int8,
    run_onnx_logits,
)


def calculate_directory_size_mb(directory: Path) -> float:
    total_bytes = sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file()
    )
    return total_bytes / (1024**2)


def main() -> None:
    config = load_config()

    artifacts_directory = resolve_project_path(
        config["paths"]["artifacts_dir"]
    )
    reports_directory = resolve_project_path(
        config["paths"]["reports_dir"]
    )

    berturk_directory = artifacts_directory / "berturk"
    onnx_directory = artifacts_directory / "berturk_onnx"
    int8_directory = artifacts_directory / "berturk_int8"

    fp32_onnx_path = (
        onnx_directory / "berturk_fp32.onnx"
    )
    int8_onnx_path = (
        int8_directory / "berturk_int8.onnx"
    )

    report_path = (
        reports_directory
        / "metrics"
        / "berturk_quantization.json"
    )

    provider = str(
        config["quantization"]["execution_provider"]
    )
    max_length = int(config["transformer"]["max_length"])

    print("Loading the fine-tuned BERTurk model...")

    classifier = BERTurkClassifier.load(
        artifact_directory=berturk_directory,
        device="cpu",
    )
    classifier.model.eval()

    sample_text = (
        "Bugün arkadaşlarımla güzel vakit geçirdim."
    )
    encoded_inputs = classifier.tokenizer(
        sample_text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    if "token_type_ids" not in encoded_inputs:
        encoded_inputs["token_type_ids"] = torch.zeros_like(
            encoded_inputs["input_ids"]
        )

    sample_inputs = {
        name: tensor.cpu()
        for name, tensor in encoded_inputs.items()
        if name
        in {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }
    }

    with torch.inference_mode():
        pytorch_logits = (
            classifier.model(**sample_inputs)
            .logits.detach()
            .cpu()
            .numpy()
        )

    print("Exporting the FP32 model to ONNX...")
    export_started_at = perf_counter()

    export_sequence_classifier_to_onnx(
        model=classifier.model,
        sample_inputs=sample_inputs,
        output_path=fp32_onnx_path,
        opset_version=18,
    )

    export_seconds = perf_counter() - export_started_at

    print("Verifying PyTorch and FP32 ONNX outputs...")

    fp32_onnx_logits = run_onnx_logits(
        model_path=fp32_onnx_path,
        inputs=sample_inputs,
        provider=provider,
    )

    np.testing.assert_allclose(
        pytorch_logits,
        fp32_onnx_logits,
        rtol=1e-4,
        atol=1e-4,
    )

    print("Applying dynamic INT8 quantization...")
    quantization_started_at = perf_counter()

    quantize_onnx_dynamic_int8(
        source_path=fp32_onnx_path,
        output_path=int8_onnx_path,
    )

    quantization_seconds = (
        perf_counter() - quantization_started_at
    )

    print("Verifying the INT8 ONNX model...")

    int8_logits = run_onnx_logits(
        model_path=int8_onnx_path,
        inputs=sample_inputs,
        provider=provider,
    )

    pytorch_label = int(np.argmax(pytorch_logits, axis=-1)[0])
    fp32_onnx_label = int(
        np.argmax(fp32_onnx_logits, axis=-1)[0]
    )
    int8_label = int(np.argmax(int8_logits, axis=-1)[0])

    fp32_max_difference = float(
        np.max(
            np.abs(pytorch_logits - fp32_onnx_logits)
        )
    )
    int8_max_difference = float(
        np.max(
            np.abs(pytorch_logits - int8_logits)
        )
    )

    if not (
        pytorch_label == fp32_onnx_label == int8_label
    ):
        raise RuntimeError(
            "The exported models disagree on the smoke-test label."
        )

    int8_directory.mkdir(parents=True, exist_ok=True)
    classifier.tokenizer.save_pretrained(int8_directory)

    source_size_mb = calculate_directory_size_mb(
        berturk_directory
    )
    fp32_size_mb = calculate_file_size_mb(
        fp32_onnx_path
    )
    int8_size_mb = calculate_file_size_mb(
        int8_onnx_path
    )

    reduction_percent = (
        (fp32_size_mb - int8_size_mb)
        / fp32_size_mb
        * 100.0
    )
    compression_ratio = fp32_size_mb / int8_size_mb

    metadata = {
        "model_name": "BERTurk Dynamic INT8 ONNX",
        "source_model": "BERTurk Fine-Tuned",
        "onnx_file": int8_onnx_path.name,
        "max_length": max_length,
        "labels": {
            "0": "clean",
            "1": "offensive",
        },
        "quantization": {
            "method": "dynamic_int8",
            "weight_type": "QInt8",
            "per_channel": False,
            "reduce_range": False,
        },
        "execution_provider": provider,
        "input_names": [
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ],
        "output_name": "logits",
    }

    metadata_path = int8_directory / "metadata.json"
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = {
        "source_model": "BERTurk Fine-Tuned",
        "quantized_model": "BERTurk Dynamic INT8 ONNX",
        "source_model_size_mb": round(source_size_mb, 6),
        "fp32_onnx_size_mb": round(fp32_size_mb, 6),
        "int8_onnx_size_mb": round(int8_size_mb, 6),
        "size_reduction_percent": round(
            reduction_percent,
            6,
        ),
        "compression_ratio": round(
            compression_ratio,
            6,
        ),
        "export_seconds": round(export_seconds, 6),
        "quantization_seconds": round(
            quantization_seconds,
            6,
        ),
        "verification": {
            "sample_text": sample_text,
            "pytorch_label": pytorch_label,
            "fp32_onnx_label": fp32_onnx_label,
            "int8_label": int8_label,
            "fp32_max_absolute_logit_difference": round(
                fp32_max_difference,
                8,
            ),
            "int8_max_absolute_logit_difference": round(
                int8_max_difference,
                8,
            ),
            "labels_match": True,
        },
        "artifacts": {
            "fp32_onnx": str(fp32_onnx_path),
            "int8_onnx": str(int8_onnx_path),
            "metadata": str(metadata_path),
        },
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("BERTurk ONNX export and quantization completed.")
    print(f"FP32 ONNX: {fp32_onnx_path}")
    print(f"INT8 ONNX: {int8_onnx_path}")
    print(f"Report: {report_path}")
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()