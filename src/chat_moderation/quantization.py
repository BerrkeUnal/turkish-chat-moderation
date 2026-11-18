from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn


class _SequenceClassifierLogitsWrapper(nn.Module):
    """Expose only classification logits during ONNX export."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return outputs.logits


def export_sequence_classifier_to_onnx(
    model: nn.Module,
    sample_inputs: Mapping[str, torch.Tensor],
    output_path: Path,
    opset_version: int = 18,
) -> None:
    """Export a fixed-shape sequence classifier to ONNX."""

    required_inputs = {
        "input_ids",
        "attention_mask",
        "token_type_ids",
    }
    missing_inputs = required_inputs.difference(sample_inputs)

    if missing_inputs:
        raise ValueError(
            f"Missing ONNX sample inputs: {sorted(missing_inputs)}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.to("cpu")
    model.eval()

    wrapper = _SequenceClassifierLogitsWrapper(model)
    wrapper.eval()

    export_arguments = (
        sample_inputs["input_ids"].cpu(),
        sample_inputs["attention_mask"].cpu(),
        sample_inputs["token_type_ids"].cpu(),
    )

    torch.onnx.export(
        model=wrapper,
        args=export_arguments,
        f=str(output_path),
        input_names=[
            "input_ids",
            "attention_mask",
            "token_type_ids",
        ],
        output_names=["logits"],
        dynamic_shapes=(
            {1: "sequence_length"},
            {1: "sequence_length"},
            {1: "sequence_length"},
        ),
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=True,
    )

    validate_onnx_model(output_path)


def _remove_initializer_value_info(
    model_path: Path,
) -> None:
    """Remove stale initializer shapes before dynamic quantization."""

    model = onnx.load(
        str(model_path),
        load_external_data=False,
    )
    initializer_names = {
        initializer.name
        for initializer in model.graph.initializer
    }
    retained_value_info = [
        value_info
        for value_info in model.graph.value_info
        if value_info.name not in initializer_names
    ]

    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    onnx.save(model, str(model_path))


def quantize_onnx_dynamic_int8(
    source_path: Path,
    output_path: Path,
) -> None:
    """Apply weight-only dynamic INT8 quantization."""

    if not source_path.exists():
        raise FileNotFoundError(
            f"ONNX source model not found: {source_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Dynamo export can retain stale initializer shape metadata.
    _remove_initializer_value_info(source_path)
    validate_onnx_model(source_path)

    quantize_dynamic(
        model_input=source_path,
        model_output=output_path,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=False,
        reduce_range=False,
        weight_type=QuantType.QInt8,
        use_external_data_format=False,
        extra_options={
            "MatMulConstBOnly": True,
        },
    )

    validate_onnx_model(output_path)


def validate_onnx_model(model_path: Path) -> None:
    """Validate an ONNX graph using the official checker."""

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found: {model_path}"
        )

    onnx.checker.check_model(str(model_path))


def create_onnx_session(
    model_path: Path,
    provider: str = "CPUExecutionProvider",
) -> ort.InferenceSession:
    """Create an optimized ONNX Runtime inference session."""

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=[provider],
    )


def run_onnx_logits(
    model_path: Path,
    inputs: Mapping[str, torch.Tensor],
    provider: str = "CPUExecutionProvider",
) -> np.ndarray:
    """Run one ONNX inference and return its logits."""

    session = create_onnx_session(
        model_path=model_path,
        provider=provider,
    )
    expected_input_names = {
        input_info.name for input_info in session.get_inputs()
    }
    missing_inputs = expected_input_names.difference(inputs)

    if missing_inputs:
        raise ValueError(
            f"Missing ONNX Runtime inputs: {sorted(missing_inputs)}"
        )

    runtime_inputs = {
        name: inputs[name]
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
        for name in expected_input_names
    }

    outputs = session.run(["logits"], runtime_inputs)
    return np.asarray(outputs[0])


def calculate_file_size_mb(file_path: Path) -> float:
    """Return a file size in binary megabytes."""

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Model file not found: {file_path}"
        )

    return file_path.stat().st_size / (1024**2)