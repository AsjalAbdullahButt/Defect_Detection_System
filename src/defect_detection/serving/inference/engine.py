"""ONNX Runtime adapter behind a small ``Predictor`` protocol.

The service layer depends only on the protocol (batch in, logits out), so the runtime can be
swapped (OpenVINO, TensorRT) or replaced by a fake in tests without touching business logic.
Loading verifies the artifact before anything is trusted: SHA256SUMS first, then the
metadata schema, then that the graph's input/output match the metadata.
"""

from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from defect_detection.core.hashing import verify_sha256sums
from defect_detection.core.model_meta import ModelMeta

META_FILE = "model_meta.json"


class ModelIntegrityError(RuntimeError):
    """The model directory failed verification; the server must not start."""


class Predictor(Protocol):
    """Anything that maps a preprocessed NCHW float32 batch to (batch, 2) logits."""

    def predict_logits(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Raw logits for (normal, defective)."""
        ...


def load_verified_meta(model_dir: Path) -> ModelMeta:
    """Checksums, then schema; raises ModelIntegrityError on any problem."""
    problems = verify_sha256sums(model_dir)
    if problems:
        raise ModelIntegrityError(f"model directory failed verification: {problems}")
    try:
        meta = ModelMeta.model_validate_json((model_dir / META_FILE).read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ModelIntegrityError(f"{META_FILE} is invalid: {exc}") from exc
    if meta.onnx is None:
        raise ModelIntegrityError(f"{META_FILE} has no 'onnx' section; export the model first")
    return meta


class OnnxPredictor:
    """CPU ONNX Runtime session configured from the metadata."""

    def __init__(self, model_dir: Path, meta: ModelMeta, threads: int) -> None:
        if meta.onnx is None:
            raise ModelIntegrityError("metadata has no 'onnx' section")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_dir / meta.onnx.file), options, providers=["CPUExecutionProvider"]
        )
        self._input = meta.onnx.input_name
        self._output = meta.onnx.output_name
        inputs = {i.name: i.shape for i in self._session.get_inputs()}
        size = meta.input.image_size
        if inputs.get(self._input, [None])[1:] != [3, size, size]:
            raise ModelIntegrityError(f"graph input {inputs} does not match metadata")

    def predict_logits(self, batch: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """Run the graph on a preprocessed batch."""
        outputs = self._session.run([self._output], {self._input: batch})
        logits: npt.NDArray[np.float32] = np.asarray(outputs[0], dtype=np.float32)
        return logits
