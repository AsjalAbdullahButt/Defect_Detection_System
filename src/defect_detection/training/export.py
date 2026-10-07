"""V5: export a calibrated run to a versioned, verified ONNX artifact.

Output (immutable once written): ``models/<model_version>/``

* ``model.onnx``: logits for (normal, defective), dynamic batch axis;
* ``model_meta.json``: the V3 candidate meta + ONNX description + V4 test-metric summary;
* ``SHA256SUMS``: checksums of both files, verified by serving at startup.

The graph outputs raw logits; temperature and threshold stay in the metadata, so recalibrating
never requires re-exporting the network. Export fails (and nothing is kept) if
``onnx.checker`` rejects the graph or PyTorch and ONNX Runtime logits differ by more than
``parity_atol`` on validation images.
"""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch

from defect_detection.config import ProjectConfig
from defect_detection.core.hashing import write_sha256sums
from defect_detection.core.model_meta import ModelMeta, OnnxSpec
from defect_detection.data.dataset import SplitDataset
from defect_detection.training.operating_point import CANDIDATE_META
from defect_detection.training.trainer import TrainedModel, load_trained_model, resolve_device

MODEL_FILE = "model.onnx"
META_FILE = "model_meta.json"
INPUT_NAME, OUTPUT_NAME = "input", "logits"
PARITY_BATCH_SIZES = (1, 7, 32)  # 7: an odd size the tracer never saw, exercises the dynamic axis


def onnx_session(path: Path, threads: int) -> ort.InferenceSession:
    """CPU ONNX Runtime session with a fixed intra-op thread count and full graph optimisation."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def validation_batch(
    config: ProjectConfig, trained: TrainedModel, n: int
) -> npt.NDArray[np.float32]:
    """The first ``n`` VAL images through the shared (serving) preprocessing, stacked NCHW."""
    val = SplitDataset(
        config.data.processed_dir / "splits.csv", config.data.raw_dir, "val", trained.spec
    )
    return np.stack([val[i][0].numpy() for i in range(min(n, len(val)))])


@torch.no_grad()
def parity(
    trained: TrainedModel, session: ort.InferenceSession, images: npt.NDArray[np.float32]
) -> dict[str, Any]:
    """Max |PyTorch logits - ONNX logits| at several batch sizes."""
    trained.model.eval()
    worst = 0.0
    for batch in PARITY_BATCH_SIZES:
        chunk = images[:batch]
        reference = trained.model(torch.from_numpy(chunk)).numpy()
        exported = session.run([OUTPUT_NAME], {INPUT_NAME: chunk})[0]
        worst = max(worst, float(np.abs(reference - exported).max()))
    return {
        "max_abs_logit_diff": worst,
        "images": len(images),
        "batch_sizes": list(PARITY_BATCH_SIZES),
    }


def export_run(config: ProjectConfig, run_dir: Path) -> dict[str, Any]:
    """Export ``run_dir`` to ``models/<model_version>/``; returns a summary."""
    candidate = ModelMeta.model_validate_json(
        (run_dir / CANDIDATE_META).read_text(encoding="utf-8")
    )
    out = config.export.models_dir / candidate.model_version
    if out.exists():
        raise FileExistsError(f"{out} exists; model versions are immutable")
    trained = load_trained_model(run_dir, resolve_device("cpu"))
    trained.model.eval()
    size = trained.spec.image_size
    out.mkdir(parents=True)
    try:
        torch.onnx.export(
            trained.model,
            (torch.zeros(1, 3, size, size),),
            str(out / MODEL_FILE),
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            dynamic_axes={INPUT_NAME: {0: "batch"}, OUTPUT_NAME: {0: "batch"}},
            opset_version=config.export.opset,
            do_constant_folding=True,
            dynamo=False,  # TorchScript exporter: mature for CNNs, no onnxscript dependency
        )
        onnx.checker.check_model(onnx.load(str(out / MODEL_FILE)), full_check=True)
        images = validation_batch(config, trained, config.export.parity_images)
        check = parity(trained, onnx_session(out / MODEL_FILE, threads=1), images)
        if check["max_abs_logit_diff"] > config.export.parity_atol:
            raise RuntimeError(f"PyTorch/ONNX parity failed: {check}")
    except Exception:
        shutil.rmtree(out)  # never leave a half-written model version behind
        raise

    test_metrics = None
    metrics_path = run_dir / "test_metrics.json"
    if metrics_path.exists():
        results = json.loads(metrics_path.read_text(encoding="utf-8"))["results"]
        test_metrics = {
            split: {
                name: ci["point"]
                for name, ci in r["at_operating_threshold"].items()
                if name != "_meta"
            }
            for split, r in results.items()
        }
    meta = candidate.model_copy(
        update={
            "onnx": OnnxSpec(
                file=MODEL_FILE,
                opset=config.export.opset,
                input_name=INPUT_NAME,
                output_name=OUTPUT_NAME,
            ),
            "test_metrics": test_metrics,
        }
    )
    ModelMeta.model_validate(meta.model_dump())  # re-run validators on the final object
    (out / META_FILE).write_text(meta.model_dump_json(indent=2), encoding="utf-8")
    write_sha256sums(out, [MODEL_FILE, META_FILE])
    return {
        "model_dir": out.as_posix(),
        "model_version": meta.model_version,
        "onnx_bytes": (out / MODEL_FILE).stat().st_size,
        "parity": check,
        "includes_test_metrics": test_metrics is not None,
    }
