"""V5: CPU latency/throughput, PyTorch vs ONNX Runtime, plus optional INT8 dynamic quantisation.

Fairness rules: same real VAL images, same thread count for both runtimes, warm-up runs
discarded, wall-clock time per call (``time.perf_counter``). Latency is reported as p50/p95/p99
over ``runs`` calls; throughput = batch size / median call time.

INT8 dynamic quantisation (weights to int8, activations quantised on the fly) is tried as an
experiment: it is written next to the run, never into the model registry, and its accuracy
change is measured on the full validation split before anyone could choose to ship it.
"""

import json
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from defect_detection.config import ProjectConfig
from defect_detection.core.model_meta import ModelMeta
from defect_detection.data.dataset import SplitDataset
from defect_detection.training.calibration import defect_probability
from defect_detection.training.evaluate import classification_metrics
from defect_detection.training.export import (
    INPUT_NAME,
    META_FILE,
    MODEL_FILE,
    OUTPUT_NAME,
    onnx_session,
    validation_batch,
)
from defect_detection.training.trainer import load_trained_model, resolve_device

INT8_FILE = "model.int8.onnx"


def time_calls(fn: Callable[[], object], warmup: int, runs: int) -> dict[str, float]:
    """p50/p95/p99/mean milliseconds of ``fn()`` after ``warmup`` discarded calls."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    p50, p95, p99 = np.percentile(samples, [50, 95, 99])
    return {
        "p50_ms": float(p50),
        "p95_ms": float(p95),
        "p99_ms": float(p99),
        "mean_ms": float(np.mean(samples)),
    }


def tile(images: npt.NDArray[np.float32], batch: int) -> npt.NDArray[np.float32]:
    """A batch of exactly ``batch`` real images (repeating if fewer are available)."""
    reps = -(-batch // len(images))
    return np.ascontiguousarray(np.concatenate([images] * reps)[:batch])


def run_benchmark(config: ProjectConfig, run_dir: Path, model_dir: Path) -> dict[str, Any]:
    """Benchmark the exported model in ``model_dir`` against its source checkpoint."""
    bc = config.benchmark
    torch.set_num_threads(bc.threads)
    meta = ModelMeta.model_validate_json((model_dir / META_FILE).read_text(encoding="utf-8"))
    device = resolve_device("cpu")
    trained = load_trained_model(run_dir, device)
    trained.model.eval()
    images = validation_batch(config, trained, max(bc.batch_sizes))

    int8_path = run_dir / INT8_FILE
    quantize_dynamic(str(model_dir / MODEL_FILE), str(int8_path), weight_type=QuantType.QInt8)
    sessions = {
        "onnxruntime_fp32": onnx_session(model_dir / MODEL_FILE, bc.threads),
        "onnxruntime_int8": onnx_session(int8_path, bc.threads),
    }

    results: dict[str, Any] = {
        "threads": bc.threads,
        "warmup": bc.warmup,
        "runs": bc.runs,
        "latency": {},
    }
    for batch in bc.batch_sizes:
        x = tile(images, batch)
        tensor = torch.from_numpy(x)

        def torch_call(t: torch.Tensor = tensor) -> object:
            with torch.no_grad():
                return trained.model(t)

        def onnx_call(session: ort.InferenceSession, inp: npt.NDArray[np.float32] = x) -> object:
            return session.run([OUTPUT_NAME], {INPUT_NAME: inp})

        timings = {"pytorch_fp32": time_calls(torch_call, bc.warmup, bc.runs)}
        for name, session in sessions.items():
            timings[name] = time_calls(partial(onnx_call, session), bc.warmup, bc.runs)
        for t in timings.values():
            t["images_per_s"] = batch / (t["p50_ms"] / 1000)
        results["latency"][f"batch_{batch}"] = timings

    # Accuracy of each runtime on the FULL validation split at the shipped operating point.
    # Val is streamed in chunks (decoded once, every runtime scores each chunk) to bound memory.
    val = SplitDataset(
        config.data.processed_dir / "splits.csv", config.data.raw_dir, "val", trained.spec
    )
    labels = np.array(val.labels, dtype=np.int64)
    logits: dict[str, list[npt.NDArray[np.float32]]] = {
        "pytorch_fp32": [],
        **{n: [] for n in sessions},
    }
    for start in range(0, len(val), 64):
        chunk = np.stack([val[i][0].numpy() for i in range(start, min(start + 64, len(val)))])
        with torch.no_grad():
            logits["pytorch_fp32"].append(trained.model(torch.from_numpy(chunk)).numpy())
        for name, session in sessions.items():
            logits[name].append(session.run([OUTPUT_NAME], {INPUT_NAME: chunk})[0])
    probs_by = {
        name: defect_probability(np.concatenate(parts).astype(np.float64), meta.temperature)
        for name, parts in logits.items()
    }
    reference = probs_by["pytorch_fp32"]
    accuracy: dict[str, Any] = {}
    for name, probs in probs_by.items():
        m = classification_metrics(labels, probs, meta.threshold)
        accuracy[name] = {
            **{k: m[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1")},
            "max_abs_prob_diff_vs_pytorch": float(np.abs(probs - reference).max()),
            "decisions_changed_vs_pytorch": int(
                ((probs >= meta.threshold) != (reference >= meta.threshold)).sum()
            ),
        }
    results["val_accuracy_at_operating_threshold"] = accuracy
    results["file_mb"] = {
        "onnx_fp32": round((model_dir / MODEL_FILE).stat().st_size / 1e6, 2),
        "onnx_int8": round(int8_path.stat().st_size / 1e6, 2),
    }
    results["model_version"] = meta.model_version
    return results


def render_markdown(results: dict[str, Any]) -> str:
    """Human-readable benchmark report."""
    lines = [
        "# CPU benchmark",
        "",
        f"Model `{results['model_version']}` · {results['threads']} threads · "
        f"{results['warmup']} warm-up + {results['runs']} timed calls per cell · "
        "real validation images.",
        "",
        "## Latency and throughput",
        "",
        "| Batch | Runtime | p50 ms | p95 ms | p99 ms | images/s |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for batch, timings in results["latency"].items():
        for runtime, t in timings.items():
            lines.append(
                f"| {batch.removeprefix('batch_')} | {runtime} | {t['p50_ms']:.2f} | "
                f"{t['p95_ms']:.2f} | {t['p99_ms']:.2f} | {t['images_per_s']:.1f} |"
            )
    lines += [
        "",
        "## Validation accuracy at the operating threshold",
        "",
        "| Runtime | PR-AUC | ROC-AUC | Precision | Recall | F1 | max Δp vs PyTorch "
        "| decisions changed |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for runtime, a in results["val_accuracy_at_operating_threshold"].items():
        lines.append(
            f"| {runtime} | {a['pr_auc']:.4f} | {a['roc_auc']:.4f} | {a['precision']:.4f} | "
            f"{a['recall']:.4f} | {a['f1']:.4f} | {a['max_abs_prob_diff_vs_pytorch']:.2e} | "
            f"{a['decisions_changed_vs_pytorch']} |"
        )
    lines += ["", f"File sizes (MB): {json.dumps(results['file_mb'])}", ""]
    return "\n".join(lines)
