"""V5: SHA256SUMS integrity, ONNX export + parity, immutability, CPU benchmark (tiny model)."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from defect_detection.cli import app
from defect_detection.config import ProjectConfig
from defect_detection.core.hashing import SHA256SUMS, verify_sha256sums, write_sha256sums
from defect_detection.core.model_meta import ModelMeta
from defect_detection.training.benchmark import render_markdown, run_benchmark, tile
from defect_detection.training.export import META_FILE, MODEL_FILE, export_run, onnx_session
from tests.conftest import make_config
from tests.fixtures.synthetic import build_planted_dataset


def test_sha256sums_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"weights")
    (tmp_path / "b.json").write_text("{}", encoding="utf-8")
    write_sha256sums(tmp_path, ["a.bin", "b.json"])
    assert verify_sha256sums(tmp_path) == []
    (tmp_path / "a.bin").write_bytes(b"weightz")
    assert verify_sha256sums(tmp_path) == ["a.bin checksum mismatch"]
    (tmp_path / "b.json").unlink()
    assert "b.json missing" in verify_sha256sums(tmp_path)


@pytest.mark.parametrize(
    "line",
    ["nonsense", f"{'0' * 64}  ../escape.onnx", f"{'0' * 64}  .hidden", f"{'0' * 63}  a.bin"],
)
def test_sha256sums_rejects_malformed_or_unsafe_lines(tmp_path: Path, line: str) -> None:
    (tmp_path / SHA256SUMS).write_text(line + "\n", encoding="utf-8")
    assert verify_sha256sums(tmp_path)[0].startswith("malformed line")


def test_missing_manifest_is_reported(tmp_path: Path) -> None:
    assert verify_sha256sums(tmp_path) == [f"{SHA256SUMS} missing"]


@pytest.fixture(scope="module")
def calibrated(tmp_path_factory: pytest.TempPathFactory) -> tuple[ProjectConfig, Path, Path]:
    root = tmp_path_factory.mktemp("v5")
    config = make_config(root)
    build_planted_dataset(config.data.raw_dir)
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        for command in ("manifest", "split", "audit", "train", "calibrate"):
            result = CliRunner().invoke(app, [command, "--config", str(config_file)])
            assert result.exit_code == 0, f"{command}: {result.output}"
    run_dir = config.train.output_dir / (config.train.output_dir / "LATEST").read_text(
        encoding="utf-8"
    )
    return config, run_dir, config_file


@pytest.fixture(scope="module")
def exported(calibrated: tuple) -> dict:
    config, run_dir, _ = calibrated
    return export_run(config, run_dir)


def test_export_writes_verified_artifact(exported: dict, calibrated: tuple) -> None:
    config, _, _ = calibrated
    model_dir = Path(exported["model_dir"])
    assert {p.name for p in model_dir.iterdir()} == {MODEL_FILE, META_FILE, SHA256SUMS}
    assert verify_sha256sums(model_dir) == []
    assert exported["parity"]["passed"]
    assert exported["parity"]["max_tolerance_used"] <= 1.0
    meta = ModelMeta.model_validate_json((model_dir / META_FILE).read_text(encoding="utf-8"))
    assert meta.onnx is not None
    assert meta.onnx.opset == config.export.opset
    assert meta.test_metrics is None  # V4 was not run in this fixture


def test_exported_graph_has_dynamic_batch(exported: dict, calibrated: tuple) -> None:
    config, _, _ = calibrated
    session = onnx_session(Path(exported["model_dir"]) / MODEL_FILE, threads=1)
    size = config.preprocess.image_size
    for batch in (1, 5):
        out = session.run(["logits"], {"input": np.zeros((batch, 3, size, size), np.float32)})[0]
        assert out.shape == (batch, 2)


def test_model_versions_are_immutable(exported: dict, calibrated: tuple) -> None:
    config, run_dir, _ = calibrated
    with pytest.raises(FileExistsError, match="immutable"):
        export_run(config, run_dir)


def test_benchmark_runs_and_renders(exported: dict, calibrated: tuple) -> None:
    config, run_dir, _ = calibrated
    small = config.model_copy(
        update={
            "benchmark": config.benchmark.model_copy(
                update={"warmup": 1, "runs": 10, "batch_sizes": (1, 4)}
            )
        }
    )
    results = run_benchmark(small, run_dir, Path(exported["model_dir"]))
    assert set(results["latency"]) == {"batch_1", "batch_4"}
    for timings in results["latency"].values():
        assert set(timings) == {"pytorch_fp32", "onnxruntime_fp32", "onnxruntime_int8"}
        assert all(t["p50_ms"] <= t["p99_ms"] for t in timings.values())
    # fp32 ONNX matches PyTorch to float rounding. A decision can still flip for an image whose
    # probability sits within that rounding of the threshold, which the untrained tiny model
    # does constantly, so parity is asserted on probabilities, not on decision counts.
    fp32 = results["val_accuracy_at_operating_threshold"]["onnxruntime_fp32"]
    assert fp32["max_abs_prob_diff_vs_pytorch"] < 1e-4
    assert "| 4 | onnxruntime_int8 |" in render_markdown(results)


def test_tile_repeats_real_images() -> None:
    images = np.arange(2 * 3 * 2 * 2, dtype=np.float32).reshape(2, 3, 2, 2)
    batch = tile(images, 5)
    assert batch.shape == (5, 3, 2, 2)
    assert np.array_equal(batch[2], images[0])


def test_cli_export_requires_calibration(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    (config.train.output_dir / "empty_run").mkdir(parents=True)
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    result = CliRunner().invoke(
        app,
        [
            "export",
            "--config",
            str(config_file),
            "--run",
            str(config.train.output_dir / "empty_run"),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, FileNotFoundError)
