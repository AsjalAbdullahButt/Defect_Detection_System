"""V4 end to end on a tiny run: one-shot test evaluation, its guards, Grad-CAM, PatchCore."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.figure import Figure
from PIL import Image
from typer.testing import CliRunner

from defect_detection.cli import app
from defect_detection.config import ProjectConfig
from defect_detection.training import error_analysis as ea
from defect_detection.training.anomaly_baseline import (
    RESULT_FILE,
    PatchCore,
    greedy_coreset,
    run_anomaly_baseline,
)
from defect_detection.training.one_shot_test import (
    MARKER,
    AlreadyEvaluatedError,
    evaluate_once,
)
from defect_detection.training.trainer import load_trained_model
from tests.conftest import make_config
from tests.fixtures.synthetic import build_planted_dataset, smooth_image


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> tuple[ProjectConfig, Path, Path]:
    """manifest -> split -> audit -> train -> calibrate through the CLI, in a temp dir."""
    root = tmp_path_factory.mktemp("v4")
    config = make_config(root)
    build_planted_dataset(config.data.raw_dir)
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        runner = CliRunner()
        for command in ("manifest", "split", "audit", "train", "calibrate"):
            result = runner.invoke(app, [command, "--config", str(config_file)])
            assert result.exit_code == 0, f"{command}: {result.output}"
    latest = (config.train.output_dir / "LATEST").read_text(encoding="utf-8")
    return config, config.train.output_dir / latest, config_file


@pytest.fixture(scope="module")
def evaluated(project: tuple[ProjectConfig, Path, Path]) -> dict:
    config, run_dir, _ = project
    return evaluate_once(config, run_dir)


def test_evaluation_outputs(evaluated: dict, project: tuple) -> None:
    _, run_dir, _ = project
    for name in (
        "test_metrics.json",
        "test_predictions.csv",
        "test_curves.png",
        "test_confusion.png",
        MARKER,
    ):
        assert (run_dir / name).exists(), name
    assert set(evaluated["results"]) == {"test", "external_test"}
    test = evaluated["results"]["test"]
    ci = test["at_operating_threshold"]["f1"]
    assert ci["low"] <= ci["point"] <= ci["high"]
    assert sum(b["n"] for b in test["by_similarity_to_train"]) == test["n"]
    cm = test["confusion"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == test["n"]
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    assert set(predictions["split"]) == {"test", "external_test"}


def test_second_evaluation_is_refused(evaluated: dict, project: tuple) -> None:
    config, run_dir, config_file = project
    with pytest.raises(AlreadyEvaluatedError, match="used once"):
        evaluate_once(config, run_dir)
    result = CliRunner().invoke(
        app, ["evaluate", "--config", str(config_file), "--run", str(run_dir)]
    )
    assert result.exit_code == 1


def test_evaluation_refuses_a_changed_test_set(project: tuple, tmp_path: Path) -> None:
    config, run_dir, _ = project
    copy = tmp_path / "run_copy"
    copy.mkdir()
    for f in ("best.pt", "model_meta.candidate.json"):
        (copy / f).write_bytes((run_dir / f).read_bytes())
    audit_path = config.data.processed_dir / "leakage_audit.json"
    original = audit_path.read_text(encoding="utf-8")
    try:
        tampered = json.loads(original)
        tampered["test_set_sha256"] = "0" * 64
        audit_path.write_text(json.dumps(tampered), encoding="utf-8")
        with pytest.raises(RuntimeError, match="differs from the audited one"):
            evaluate_once(config, copy)
    finally:
        audit_path.write_text(original, encoding="utf-8")


def test_grad_cam_and_galleries(evaluated: dict, project: tuple) -> None:
    config, run_dir, _ = project
    trained = load_trained_model(run_dir, __import__("torch").device("cpu"))
    cam = ea.grad_cam(trained.model, smooth_image(1, size=100), trained.spec)
    assert cam.shape == (trained.spec.image_size, trained.spec.image_size)
    assert cam.min() >= 0.0
    assert cam.max() <= 1.0
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    fig = ea.gallery(predictions.head(3), config.data.raw_dir, trained.model, trained.spec, "x")
    assert isinstance(fig, Figure)
    assert isinstance(ea.confidence_histograms(predictions, 0.5, (0.4, 0.6)), Figure)
    errors = ea.error_table(predictions)
    assert (errors["error_type"].str.startswith(("FN", "FP"))).all()


@pytest.mark.parametrize("seed", range(5))
def test_greedy_coreset_is_a_2_approximation(seed: int) -> None:
    """Greedy k-center covers within 2x the optimal radius (0.1 here: centres 0.1, 5, 10)."""
    points = np.array([[0.0], [0.1], [0.2], [5.0], [10.0]], dtype=np.float32)
    chosen = greedy_coreset(points, size=3, projection_dim=1, seed=seed)
    radius = np.abs(points - chosen.T).min(axis=1).max()
    assert len(chosen) == 3
    assert radius <= 2 * 0.1 + 1e-6


def test_patchcore_scores_outliers_higher(project: tuple) -> None:
    config, _, _ = project
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from defect_detection.core.preprocessing import PreprocessSpec, preprocess

    spec = PreprocessSpec(64, config.preprocess.mean, config.preprocess.std)
    normal = torch.stack([torch.from_numpy(preprocess(smooth_image(s), spec)) for s in range(8)])
    weird = torch.from_numpy(preprocess(Image.new("RGB", (64, 64), (255, 0, 0)), spec))[None]
    model = PatchCore(config.anomaly, seed=0, pretrained=False)
    model.fit(DataLoader(TensorDataset(normal, torch.zeros(8, dtype=torch.long)), batch_size=4))
    queries = torch.cat([normal[:2], weird])
    _, scores, ms = model.score(
        DataLoader(TensorDataset(queries, torch.tensor([0, 0, 1])), batch_size=3)
    )
    assert scores[2] > scores[:2].max()
    assert ms > 0


def test_anomaly_baseline_runs_once(project: tuple, tmp_path: Path) -> None:
    config, _, _ = project
    results = run_anomaly_baseline(config, tmp_path)
    assert {"fit", "threshold", "val", "test", "external_test"} <= results.keys()
    assert (tmp_path / RESULT_FILE).exists()
    with pytest.raises(RuntimeError, match="already evaluated"):
        run_anomaly_baseline(config, tmp_path)
