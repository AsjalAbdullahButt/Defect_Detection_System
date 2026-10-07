"""Helpers for serving tests: a real ONNX model exported by the production CLI pipeline."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from defect_detection.cli import app as cli
from defect_detection.config import ProjectConfig
from defect_detection.serving.settings import Settings
from tests.fixtures.synthetic import build_planted_dataset


@dataclass(frozen=True)
class ExportedModel:
    """Paths of a fully built synthetic project."""

    config: ProjectConfig
    run_dir: Path
    model_dir: Path


def build_exported_model(config: ProjectConfig, root: Path) -> ExportedModel:
    """manifest -> split -> audit -> train -> calibrate -> export on the planted dataset."""
    build_planted_dataset(config.data.raw_dir)
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(root)
        for command in ("manifest", "split", "audit", "train", "calibrate", "export"):
            result = CliRunner().invoke(cli, [command, "--config", str(config_file)])
            assert result.exit_code == 0, f"{command}: {result.output}"
    run_name = (config.train.output_dir / "LATEST").read_text(encoding="utf-8")
    return ExportedModel(
        config, config.train.output_dir / run_name, config.export.models_dir / run_name
    )


def make_settings(model_dir: Path, **overrides: object) -> Settings:
    """Settings for tests: no metrics port, development mode unless overridden."""
    values: dict[str, object] = {
        "model_dir": model_dir,
        "metrics_port": None,
        "environment": "development",
        "min_image_side": 16,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]
