import subprocess
import sys

import pytest
from pydantic import ValidationError

from defect_detection.core.model_meta import ModelMeta


def meta_dict(**overrides: object) -> dict:
    base = {
        "model_version": "20261007T000000Z_abc",
        "backbone": "efficientnet_b0",
        "input": {"image_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "temperature": 1.3,
        "threshold": 0.4,
        "review_band": {"low": 0.2, "high": 0.7},
        "threshold_policy": "test",
        "val_metrics": {"pr_auc": 0.99},
        "git_commit": "abc",
        "config_hash": "def",
        "created_at": "2026-10-07T00:00:00+00:00",
    }
    return {**base, **overrides}


def test_valid_meta_round_trips_through_json() -> None:
    meta = ModelMeta.model_validate(meta_dict())
    assert ModelMeta.model_validate_json(meta.model_dump_json()) == meta
    assert meta.class_names == ("normal", "defective")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"threshold": 0.9}, "inside the review band"),
        ({"class_names": ["ok", "bad"]}, "class names"),
        ({"temperature": 0}, "greater than 0"),
        ({"unexpected": 1}, "Extra inputs"),
        ({"schema_version": 2}, "schema_version"),
    ],
)
def test_invalid_meta_is_rejected(overrides: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ModelMeta.model_validate(meta_dict(**overrides))


def test_model_meta_is_torch_free() -> None:
    code = "import sys, defect_detection.core.model_meta; sys.exit('torch' in sys.modules)"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0  # noqa: S603
