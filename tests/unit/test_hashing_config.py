from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from defect_detection.config import ProjectConfig, config_hash, load_config
from defect_detection.core.hashing import sha256_bytes, sha256_file, sha256_of_hashes

REPO_CONFIG = Path("configs/train.yaml")


def test_sha256_file_matches_bytes(tmp_path: Path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"x" * 3_000_000)  # spans several read chunks
    assert sha256_file(path) == sha256_bytes(b"x" * 3_000_000)


def test_sha256_of_hashes_is_order_independent() -> None:
    assert sha256_of_hashes(["b", "a"]) == sha256_of_hashes(["a", "b"])
    assert sha256_of_hashes(["a"]) != sha256_of_hashes(["a", "b"])


def test_repo_config_is_valid() -> None:
    config = load_config(REPO_CONFIG)
    assert config.seed == 42
    assert config.dedupe.similarity_threshold == 0.99
    assert config.data.external_test_dirs == ("casting_512x512",)


def _raw() -> dict:
    return yaml.safe_load(REPO_CONFIG.read_text(encoding="utf-8"))


def test_config_rejects_unknown_keys() -> None:
    raw = _raw()
    raw["split"]["tets"] = 0.1
    with pytest.raises(ValidationError):
        ProjectConfig.model_validate(raw)


def test_config_rejects_ratios_not_summing_to_one() -> None:
    raw = _raw()
    raw["split"]["test"] = 0.3
    with pytest.raises(ValidationError, match="sum to 1"):
        ProjectConfig.model_validate(raw)


def test_config_rejects_alias_to_unknown_class() -> None:
    raw = _raw()
    raw["data"]["class_aliases"]["weird"] = "scratched"
    with pytest.raises(ValidationError, match="unknown classes"):
        ProjectConfig.model_validate(raw)


def test_config_hash_ignores_key_order() -> None:
    raw = _raw()
    reordered = dict(reversed(list(raw.items())))
    assert config_hash(ProjectConfig.model_validate(raw)) == config_hash(
        ProjectConfig.model_validate(reordered)
    )
    raw["seed"] = 7
    assert config_hash(ProjectConfig.model_validate(raw)) != config_hash(load_config(REPO_CONFIG))
