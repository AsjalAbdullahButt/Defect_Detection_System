from pathlib import Path

from typer.testing import CliRunner

from defect_detection.cli import app
from defect_detection.data.inventory import (
    NO_SPLIT,
    detect_split,
    identify_dataset,
    scan_dataset,
    summarize,
)
from tests.fixtures.synthetic import write_image


def test_counts_per_split_and_class(casting_like_dataset: Path) -> None:
    inventory = scan_dataset(casting_like_dataset)
    assert inventory.counts() == {
        "test": {"def_front": 2, "ok_front": 1},
        "train": {"def_front": 3, "ok_front": 3},  # includes the truncated file
    }
    assert inventory.other_files == ["notes.txt"]


def test_truncated_image_is_flagged_corrupt(casting_like_dataset: Path) -> None:
    corrupt = scan_dataset(casting_like_dataset).corrupt
    assert [r.rel_path for r in corrupt] == ["train/ok_front/truncated.jpeg"]
    assert corrupt[0].error is not None


def test_metadata_of_valid_image(casting_like_dataset: Path) -> None:
    record = next(r for r in scan_dataset(casting_like_dataset).valid)
    assert (record.format, record.width, record.height, record.mode) == ("JPEG", 300, 300, "RGB")


def test_detect_split_aliases() -> None:
    assert detect_split(("Validation", "ok")) == "val"
    assert detect_split(("casting", "train", "ok")) == "train"
    assert detect_split(("ok",)) == NO_SPLIT


def test_extension_format_mismatch_reported(tmp_path: Path) -> None:
    write_image(tmp_path / "a" / "fake.jpg", fmt="PNG")
    summary = summarize(scan_dataset(tmp_path))
    assert summary["extension_format_mismatch"] == ["a/fake.jpg"]


def test_casting_signature_identified_with_medium_confidence(casting_like_dataset: Path) -> None:
    match = identify_dataset(scan_dataset(casting_like_dataset))
    assert match is not None
    assert "casting" in match.name
    assert match.confidence == "medium"  # counts differ from the full release


def test_unknown_dataset_not_identified(tmp_path: Path) -> None:
    write_image(tmp_path / "good" / "x.png", fmt="PNG")
    assert identify_dataset(scan_dataset(tmp_path)) is None


def test_cli_inspect_writes_json(casting_like_dataset: Path, tmp_path: Path) -> None:
    out = tmp_path / "inv.json"
    result = CliRunner().invoke(app, ["inspect", str(casting_like_dataset), "--json-out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Corrupt/unreadable: 1" in result.output
    assert out.exists()


def test_cli_inspect_empty_dir_fails(tmp_path: Path) -> None:
    result = CliRunner().invoke(app, ["inspect", str(tmp_path)])
    assert result.exit_code == 1
