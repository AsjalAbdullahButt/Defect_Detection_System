import pytest
from matplotlib.figure import Figure

from defect_detection.data import eda
from tests.conftest import PipelineRun


@pytest.fixture(scope="module")
def processed(official_run: PipelineRun) -> PipelineRun:
    out = official_run.config.data.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    official_run.manifest.to_csv(out / "manifest.csv", index=False)
    official_run.splits.to_csv(out / "splits.csv", index=False)
    return official_run


def test_eda_frame_never_contains_test(processed: PipelineRun) -> None:
    frame = eda.load_eda_frame(processed.config.data.processed_dir)
    assert set(frame["split"]) == {"train", "val"}
    assert {"width", "height"} <= set(frame.columns)


def test_split_counts_include_test_counts_only(processed: PipelineRun) -> None:
    counts = eda.split_counts(processed.config.data.processed_dir)
    assert "test" in counts.index
    assert "excluded:exact_duplicate" in counts.index


def test_lighting_shortcut_table(processed: PipelineRun) -> None:
    frame = eda.load_eda_frame(processed.config.data.processed_dir)
    stats = eda.image_stats(frame, processed.config.data.raw_dir)
    table = eda.lighting_shortcut(stats)
    assert list(table.index) == ["brightness", "contrast"]
    assert table["separability_auc"].between(0.5, 1).all()
    assert isinstance(eda.feature_histograms(stats), Figure)


def test_figures_render(processed: PipelineRun) -> None:
    frame = eda.load_eda_frame(processed.config.data.processed_dir)
    raw = processed.config.data.raw_dir
    assert isinstance(eda.sample_grid(frame, raw, per_class=4), Figure)
    assert isinstance(eda.cluster_grid(frame, raw), Figure)
