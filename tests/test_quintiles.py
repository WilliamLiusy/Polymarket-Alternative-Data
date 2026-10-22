from pathlib import Path

import pytest

from baseline.quintiles import resolve_paths


def completed_run(root, name):
    run = root / name
    run.mkdir(parents=True)
    (run / "summary.json").write_text("{}")
    (run / "predictions.parquet").touch()
    return run


def test_default_paths_select_latest_completed_timestamp_run(tmp_path):
    completed_run(tmp_path, "20261019T122626585000")
    latest = completed_run(tmp_path, "20261022T120000000000")
    completed_run(tmp_path, "comparison")
    incomplete = tmp_path / "20261023T120000000000"
    incomplete.mkdir()
    (incomplete / "predictions.parquet").touch()
    no_predictions = tmp_path / "20261024T120000000000"
    no_predictions.mkdir()
    (no_predictions / "summary.json").write_text("{}")
    (tmp_path / "predictions.parquet").touch()

    assert resolve_paths(output_root=tmp_path) == (
        latest / "predictions.parquet", latest / "quintiles",
    )


def test_default_paths_fall_back_to_legacy_predictions(tmp_path):
    predictions = tmp_path / "predictions.parquet"
    predictions.touch()
    incomplete = tmp_path / "20261022T120000000000"
    incomplete.mkdir()
    (incomplete / "predictions.parquet").touch()

    assert resolve_paths(output_root=tmp_path) == (predictions, tmp_path / "quintiles")


@pytest.mark.parametrize("root_exists", [True, False])
def test_default_paths_explain_missing_completed_run(tmp_path, root_exists):
    root = tmp_path if root_exists else tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="pass --predictions explicitly"):
        resolve_paths(output_root=root)


def test_explicit_predictions_default_to_analysis_beside_predictions(tmp_path):
    predictions = Path("custom/run/predictions.parquet")
    assert resolve_paths(predictions, output_root=tmp_path) == (
        predictions, predictions.parent / "quintiles",
    )


def test_explicit_paths_are_preserved(tmp_path):
    predictions, output_dir = Path("custom/predictions.parquet"), Path("custom/analysis")
    assert resolve_paths(predictions, output_dir, output_root=tmp_path) == (predictions, output_dir)


def test_explicit_output_with_default_predictions(tmp_path):
    latest = completed_run(tmp_path, "20261022T120000000000")
    output_dir = tmp_path / "analysis"
    assert resolve_paths(output_dir=output_dir, output_root=tmp_path) == (
        latest / "predictions.parquet", output_dir,
    )
