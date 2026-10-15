import json

import numpy as np
import pandas as pd
import pytest

from baseline.train import assemble_month, label_cutoff, monthly_windows, summarize_ic, _sha256, _verified_month


def settings():
    return {
        "train_start": "2025-01-01",
        "first_test_month": "2026-04-01",
        "research_end": "2026-09-18",
    }


def test_monthly_training_expands_from_the_same_start():
    calendar = pd.bdate_range("2025-01-01", "2026-10-01")
    windows = monthly_windows(settings(), calendar)
    assert len(windows) == 6
    assert {window["train_start"] for window in windows} == {"2025-01-01"}
    assert windows[0]["label_available_end"] == "2026-03-31"
    assert windows[0]["train_end"] == "2026-03-27"
    assert windows[1]["label_available_end"] == "2026-04-30"
    assert windows[1]["train_end"] == "2026-04-28"
    assert windows[3]["label_available_end"] == "2026-06-30"
    assert windows[3]["train_end"] == "2026-06-26"
    assert windows[-1]["label_available_end"] == "2026-08-31"
    assert windows[-1]["test_end"] == "2026-09-18"
    assert windows[-1]["evaluation_end"] == "2026-09-16"


def test_purge_uses_market_calendar_including_holidays():
    calendar = pd.DatetimeIndex(["2026-04-27", "2026-04-28", "2026-04-29", "2026-04-30", "2026-05-06"])
    assert label_cutoff(calendar, "2026-05-05") == pd.Timestamp("2026-04-28")
    assert label_cutoff(calendar, "2026-05-06") == pd.Timestamp("2026-04-29")


def make_cache(directory):
    columns = pd.MultiIndex.from_tuples([("feature", "X"), ("label", "LABEL0"), ("raw_label", "LABEL0")])
    for year in (2025, 2026):
        dates = pd.bdate_range(f"{year}-01-01", f"{year}-12-31")
        index = pd.MultiIndex.from_product([dates, ["SH600000", "SZ000001"]], names=["datetime", "instrument"])
        values = np.ones((len(index), 3), dtype=np.float32)
        values[:, 1] = np.tile([-1, 1], len(dates))
        frame = pd.DataFrame(values, index=index, columns=columns)
        missing_date = pd.Timestamp(f"{year}-03-02") + pd.offsets.BDay(0)
        frame.loc[(missing_date, "SH600000"), ("label", "LABEL0")] = np.nan
        frame.to_parquet(directory / f"{year}.parquet")


def test_training_retains_all_prior_months_and_only_purges_current_boundary(tmp_path):
    make_cache(tmp_path)
    calendar = pd.bdate_range("2025-01-01", "2026-10-01")
    windows = monthly_windows(settings(), calendar)
    train_indices = []
    for position in (0, 1, 3):
        window = windows[position]
        frame, raw = assemble_month(tmp_path, window)
        dates = frame.index.get_level_values("datetime")
        train_indices.append(frame.index[dates <= window["train_end"]])
        assert (pd.Timestamp("2026-03-02"), "SH600000") not in frame.index
        assert raw.index.get_level_values("datetime").min() == pd.Timestamp(window["test_start"])
        assert set(frame.columns.get_level_values(0)) == {"feature", "label"}

    april, may, july = train_indices
    assert set(april) < set(may) < set(july)
    for date in ("2025-12-30", "2025-12-31"):
        assert (pd.Timestamp(date), "SH600000") in april
    for date in ("2026-01-15", "2026-02-16", "2026-03-16"):
        assert (pd.Timestamp(date), "SH600000") in july
    for date in ("2026-03-30", "2026-03-31"):
        row = (pd.Timestamp(date), "SH600000")
        assert row not in april
        assert row in may
    for date in ("2026-04-29", "2026-04-30"):
        row = (pd.Timestamp(date), "SH600000")
        assert row not in may
        assert row in july

    # Every valid historical row belongs to July training, including Jan–Mar.
    cached = pd.concat([pd.read_parquet(tmp_path / f"{year}.parquet") for year in (2025, 2026)])
    expected = cached.loc[:"2026-06-26"].dropna(subset=[("label", "LABEL0")]).index
    pd.testing.assert_index_equal(july, expected)


def test_last_month_keeps_predictions_but_masks_labels_beyond_project_end(tmp_path):
    make_cache(tmp_path)
    calendar = pd.bdate_range("2025-01-01", "2026-10-01")
    window = monthly_windows(settings(), calendar)[-1]
    frame, raw = assemble_month(tmp_path, window)
    late = frame.index.get_level_values("datetime") >= "2026-09-17"
    assert late.sum() == 4
    assert frame.loc[late, ("label", "LABEL0")].isna().all()
    assert raw.loc[raw.index.get_level_values("datetime") >= "2026-09-17"].isna().all()
    assert frame.index.get_level_values("datetime").max() == pd.Timestamp("2026-09-18")


def test_overall_ic_weights_days_not_months():
    daily = pd.DataFrame({"ic": [1.0, 0.0, 0.0, np.nan], "rank_ic": [1.0, 0.0, 0.0, np.nan], "n": [3, 3, 3, 0]})
    summary = summarize_ic(daily)
    assert summary["ic_mean"] == pytest.approx(1 / 3)
    assert summary["evaluated_days"] == 3
    assert summary["prediction_days"] == 4


def test_resume_checks_fingerprint_and_artifact_hash(tmp_path):
    artifact = tmp_path / "model.txt"
    artifact.write_text("model")
    (tmp_path / "completed.json").write_text(json.dumps({"fingerprint": "abc", "artifacts": {"model.txt": _sha256(artifact)}}))
    assert _verified_month(tmp_path, "abc")
    with pytest.raises(ValueError, match="different inputs"):
        _verified_month(tmp_path, "def")
    artifact.write_text("changed")
    with pytest.raises(ValueError, match="modified result"):
        _verified_month(tmp_path, "abc")
