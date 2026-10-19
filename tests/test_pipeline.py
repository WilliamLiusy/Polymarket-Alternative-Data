"""Offline integration: synthetic prices are test fixtures, never research data."""

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline.__main__ import load_config
from baseline.prepare import prepare


def make_raw_fixture(tmp_path):
    config = load_config("configs/baseline.yaml")
    for key in ("data_dir", "provider_dir", "cache_dir", "output_dir"):
        config[key] = str(tmp_path / key)
    config.update(research_start="2025-01-01", train_start="2025-01-01")
    config["model"].update(epochs=1, hidden_size=8, batch_size=128)
    root = Path(config["data_dir"])
    (root / "instruments").mkdir(parents=True)
    (root / "data").mkdir()
    calendar = pd.bdate_range("2024-09-02", "2026-09-30")
    (root / "calendar.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
    stocks = [f"SH60000{i}" for i in range(8)]
    for day in calendar[(calendar >= "2025-01-01") & (calendar <= "2026-09-30")]:
        members = stocks[:-1] if "2025-05-01" <= str(day.date()) <= "2025-06-30" else stocks
        (root / "instruments" / f"{day:%Y-%m-%d}.txt").write_text("\n".join(members) + "\n")
    rng = np.random.default_rng(2026)
    for number, stock in enumerate(stocks):
        close = (10 + number) * np.exp(np.cumsum(rng.normal(0.0002, 0.012, len(calendar))))
        factor = np.where(calendar >= "2025-07-01", 2.0, 1.0)
        volume = rng.integers(1000, 10000, len(calendar))
        frame = pd.DataFrame({
            "日期": calendar.strftime("%Y-%m-%d"), "股票代码": stock[2:],
            "开盘": close * 0.998, "最高": close * 1.01, "最低": close * 0.99,
            "收盘": close, "成交量": volume,
            "成交额": close / factor * volume * 100,
            "振幅": 2.0, "涨跌幅": 0.1, "涨跌额": 0.01, "换手率": 1.0,
            "raw_close": close / factor,
        })
        if number == 3:
            frame = frame[frame["日期"] != "2025-06-10"]
        frame.to_csv(root / "data" / f"{stock}.csv", index=False)
    return config, calendar


def observe_model_calls(monkeypatch):
    """Observe real Qlib/MLP training without substituting a fake learner."""
    from qlib.data.dataset.handler import DataHandlerLP

    from baseline.model import MLPModel

    fits = []
    original_fit, original_predict = MLPModel.fit, MLPModel.predict

    def fit(model, dataset, **kwargs):
        assert model.network is None, "Each month must initialize a fresh model"
        fits.append({
            "model": model,
            "dataset": dataset,
            "indices": {
                name: dataset.prepare(name, col_set=DataHandlerLP.CS_RAW,
                                      data_key=DataHandlerLP.DK_L).index.copy()
                for name in dataset.segments
            },
        })
        return original_fit(model, dataset, **kwargs)

    monkeypatch.setattr(MLPModel, "fit", fit)
    return fits, original_predict


def assert_normalization_matches(model, expected):
    means, scales = [], []
    for column in expected["feature"].to_numpy(dtype="float64").T:
        finite = column[np.isfinite(column)]
        means.append(finite.mean() if len(finite) else 0)
        scale = finite.std() if len(finite) else 0
        scales.append(scale if scale >= 1e-6 else 1)
    np.testing.assert_allclose(model.alpha_mean, np.asarray(means, dtype="float32"), rtol=1e-6)
    np.testing.assert_allclose(model.alpha_scale, np.asarray(scales, dtype="float32"), rtol=1e-6)


def test_raw_prices_to_six_fixed_epoch_qlib_models(tmp_path, monkeypatch):
    from baseline.train import train

    config, calendar = make_raw_fixture(tmp_path)
    config["model"]["epochs"] = 3
    prepare(config)
    cache = Path(config["cache_dir"])
    panel = pd.concat([pd.read_parquet(cache / f"{year}.parquet") for year in (2025, 2026)])
    assert len(panel["feature"].columns) == 158
    assert set(panel.dtypes) == {np.dtype("float32")}

    # An exited member is absent, and its earlier history is usable upon reentry.
    assert (pd.Timestamp("2025-06-02"), "SH600007") not in panel.index
    assert (pd.Timestamp("2025-07-01"), "SH600007") in panel.index
    means = panel["label"]["LABEL0"].groupby(level="datetime").mean().dropna()
    stds = panel["label"]["LABEL0"].groupby(level="datetime").std().dropna()
    np.testing.assert_allclose(means, 0, atol=2e-6)
    np.testing.assert_allclose(stds, 1, atol=2e-6)

    # Check T+1 to T+2 labels against raw input.
    raw = pd.read_csv(Path(config["data_dir"]) / "data/SH600000.csv", index_col="日期")
    expected = raw.loc["2026-04-03", "收盘"] / raw.loc["2026-04-02", "收盘"] - 1
    np.testing.assert_allclose(panel.loc[(pd.Timestamp("2026-04-01"), "SH600000"),
                                       ("raw_label", "LABEL0")], expected, atol=2e-7)
    assert panel.loc["2026-09-29":"2026-09-30", "label"].isna().all().all()

    fits, original_predict = observe_model_calls(monkeypatch)
    summary = train(config)
    output = Path(config["output_dir"])
    assert len(fits) == len({id(call["model"]) for call in fits}) == 6
    assert len(list(output.rglob("model.pt"))) == 6
    assert len(list(output.rglob("training_history.csv"))) == 6
    assert not list(output.rglob("selection_training_history.csv"))
    assert not list(output.rglob("validation_daily_ic.csv"))
    assert not list(output.rglob("validation_metrics.json"))
    assert summary["months"] == 6
    assert summary["model_class"] == "baseline.model.MLPModel"
    assert summary["training_scheme"] == "expanding_monthly_fixed_epochs"
    predictions = pd.read_parquet(output / "predictions.parquet")
    daily = pd.read_csv(output / "daily_ic.csv", index_col="datetime")
    first_day = predictions.xs(pd.Timestamp("2026-04-01")).dropna(subset=["score", "label"])
    expected_ic = np.corrcoef(first_day["score"], first_day["label"])[0, 1]
    np.testing.assert_allclose(daily.loc["2026-04-01", "ic"], expected_ic, atol=1e-7)
    assert summary["ic_mean"] == daily["ic"].mean()

    # September is complete, including predictions whose T+2 labels are unavailable.
    september = predictions.loc["2026-09-01":"2026-09-30"]
    assert september.index.get_level_values("datetime").unique().equals(pd.bdate_range("2026-09-01", "2026-09-30"))
    for day in ("2026-09-29", "2026-09-30"):
        rows = predictions.xs(pd.Timestamp(day))
        assert len(rows) == 8
        assert rows["score"].notna().all()
        assert rows[["label", "raw_label"]].isna().all().all()

    runs = [json.loads((output / f"2026-{month:02d}" / "run.json").read_text()) for month in range(4, 10)]
    training_counts = [run["actual"]["train"]["samples"] for run in runs]
    assert all(earlier < later for earlier, later in zip(training_counts, training_counts[1:]))
    for call, run in zip(fits, runs):
        model, window = call["model"], run["window"]
        test_start = pd.Timestamp(window["test_start"])
        # Derive the full eligible history directly from cache and trading dates.
        final_end = calendar[calendar < test_start][-3]
        expected_train = panel.loc["2025-01-01":final_end].dropna(subset=[("label", "LABEL0")])
        pd.testing.assert_index_equal(call["indices"]["train"], expected_train.index)
        assert set(call["indices"]) == set(run["segments"]) == set(run["actual"]) == {"train", "test"}
        assert not {"selection", "configured_model", "best_epoch"}.intersection(run)
        assert run["actual"]["train"]["samples"] == len(expected_train)
        assert run["actual"]["train"]["first_date"] == "2025-01-01"
        assert model.trained_epochs == run["trained_epochs"] == len(model.history) == 3
        assert run["model"] == config["model"]
        assert all(set(epoch) == {"epoch", "train_mse"} for epoch in model.history)
        assert_normalization_matches(model, expected_train)

        # Dates previously removed at the internal holdout boundary are retained.
        for day in ("2025-12-30", "2025-12-31", "2026-01-15", "2026-02-16", "2026-03-16"):
            assert (pd.Timestamp(day), "SH600000") in call["indices"]["train"]
        train_days = expected_train.index.get_level_values("datetime").unique()
        assert (calendar[calendar.get_indexer(train_days) + 2] < test_start).all()
        for day in calendar[(calendar > final_end) & (calendar < test_start)]:
            assert (day, "SH600000") not in call["indices"]["train"]

        month_dir = output / window["month"]
        with (month_dir / "model.pkl").open("rb") as stream:
            restored = pickle.load(stream)
        exported = pd.read_parquet(month_dir / "predictions.parquet")
        pd.testing.assert_series_equal(original_predict(restored, call["dataset"]), exported["score"])
        np.testing.assert_array_equal(restored.alpha_mean, model.alpha_mean)
        pd.testing.assert_frame_equal(pd.read_csv(month_dir / "training_history.csv"),
                                      pd.DataFrame(model.history))
    assert runs[0]["window"]["train_end"] == "2026-03-27"
    assert runs[3]["window"]["train_end"] == "2026-06-26"
    assert runs[3]["window"]["label_available_end"] == "2026-06-30"
    assert runs[-1]["window"]["test_end"] == "2026-09-30"

    # Repeated execution should reuse models instead of silently retraining.
    before = {path: path.stat().st_mtime_ns for path in output.rglob("model.pt")}
    train(config)
    assert len(fits) == 6
    assert before == {path: path.stat().st_mtime_ns for path in before}

    # A completion marker from the old validation/refit protocol cannot be reused.
    cache_files = []
    for year in (2025, 2026):
        path = cache / f"{year}.parquet"
        stat = path.stat()
        cache_files.append((path.name, stat.st_size, stat.st_mtime_ns))
    legacy_identity = {
        "model": "alpha158_mlp_v2", "training_scheme": "expanding_monthly_validate_then_refit",
        "config": config, "cache": json.loads((cache / "manifest.json").read_text()),
        "files": cache_files,
        "calendar": hashlib.sha256((Path(config["provider_dir"]) / "calendars/day.txt").read_bytes()).hexdigest(),
    }
    legacy_fingerprint = hashlib.sha256(json.dumps(legacy_identity, sort_keys=True).encode()).hexdigest()
    marker_path = output / "2026-04/completed.json"
    marker = json.loads(marker_path.read_text())
    assert legacy_fingerprint != marker["fingerprint"]
    marker["fingerprint"] = legacy_fingerprint
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="different inputs/config; use --refresh"):
        train(config)
    assert len(fits) == 6
    assert before == {path: path.stat().st_mtime_ns for path in before}
