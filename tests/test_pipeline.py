"""Offline integration: synthetic prices are test fixtures, never research data."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.__main__ import load_config
from baseline.prepare import prepare


def make_raw_fixture(tmp_path):
    config = load_config("configs/baseline.yaml")
    for key in ("data_dir", "provider_dir", "cache_dir", "output_dir"):
        config[key] = str(tmp_path / key)
    config.update(research_start="2025-01-01", train_start="2025-01-01")
    config["model"].update(num_boost_round=5, num_leaves=8, max_depth=3,
                           min_data_in_leaf=5, lambda_l1=0, lambda_l2=0)
    root = Path(config["data_dir"])
    (root / "instruments").mkdir(parents=True)
    (root / "data").mkdir()
    calendar = pd.bdate_range("2024-09-02", "2026-09-30")
    (root / "calendar.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
    stocks = [f"SH60000{i}" for i in range(8)]
    for day in calendar[(calendar >= "2025-01-01") & (calendar <= "2026-09-18")]:
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


def test_raw_prices_to_six_qlib_models(tmp_path):
    from baseline.train import train

    config, calendar = make_raw_fixture(tmp_path)
    prepare(config)
    cache = Path(config["cache_dir"])
    panels = [pd.read_parquet(cache / f"{year}.parquet") for year in (2025, 2026)]
    panel = pd.concat(panels)
    assert len(panel["feature"].columns) == 158
    assert set(panel.dtypes) == {np.dtype("float32")}

    # An exited member is absent, and its earlier history is usable upon reentry.
    assert (pd.Timestamp("2025-06-02"), "SH600007") not in panel.index
    assert (pd.Timestamp("2025-07-01"), "SH600007") in panel.index
    means = panel["label"]["LABEL0"].groupby(level="datetime").mean().dropna()
    stds = panel["label"]["LABEL0"].groupby(level="datetime").std().dropna()
    np.testing.assert_allclose(means, 0, atol=2e-6)
    np.testing.assert_allclose(stds, 1, atol=2e-6)

    # Check the expression against raw input, not another copy of the implementation.
    raw = pd.read_csv(Path(config["data_dir"]) / "data/SH600000.csv", index_col="日期")
    expected = raw.loc["2026-04-03", "收盘"] / raw.loc["2026-04-02", "收盘"] - 1
    np.testing.assert_allclose(panel.loc[(pd.Timestamp("2026-04-01"), "SH600000"),
                                       ("raw_label", "LABEL0")], expected, atol=2e-7)
    tail = panel.loc[pd.IndexSlice[pd.Timestamp("2026-09-17"):pd.Timestamp("2026-09-18"), :], "label"]
    assert tail.isna().all().all()

    train(config)
    output = Path(config["output_dir"])
    # Six artifacts prove real model fitting and recorders ran without a valid segment.
    assert len(list(output.rglob("model.txt"))) == 6
    assert len(list(output.rglob("predictions.parquet"))) >= 6
    assert (output / "summary.json").is_file()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["months"] == 6
    predictions = pd.read_parquet(output / "predictions.parquet")
    daily = pd.read_csv(output / "daily_ic.csv", index_col="datetime")
    first_day = predictions.xs(pd.Timestamp("2026-04-01")).dropna(subset=["score", "label"])
    expected_ic = np.corrcoef(first_day["score"], first_day["label"])[0, 1]
    np.testing.assert_allclose(daily.loc["2026-04-01", "ic"], expected_ic, atol=1e-7)
    assert summary["ic_mean"] == daily["ic"].mean()
    assert len(predictions.xs(pd.Timestamp("2026-09-18"))) == 8
    assert predictions.xs(pd.Timestamp("2026-09-18"))["label"].isna().all()

    runs = [json.loads((output / f"2026-{month:02d}" / "run.json").read_text()) for month in range(4, 10)]
    training_counts = [run["actual"]["train"]["samples"] for run in runs]
    assert all(earlier < later for earlier, later in zip(training_counts, training_counts[1:]))
    for run in runs:
        window = run["window"]
        expected_train = panel.loc[window["train_start"]:window["train_end"]].dropna(subset=[("label", "LABEL0")])
        assert run["actual"]["train"]["samples"] == len(expected_train)
        assert run["actual"]["train"]["first_date"] == "2025-01-01"
    assert runs[3]["window"]["label_available_end"] == "2026-06-30"

    # Repeated execution should reuse models instead of silently retraining.
    before = {path: path.stat().st_mtime_ns for path in output.rglob("model.txt")}
    train(config)
    assert before == {path: path.stat().st_mtime_ns for path in before}
