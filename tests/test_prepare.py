import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from baseline.prepare import (
    FIELDS, build_provider, convert_bars, membership_intervals, prepare,
    read_inputs, source_fingerprint, write_binary,
)


def bars_frame(dates, close=None):
    if close is None:
        close = np.arange(len(dates), dtype=float) + 20
    close = np.asarray(close)
    return pd.DataFrame({"日期": dates, "开盘": close, "收盘": close, "最高": close * 1.1,
                         "最低": close * .9, "成交量": 100., "成交额": close / 2 * 10000,
                         "raw_close": close / 2})


def make_inputs(tmp_path):
    dates = pd.bdate_range("2025-08-01", "2026-01-16")
    start, end = pd.Timestamp("2025-12-22"), pd.Timestamp("2026-01-15")
    data = tmp_path / "raw"
    (data / "data").mkdir(parents=True)
    (data / "instruments").mkdir()
    (data / "calendar.txt").write_text("\n".join(dates.strftime("%Y-%m-%d")) + "\n")
    codes = ["SH600000", "SH600001", "SZ000001"]
    for day in dates[(dates >= start) & (dates <= end)]:
        members = codes if day < pd.Timestamp("2025-12-30") or day >= pd.Timestamp("2026-01-06") else codes[1:]
        (data / "instruments" / f"{day:%Y-%m-%d}.txt").write_text("\n".join(members) + "\n")
    for i, code in enumerate(codes):
        close = (20 + np.arange(len(dates)) * (i + 1) * .1) ** (1 + i * .1)
        bars_frame(dates, close).to_csv(data / "data" / f"{code}.csv", index=False)
    return {"data_dir": str(data), "provider_dir": str(tmp_path / "provider"),
            "cache_dir": str(tmp_path / "cache"), "research_start": str(start.date()),
            "research_end": str(end.date()), "qlib": {"kernels": 1}}


def test_adjustment_volume_and_vwap_are_consistent():
    frame = bars_frame(pd.bdate_range("2026-01-05", periods=3), [20, 22, 24])
    frame.loc[1, "成交量"] = 0  # Suspension (including stale quoted prices).
    frame.loc[2, "raw_close"] = 0
    bars = convert_bars(frame)
    first = bars.iloc[0]
    assert first["factor"] == 2
    assert first["volume"] == 5000
    assert first["vwap"] == 20
    assert first["vwap"] * first["volume"] == first["amount"]
    assert bars.iloc[1:].isna().all().all()
    assert all(dtype == np.dtype("float32") for dtype in bars.dtypes)


def test_binary_preserves_missing_market_days(tmp_path):
    calendar = pd.bdate_range("2026-01-01", periods=5)
    bars = convert_bars(bars_frame(calendar[[1, 3]], [20, 24]))
    length = write_binary(bars, calendar, tmp_path / "features")
    assert length == 3
    actual = np.fromfile(tmp_path / "features/close.day.bin", dtype="<f4")
    np.testing.assert_allclose(actual, [1, 20, np.nan, 24], equal_nan=True)
    assert len(list((tmp_path / "features").glob("*.bin"))) == len(FIELDS)


def test_provider_masks_adjusted_anomalies_and_keeps_other_stock_history(tmp_path):
    config = make_inputs(tmp_path)
    source_path = Path(config["data_dir"]) / "data/SH600001.csv"
    source = pd.read_csv(source_path)
    for day, column, value in [("2025-12-24", "收盘", -0.34),
                               ("2025-12-25", "最低", 0.0),
                               ("2025-12-26", "最高", np.inf)]:
        source.loc[source["日期"] == day, column] = value
    source.to_csv(source_path, index=False)
    cache = prepare(config)
    calendar = pd.DatetimeIndex(pd.to_datetime(
        (Path(config["provider_dir"]) / "calendars/day.txt").read_text().splitlines()))
    bad_positions = calendar.get_indexer(pd.to_datetime(["2025-12-24", "2025-12-25", "2025-12-26"]))
    for field in FIELDS:
        binary = np.fromfile(Path(config["provider_dir"]) / f"features/sh600001/{field}.day.bin", dtype="<f4")
        assert np.isnan(binary[1 + bad_positions - int(binary[0])]).all()
    panel = pd.read_parquet(cache / "2025.parquet")
    # T+1/T+2 labels involving an invalid close remain missing, not negative-price returns.
    for day in ["2025-12-22", "2025-12-23", "2025-12-24", "2025-12-25"]:
        assert np.isnan(panel.loc[(pd.Timestamp(day), "SH600001"), ("raw_label", "LABEL0")])
    assert np.isfinite(panel.loc[(pd.Timestamp("2025-12-29"), "SH600001"), ("raw_label", "LABEL0")])
    assert "SH600001" in (Path(config["provider_dir"]) / "instruments/csi300.txt").read_text()
    pd.testing.assert_frame_equal(pd.read_csv(source_path), source)


def test_membership_intervals_keep_exit_and_reentry():
    days = pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"])
    membership = {days[0]: {"A", "B"}, days[1]: {"B"}, days[2]: {"A", "B"}, days[3]: {"A"}}
    assert membership_intervals(membership) == [
        ("A", "2026-01-02", "2026-01-02"),
        ("A", "2026-01-06", "2026-01-07"),
        ("B", "2026-01-02", "2026-01-06"),
    ]


def test_inputs_reject_missing_membership_and_historical_bars(tmp_path):
    config = make_inputs(tmp_path)
    missing = Path(config["data_dir"]) / "instruments/2025-12-22.txt"
    contents = missing.read_text()
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="membership"):
        read_inputs(config)
    missing.write_text(contents)
    (Path(config["data_dir"]) / "data/SH600000.csv").unlink()
    with pytest.raises(FileNotFoundError, match="historical member"):
        read_inputs(config)


def test_provider_rebuild_does_not_erase_unknown_directories(tmp_path):
    config = make_inputs(tmp_path)
    calendar, membership = read_inputs(config)
    provider = Path(config["provider_dir"])
    provider.mkdir()
    (provider / "unrelated.txt").write_text("user file")
    with pytest.raises(ValueError, match="unrecognized"):
        build_provider(config, calendar, membership, source_fingerprint(config, membership))
    assert (provider / "unrelated.txt").read_text() == "user file"


def test_prepare_rejects_failed_refresh_and_missing_completion_markers(tmp_path):
    config = make_inputs(tmp_path)
    data = Path(config["data_dir"])
    failures = data / "failures.json"
    failures.write_text(json.dumps([{"stage": "price", "item": "SH600000"}]))
    with pytest.raises(ValueError, match="Download is incomplete"):
        read_inputs(config)
    failures.write_text("[]")
    calendar, membership = read_inputs(config)
    metadata = {
        "calendar": {"start": "2025-08-01", "end": "2026-01-16"},
        "memberships": {f"{day:%Y-%m-%d}": {"count": len(members)} for day, members in membership.items()},
        "prices": {code: {"start": "2025-08-01", "end": "2026-01-16", "adjust": "hfq"}
                   for code in set.union(*membership.values())},
    }
    path = data / "metadata.json"
    path.write_text(json.dumps(metadata))
    read_inputs(config)
    # A refresh clears the completion marker before fetching; the old CSV remains.
    metadata["prices"].pop("SH600000")
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="completed price metadata"):
        read_inputs(config)


def test_provider_rejects_bars_only_outside_historical_membership(tmp_path):
    config = make_inputs(tmp_path)
    path = Path(config["data_dir"]) / "data/SH600000.csv"
    frame = pd.read_csv(path)
    frame.loc[frame["日期"] < config["research_start"]].to_csv(path, index=False)
    calendar, membership = read_inputs(config)
    with pytest.raises(ValueError, match="no usable bars while it belongs"):
        build_provider(config, calendar, membership, source_fingerprint(config, membership))


def limit_first_member_to_three_days(config):
    root = Path(config["data_dir"])
    member_days = pd.to_datetime(["2025-12-22", "2025-12-23", "2025-12-24"])
    for path in (root / "instruments").glob("*.txt"):
        if pd.Timestamp(path.stem) not in member_days:
            codes = [code for code in path.read_text().splitlines() if code != "SH600000"]
            path.write_text("\n".join(codes) + "\n")
    return member_days


@pytest.mark.parametrize("mode", ["absent", "zero_volume"])
def test_provider_preserves_entire_suspended_member_window_between_real_bars(tmp_path, caplog, mode):
    config = make_inputs(tmp_path)
    member_days = limit_first_member_to_three_days(config)
    path = Path(config["data_dir"]) / "data/SH600000.csv"
    source = pd.read_csv(path)
    mask = pd.to_datetime(source["日期"]).isin(member_days)
    if mode == "absent":
        source = source.loc[~mask]
    else:
        source.loc[mask, ["成交量", "成交额"]] = 0
    source.to_csv(path, index=False)
    saved_source = path.read_bytes()
    cache = prepare(config)
    provider = Path(config["provider_dir"])
    calendar = pd.DatetimeIndex(pd.to_datetime((provider / "calendars/day.txt").read_text().splitlines()))
    positions = calendar.get_indexer(member_days)
    for field in FIELDS:
        binary = np.fromfile(provider / f"features/sh600000/{field}.day.bin", dtype="<f4")
        assert np.isnan(binary[1 + positions - int(binary[0])]).all()
    assert "SH600000\t2025-12-22\t2025-12-24\n" in (provider / "instruments/csi300.txt").read_text()
    manifest = json.loads((provider / "manifest.json").read_text())
    audit = manifest["zero_member_bars"]["SH600000"]
    assert "SH600000" in manifest["stock_lengths"]
    assert audit["member_dates"] == member_days.strftime("%Y-%m-%d").tolist()
    assert audit["last_usable_bar_before"] == "2025-12-19"
    assert audit["first_usable_bar_after"] == "2025-12-25"
    assert audit["zero_volume_rows"] == (0 if mode == "absent" else 3)
    panel = pd.read_parquet(cache / "2025.parquet")
    actual = panel.index[panel.index.get_level_values("instrument") == "SH600000"].get_level_values("datetime")
    pd.testing.assert_index_equal(actual, member_days, check_names=False)
    assert path.read_bytes() == saved_source
    assert "retain 3 member dates without usable bars" in caplog.text


@pytest.mark.parametrize("column, value", [("收盘", -1), ("最低", 0), ("最高", np.inf),
                                        ("raw_close", np.nan), ("成交量", -1)])
def test_bracketed_member_window_with_invalid_source_values_still_fails(tmp_path, column, value):
    config = make_inputs(tmp_path)
    member_days = limit_first_member_to_three_days(config)
    path = Path(config["data_dir"]) / "data/SH600000.csv"
    source = pd.read_csv(path)
    source.loc[pd.to_datetime(source["日期"]).isin(member_days), column] = value
    source.to_csv(path, index=False)
    calendar, membership = read_inputs(config)
    with pytest.raises(ValueError, match="no usable bars while it belongs"):
        build_provider(config, calendar, membership, source_fingerprint(config, membership))


def test_complete_membership_cache_keeps_rows_after_last_bar_for_predictions_only(tmp_path):
    from baseline.model import MLPModel
    from baseline.train import assemble_month, monthly_windows
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    config = make_inputs(tmp_path)
    config.update(train_start=config["research_start"], first_test_month="2026-01-01")
    path = Path(config["data_dir"]) / "data/SH600001.csv"
    source = pd.read_csv(path)
    source = source.loc[source["日期"] <= "2025-12-23"]
    source.to_csv(path, index=False)
    source_bytes = path.read_bytes()
    calendar, membership = read_inputs(config)
    cache = prepare(config)
    panel = pd.concat([pd.read_parquet(cache / f"{year}.parquet") for year in (2025, 2026)])
    expected_index = pd.MultiIndex.from_tuples(
        [(day, code) for day in sorted(membership) for code in sorted(membership[day])],
        names=["datetime", "instrument"],
    )
    pd.testing.assert_index_equal(panel.index, expected_index)
    assert panel.index.is_unique and panel.index.is_monotonic_increasing
    assert len(panel) == sum(map(len, membership.values()))
    assert set(panel.dtypes) == {np.dtype("float32")}
    beyond_source = ((panel.index.get_level_values("instrument") == "SH600001")
                     & (panel.index.get_level_values("datetime") > "2025-12-23"))
    assert beyond_source.any() and panel.loc[beyond_source].isna().all().all()
    window, = monthly_windows(config, calendar)
    frame, _ = assemble_month(cache, window)
    dates = frame.index.get_level_values("datetime")
    training = frame.loc[dates <= window["train_end"]]
    assert np.isfinite(training[("label", "LABEL0")]).all()
    assert "SH600001" not in training.index.get_level_values("instrument")
    test_index = expected_index[expected_index.get_level_values("datetime") >= window["test_start"]]
    pd.testing.assert_index_equal(frame.index[dates >= window["test_start"]], test_index)
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments={
        "train": (window["train_start"], window["train_end"]),
        "test": (window["test_start"], window["test_end"]),
    })
    model = MLPModel(epochs=1, hidden_size=4, batch_size=16, num_threads=1)
    model.fit(dataset)
    pred = model.predict(dataset)
    pd.testing.assert_index_equal(pred.index, test_index)
    assert np.isfinite(pred).all()
    assert len(pred.xs("SH600001", level="instrument")) == sum(
        day >= pd.Timestamp(window["test_start"]) and "SH600001" in codes
        for day, codes in membership.items()
    )
    assert path.read_bytes() == source_bytes


def test_qlib_alpha158_labels_cache_and_inference_tail(tmp_path):
    config = make_inputs(tmp_path)
    cache = prepare(config)
    panel = pd.concat([pd.read_parquet(cache / f"{year}.parquet") for year in (2025, 2026)])
    assert panel.index.names == ["datetime", "instrument"]
    assert panel["feature"].shape[1] == 158
    assert all(dtype == np.dtype("float32") for dtype in panel.dtypes)
    assert (pd.Timestamp("2026-01-05"), "SH600000") not in panel.index
    assert (pd.Timestamp("2026-01-06"), "SH600000") in panel.index
    # Last two research days remain forecastable, but have no observable T+2 label.
    tail = panel.loc[pd.IndexSlice["2026-01-14":"2026-01-15", :], :]
    assert len(tail) == 6
    assert tail["raw_label"].isna().all().all()
    assert tail["feature"].notna().any(axis=1).all()
    # Dec 31 labels read Jan 1/2 from the same complete provider across year chunks.
    day = pd.Timestamp("2025-12-31")
    actual = panel.loc[day, ("raw_label", "LABEL0")]
    expected = []
    for code in actual.index:
        source = pd.read_csv(Path(config["data_dir"]) / "data" / f"{code}.csv", index_col="日期")
        expected.append(source.loc["2026-01-02", "收盘"] / source.loc["2026-01-01", "收盘"] - 1)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    raw = panel["raw_label"]["LABEL0"]
    expected_z = raw.groupby(level="datetime").transform(lambda series: (series - series.mean()) / series.std(ddof=1))
    np.testing.assert_allclose(panel["label"]["LABEL0"], expected_z, atol=1e-6, equal_nan=True)
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["years"] == [2025, 2026]
    assert manifest["feature_count"] == 158
    assert manifest["index_policy"] == "complete_historical_membership"
    assert manifest["index_version"] == 1
    from baseline.prepare import _cache_complete

    assert _cache_complete(cache, manifest, manifest["fingerprint"], [2025, 2026])
    legacy_manifest = {key: value for key, value in manifest.items() if key not in {"index_policy", "index_version"}}
    assert not _cache_complete(cache, legacy_manifest, manifest["fingerprint"], [2025, 2026])
    assert not _cache_complete(cache, {**manifest, "index_version": 0}, manifest["fingerprint"], [2025, 2026])
    # Completed cache is reused; a missing year cannot be silently reused.
    before = (cache / "2025.parquet").stat().st_mtime_ns
    assert prepare(config) == cache
    assert (cache / "2025.parquet").stat().st_mtime_ns == before
    (cache / "2026.parquet").unlink()
    prepare(config)
    assert (cache / "2026.parquet").is_file()
