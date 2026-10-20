import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from polymarket_ext.factors import CATEGORY_FACTORS, FACTOR_COLUMNS, align_prices, compute_factors, prepare


def market(market_id="m1", category="threshold", **kwargs):
    return {
        "event_id": "event1", "market_id": market_id, "category": category,
        "side": None, "token_id": f"yes-{market_id}", "no_token_id": f"no-{market_id}",
        "condition_id": f"condition-{market_id}", "question": "Oil price?", "event_title": "Oil",
        "underlying": "WTI", "price_basis": "settlement", "slug": market_id,
        "opened_at": "2026-01-01T00:00:00Z", "metadata_available_at": "2026-01-01T00:00:00Z",
        "closed_at": None, "observation_start": "2026-01-01T00:00:00Z",
        "observation_end": "2026-03-01T00:00:00Z", "expires_at": "2026-03-01T00:00:00Z",
        "threshold": 80., "lower": None, "upper": None, "record_threshold": None,
        "market_dir": f"{category}/event1/{market_id}",
        **({"time_extraction_version": 1} if category == "direction" else {}), **kwargs,
    }


def quotes(markets, dates, values):
    rows = []
    for item, probabilities in zip(markets, values):
        for day, probability in zip(dates, probabilities):
            rows.append({"token_id": item["token_id"],
                         "timestamp": int((pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=7)).timestamp()),
                         "price": probability})
    return pd.DataFrame(rows)


def panel(markets, dates, values, **kwargs):
    aligned = align_prices(pd.DataFrame(markets), quotes(markets, dates, values), dates, **kwargs)
    return compute_factors(aligned, dates)


def test_align_beijing_close_opening_expiry_and_metadata_availability():
    days = pd.bdate_range("2026-01-05", periods=3)
    markets = [market("open", opened_at="2026-01-05T06:59:00Z", closed_at="2026-01-07T07:00:00Z"),
               market("later", opened_at="2026-01-05T07:00:01Z"),
               market("metadata", metadata_available_at="2026-01-06T07:00:01Z"),
               market("unknown", metadata_available_at=None)]
    prices = pd.DataFrame([
        {"token_id": "yes-open", "timestamp": "2026-01-05T06:58:00Z", "price": .1},
        {"token_id": "yes-open", "timestamp": "2026-01-05T07:00:00Z", "price": .3},
        {"token_id": "yes-open", "timestamp": "2026-01-05T07:00:01Z", "price": .9},
    ])
    result = align_prices(pd.DataFrame(markets), prices, days)
    first = result.set_index(["datetime", "market_id"])
    assert first.loc[(days[0], "open"), "probability"] == .3
    assert first.loc[(days[1], "open"), "probability"] == .9
    assert (days[0], "later") not in first.index
    assert (days[1], "later") in first.index
    assert (days[1], "metadata") not in first.index
    assert (days[2], "metadata") in first.index
    assert (days[2], "open") not in first.index
    assert "unknown" not in result["market_id"].values
    assert (result["quote_time"].dropna() <= result.loc[result["quote_time"].notna(), "cutoff"]).all()
    assert first.loc[(days[0], "open"), "cutoff"] == pd.Timestamp("2026-01-05T07:00:00Z")


def test_latest_invalid_quote_is_not_replaced_and_age_is_audited():
    days = pd.bdate_range("2026-01-05", periods=2)
    prices = pd.DataFrame([
        {"token_id": "yes-m1", "timestamp": "2026-01-05T05:59:00Z", "price": .2, "resolution_seconds": 60},
        {"token_id": "yes-m1", "timestamp": "2026-01-05T06:59:00Z", "price": 1.2, "resolution_seconds": 60},
        {"token_id": "yes-m1", "timestamp": "2026-01-05T07:01:00Z", "price": .4, "resolution_seconds": 300},
    ])
    result = align_prices(pd.DataFrame([market()]), prices, days, max_quote_age="1h")
    assert result["probability"].isna().all()
    assert result["quote_valid"].eq(False).all()
    assert result.loc[0, "price"] == 1.2
    assert result.loc[0, "quote_age_seconds"] == 60
    assert result.loc[1, "quote_resolution_seconds"] == 300
    assert result.loc[1, "price"] == .4


@pytest.mark.parametrize("resolution", [60, 300, 1800, 10800])
def test_data_v2_buckets_become_available_only_when_complete(resolution):
    days = pd.DatetimeIndex(["2026-01-05"])
    cutoff = pd.Timestamp("2026-01-05T07:00:00Z")
    ids = ["before", "boundary", "pending", "exact", "clob", "local"]
    rows = []
    for name, offset in (("before", -1), ("boundary", 0), ("pending", 1)):
        rows.append({"token_id": f"yes-{name}", "price": .6, "source": "data_v2",
                     "timestamp": cutoff - pd.Timedelta(seconds=resolution) + pd.Timedelta(seconds=offset),
                     "resolution_seconds": resolution})
    # An unfinished new bucket cannot hide an older available observation.
    rows.append({"token_id": "yes-pending", "price": .2, "source": "clob",
                 "timestamp": cutoff - pd.Timedelta(hours=4), "resolution_seconds": 60})
    for name, source in (("exact", "data_v2"), ("clob", "clob"), ("local", None)):
        rows.append({"token_id": f"yes-{name}", "price": .4, "source": source,
                     "timestamp": cutoff, "resolution_seconds": 0 if name == "exact" else resolution})
    result = align_prices(pd.DataFrame([market(name) for name in ids]), pd.DataFrame(rows), days)
    result = result.set_index("market_id")
    assert result.loc[["before", "boundary"], "probability"].eq(.6).all()
    assert result.loc["before", "quote_available_at"] == cutoff - pd.Timedelta(seconds=1)
    assert result.loc["boundary", "quote_available_at"] == cutoff
    assert result.loc["pending", "probability"] == .2
    assert result.loc["pending", "quote_source"] == "clob"
    assert result.loc[["exact", "clob", "local"], "probability"].eq(.4).all()
    assert result.loc[["exact", "clob", "local"], "quote_available_at"].eq(cutoff).all()
    assert (result["quote_available_at"] <= result["cutoff"]).all()


def test_delayed_older_bucket_never_replaces_newer_observation():
    days = pd.bdate_range("2026-01-05", periods=2)
    prices = pd.DataFrame([
        {"token_id": "yes-m1", "timestamp": "2026-01-05T06:00:00Z", "price": .9,
         "resolution_seconds": 10800, "source": "data_v2"},
        {"token_id": "yes-m1", "timestamp": "2026-01-05T06:59:00Z", "price": .4,
         "resolution_seconds": 60, "source": "clob"},
    ])
    result = align_prices(pd.DataFrame([market()]), prices, days)
    # The coarse point arrives at 09:00, after the first cutoff, but its older
    # observation timestamp also prevents it from replacing CLOB the next day.
    assert result["probability"].eq(.4).all()
    assert result["quote_source"].eq("clob").all()
    assert result["quote_time"].eq(pd.Timestamp("2026-01-05T06:59:00Z")).all()


def test_data_v2_availability_precedes_price_validation_without_backfill():
    days = pd.DatetimeIndex(["2026-01-05"])
    prices = pd.DataFrame([
        {"token_id": "yes-m1", "timestamp": "2026-01-05T06:58:00Z", "price": .2,
         "resolution_seconds": 60, "source": "clob"},
        {"token_id": "yes-m1", "timestamp": "2026-01-05T06:59:00Z", "price": 1.2,
         "resolution_seconds": 60, "source": "data_v2"},
        {"token_id": "yes-m1", "timestamp": "2026-01-05T07:00:00Z", "price": .7,
         "resolution_seconds": 60, "source": "data_v2"},
    ])
    result = align_prices(pd.DataFrame([market()]), prices, days)
    assert result.loc[0, "price"] == 1.2
    assert result.loc[0, "quote_available_at"] == pd.Timestamp("2026-01-05T07:00:00Z")
    assert not result.loc[0, "quote_valid"]
    assert np.isnan(result.loc[0, "probability"])


def test_threshold_scale_uses_available_metadata_and_does_not_skip_missing_quote():
    days = pd.bdate_range("2026-01-05", periods=2)
    markets = [market("a", threshold=60), market("b", threshold=80), market("c", threshold=100),
               market("future", threshold=1000, opened_at="2026-01-06T00:00:00Z")]
    result = panel(markets, days, [[.8, .8], [np.nan, .5], [.2, .2], [.01, .01]])
    one = result.loc[result["datetime"] == days[0]].set_index("market_id")
    assert one["group_center"].eq(80).all()
    assert one["group_scale"].eq(80).all()
    assert one["group_boundary_count"].eq(3).all()
    assert one.loc["a", "THRESHOLD_REL_POSITION"] == -.25
    assert one["THRESHOLD_ADJACENT_MASS"].isna().all()
    two = result.loc[result["datetime"] == days[1]].set_index("market_id")
    assert two["group_center"].eq(90).all()
    assert two.loc["a", "THRESHOLD_ADJACENT_MASS"] == pytest.approx(.3)
    assert two.loc["a", "THRESHOLD_REL_DENSITY"] == pytest.approx(.3 / (20 / 90))
    assert np.isnan(two.loc["future", "THRESHOLD_ADJACENT_MASS"])


def test_threshold_duplicates_and_negative_masses_are_not_probabilities():
    days = pd.bdate_range("2026-01-05", periods=1)
    markets = [market("a", threshold=60), market("b", threshold=80), market("duplicate", threshold=80),
               market("c", threshold=100), market("d", threshold=120)]
    result = panel(markets, days, [[.8], [.5], [.5], [.2], [.3]])
    assert result["THRESHOLD_ADJACENT_MASS"].isna().all()
    assert result["THRESHOLD_REL_DENSITY"].isna().all()
    assert result["group_boundary_count"].eq(4).all()
    tiny = panel([market("a", threshold=60), market("b", threshold=80)], days, [[.5], [.5 + 1e-10]])
    assert tiny.iloc[0]["THRESHOLD_ADJACENT_MASS"] == 0
    assert tiny.iloc[0]["THRESHOLD_REL_DENSITY"] == 0


@pytest.mark.parametrize("different", [
    {"event_id": "other"}, {"underlying": "Brent"}, {"price_basis": "spot"},
    {"observation_start": "2026-01-02T00:00:00Z"},
    {"observation_end": "2026-02-01T00:00:00Z"}, {"expires_at": "2026-02-01T00:00:00Z"},
])
def test_different_contract_definitions_are_never_grouped(different):
    days = pd.bdate_range("2026-01-05", periods=1)
    result = panel([market("a", threshold=60), market("b", threshold=80, **different)], days, [[.8], [.2]])
    assert result["THRESHOLD_REL_POSITION"].isna().all()
    assert result["THRESHOLD_ADJACENT_MASS"].isna().all()


def test_touch_groups_high_and_low_separately_and_retains_signed_changes():
    days = pd.bdate_range("2026-01-05", periods=2)
    markets = [market("h1", "touch", side="high", threshold=100),
               market("h2", "touch", side="high", threshold=120),
               market("l1", "touch", side="low", threshold=40),
               market("l2", "touch", side="low", threshold=60)]
    result = panel(markets, days, [[.4, .6], [.2, .3], [.1, .5], [.3, .8]])
    last = result.loc[result["datetime"] == days[1]].set_index("market_id")
    assert last.loc["h1", "group_center"] == 110
    assert last.loc["l1", "group_center"] == 50
    assert last.loc["h1", "HIGH_D1"] == pytest.approx(.2)
    assert last.loc["l1", "LOW_D1"] == pytest.approx(.4)
    assert last.loc["h1", "HIGH_REL_POSITION"] == pytest.approx(-10 / 110)
    assert last.loc["l1", "LOW_REL_POSITION"] == -.2
    assert last.loc[["h1", "h2"], "LOW_PROB"].isna().all()


def test_range_open_tails_widths_partition_coordinate_and_density_can_exceed_one():
    days = pd.bdate_range("2026-01-05", periods=1)
    markets = [market("a", "range", lower=-np.inf, upper=60),
               market("b", "range", lower=60, upper=100),
               market("c", "range", lower=100, upper=np.inf)]
    result = panel(markets, days, [[.1], [.7], [.2]]).set_index("market_id")
    assert result["group_center"].eq(80).all()
    assert result["group_scale"].eq(80).all()
    assert result["RANGE_REL_WIDTH"].sum() == pytest.approx(1)
    assert result.loc["a", "RANGE_REL_WIDTH"] == pytest.approx(.5 + np.arctan(-.25) / np.pi)
    assert result.loc["c", "RANGE_REL_WIDTH"] == pytest.approx(.5 - np.arctan(.25) / np.pi)
    assert result.loc["b", "RANGE_REL_DENSITY"] > 1


def test_range_single_boundary_scale_and_missing_quote_or_bad_boundary():
    days = pd.bdate_range("2026-01-05", periods=1)
    singleton = panel([market("a", "range", lower=-np.inf, upper=80)], days, [[.4]])
    assert singleton.loc[0, "RANGE_REL_WIDTH"] == .5
    assert singleton.loc[0, "RANGE_REL_DENSITY"] == .8
    markets = [market("missing", "range", lower=70, upper=80),
               market("bad", "range", lower=90, upper=80),
               market("unknown", "range", lower=None, upper=80)]
    bad = panel(markets, days, [[np.nan], [.3], [.2]])
    assert bad[["RANGE_REL_WIDTH", "RANGE_REL_DENSITY"]].isna().all().all()
    zero = panel([market("zero", "range", lower=-np.inf, upper=0)], days, [[.5]])
    assert np.isnan(zero.loc[0, "RANGE_REL_WIDTH"])


def test_differences_require_contiguous_valid_calendar_days_and_same_token():
    days = pd.bdate_range("2026-01-05", periods=7)
    markets = [market("record", "record", record_threshold=147.27),
               market("gap", "range", lower=60, upper=100)]
    result = panel(markets, days, [[.1, .2, .3, .4, .5, .6, .8], [.1, .2, np.nan, .4, .5, .6, .8]])
    record = result.loc[result["category"] == "record"].reset_index(drop=True)
    assert record.loc[:4, "HIGH_D5"].isna().all()
    assert record.loc[5, "HIGH_D5"] == pytest.approx(.5)
    assert record.loc[6, "HIGH_D5"] == pytest.approx(.6)
    assert record["record_threshold"].eq(147.27).all()
    gap = result.loc[result["category"] == "range"].reset_index(drop=True)
    assert gap.loc[[0, 2, 3], "D1"].isna().all()
    assert gap["D5"].isna().all()
    assert gap.loc[6, "D1"] == pytest.approx(.2)
    aligned = align_prices(pd.DataFrame([markets[0]]), quotes([markets[0]], days, [[.1] * 7]), days)
    aligned = aligned.drop(index=2)  # Absent rows also break the six-cutoff requirement.
    missing_day = compute_factors(aligned, days)
    assert missing_day["HIGH_D5"].isna().all()
    switched = align_prices(pd.DataFrame([markets[0]]), quotes([markets[0]], days, [[.1] * 7]), days)
    switched.loc[3:, "token_id"] = "new-token"
    assert compute_factors(switched, days)["HIGH_D5"].isna().all()


def test_direction_entropy_boundaries_and_unused_features_remain_missing():
    days = pd.bdate_range("2026-01-05", periods=3)
    result = panel([market("direction", "direction")], days, [[0, .5, 1]])
    np.testing.assert_allclose(result["ENTROPY"], [0, np.log(2), 0])
    np.testing.assert_allclose(result["DIRECTION_SCORE"], [-1, 0, 1])
    np.testing.assert_allclose(result["DIRECTION_CERTAINTY"], [1, 0, 1])
    unused = set(FACTOR_COLUMNS) - set(CATEGORY_FACTORS["direction"])
    assert result[list(unused)].isna().all().all()


def test_direction_time_factors_use_calendar_seconds_without_requiring_a_quote():
    days = pd.DatetimeIndex(["2026-01-05"])
    item = market("direction", "direction", observation_start="2026-01-04T07:00:00Z",
                  observation_end="2026-01-06T07:00:00Z")
    result = compute_factors(align_prices(pd.DataFrame([item]), pd.DataFrame(), days), days)
    assert CATEGORY_FACTORS["direction"] == (
        "PROB", "ENTROPY", "DIRECTION_SCORE", "DIRECTION_CERTAINTY", "WINDOW_LENGTH", "TIME_TO_END",
    )
    assert result.loc[0, list(CATEGORY_FACTORS["direction"][:4])].isna().all()
    assert result.loc[0, "WINDOW_LENGTH"] == pytest.approx(np.log(3))
    assert result.loc[0, "TIME_TO_END"] == pytest.approx(np.log(2))
    # These features belong only to direction, even if another category has times.
    threshold = panel([market("threshold")], days, [[.4]])
    assert threshold[["WINDOW_LENGTH", "TIME_TO_END"]].isna().all().all()


def test_direction_time_factors_preserve_partial_times_and_invalid_window():
    days = pd.DatetimeIndex(["2026-01-05"])
    end = "2026-01-06T07:00:00Z"
    markets = [market("no_start", "direction", observation_start=None, observation_end=end),
               market("no_end", "direction", observation_end=None),
               market("no_times", "direction", observation_start=None, observation_end=None),
               market("zero_window", "direction", observation_start=end, observation_end=end),
               market("reversed", "direction", observation_start="2026-01-07T07:00:00Z",
                      observation_end=end)]
    result = panel(markets, days, [[.6]] * len(markets)).set_index("market_id")
    assert result["PROB"].eq(.6).all()
    assert result["WINDOW_LENGTH"].isna().all()
    assert result.loc[["no_end", "no_times"], "TIME_TO_END"].isna().all()
    np.testing.assert_allclose(result.loc[["no_start", "zero_window", "reversed"], "TIME_TO_END"],
                               np.log(2))


def test_direction_future_window_weekend_and_dst_use_actual_elapsed_seconds():
    days = pd.DatetimeIndex(["2026-03-06", "2026-03-09"])
    item = market("direction", "direction", observation_start="2026-03-06T16:00:00-05:00",
                  observation_end="2026-03-09T16:00:00-04:00", expires_at="2026-04-01T00:00:00Z")
    result = panel([item], days, [[.4, .6]])
    # Friday 21:00 UTC to Monday 20:00 UTC is 71 hours, crossing the spring DST shift.
    np.testing.assert_allclose(result["WINDOW_LENGTH"], np.log1p(71 / 24))
    np.testing.assert_allclose(result["TIME_TO_END"], np.log1p(np.array([85, 13]) / 24))
    assert result.loc[0, "TIME_TO_END"] > result.loc[0, "WINDOW_LENGTH"]


def test_direction_known_comparison_end_excludes_expired_markets_at_close():
    days = pd.DatetimeIndex(["2026-01-05"])
    markets = [market("ended", "direction", observation_end="2026-01-05T06:59:59Z"),
               market("at_close", "direction", observation_end="2026-01-05T07:00:00Z"),
               market("active", "direction", observation_end="2026-01-05T07:00:01Z")]
    result = panel(markets, days, [[.5]] * len(markets))
    assert result["market_id"].tolist() == ["active"]
    assert result.iloc[0]["TIME_TO_END"] == pytest.approx(np.log1p(1 / 86400))


def test_prepare_reads_layout_preserves_warmup_and_invalidates_changed_inputs(tmp_path):
    days = pd.bdate_range("2026-01-05", periods=7)
    item = market("record", "record", record_threshold=147.27)
    directory = tmp_path / "data" / item["market_dir"]
    directory.mkdir(parents=True)
    price_file = directory / "yes_prices.csv"
    history = quotes([item], days, [[.1, .2, .3, .4, .5, .6, .8]])
    history[["timestamp", "price"]].to_csv(price_file, index=False)
    (tmp_path / "data/markets.jsonl").write_text(json.dumps(item) + "\n")
    calendar_file = tmp_path / "calendar.txt"
    calendar_file.write_text("\n".join(days.strftime("%Y-%m-%d")))
    config = {"data_dir": str(tmp_path / "data"), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(calendar_file), "research_start": "2026-01-12", "research_end": "2026-01-13"}
    path = prepare(config)
    result = pd.read_parquet(path / "factors.parquet")
    assert result["datetime"].tolist() == list(days[-2:])
    np.testing.assert_allclose(result["HIGH_D5"], [.5, .6])
    before = (path / "factors.parquet").stat().st_mtime_ns
    assert prepare(config) == path
    assert (path / "factors.parquet").stat().st_mtime_ns == before
    history.loc[6, "price"] = .9
    history[["timestamp", "price"]].to_csv(price_file, index=False)
    prepare(config)
    result = pd.read_parquet(path / "factors.parquet")
    assert result.iloc[-1]["HIGH_PROB"] == .9
    assert result.iloc[-1]["HIGH_D5"] == pytest.approx(.7)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["rows"] == 2
    assert manifest["valid_quotes"] == 2
    assert manifest["cutoff"] == "15:00 Asia/Shanghai"
    assert set(manifest["files"]) == {"aligned.parquet", "factors.parquet"}
    assert manifest["source_warnings"]
    assert manifest["category_coverage"]["record"] == {
        "first_valid_date": "2026-01-12", "last_valid_date": "2026-01-13",
        "valid_rows": 2, "quoted_markets": 1, "trading_days_with_quotes": 2,
    }
    assert manifest["category_coverage"]["direction"]["first_valid_date"] is None
    assert "historical contract metadata versions" in manifest["metadata_vintage_limitation"]


@pytest.mark.parametrize("include_resolution", [False, True])
def test_prepare_per_market_alignment_matches_joint_history_and_global_groups(
        tmp_path, monkeypatch, include_resolution):
    import polymarket_ext.factors as factor_module

    days = pd.bdate_range("2026-01-05", periods=7)
    markets = [market("threshold_low", threshold=60),
               market("threshold_middle", threshold=80),
               market("threshold_high", threshold=100),
               market("range_left", "range", lower=-np.inf, upper=60),
               market("range_middle", "range", lower=60, upper=100),
               market("range_right", "range", lower=100, upper=np.inf),
               market("unquoted", "record", record_threshold=147.27)]
    probabilities = [[.9] * 7, [.6] * 7, [.2] * 7,
                     [.1, .2, .3, .4, .5, .6, .7],
                     [.8, .7, .6, .5, .4, .3, .2], [.1] * 7, []]
    histories = []
    for index, (item, values) in enumerate(zip(markets, probabilities)):
        history = quotes([item], days, [values]).reindex(columns=["token_id", "timestamp", "price"])
        if item["market_id"] == "threshold_middle":
            history = history.drop(index=5)
        if item["market_id"] == "range_middle":
            history = history.drop(index=2)
        # Mix modern and legacy CSVs in the same preparation run, and check
        # separately that an entirely legacy collection is still accepted.
        if include_resolution and index % 2 == 0:
            history["resolution_seconds"] = 60
            history["source"] = "data_v2"
            history["timestamp"] -= 60
        history["unused_api_payload"] = "not a factor input"
        directory = tmp_path / "data" / item["market_dir"]
        directory.mkdir(parents=True)
        history.drop(columns="token_id").to_csv(directory / "yes_prices.csv", index=False)
        histories.append(history)
    (tmp_path / "data/markets.jsonl").write_text(
        "".join(json.dumps(item) + "\n" for item in markets))
    calendar_file = tmp_path / "calendar.txt"
    calendar_file.write_text("\n".join(days.strftime("%Y-%m-%d")))
    config = {"data_dir": str(tmp_path / "data"), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(calendar_file), "max_quote_age": "1h",
              "research_start": str(days[5].date()), "research_end": str(days[6].date())}

    # The public functions provide the original all-histories-at-once oracle.
    expected_aligned = align_prices(pd.DataFrame(markets), pd.concat(histories, ignore_index=True),
                                    days, max_quote_age="1h")
    expected_factors = compute_factors(expected_aligned, days)
    calls = []

    def align_one_market(metadata, history, calendar, max_quote_age=None):
        assert len(metadata) == 1
        assert set(history.columns) <= {"timestamp", "price", "resolution_seconds", "source", "token_id"}
        assert set(history["token_id"]) <= {metadata.iloc[0]["token_id"]}
        calls.append(metadata.iloc[0]["market_id"])
        return align_prices(metadata, history, calendar, max_quote_age)

    monkeypatch.setattr(factor_module, "align_prices", align_one_market)
    destination = prepare(config)
    for name, expected in (("aligned", expected_aligned), ("factors", expected_factors)):
        expected = expected.loc[expected["datetime"] >= days[5]].reset_index(drop=True)
        pd.testing.assert_frame_equal(pd.read_parquet(destination / f"{name}.parquet"),
                                      expected, check_dtype=False)
    assert calls == [item["market_id"] for item in markets]

    result = pd.read_parquet(destination / "factors.parquet").set_index(["datetime", "market_id"])
    first, last = result.loc[days[5]], result.loc[days[6]]
    # Missing quotes retain their market and its boundary; adjacency never skips it.
    assert np.isnan(first.loc["threshold_middle", "PROB"])
    assert np.isnan(first.loc["threshold_low", "THRESHOLD_ADJACENT_MASS"])
    assert first.loc["threshold_middle", "THRESHOLD_REL_POSITION"] == 0
    assert last.loc["threshold_low", "THRESHOLD_ADJACENT_MASS"] == pytest.approx(.3)
    assert last.loc["threshold_low", "THRESHOLD_REL_DENSITY"] == pytest.approx(1.2)
    range_rows = first.loc[["range_left", "range_middle", "range_right"]]
    assert range_rows["group_center"].eq(80).all()
    assert range_rows["RANGE_REL_WIDTH"].sum() == pytest.approx(1)
    assert first.loc["range_left", "D5"] == pytest.approx(.5)
    assert last.loc["range_left", "D1"] == pytest.approx(.1)
    assert first.loc["range_middle", "D1"] == pytest.approx(-.1)
    assert result.xs("range_middle", level="market_id")["D5"].isna().all()
    assert result.xs("unquoted", level="market_id")[list(CATEGORY_FACTORS["record"])].isna().all().all()
    if include_resolution:
        assert first.loc["threshold_low", "quote_resolution_seconds"] == 60
        assert first.loc["threshold_low", "quote_source"] == "data_v2"
        assert first.loc["threshold_low", "quote_available_at"] == first.loc["threshold_low", "cutoff"]
    else:
        assert result["quote_resolution_seconds"].isna().all()


def test_prepare_direction_times_cross_research_end_and_reject_legacy_catalog(tmp_path):
    days = pd.DatetimeIndex(["2026-09-28", "2026-09-29", "2026-09-30"])
    item = market("direction", "direction", observation_start="2026-09-01T00:00:00Z",
                  observation_end="2026-10-01T00:00:00Z", expires_at="2026-10-02T00:00:00Z")
    directory = tmp_path / item["market_dir"]
    directory.mkdir(parents=True)
    quotes([item], days, [[np.nan] * len(days)]).to_csv(directory / "yes_prices.csv", index=False)
    catalog = tmp_path / "markets.jsonl"
    catalog.write_text(json.dumps(item) + "\n")
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("\n".join(days.strftime("%Y-%m-%d")))
    config = {"data_dir": str(tmp_path), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(calendar), "research_start": "2026-09-29", "research_end": "2026-09-30"}
    path = prepare(config)
    result = pd.read_parquet(path / "factors.parquet")
    assert result["datetime"].tolist() == list(days[-2:])
    assert result["PROB"].isna().all()
    np.testing.assert_allclose(result["WINDOW_LENGTH"], np.log1p(30))
    np.testing.assert_allclose(result["TIME_TO_END"], np.log1p(np.array([41, 17]) / 24))
    assert json.loads((path / "manifest.json").read_text())["version"] == 3
    # A previously created panel cannot bypass reclassification of old timestamps.
    item.pop("time_extraction_version")
    catalog.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="outdated catalog; rerun classify"):
        prepare(config)
    item["time_extraction_version"] = 0
    catalog.write_text(json.dumps(item) + "\n")
    with pytest.raises(ValueError, match="outdated catalog; rerun classify"):
        prepare(config, refresh=True)


def test_empty_catalog_produces_auditable_empty_panels(tmp_path):
    (tmp_path / "markets.jsonl").write_text("")
    (tmp_path / "calendar.txt").write_text("2026-01-05\n")
    config = {"data_dir": str(tmp_path), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(tmp_path / "calendar.txt"),
              "research_start": "2026-01-05", "research_end": "2026-01-05"}
    path = prepare(config)
    result = pd.read_parquet(path / "factors.parquet")
    assert result.empty
    assert set(FACTOR_COLUMNS).issubset(result.columns)


def test_prepare_rejects_incomplete_collection_even_if_a_cached_panel_exists(tmp_path):
    days = pd.bdate_range("2026-01-05", periods=2)
    item = market("direction", "direction")
    directory = tmp_path / item["market_dir"]
    directory.mkdir(parents=True)
    quotes([item], days, [[.4, .6]]).to_csv(directory / "yes_prices.csv", index=False)
    (tmp_path / "markets.jsonl").write_text(json.dumps(item) + "\n")
    (tmp_path / "calendar.txt").write_text("\n".join(days.strftime("%Y-%m-%d")))
    config = {"data_dir": str(tmp_path), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(tmp_path / "calendar.txt"),
              "research_start": "2026-01-05", "research_end": "2026-01-06"}
    prepare(config)
    collection_path = tmp_path / "collection.json"
    collection_path.write_text(json.dumps({"complete": False, "failed_markets": ["direction"]}))
    with pytest.raises(ValueError, match="Collection is incomplete"):
        prepare(config)
    collection_path.write_text(json.dumps({"complete": True, "failed_markets": []}))
    with pytest.raises(ValueError, match="no completion metadata"):
        prepare(config)
    coverage_path = directory / "coverage.json"
    coverage_path.write_text(json.dumps({"api_complete": False, "yes": {"api_complete": True}}))
    with pytest.raises(ValueError, match="Market collection is incomplete"):
        prepare(config)
    coverage_path.write_text(json.dumps({"api_complete": True, "yes": {"api_complete": True}}))
    prepare(config)
    manifest = json.loads((tmp_path / "panel/manifest.json").read_text())
    assert manifest["source_warnings"] == []
    price_path = directory / "yes_prices.csv"
    info = price_path.stat()
    coverage_path.write_text(json.dumps({"api_complete": True, "yes": {"api_complete": True},
                                        "artifacts": {"yes_prices.csv": {"size": info.st_size, "mtime_ns": info.st_mtime_ns}}}))
    price_path.write_text(price_path.read_text() + "\n")
    with pytest.raises(ValueError, match="no longer matches completed coverage"):
        prepare(config)
    config["research_end"] = "2026-01-07"
    with pytest.raises(ValueError, match="Calendar must cover"):
        prepare(config)


def test_prepare_can_omit_not_yet_open_future_markets_without_a_history(tmp_path):
    item = market("future", opened_at="2026-05-01T00:00:00Z", observation_end="2026-06-01T00:00:00Z")
    (tmp_path / "markets.jsonl").write_text(json.dumps(item) + "\n")
    (tmp_path / "calendar.txt").write_text("2026-01-05\n")
    (tmp_path / "collection.json").write_text(json.dumps({"complete": True, "failed_markets": [], "skipped_future_markets": ["future"]}))
    config = {"data_dir": str(tmp_path), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(tmp_path / "calendar.txt"),
              "research_start": "2026-01-05", "research_end": "2026-01-05"}
    path = prepare(config)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["catalog_markets"] == 1
    assert manifest["markets"] == 0
    assert pd.read_parquet(path / "factors.parquet").empty
