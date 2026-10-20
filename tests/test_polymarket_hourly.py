import json

import numpy as np
import pandas as pd
import pytest

from polymarket_ext.factors import (
    CATEGORY_FACTORS, HOURLY_FACTORS, HOURLY_LIMITS, align_prices, compute_factors, prepare,
)
from polymarket_ext.model import PolymarketModel


DAY = pd.DatetimeIndex(["2026-01-05"])
CLOSE = pd.Timestamp("2026-01-05T15:00:00+08:00")


def metadata(category="threshold", **overrides):
    return {
        "market_id": "m", "token_id": "yes", "event_id": "event", "category": category,
        "opened_at": "2026-01-01T00:00:00Z", "metadata_available_at": "2026-01-01T00:00:00Z",
        "observation_start": "2026-01-01T00:00:00Z", "observation_end": "2026-02-01T00:00:00Z",
        "closed_at": None, "expires_at": None, "side": None, "threshold": 80,
        "lower": 70, "upper": 80, "underlying": "WTI", "price_basis": "settlement",
        "market_dir": "market", "time_extraction_version": 1, **overrides,
    }


def quote(time, price, resolution=60, source="clob", token="yes"):
    stamp = pd.Timestamp(f"2026-01-05T{time}:00+08:00") if isinstance(time, str) else time
    return {"token_id": token, "timestamp": int(stamp.timestamp()), "price": price,
            "resolution_seconds": resolution, "source": source}


def factors(rows, item=None, **kwargs):
    aligned = align_prices(pd.DataFrame([metadata() if item is None else item]),
                           pd.DataFrame(rows), DAY, **kwargs)
    return compute_factors(aligned, DAY)


@pytest.mark.parametrize("category,side,prefix", [
    ("direction", None, ""), ("threshold", None, ""), ("range", None, ""),
    ("touch", "high", "HIGH_"), ("touch", "low", "LOW_"), ("record", None, "HIGH_"),
])
def test_hourly_changes_use_elapsed_endpoints_and_correct_category(category, side, prefix):
    rows = [quote("09:00", .2), quote("10:30", .9), quote("12:00", .4),
            quote("14:00", .65), quote("14:01", .8), quote("15:00", .6), quote("15:01", .99)]
    result = factors(rows, metadata(category, side=side)).iloc[0]
    for hours, expected in ((1, -.05), (3, .2), (6, .4)):
        assert result[f"{prefix}H{hours}"] == pytest.approx(expected)
    if category == "touch":
        other = "LOW_" if side == "high" else "HIGH_"
        assert all(pd.isna(result[f"{other}H{hours}"]) for hours in HOURLY_LIMITS)
    assert result["hourly_1h_quote_time"] == pd.Timestamp("2026-01-05T14:00:00+08:00")


def test_zero_change_is_observed_and_intermediate_gaps_do_not_block_h6():
    result = factors([quote("09:00", .2), quote("12:00", np.nan),
                      quote("14:00", np.nan), quote("15:00", .2)]).iloc[0]
    assert result["H6"] == 0
    assert pd.isna(result["H1"]) and pd.isna(result["H3"])


def test_bucket_end_age_is_independent_of_daily_age_and_h1_resolution_limit():
    rows = [quote("05:00", .1, 10800, "data_v2"), quote("08:00", .2, 10800, "data_v2"),
            quote("11:00", .6, 10800, "data_v2"), quote("14:00", .99, 10800, "data_v2")]
    result = factors(rows, max_quote_age="30min").iloc[0]
    assert pd.isna(result["PROB"]) and pd.isna(result["H1"])
    assert result["H3"] == pytest.approx(.4)
    assert result["H6"] == pytest.approx(.5)
    assert result["quote_age_seconds"] == 4 * 3600
    assert result["hourly_0h_age_seconds"] == 3600
    assert result["hourly_0h_quote_available_at"] == pd.Timestamp("2026-01-05T14:00:00+08:00")


def test_completed_bucket_boundary_is_inclusive_at_each_endpoint():
    rows = [quote("13:00", .2, 3600, "data_v2"), quote("13:01", .99, 3600, "data_v2"),
            quote("14:00", .6, 3600, "data_v2"), quote("14:01", .99, 3600, "data_v2")]
    result = factors(rows).iloc[0]
    assert result["H1"] == pytest.approx(.4)
    assert result["hourly_1h_price"] == .2
    assert result["hourly_0h_price"] == .6


@pytest.mark.parametrize("resolution", [0, 3600, 3601, 10800, 10801, np.nan, np.inf, -1])
def test_hourly_resolution_limits_and_unknown_resolution(resolution):
    rows = [quote("09:00", .1, resolution), quote("12:00", .2, resolution),
            quote("14:00", .3, resolution), quote("15:00", .4, resolution)]
    result = factors(rows).iloc[0]
    for hours, limit in HOURLY_LIMITS.items():
        assert np.isfinite(result[f"H{hours}"]) == (0 <= resolution <= limit)
    # Invalid hourly resolution does not change the pre-existing daily probability.
    assert result["PROB"] == .4


@pytest.mark.parametrize("hours", [1, 3, 6])
@pytest.mark.parametrize("offset", [0, 1])
def test_hourly_age_limit_is_inclusive_at_both_endpoints(hours, offset):
    age = pd.Timedelta(seconds=HOURLY_LIMITS[hours] + offset)
    result = factors([quote(CLOSE - age - pd.Timedelta(hours=hours), .2),
                      quote(CLOSE - age, .6)]).iloc[0]
    assert np.isfinite(result[f"H{hours}"]) == (offset == 0)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -0.1, 1.1])
@pytest.mark.parametrize("endpoint", ["current", "lag"])
def test_invalid_latest_endpoint_never_falls_back_to_an_older_valid_quote(bad, endpoint):
    rows = [quote("09:00", .1), quote("12:00", .2), quote("13:59", .3), quote("14:00", .4),
            quote("14:59", .5), quote("15:00", .6)]
    rows[-1 if endpoint == "current" else 3]["price"] = bad
    result = factors(rows).iloc[0]
    assert pd.isna(result["H1"])
    if endpoint == "lag":
        assert result["H3"] == pytest.approx(.4)
        assert result["H6"] == pytest.approx(.5)


def test_hourly_lags_require_market_and_metadata_availability():
    rows = [quote("09:00", .1), quote("12:00", .2), quote("14:00", .3), quote("15:00", .4)]
    opened = factors(rows, metadata(opened_at="2026-01-05T10:00:00+08:00")).iloc[0]
    assert pd.isna(opened["H6"]) and opened["H3"] == pytest.approx(.2)
    late_metadata = factors(rows, metadata(metadata_available_at="2026-01-05T13:00:00+08:00")).iloc[0]
    assert pd.isna(late_metadata["H3"]) and late_metadata["H1"] == pytest.approx(.1)
    for boundary in ("observation_end", "expires_at", "closed_at"):
        assert factors(rows, metadata(**{boundary: CLOSE.isoformat()})).empty


def test_hourly_endpoints_do_not_cross_tokens_or_use_future_quotes():
    items = pd.DataFrame([metadata(), metadata(market_id="other", token_id="other")])
    rows = [quote("15:00", .6), quote("09:00", .2, token="other")]
    result = compute_factors(align_prices(items, pd.DataFrame(rows), DAY), DAY)
    assert result["H6"].isna().all()
    result = factors([quote("09:00", .1), quote(CLOSE - pd.Timedelta(minutes=59, seconds=59), .3),
                      quote("15:00", .6)]).iloc[0]
    assert pd.isna(result["H1"])


def test_delayed_older_bucket_does_not_displace_fine_hourly_observation():
    result = factors([quote("12:00", .99, 10800, "data_v2"), quote("13:59", .3),
                      quote("14:59", .6)]).iloc[0]
    assert result["H1"] == pytest.approx(.3)
    assert result["hourly_0h_quote_source"] == "clob"


@pytest.mark.parametrize("category", CATEGORY_FACTORS)
def test_model_rejects_missing_hourly_columns_but_accepts_nan_values(category):
    panel = pd.DataFrame([{"datetime": DAY[0], "category": category, "market_id": "m",
                           **dict.fromkeys(CATEGORY_FACTORS[category], np.nan)}])
    PolymarketModel(panel)
    for factor in HOURLY_FACTORS[category]:
        with pytest.raises(ValueError, match="hourly factor panel"):
            PolymarketModel(panel.drop(columns=factor))


def test_prepare_persists_hourly_evidence_coverage_and_cache_policy(tmp_path):
    item = metadata()
    (tmp_path / "market").mkdir()
    (tmp_path / "markets.jsonl").write_text(json.dumps(item) + "\n")
    (tmp_path / "calendar.txt").write_text("2026-01-05\n")
    pd.DataFrame([quote("09:00", .1), quote("12:00", .2), quote("14:00", .3),
                  quote("15:00", .4)]).to_csv(tmp_path / "market/yes_prices.csv", index=False)
    config = {"data_dir": str(tmp_path), "panel_dir": str(tmp_path / "panel"),
              "calendar_path": str(tmp_path / "calendar.txt"),
              "research_start": "2026-01-05", "research_end": "2026-01-05"}
    destination = prepare(config)
    manifest = json.loads((destination / "manifest.json").read_text())
    assert manifest["version"] == 4
    assert manifest["hourly_limits_seconds"] == {"1": 3600, "3": 10800, "6": 10800}
    assert manifest["hourly_age_basis"] == "quote_available_at"
    assert manifest["hourly_coverage"]["threshold"]["H6"] == {
        "valid_rows": 1, "markets": 1, "trading_days": 1,
    }
    assert manifest["hourly_coverage"]["touch"]["HIGH_H1"]["valid_rows"] == 0
    result = pd.read_parquet(destination / "factors.parquet")
    assert result.loc[0, "H6"] == pytest.approx(.3)
    assert result.loc[0, "hourly_6h_quote_resolution_seconds"] == 60
    before = (destination / "factors.parquet").stat().st_mtime_ns
    prepare(config)
    assert (destination / "factors.parquet").stat().st_mtime_ns == before
