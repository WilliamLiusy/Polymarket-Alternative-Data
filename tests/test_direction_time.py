"""Direction comparisons use contract price times, independent of listing times."""
import json

import numpy as np
import pandas as pd
import pytest

from polymarket_ext.classify import classify, classify_market
from polymarket_ext.direction_time import extract_window


def source(rule, question="WTI Up or Down from June 1 to June 30?"):
    market = {"id": "m", "question": question, "description": rule,
              "outcomes": ["Up", "Down"], "clobTokenIds": ["u", "d"], "conditionId": "c",
              "createdAt": "2026-05-01T00:00:00Z", "startDate": "2026-05-02T00:00:00Z",
              "eventStartTime": "2026-05-03T00:00:00Z", "endDate": "2026-07-10T00:00:00Z"}
    return {"id": "e", "title": question, "markets": [market]}, market


def test_explicit_multiday_interval_and_evidence():
    event, market = source("WTI increases from June 1, 2026 at 4:00 PM ET to June 30, 2026 at 4:00 PM ET.")
    row, reason = classify_market(event, market)
    assert reason is None and row["category"] == "direction"
    assert row["observation_start"] == "2026-06-01T20:00:00+00:00"
    assert row["observation_end"] == "2026-06-30T20:00:00+00:00"
    assert row["opened_at"] == market["startDate"].replace("Z", "+00:00")
    assert row["expires_at"] == market["endDate"].replace("Z", "+00:00")
    assert row["time_extraction"]["status"] == "complete"
    assert row["time_extraction"]["start_evidence"][0]["text"] == market["description"]


@pytest.mark.parametrize("zone,hour", [("ET", "20"), ("EDT", "20"), ("EST", "21"), ("UTC", "16")])
def test_timezones_are_not_interchangeable(zone, hour):
    event, market = source(f"Up if WTI rises from June 1, 2026 at 4:00 PM {zone} to June 2, 2026 at 4:00 PM {zone}.")
    row = extract_window(event, market)
    assert row["observation_end"] == f"2026-06-02T{hour}:00:00+00:00"


def test_iso_window_and_comparison_order():
    event, market = source("Up if the price on 2026-07-01T00:00:00Z is higher than the price on 2026-06-01T00:00:00Z.")
    row = extract_window(event, market)
    assert row["observation_start"] == "2026-06-01T00:00:00+00:00"
    assert row["observation_end"] == "2026-07-01T00:00:00+00:00"
    market["description"] = "Up if the price on 2026-06-01T00:00:00Z is lower than the price on 2026-07-01T00:00:00Z."
    assert extract_window(event, market)["observation_end"] == "2026-07-01T00:00:00+00:00"


@pytest.mark.parametrize("target,previous,end", [
    ("March 9, 2026", "2026-03-06T22:00:00+00:00", "2026-03-09T21:00:00+00:00"),
    ("April 6, 2026", "2026-04-02T21:00:00+00:00", "2026-04-06T21:00:00+00:00"),
    ("July 6, 2026", "2026-07-03T18:30:00+00:00", "2026-07-06T21:00:00+00:00"),
])
def test_previous_cme_session_handles_weekend_dst_holiday_and_early_close(target, previous, end):
    rule = (f'Up if the Close price for WTI on {target} is higher than the Close price on '
            'the most recent prior trading day. The primary exchange is CME NYMEX.')
    event, market = source(rule)
    row = extract_window(event, market)
    assert row["observation_start"] == previous
    assert row["observation_end"] == end


def test_open_to_close_and_monthly_session_endpoints():
    event, market = source("Up if the closing price exceeds the opening price on June 1, 2026. CME NYMEX CL futures.")
    row = extract_window(event, market)
    assert row["observation_start"] == "2026-05-31T22:00:00+00:00"
    assert row["observation_end"] == "2026-06-01T21:00:00+00:00"
    market["description"] = ("Up if the closing price on the last trading day of June 2026 is higher than "
                             "the closing price on the first trading day of June 2026. CME NYMEX CL futures.")
    row = extract_window(event, market)
    assert row["observation_start"] == "2026-06-01T21:00:00+00:00"
    assert row["observation_end"] == "2026-06-30T21:00:00+00:00"


@pytest.mark.parametrize("rule", [
    "WTI Up or Down this month?",
    "Up if WTI rises from June 1, 2026 to June 30, 2026.",
    "Up if WTI rises from June 1, 2026 at 4:00 PM to June 30, 2026 at 4:00 PM.",
    "Up if the closing price on June 1, 2026 is higher than the prior trading day. ICE Futures.",
    "Up if the settlement price on June 1, 2026 is higher than the previous trading day. CME NYMEX.",
    "Orders can be placed from June 1, 2026 at 9:00 AM ET to June 30, 2026 at 4:00 PM ET.",
    "This Up/Down market accepts orders from June 1, 2026 at 9:00 AM ET to June 30, 2026 at 4:00 PM ET.",
    "Up if opening price on June 1, 2026 is lower than closing price on the last trading day of June 2026. CME NYMEX.",
    "Up if opening price on June 1 is lower than closing price on June 30, 2026. CME NYMEX.",
])
def test_uncertain_times_never_fall_back_to_platform_dates(rule):
    event, market = source(rule)
    row, reason = classify_market(event, market)
    assert row is not None and reason is None
    assert row["observation_start"] is None and row["observation_end"] is None


def test_partial_endpoint_and_verified_override():
    event, market = source("Up if the closing price on June 30, 2026 at 4:00 PM ET exceeds the reference price.")
    row = extract_window(event, market)
    assert row["observation_start"] is None
    assert row["observation_end"] == "2026-06-30T20:00:00+00:00"
    row = extract_window(event, market, {"observation_start": "2026-06-01T16:00:00-04:00"})
    assert row["observation_start"] == "2026-06-01T20:00:00+00:00"
    assert row["time_extraction"]["start_evidence"][0]["source"] == "override"
    with pytest.raises(ValueError, match="timezone"):
        extract_window(event, market, {"observation_start": "2026-06-01"})


def test_conflicting_rules_are_flagged_and_manual_override_can_resolve():
    event, market = source("Up from June 1, 2026 at 4:00 PM ET to June 30, 2026 at 4:00 PM ET.")
    event["description"] = "Up from June 2, 2026 at 4:00 PM ET to June 30, 2026 at 4:00 PM ET."
    row = extract_window(event, market)
    assert row["observation_start"] is None and row["observation_end"] is not None
    assert row["time_extraction"]["issues"][0]["reason"] == "conflicting_endpoints"
    row = extract_window(event, market, {"observation_start": "2026-06-01T20:00:00Z"})
    assert row["time_extraction"]["status"] == "complete"


@pytest.mark.parametrize("question", [
    "WTI Up or Down this week?", "WTI Up or Down this month?",
    "WTI Up or Down from June 1 to June 30?", "WTI Up or Down on June 1, 2026 at 4:00 PM ET?",
    "WTI Up or Down over 48 hours?",
])
def test_multiday_and_daily_with_clock_are_included(question):
    event, market = source("", question)
    row, reason = classify_market(event, market)
    assert row is not None and reason is None


@pytest.mark.parametrize("question", ["WTI Up or Down next 4 hours?", "WTI Up or Down hourly?", "WTI Up or Down next 15 minutes?",
    "WTI Up or Down 15m?", "WTI Up or Down 1h?",
    "WTI Up or Down from June 1, 2026 at 9:00 AM ET to June 1, 2026 at noon ET?"])
def test_explicit_intraday_products_remain_out_of_scope(question):
    event, market = source("", question)
    row, reason = classify_market(event, market)
    assert row is None and reason == "intraday_direction"


def test_intraday_rule_is_not_hidden_by_daily_looking_title():
    event, market = source("Up if price increases from June 1, 2026 at 9:00 AM ET to June 1, 2026 at 10:00 AM ET.",
                           "WTI Up or Down on June 1, 2026?")
    row, reason = classify_market(event, market)
    assert row is None and reason == "intraday_direction"


def test_price_candle_resolution_does_not_exclude_multiday_market():
    event, market = source("Up if the 1-minute candle price on 2026-07-01T00:00:00Z is higher than the 1-minute candle price on 2026-06-01T00:00:00Z.")
    row, reason = classify_market(event, market)
    assert row is not None and reason is None
    assert row["observation_end"] == "2026-07-01T00:00:00+00:00"


@pytest.mark.parametrize("start", ["March 8, 2026 at 2:30 AM ET", "November 1, 2026 at 1:30 AM ET"])
def test_dst_ambiguity_is_missing_without_losing_known_end(start):
    event, market = source(f"Compare the price from {start} to December 1, 2026 at 4:00 PM ET.")
    row = extract_window(event, market)
    assert row["observation_start"] is None
    assert row["observation_end"] == "2026-12-01T21:00:00+00:00"


def test_each_endpoint_uses_its_own_price_type():
    event, market = source("Compare the opening price on the first trading day of June 2026 to the closing price on the last trading day of June 2026. CME NYMEX.")
    row = extract_window(event, market)
    assert row["observation_start"] == "2026-05-31T22:00:00+00:00"
    assert row["observation_end"] == "2026-06-30T21:00:00+00:00"
    market["description"] = "Up if the closing price on June 2, 2026 is higher than the opening price on the previous trading day. CME NYMEX."
    row = extract_window(event, market)
    assert row["observation_start"] == "2026-05-31T22:00:00+00:00"
    assert row["observation_end"] == "2026-06-02T21:00:00+00:00"


def test_explicit_clock_before_date_is_not_replaced_by_exchange_close():
    event, market = source("Up if the closing price at 2:30 PM ET on June 2, 2026 exceeds the reference price. CME NYMEX.")
    row = extract_window(event, market)
    assert row["observation_end"] == "2026-06-02T18:30:00+00:00"


def test_reversed_interval_preserves_independent_end_for_time_to_end():
    event, market = source("Up from June 30, 2026 at noon UTC to June 1, 2026 at noon UTC.")
    row = extract_window(event, market)
    assert row["observation_start"] is None
    assert row["observation_end"] == "2026-06-01T12:00:00+00:00"
    assert row["time_extraction"]["issues"][0]["reason"] == "nonpositive_comparison_window"


def test_review_keeps_incomplete_market_and_includes_evidence(tmp_path):
    event, _ = source("")
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "events.jsonl").write_text(json.dumps(event) + "\n")
    report = classify({"data_dir": str(tmp_path)})
    assert report["selected_markets"] == 1
    assert report["direction_time_status"] == {"missing": 1}
    row = json.loads((tmp_path / "markets.jsonl").read_text())
    assert row["time_extraction_version"] == 1
    review = json.loads((tmp_path / "classification_review.json").read_text())
    assert review[0]["included"] is True


def test_classification_to_factor_panel_preserves_time_without_quotes(tmp_path):
    from polymarket_ext.factors import prepare

    event, market = source("Up from June 1, 2026 at noon UTC to June 30, 2026 at noon UTC.")
    event["markets"].append({**market, "id": "unknown", "clobTokenIds": ["u2", "d2"], "description": ""})
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "events.jsonl").write_text(json.dumps(event) + "\n")
    classify({"data_dir": str(tmp_path)})
    rows = [json.loads(line) for line in (tmp_path / "markets.jsonl").read_text().splitlines()]
    for row in rows:
        pd.DataFrame(columns=["timestamp", "price"]).to_csv(tmp_path / row["market_dir"] / "yes_prices.csv", index=False)
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("2026-05-29\n2026-06-01\n2026-06-02\n")
    directory = prepare({"data_dir": str(tmp_path), "panel_dir": str(tmp_path / "panel"),
                         "calendar_path": str(calendar), "research_start": "2026-06-01",
                         "research_end": "2026-06-02"})
    panel = pd.read_parquet(directory / "factors.parquet")
    known = panel.loc[panel.market_id == "m"]
    unknown = panel.loc[panel.market_id == "unknown"]
    assert not panel.quote_valid.any() and panel.PROB.isna().all()
    np.testing.assert_allclose(known.WINDOW_LENGTH, np.log1p(29))
    np.testing.assert_allclose(known.TIME_TO_END, np.log1p([29 + 5 / 24, 28 + 5 / 24]))
    assert unknown.WINDOW_LENGTH.isna().all() and unknown.TIME_TO_END.isna().all()
    assert len(known) == len(unknown) == 2
