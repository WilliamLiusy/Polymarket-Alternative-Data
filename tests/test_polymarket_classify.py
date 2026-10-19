import copy
import json

import pandas as pd
import pytest

from polymarket_ext.classify import classify_event, classify_market


def event(question, **extra):
    market = {"id": "m1", "question": question, "conditionId": "condition",
              "outcomes": '["No", "Yes"]', "clobTokenIds": '["n", "y"]',
              "createdAt": "2026-06-01T12:00:00Z", "startDate": "2026-06-01T13:00:00Z",
              "endDate": "2026-07-01T00:00:00Z", **extra}
    return {"id": "e1", "title": question, "startDate": "2026-06-01T12:00:00Z", "markets": [market]}


def test_outcome_names_not_token_order_and_closed_market_retained():
    source = event("Will Crude Oil (CL) settle over $60 on the final trading day of June 2026?", closed=True)
    row = classify_event(source)[0]
    assert row["category"] == "threshold"
    assert row["threshold"] == 60
    assert row["token_id"] == "y"
    assert row["no_token_id"] == "n"
    assert row["opened_at"] == "2026-06-01T13:00:00+00:00"


def test_daily_direction_uses_up_and_excludes_gasoline():
    source = event("WTI Up or Down on June 2?", outcomes=["Down", "Up"], clobTokenIds=["down", "up"])
    assert classify_event(source)[0]["token_id"] == "up"
    assert not classify_event(event("Will gasoline close above $4?"))
    assert not classify_event(event("Will Brent crude oil close above $90?"))
    monthly = classify_event(event("WTI Up or Down this month?", outcomes=["Up", "Down"]))[0]
    assert monthly["category"] == "direction"
    assert monthly["observation_start"] is None and monthly["observation_end"] is None
    assert monthly["time_extraction_version"] == 1


def test_touch_high_low_and_ambiguous_hit_requires_override():
    for verb, side in (("reach", "high"), ("dip to", "low")):
        row = classify_event(event(f"Will WTI {verb} $90 by June 30?"))[0]
        assert (row["category"], row["side"]) == ("touch", side)
    source = event("Will WTI hit $90 by June 30?")
    assert not classify_event(source)
    assert classify_event(source, {"m1": {"category": "touch", "threshold": 90, "side": "high"}})[0]["side"] == "high"


def test_finite_ranges_and_open_tails_share_category():
    source = event("What price will Crude Oil (CL) settle at end of June?")
    questions = ["Will Crude Oil settle below $60?", "Will Crude Oil settle between $60 and $70?", "Will Crude Oil settle above $70?"]
    template = source["markets"][0]
    source["markets"] = [{**copy.deepcopy(template), "id": str(i), "question": q} for i, q in enumerate(questions)]
    rows = classify_event(source)
    assert [row["category"] for row in rows] == ["range"] * 3
    assert [(row["lower"], row["upper"]) for row in rows] == [("-inf", 60), (60, 70), (70, "inf")]


def test_record_requires_rule_threshold_and_does_not_extract_year():
    source = event("Will WTI reach an all-time high in 2026?")
    assert not classify_event(source)
    source["markets"][0]["description"] = "The historical record high is $147.27, reached in 2008."
    row = classify_event(source)[0]
    assert row["category"] == "record" and row["record_threshold"] == 147.27


@pytest.mark.parametrize("deadline", (
    "March 2026", "April 2026", "on or before the specified date",
))
@pytest.mark.parametrize("threshold", (147.27, 151.75))
def test_record_extracts_explicit_cme_yes_condition_without_record_word(deadline, threshold):
    rule = (
        'This market will resolve to "Yes" if, on any trading day after market creation, '
        "the official daily high price published by the CME Group for the Active Month "
        f"(front month) of CME Crude Oil (CL) futures is greater than ${threshold} "
        f'by the final trading day of {deadline}. Otherwise, this market will resolve to "No".\n\n'
        "For example, if the spot month expires on a Friday the next listed contract "
        "will be considered the Active Month on the Wednesday prior to expiration."
    )
    row = classify_event(event("Crude Oil all time high by March 31?", description=rule))[0]
    assert row["category"] == "record"
    assert row["record_threshold"] == threshold


@pytest.mark.parametrize("rule", (
    'This market will resolve to "Yes" if crude oil makes a new high in 2026. '
    'Otherwise, this market will resolve to "No". Example: the high price is greater than $100.',
    'This market will resolve to "Yes" if the daily high price is greater than $100 '
    'or the daily high price is greater than $200. Otherwise, this market will resolve to "No".',
))
def test_record_does_not_infer_threshold_from_unrelated_or_ambiguous_prices(rule):
    source = event("Will WTI reach an all-time high in 2026?", description=rule)
    row, reason = classify_market(source, source["markets"][0])
    assert row is None and reason == "missing_record_threshold"


def test_missing_lifecycle_is_reviewed_and_touch_window_is_preserved():
    source = event("Will WTI reach $100 by July?", description="From June 3, 2026 at 09:30 AM ET until July 1, 2026.")
    assert classify_event(source)[0]["observation_start"] == "2026-06-03T13:30:00+00:00"
    source["markets"][0].pop("endDate")
    row, reason = classify_market(source, source["markets"][0])
    assert row is None and reason == "missing_identity_or_lifecycle"


def test_intraperiod_above_is_touch_and_multiple_barriers_are_rejected():
    source = event("Will CL be above $100 at any time in July?")
    row = classify_event(source)[0]
    assert (row["category"], row["side"]) == ("touch", "high")
    assert not classify_event(event("Will CL be above $100 by July?"))
    assert not classify_event(event("Will CL reach $100 or fall below $50?"))


def test_explicit_contracts_do_not_share_price_basis():
    first = event("Will CL close above $100?", description="Settlement price of CLZ6")
    second = event("Will CL close above $100?", description="Settlement price of CLF7")
    assert classify_event(first)[0]["price_basis"] != classify_event(second)[0]["price_basis"]


def test_monthly_touch_uses_observation_session_and_separates_late_barriers():
    rule = ("At any point after market creation and during a trading session of October 2026. "
            "Active Month of ICE Futures Europe WTI Crude Futures. Sessions begin at 8:00 PM ET.")
    source = event("Will WTI hit (HIGH) $150 in October?", description=rule,
                   createdAt="2026-09-25T04:02:01Z", endDate="2026-11-01T03:59:59Z")
    first = classify_event(source)[0]
    assert first["observation_start"] == "2026-10-01T00:00:00+00:00"
    source["markets"][0]["createdAt"] = "2026-10-03T10:00:00Z"
    assert classify_event(source)[0]["observation_start"] == "2026-10-03T10:00:00+00:00"


def test_import_classify_prepare_preserves_open_range_bounds(tmp_path):
    from polymarket_ext.classify import classify
    from polymarket_ext.collect import discover
    from polymarket_ext.factors import prepare

    source = event("What price will Crude Oil (CL) settle at end of June?")
    template = source["markets"][0]
    source["markets"] = [{**template, "id": str(i), "question": question,
                           "createdAt": "2026-06-01T00:00:00Z", "startDate": "2026-06-01T01:00:00Z",
                           "clobTokenIds": [f"no-{i}", f"yes-{i}"]}
                          for i, question in enumerate(("Will CL settle below $60?",
                              "Will CL settle between $60 and $70?", "Will CL settle above $70?"))]
    source_path = tmp_path / "events.json"
    source_path.write_text(json.dumps([source]))
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("2026-05-29\n2026-06-01\n2026-06-02\n")
    root = tmp_path / "raw_data"
    config = {"data_dir": str(root), "panel_dir": str(tmp_path / "panels"),
              "collection": {"events_file": str(source_path)}, "calendar_path": str(calendar),
              "research_start": "2026-06-01", "research_end": "2026-06-02"}
    assert discover(config)["event_count"] == 1
    assert classify(config)["categories"] == {"range": 3}
    for row in map(json.loads, (root / "markets.jsonl").read_text().splitlines()):
        pd.DataFrame({"timestamp": [1780297140], "price": [0.3]}).to_csv(
            root / row["market_dir"] / "yes_prices.csv", index=False)
    output = prepare(config)
    panel = pd.read_parquet(output / "factors.parquet")
    first_day = panel.loc[panel.datetime == "2026-06-01"]
    assert len(first_day) == 3
    assert first_day.RANGE_REL_WIDTH.sum() == pytest.approx(1.0)
    assert first_day.RANGE_REL_DENSITY.notna().all()
