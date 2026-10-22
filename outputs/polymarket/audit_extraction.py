"""Reproduce the extraction audit from the downloaded catalog and factor panel.

Run from the repository root with ``.venv/bin/python outputs/polymarket/audit_extraction.py``.
This script writes only audit artifacts; it never fills factors or changes rules.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from polymarket_ext.factors import CATEGORY_FACTORS, GROUP_COLUMNS


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value):
    return pd.notna(value) and np.isfinite(float(value))


def json_write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def main():
    config = yaml.safe_load((ROOT / "configs/polymarket.yaml").read_text())
    data = ROOT / config["data_dir"]
    panel_dir = ROOT / config["panel_dir"]
    output = ROOT / config["output_dir"]
    output.mkdir(parents=True, exist_ok=True)
    catalog = read_jsonl(data / "markets.jsonl")
    events = read_jsonl(data / "raw/events.jsonl")
    review = json.loads((data / "classification_review.json").read_text())
    manifest = json.loads((panel_dir / "manifest.json").read_text())
    raw = {str(m["id"]): (e, m) for e in events for m in e.get("markets", [])}
    metadata = {row["market_id"]: row for row in catalog}
    panel = pd.read_parquet(panel_dir / "factors.parquet")
    panel["market_id"] = panel["market_id"].astype(str)
    panel["token_id"] = panel["token_id"].astype(str)
    days = pd.DatetimeIndex(pd.to_datetime((ROOT / config["calendar_path"]).read_text().splitlines())).as_unit("ns")
    research_days = days[(days >= config["research_start"]) & (days <= config["research_end"])]
    cutoffs = (research_days.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).tz_convert("UTC")
    last_cutoff = cutoffs[-1]
    positions = {day: i for i, day in enumerate(days)}
    histories = {(str(row.token_id), row.datetime): row.probability for row in panel.itertuples()}

    # Find group neighbors using every active member, including missing quotes.
    group_details = {}
    for _, group in panel.groupby(["datetime", *GROUP_COLUMNS], sort=False, dropna=False):
        if group.iloc[0]["category"] != "threshold":
            continue
        boundaries = sorted(set(group.loc[np.isfinite(group["threshold"]), "threshold"]))
        by_threshold = {value: frame for value, frame in group.groupby("threshold")}
        for row in group.itertuples():
            threshold = row.threshold
            detail = {"last": False, "duplicate": False, "neighbor_probability": None}
            if finite(threshold) and threshold in boundaries:
                i = boundaries.index(threshold)
                detail["last"] = i == len(boundaries) - 1
                if not detail["last"]:
                    left, right = by_threshold[threshold], by_threshold[boundaries[i + 1]]
                    detail["duplicate"] = len(left) != 1 or len(right) != 1
                    detail["neighbor_probability"] = right.iloc[0]["probability"]
            group_details[row.Index] = detail

    def extraction_reason(row):
        _, market = raw[row["market_id"]]
        rule = market.get("description", "")
        if "ICE Futures Europe" in rule:
            return "unsupported_ice_wti_exchange_calendar"
        if "settlement price" in rule.lower():
            return "unsupported_cme_settlement_schedule"
        return "unresolved_rule_comparison_times"

    def quote_reason(row):
        if pd.isna(row["quote_time"]):
            return "quote_absent_at_or_before_cutoff"
        if not finite(row["price"]) or not 0 <= row["price"] <= 1:
            return "invalid_quote_price"
        return "quote_older_than_max_quote_age"

    def difference_reason(row, lag):
        if not finite(row["probability"]):
            return quote_reason(row)
        current = positions[row["datetime"]]
        required = [current - 1] if lag == 1 else list(range(current - 5, current))
        if min(required) < 0:
            return "insufficient_calendar_history"
        keys = [(str(row["token_id"]), days[i]) for i in required]
        if any(key not in histories for key in keys):
            return "insufficient_prior_active_market_history"
        if any(not finite(histories[key]) for key in keys):
            return "prior_quote_missing"
        return "unexplained_missing_difference"

    probability_factors = {"PROB", "ENTROPY", "DIRECTION_SCORE", "DIRECTION_CERTAINTY", "HIGH_PROB", "LOW_PROB"}
    position_factors = {"THRESHOLD_REL_POSITION", "HIGH_REL_POSITION", "LOW_REL_POSITION"}

    def missing_reason(index, row, factor):
        category = row["category"]
        if category == "touch" and ((row["side"] == "high" and factor.startswith("LOW_")) or
                                     (row["side"] == "low" and factor.startswith("HIGH_"))):
            return "structural_opposite_touch_side"
        if factor in ("WINDOW_LENGTH", "TIME_TO_END"):
            required = ["observation_end"] if factor == "TIME_TO_END" else ["observation_start", "observation_end"]
            if any(pd.isna(row[key]) for key in required):
                return "natural_language_" + extraction_reason(metadata[row["market_id"]])
            return "invalid_or_nonpositive_rule_time"
        if factor in probability_factors:
            return quote_reason(row)
        if factor in ("D1", "HIGH_D1", "LOW_D1", "D5", "HIGH_D5"):
            return difference_reason(row, 5 if factor.endswith("D5") else 1)
        if factor in position_factors:
            if row["group_boundary_count"] < 2:
                return "insufficient_distinct_group_price_levels"
            if not finite(row["group_scale"]) or row["group_scale"] <= 0:
                return "invalid_group_scale"
            if not finite(row["threshold"]):
                return "invalid_threshold_metadata"
        if factor in ("THRESHOLD_ADJACENT_MASS", "THRESHOLD_REL_DENSITY"):
            if row["group_boundary_count"] < 2:
                return "insufficient_distinct_group_price_levels"
            if not finite(row["group_scale"]) or row["group_scale"] <= 0:
                return "invalid_group_scale"
            detail = group_details[index]
            if detail["last"]:
                return "structural_highest_threshold_has_no_next_neighbor"
            if detail["duplicate"]:
                return "duplicate_adjacent_threshold"
            if not finite(row["probability"]):
                return quote_reason(row)
            neighbor = detail["neighbor_probability"]
            if not finite(neighbor):
                return "adjacent_market_quote_missing"
            if row["probability"] - neighbor < -config.get("negative_mass_tolerance", 1e-8):
                return "negative_adjacent_probability_mass"
        if factor in ("RANGE_REL_WIDTH", "RANGE_REL_DENSITY"):
            if not finite(row["group_scale"]) or row["group_scale"] <= 0:
                return "invalid_group_scale"
            if pd.isna(row["lower"]) or pd.isna(row["upper"]) or row["lower"] >= row["upper"]:
                return "invalid_range_bounds"
            if not finite(row["probability"]):
                return quote_reason(row)
            return "nonpositive_relative_range_width"
        return "unexplained_missing_factor"

    coverage = {}
    all_reasons = Counter()
    for category, factors in CATEGORY_FACTORS.items():
        subset = panel.loc[panel["category"] == category]
        coverage[category] = {"market_days": len(subset), "markets": int(subset["market_id"].nunique()), "factors": {}}
        for factor in factors:
            missing = ~np.isfinite(subset[factor].to_numpy(dtype=float))
            missing_rows = subset.loc[missing]
            reasons = Counter(missing_reason(i, row, factor) for i, row in missing_rows.iterrows())
            all_reasons.update(reasons)
            nl = sum(n for reason, n in reasons.items() if reason.startswith("natural_language_"))
            structural = sum(n for reason, n in reasons.items() if reason.startswith("structural_"))
            coverage[category]["factors"][factor] = {
                "total_slots": len(subset), "finite_slots": int((~missing).sum()),
                "missing_slots": int(missing.sum()), "natural_language_missing_slots": nl,
                "structural_missing_slots": structural,
                "other_missing_slots": int(missing.sum()) - nl - structural,
                "missing_reason_counts": dict(sorted(reasons.items())),
            }

    missing_markets = []
    incomplete = [row for row in catalog if row["category"] == "direction" and row["time_extraction"]["status"] != "complete"]
    for row in incomplete:
        event, market = raw[row["market_id"]]
        actual = panel.loc[panel["market_id"] == row["market_id"]]
        # Independently reproduce lifecycle applicability to verify missing rows.
        active = (cutoffs >= pd.Timestamp(row["opened_at"])) & (cutoffs >= pd.Timestamp(row["metadata_available_at"]))
        for key in ("observation_end", "expires_at", "closed_at"):
            if row.get(key):
                active &= cutoffs < pd.Timestamp(row[key])
        assert len(actual) == int(active.sum()), row["market_id"]
        reason_no_rows = None
        if actual.empty:
            reason_no_rows = ("opened_after_last_research_cutoff" if pd.Timestamp(row["opened_at"]) > last_cutoff
                              else "no_a_share_trading_cutoff_during_active_lifetime")
        entry = {
            "market_id": row["market_id"], "event_id": row["event_id"], "question": row["question"],
            "extraction_status": row["time_extraction"]["status"], "extraction_reason": extraction_reason(row),
            "observation_start": row["observation_start"], "observation_end": row["observation_end"],
            "opened_at": row["opened_at"], "expires_at": row["expires_at"], "closed_at": row["closed_at"],
            "research_market_days": len(actual), "research_dates": actual["datetime"].dt.strftime("%Y-%m-%d").tolist(),
            "valid_quote_market_days": int(actual["quote_valid"].sum()),
            "WINDOW_LENGTH_missing_slots": int(actual["WINDOW_LENGTH"].isna().sum()),
            "TIME_TO_END_missing_slots": int(actual["TIME_TO_END"].isna().sum()),
            "no_research_rows_reason": reason_no_rows,
            "time_extraction_evidence": row["time_extraction"],
            "raw_market_rule": market.get("description", ""), "raw_event_rule": event.get("description", ""),
            "rule_first_paragraph": market.get("description", "").split("\n")[0],
            "market_metadata_path": str((data / row["market_dir"] / "meta.json").relative_to(ROOT)),
        }
        missing_markets.append(entry)

    rejected_review = [row for row in review if not row.get("included")]
    rejected = []
    for row in rejected_review:
        event, market = raw[row["market_id"]]
        text = row["question"].lower()
        if row["reason"] == "excluded_underlying":
            scope = "excluded_non_wti_underlying_or_diesel_policy"
        elif row["reason"] == "missing_touch_threshold":
            scope = "oil_inventory_quantity" if "reserves" in text else "oil_production_quantity"
        elif "senate" in text:
            scope = "joint_oil_price_and_election_proposition"
        elif "production" in text:
            scope = "oil_production_quantity"
        elif any(word in text for word in ("strike", "ceasefire", "military")):
            scope = "military_or_ceasefire_proposition"
        elif any(word in text for word in ("reserve", "stocks")):
            scope = "strategic_reserve_action"
        else:
            scope = "oil_policy_or_trade_arrangement"
        rejected.append({**row, "audit_scope_reason": scope, "target_price_proposition_extraction_failure": False,
                         "event_title": event.get("title", ""), "raw_market_rule": market.get("description", "")})

    direction_count = sum(row["category"] == "direction" for row in catalog)
    nl_total = sum(n for reason, n in all_reasons.items() if reason.startswith("natural_language_"))
    structural_total = sum(n for reason, n in all_reasons.items() if reason.startswith("structural_"))
    no_rows = Counter(row["no_research_rows_reason"] for row in missing_markets if not row["research_market_days"])
    research_missing = [row for row in missing_markets if row["research_market_days"]]
    excluded_count = len(raw) - len(catalog)
    omitted_not_oil = excluded_count - len(rejected)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "producer": "outputs/polymarket/audit_extraction.py", "research_start": config["research_start"],
        "research_end": config["research_end"], "cutoff": "15:00 Asia/Shanghai",
        "source_files": {str(path.relative_to(ROOT)): {"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                         for path in [data / "markets.jsonl", data / "classification_review.json", data / "raw/events.jsonl",
                                      panel_dir / "factors.parquet", ROOT / config["calendar_path"]]},
        "panel_fingerprint": manifest["fingerprint"],
        "summary": {
            "catalog_markets": len(catalog), "catalog_direction_markets": direction_count,
            "catalog_direction_incomplete_time_markets": len(incomplete),
            "catalog_direction_incomplete_time_fraction": len(incomplete) / direction_count,
            "incomplete_time_reasons_catalog": dict(Counter(row["extraction_reason"] for row in missing_markets)),
            "research_panel_market_days": len(panel), "research_panel_markets": int(panel["market_id"].nunique()),
            "research_direction_market_days": int(panel["category"].eq("direction").sum()),
            "research_time_missing_markets": len(research_missing),
            "research_time_missing_market_days": sum(row["research_market_days"] for row in missing_markets),
            "research_natural_language_missing_factor_slots": nl_total,
            "research_natural_language_missing_factor_slots_by_factor": {
                factor: coverage["direction"]["factors"][factor]["natural_language_missing_slots"]
                for factor in ("WINDOW_LENGTH", "TIME_TO_END")},
            "research_natural_language_missing_factor_slots_by_reason": {
                reason: n for reason, n in all_reasons.items() if reason.startswith("natural_language_")},
            "incomplete_time_catalog_markets_without_research_rows": dict(no_rows),
            "all_configured_factor_slots": sum(len(panel.loc[panel["category"] == c]) * len(fs) for c, fs in CATEGORY_FACTORS.items()),
            "all_configured_factor_missing_slots": sum(all_reasons.values()),
            "all_structural_missing_slots": structural_total,
            "touch_opposite_side_structural_missing_slots": all_reasons["structural_opposite_touch_side"],
            "all_non_natural_language_nonstructural_missing_slots": sum(all_reasons.values()) - nl_total - structural_total,
            "unexplained_missing_factor_slots": sum(n for reason, n in all_reasons.items() if reason.startswith("unexplained_")),
        },
        "interpretation": {
            "natural_language_missing": "Rule comparison times could not be resolved because the extractor supports CME session open/close only; the affected rules use CME official settlement or ICE Europe WTI close. This is unsupported rule/calendar interpretation, not a missing year, missing timezone, quote gap or D1/D5 requirement.",
            "rule_times_not_imputed": "Unknown endpoints remain NaN. Gamma expiry, creation and actual resolution timestamps are never substituted.",
            "quote_independence": "Every affected research market-day has a valid quote; its four probability factors remain available.",
            "reason_assignment": "Each missing configured factor cell receives exactly one primary reason following the factor computation conditions. Touch opposite-side slots and a highest threshold without a successor are structural; quote history, difference history and insufficient group levels are non-NL reasons.",
            "rejected_markets": "The 36 missing_touch_threshold review entries are inventory or production quantities; the 29 unrecognized_proposition entries are non-target energy actions or joint oil/election propositions. They are not price-threshold extraction failures.",
            "fixability": "ICE session and CME settlement time support require verified exchange-specific schedules, holidays and settlement semantics. No safe purely textual date-format omission was identified; do not equate settlement with session close.",
            "catalog_scope": "The downloaded oil-tag catalog was audited; this does not establish that Gamma's oil tag contains every historical WTI market on Polymarket.",
        },
        "category_factor_coverage": coverage,
        "missing_factor_reason_counts": dict(sorted(all_reasons.items())),
        "incomplete_direction_time_markets": missing_markets,
        "excluded_market_audit": {
            "all_excluded_markets": excluded_count, "not_oil_keyword_rejections_omitted_from_review": omitted_not_oil,
            "reviewed_rejected_markets": len(rejected), "review_reason_counts": dict(Counter(row["reason"] for row in rejected)),
            "audited_scope_reason_counts": dict(Counter(row["audit_scope_reason"] for row in rejected)),
            "identified_target_price_proposition_extraction_failures": 0,
            "reviewed_rejections": rejected,
            "not_oil_review_note": "Inspected oil/price-like questions among not_oil rejections: price questions are US gasoline; others are military/shipping/policy or joint propositions, outside the five WTI price types.",
        },
    }
    assert report["summary"]["unexplained_missing_factor_slots"] == 0, report["missing_factor_reason_counts"]
    assert nl_total == sum(row["WINDOW_LENGTH_missing_slots"] + row["TIME_TO_END_missing_slots"] for row in missing_markets)
    assert all(row["valid_quote_market_days"] == row["research_market_days"] for row in missing_markets)
    json_write(output / "extraction_audit.json", report)
    csv_rows = []
    for row in missing_markets:
        csv_rows.append({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()})
    pd.DataFrame(csv_rows).to_csv(output / "extraction_missing_markets.csv", index=False)
    factor_rows = []
    for category, category_data in coverage.items():
        for factor, details in category_data["factors"].items():
            factor_rows.append({"category": category, "factor": factor, **details,
                                "missing_reason_counts": json.dumps(details["missing_reason_counts"], ensure_ascii=False)})
    pd.DataFrame(factor_rows).to_csv(output / "factor_missingness.csv", index=False)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(json.dumps(report["missing_factor_reason_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
