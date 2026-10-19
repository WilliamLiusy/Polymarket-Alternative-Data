"""Auditable rule classification; uncertain markets require explicit overrides."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import pandas as pd
import yaml

from .direction_time import extract_window, is_intraday

CATEGORIES = ("direction", "threshold", "touch", "range", "record")
CATEGORY_NAMES = dict(zip(CATEGORIES, ("区间涨跌型", "收盘超过某价位型", "期间触及高低价型", "结算价格区间型", "历史新高型")))
NUMBER = r"-?\d[\d,]*(?:\.\d+)?"
MONEY = rf"\$\s*({NUMBER})"
OIL = re.compile(r"\b(?:WTI|CL|CLL|crude\s+oil|west\s+texas\s+intermediate)\b", re.I)
EXCLUDED = re.compile(r"\b(?:gasoline|petrol|diesel|brent|natural\s+gas|RBOB)\b", re.I)


def _array(value):
    return json.loads(value) if isinstance(value, str) else (value or [])


def _timestamp(value):
    if value is None or value == "":
        return None
    return pd.to_datetime(value, utc=True).isoformat()


def _number(value):
    return float(str(value).replace(",", ""))


def _slug(value):
    return re.sub(r"[^\w.-]+", "-", str(value)).strip(".-")[:100] or "unnamed"


def _bounds(text):
    """Explicit monetary ranges and tails; dates are never price levels."""
    match = re.search(rf"(?:between\s+)?{MONEY}\s*(?:and|to|[-–—])\s*\$?\s*({NUMBER})", text, re.I)
    if match:
        return _number(match[1]), _number(match[2])
    match = re.search(rf"(?:below|under|less than|<)\s*{MONEY}|{MONEY}\s*(?:or\s+)?(?:less|lower|below)", text, re.I)
    if match:
        return "-inf", _number(next(x for x in match.groups() if x is not None))
    match = re.search(rf"(?:above|over|more than|>)\s*{MONEY}|{MONEY}\s*(?:\+|or\s+(?:more|higher|above))", text, re.I)
    if match:
        return _number(next(x for x in match.groups() if x is not None)), "inf"
    return None, None


def _record_threshold(rule):
    match = re.search(rf"(?:record|all.time high)[^.]*?{MONEY}", rule, re.I)
    if match:
        return _number(match[1])
    # CME record contracts often state only the Yes condition, without naming
    # the historical record in the rule. Read its explicit high-price barrier.
    condition = re.search(
        r'\bresolve\s+to\s+["“]Yes["”]\s+if\b(.*?)(?=\bOtherwise\b|\n\s*\n|$)',
        rule, re.I | re.S,
    )
    if condition is None:
        return None
    levels = {
        _number(value) for value in re.findall(
            rf"\b(?:daily\s+)?high\s+price\b[^$\n]*?\b(?:greater\s+than|above|over|exceeds?)\s*{MONEY}",
            condition[1], re.I,
        )
    }
    return next(iter(levels)) if len(levels) == 1 else None


def _kind(event, market):
    title = str(event.get("title", ""))
    question = str(market.get("question", ""))
    label = str(market.get("groupItemTitle", ""))
    text = question + " " + title
    rule = str(market.get("description", ""))
    if EXCLUDED.search(question) or (not OIL.search(question) and EXCLUDED.search(title)):
        return None, "excluded_underlying", {}
    if not OIL.search(text + " " + str(market.get("description", ""))):
        return None, "not_oil", {}
    # These words must occur in the proposition, not in unrelated rule boilerplate.
    if re.search(r"all[- ]time\s+high|record\s+high|new\s+(?:price\s+)?record", text, re.I):
        return "record", None, {"record_threshold": _record_threshold(rule)}
    if re.search(r"up\s*(?:or|/|and)\s*down", text, re.I):
        if is_intraday(event, market):
            return None, "intraday_direction", {}
        return "direction", None, {}
    lower, upper = _bounds(question + " " + label)
    range_event = bool(re.search(r"(?:settle|settlement|closing|close|end).*(?:price|between|range)|(?:price|range).*(?:settle|close|end)", title, re.I))
    sibling_ranges = any(
        re.search(rf"{MONEY}\s*(?:and|to|[-–—])\s*\$?\s*{NUMBER}", str(m.get("question", "")) + " " + str(m.get("groupItemTitle", "")), re.I)
        for m in event.get("markets", [])
    )
    is_finite_range = lower is not None and upper is not None and lower != "-inf" and upper != "inf"
    if (is_finite_range or sibling_ranges) and (range_event or re.search(r"settle|close|closing|settlement", text, re.I)):
        return "range", None, {"lower": lower, "upper": upper}
    touch = re.search(r"\b(?:hit|touch|reach|dip|fall\s+to|drop\s+to)\b", question, re.I)
    during = re.search(r"\bat any (?:point|time)\b", question + " " + rule, re.I)
    if touch or during:
        match = re.search(MONEY, question + " " + label)
        if not match:
            return None, "missing_touch_threshold", {}
        if len(set(re.findall(MONEY, question))) > 1:
            return None, "multiple_touch_barriers", {}
        low = bool(re.search(r"\b(?:dip|fall|drop|low|below|under)\b", question, re.I))
        high = bool(re.search(r"\b(?:reach|rise|high|above|over)\b", question, re.I))
        if low == high:
            return None, "ambiguous_touch_direction", {}
        return "touch", None, {"side": "low" if low else "high", "threshold": _number(match[1])}
    if re.search(r"\b(?:close[sd]?|settle[sd]?|closing|settlement)\b", text + " " + rule, re.I):
        match = re.search(rf"(?:above|over|higher\s+than|exceed)\s*{MONEY}", question, re.I)
        if match:
            return "threshold", None, {"threshold": _number(match[1])}
    return None, "unrecognized_proposition", {}


def _observation_start(event, market):
    for key in ("observation_start", "eventStartTime"):
        if market.get(key):
            return _timestamp(market[key])
    # Preserve explicitly dated 'from' windows, including later-added barriers.
    rule = str(market.get("description", ""))
    match = re.search(r"\bfrom\s+([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?(?:,)?\s+20\d{2})(?:\s+(?:at\s+)?(\d{1,2}:\d{2}\s*(?:AM|PM)?)(?:\s+(ET|EST|EDT|UTC))?)?", rule, re.I)
    if match:
        text = match[1] + " " + (match[2] or "00:00")
        stamp = pd.Timestamp(text)
        zone = "America/New_York" if (match[3] or "").upper() in ("ET", "EST", "EDT") else "UTC"
        return stamp.tz_localize(zone).tz_convert("UTC").isoformat()
    monthly = re.search(r"during a trading session of ([A-Za-z]+\s+20\d{2})", rule, re.I)
    if monthly:
        # First possible business-day session; holiday closures can only move it
        # later. Late-created barriers must not join a full-month probability group.
        day = pd.Timestamp("1 " + monthly[1])
        while day.dayofweek >= 5:
            day += pd.Timedelta(days=1)
        if "ice futures" in rule.lower() and re.search(r"8:00\s*PM\s*ET", rule, re.I):
            hour = 18 if day.dayofweek == 0 else 20
            opened = (day - pd.Timedelta(days=1) + pd.Timedelta(hours=hour)).tz_localize("America/New_York").tz_convert("UTC")
        else:
            opened = day.tz_localize("UTC")
        if re.search(r"after market creation", rule, re.I):
            created = pd.to_datetime(market.get("createdAt") or event.get("createdAt"), utc=True)
            if pd.notna(created):
                opened = max(opened, created)
        return opened.isoformat()
    if re.search(r"after market creation|from (?:the )?(?:market.s )?(?:creation|launch)", rule, re.I):
        return _timestamp(market.get("createdAt") or event.get("createdAt"))
    return _timestamp(event.get("startTime") or event.get("startDate") or event.get("createdAt"))


def _price_basis(event, market, category):
    rule = str(market.get("description") or event.get("description") or "").lower()
    source = str(market.get("resolutionSource") or event.get("resolutionSource") or "unknown").lower().rstrip("/")
    exchange = "ice" if "ice futures" in rule else "nymex" if "nymex" in rule else "cme" if "cme" in rule else "unspecified_exchange"
    contract = "active_month" if "active month" in rule else "front_month" if "front month" in rule or "front-month" in rule else "unspecified_contract"
    explicit = re.search(r"\bcl[fghjkmnquvxz]\d{1,4}\b|\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+20\d{2}\s+(?:futures\s+)?contract\b", rule)
    if explicit:
        contract = explicit[0]
    if re.search(r"\bspot\b", rule):
        contract += ":spot"
    price = "settlement" if "settlement" in rule and "pyth" not in rule else "close" if category in ("direction", "threshold", "range") else "intraday"
    return ":".join((source, exchange, contract, price))


def classify_market(event: dict, market: dict, override: dict | None = None) -> tuple[dict | None, str | None]:
    override = override or {}
    if override.get("exclude"):
        return None, "manual_exclusion"
    category, reason, values = _kind(event, market)
    category = override.get("category", category)
    if category is None:
        return None, reason
    if category not in CATEGORIES:
        raise ValueError(f"Unknown category {category}")
    outcomes, tokens = _array(market.get("outcomes")), _array(market.get("clobTokenIds"))
    mapping = {str(label).lower(): str(token) for label, token in zip(outcomes, tokens)}
    positive, negative = ("up", "down") if category == "direction" else ("yes", "no")
    token = override.get("token_id") or mapping.get(positive)
    no_token = override.get("no_token_id") or mapping.get(negative)
    if not token or not no_token or token == no_token:
        return None, "missing_or_nonbinary_tokens"
    created = market.get("createdAt") or event.get("createdAt")
    opened = market.get("acceptingOrdersTimestamp") or market.get("startDate") or created
    end = market.get("endDate") or event.get("endDate")
    row = {
        "event_id": str(event["id"]), "market_id": str(market["id"]),
        "event_title": event.get("title", ""), "question": market.get("question", ""),
        "event_slug": event.get("slug", ""), "slug": market.get("slug", ""),
        "category": category, "side": None, "token_id": token, "yes_token_id": token,
        "no_token_id": no_token, "condition_id": market.get("conditionId"),
        "underlying": "WTI", "price_basis": _price_basis(event, market, category),
        "opened_at": _timestamp(opened), "closed_at": _timestamp(market.get("closedTime")),
        "expires_at": _timestamp(end),
        "observation_start": None if category == "direction" else _observation_start(event, market),
        "observation_end": None if category == "direction" else _timestamp(end),
        "metadata_available_at": _timestamp(created),
        "metadata_updated_at": _timestamp(market.get("updatedAt")),
        "metadata_time_assumption": "current_contract_terms_assumed_unchanged_since_creation",
        "threshold": None, "lower": None, "upper": None, "record_threshold": None,
        "classification_source": "override" if override else "rules",
        **values,
    }
    row.update({key: value for key, value in override.items() if key != "exclude"})
    if category == "direction":
        row.update(extract_window(event, market, override))
        if is_intraday(event, market, row):
            return None, "intraday_direction"
    row["yes_token_id"] = row["token_id"]
    for key in ("opened_at", "closed_at", "expires_at", "observation_start", "observation_end", "metadata_available_at"):
        row[key] = _timestamp(row.get(key))
    for key in ("lower", "upper"):
        if row[key] is not None:
            row[key] = str(row[key]) if str(row[key]) in ("inf", "-inf") else float(row[key])
    if not row.get("condition_id") or not row.get("opened_at") or not row.get("expires_at") or not row.get("metadata_available_at"):
        return None, "missing_identity_or_lifecycle"
    if category in ("threshold", "touch") and row["threshold"] is None:
        return None, "missing_threshold"
    if category == "touch" and row["side"] not in ("high", "low"):
        return None, "missing_touch_direction"
    if category == "range" and (row["lower"] is None or row["upper"] is None or float(row["lower"]) >= float(row["upper"])):
        return None, "invalid_range"
    if category == "record" and row["record_threshold"] is None:
        return None, "missing_record_threshold"
    event_dir = f"{_slug(event.get('slug') or event.get('title'))}--{_slug(event['id'])}"
    market_dir = f"{_slug(market.get('slug') or market.get('question'))}--{_slug(market['id'])}"
    row["market_dir"] = str(Path(category) / event_dir / market_dir)
    return row, None


def classify_event(event: dict, overrides: dict | None = None) -> list[dict]:
    result = []
    for market in event.get("markets", []):
        row, _ = classify_market(event, market, (overrides or {}).get(str(market["id"])))
        if row:
            result.append(row)
    return result


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classify(config: dict, refresh: bool = False) -> dict:
    root = Path(config["data_dir"])
    source = root / "raw" / "events.jsonl"
    override_path = config.get("classification", {}).get("overrides_file")
    overrides = yaml.safe_load(Path(override_path).read_text()) or {} if override_path else {}
    overrides = {str(key): value for key, value in overrides.items()}
    rows, rejected, seen = [], [], set()
    with source.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            for market in event.get("markets", []):
                identifier = str(market["id"])
                if identifier in seen:
                    raise ValueError(f"Duplicate market in event source: {identifier}")
                seen.add(identifier)
                row, reason = classify_market(event, market, overrides.get(identifier))
                if row is None:
                    if reason != "not_oil":
                        rejected.append({"event_id": str(event["id"]), "market_id": identifier,
                                         "question": market.get("question"), "reason": reason})
                    continue
                rows.append(row)
                if row["category"] == "direction" and row["time_extraction"]["status"] != "complete":
                    rejected.append({"event_id": row["event_id"], "market_id": row["market_id"],
                                     "question": row["question"], "reason": "incomplete_direction_time",
                                     "included": True, "time_extraction": row["time_extraction"]})
                directory = root / row["market_dir"]
                _json(directory.parent / "meta.json", event)
                _json(directory / "meta.json", {"normalized": row, "raw": market})
    temporary = root / ".markets.jsonl.tmp"
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(root / "markets.jsonl")
    _json(root / "classification_review.json", rejected)
    report = {"markets_seen": len(seen), "selected_markets": len(rows),
              "selected_events": len({row["event_id"] for row in rows}),
              "categories": dict(Counter(row["category"] for row in rows)),
              "direction_time_status": dict(Counter(row["time_extraction"]["status"]
                                                     for row in rows if row["category"] == "direction")),
              "review_reasons": dict(Counter(row["reason"] for row in rejected)),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "metadata_time_assumption": "Gamma does not provide historical contract metadata versions; current terms assumed unchanged since createdAt"}
    _json(root / "classification.json", report)
    return report
