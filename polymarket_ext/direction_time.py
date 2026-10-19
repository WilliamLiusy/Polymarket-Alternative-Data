"""Extract scheduled price-comparison endpoints, never market creation times.

Explicit dated comparisons and CME CL open/close templates are supported.
Unknown exchanges, settlement schedules and ambiguous text stay missing; the
existing classification overrides can supply verified, timezone-aware endpoints.
"""
from __future__ import annotations

from datetime import timedelta, timezone
from functools import lru_cache
import re

import pandas as pd
import pandas_market_calendars as calendars
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

VERSION = 1
MONTH = (r"(?:January|February|March|April|May|June|July|August|September|"
         r"October|November|December)")
DATE = rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+20\d{{2}})"
CLOCK = r"(?:\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?|\d{1,2}\s*(?:AM|PM)|noon|midnight)"
ZONE = r"(?:UTC|GMT|EST|EDT|ET|Z|[+-]\d{2}:?\d{2})"
POINT = re.compile(rf"(?P<date>{DATE})(?:(?:T|\s+(?:at\s+)?)(?P<clock>{CLOCK})\s*(?P<zone>{ZONE})?)?", re.I)
CLOCK_FIRST = re.compile(rf"(?P<clock>{CLOCK})\s*(?P<zone>{ZONE})?\s+on\s+(?P<date>{DATE})", re.I)
MONTH_EDGE = re.compile(rf"(?P<edge>first|last)\s+trading\s+day\s+(?:of|in)\s+(?P<month>{MONTH}\s+20\d{{2}})", re.I)
PREVIOUS = re.compile(r"(?:most recent\s+)?(?:prior|previous|preceding)\s+trading\s+day", re.I)
SHORT_DATE = re.compile(rf"\b{MONTH}\s+\d{{1,2}}(?:st|nd|rd|th)?\b(?!,?\s+20\d{{2}})", re.I)


def _aware(value):
    """Overrides must carry their timezone, including an explicit UTC suffix."""
    if value is None or value == "":
        return None
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("Direction observation times must include a timezone")
    return stamp.tz_convert("UTC")


@lru_cache(maxsize=32)
def _schedule(year):
    # Product-specific calendar includes holidays, short sessions and DST.
    return calendars.get_calendar("CMEGlobex_CL").schedule(
        pd.Timestamp(year, 1, 1) - pd.Timedelta(days=14), pd.Timestamp(year, 12, 31))


def _is_cme(rule):
    return bool(re.search(r"\b(?:CME|NYMEX)\b", rule, re.I)) and not re.search(r"\bICE\b", rule, re.I)


def _kind(text):
    mentions = list(re.finditer(r"\b(open|opening|close|closing|settlement)\s+price\b|\bat\s+(?:the\s+)?(open|close)\b", text, re.I))
    if not mentions:
        return None
    kind = (mentions[-1][1] or mentions[-1][2]).lower()
    # Settlement is not the session close. Each endpoint uses its nearest price type.
    return "settlement" if kind == "settlement" else "market_open" if kind.startswith("open") else "market_close"


def _session_point(day, kind, rule):
    if kind not in ("market_open", "market_close") or not _is_cme(rule):
        return None
    schedule = _schedule(day.year)
    return schedule.loc[day, kind] if day in schedule.index else None


def _parse_point(match, kind, rule, shared_zone=None):
    day = pd.Timestamp(re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", match["date"], flags=re.I))
    clock = match["clock"]
    if clock is None:
        return _session_point(day, kind, rule)
    zone = match["zone"] or shared_zone
    if zone is None:
        return None
    clock = {"noon": "12:00", "midnight": "00:00"}.get(clock.lower(), clock)
    stamp = pd.Timestamp(f"{day.date()} {clock}")
    zone = zone.upper()
    if zone == "ET":
        zone = "America/New_York"
    elif zone in ("EST", "EDT"):
        zone = timezone(timedelta(hours=-5 if zone == "EST" else -4))
    elif zone in ("Z", "GMT"):
        zone = "UTC"
    return stamp.tz_localize(zone, ambiguous="raise", nonexistent="raise").tz_convert("UTC")


def _point(match, kind, rule, shared_zone=None):
    try:
        return _parse_point(match, kind, rule, shared_zone)
    except (ValueError, TypeError, AmbiguousTimeError, NonExistentTimeError):
        return None


def _clause_window(clause, context):
    """Return endpoint candidates only for explicit comparison constructions."""
    if not re.search(r"\b(?:up|down|rises?|increases?|decreases?|higher|lower|greater|less|exceeds?|compares?|comparison)\b", clause, re.I):
        return None, None
    if re.search(r"\b(?:accepts?\s+orders|orders?\s+(?:can|may|will)|trading\s+(?:opens?|closes?|starts?|ends?|is|will)|market\s+(?:opens?|closes?))\b", clause, re.I):
        return None, None
    clause = CLOCK_FIRST.sub(lambda match: f'{match["date"]} at {match["clock"]} {match["zone"] or ""}', clause)
    points = list(POINT.finditer(clause))
    # A partly specified second date must not turn a multi-day comparison into
    # a same-day one. These mixed templates need a verified override.
    if SHORT_DATE.search(clause) or (points and MONTH_EDGE.search(clause)):
        return None, None
    zones = {point["zone"].upper() for point in points if point["zone"]}
    shared_zone = next(iter(zones)) if len(zones) == 1 else None
    kind = _kind(clause)
    if len(points) == 2:
        first, second = points
        between = clause[first.end():second.start()]
        before = clause[:first.start()]
        if re.search(r"\bfrom\s*$", before, re.I) and re.search(r"\b(?:to|until|through)\b", between, re.I):
            return (_point(first, _kind(before) or kind, context, shared_zone),
                    _point(second, _kind(between) or kind, context, shared_zone))
        if re.search(r"\b(?:higher|lower|greater|less)\s+than\b|\b(?:compared\s+(?:with|to)|versus)\b", between, re.I):
            values = (_point(first, _kind(before) or kind, context, shared_zone),
                      _point(second, _kind(between) or kind, context, shared_zone))
            # Rules can state either B > A or A < B; chronology identifies the
            # start/end once both observations are independently known.
            return tuple(sorted(values)) if all(value is not None for value in values) else (None, None)
    if len(points) == 1 and PREVIOUS.search(clause):
        point = points[0]
        end_kind = _kind(clause[:point.start()]) or kind
        start_kind = _kind(clause[point.end():PREVIOUS.search(clause).start()]) or end_kind
        end = _point(point, end_kind, context, shared_zone)
        day = pd.Timestamp(re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", point["date"], flags=re.I))
        if end is None or not _is_cme(context) or start_kind not in ("market_open", "market_close"):
            return None, end
        prior = _schedule(day.year).loc[lambda frame: frame.index < day]
        return (prior.iloc[-1][start_kind] if len(prior) else None), end
    if len(points) == 1 and re.search(r"\b(?:open|opening)\s+price", clause, re.I) and re.search(r"\b(?:close|closing)\s+price", clause, re.I):
        if _kind(clause[points[0].end():]) and not re.search(r"\b(?:that|same)\s+day\b", clause, re.I):
            return None, None
        day = pd.Timestamp(points[0]["date"])
        start, end = _session_point(day, "market_open", context), _session_point(day, "market_close", context)
        if points[0]["clock"]:
            if _kind(clause[:points[0].start()]) == "market_open":
                start = _point(points[0], "market_open", context, shared_zone)
            else:
                end = _point(points[0], "market_close", context, shared_zone)
        return start, end
    edges = list(MONTH_EDGE.finditer(clause))
    if len(edges) == 2 and _is_cme(context):
        values = []
        for edge in edges:
            month = pd.Timestamp("1 " + edge["month"])
            schedule = _schedule(month.year)
            rows = schedule.loc[(schedule.index.year == month.year) & (schedule.index.month == month.month)]
            prefix = clause[max(0, edge.start() - 100):edge.start()]
            price_kind = _kind(prefix) or kind
            values.append(rows.iloc[0 if edge["edge"].lower() == "first" else -1][price_kind]
                          if len(rows) and price_kind in ("market_open", "market_close") else None)
        if re.search(r"\b(?:higher|lower|greater|less)\s+than\b", clause[edges[0].end():edges[1].start()], re.I):
            return tuple(sorted(values)) if all(value is not None for value in values) else (None, None)
        if edges[0]["edge"].lower() == "first" and edges[1]["edge"].lower() == "last":
            return tuple(values)
    if len(points) == 1 and re.search(r"\b(?:close|closing)\s+price|\bup\s+(?:or|/)\s+down\b", clause, re.I):
        # A known target price time is useful even if the reference is unknown.
        return None, _point(points[0], kind, context, shared_zone)
    return None, None


def extract_window(event, market, override=None):
    """Return UTC endpoints, a version marker and inspectable extraction evidence.

    Descriptions supply comparison semantics. Question/title can supply explicit
    dated windows; raw Gamma date fields alone never establish those semantics.
    Conflicting candidates are missing, not resolved by guessing source priority.
    """
    sources = [("market.description", market.get("description")),
               ("event.description", event.get("description")),
               ("market.question", market.get("question")), ("event.title", event.get("title"))]
    context = " ".join(str(text) for _, text in sources[:2] if text)
    candidates = [[], []]
    issues = []
    for source, text in sources:
        if not text:
            continue
        for clause in re.split(r"(?<=[.!?])\s+|\n+", str(text)):
            if re.search(r"\b(?:example|e\.g\.)\b", clause, re.I):
                continue
            try:
                endpoints = _clause_window(clause, context)
            except (ValueError, TypeError, KeyError, AmbiguousTimeError, NonExistentTimeError):
                issues.append({"source": source, "text": clause, "reason": "invalid_or_ambiguous_time"})
                continue
            for index, value in enumerate(endpoints):
                if value is not None:
                    candidates[index].append({"value": value.isoformat(), "source": source, "text": clause})
    values = []
    for entries in candidates:
        unique = {entry["value"] for entry in entries}
        values.append(next(iter(unique)) if len(unique) == 1 else None)
        if len(unique) > 1:
            issues.append({"reason": "conflicting_endpoints", "candidates": entries})
    for index, key in enumerate(("observation_start", "observation_end")):
        if key in (override or {}):
            stamp = _aware(override[key])
            values[index] = stamp.isoformat() if stamp is not None else None
            candidates[index] = [{"value": values[index], "source": "override", "text": "Verified contract endpoint"}]
    if all(values) and pd.Timestamp(values[0]) >= pd.Timestamp(values[1]):
        issues.append({"reason": "nonpositive_comparison_window", "start": values[0], "end": values[1]})
        values[0] = None  # Keep an independently known end; only the window is unusable.
    return {"observation_start": values[0], "observation_end": values[1],
            "time_extraction_version": VERSION,
            "time_extraction": {"status": "complete" if all(values) else "partial" if any(values) else "missing",
                                "start_evidence": candidates[0], "end_evidence": candidates[1], "issues": issues}}


def is_intraday(event, market, window=None):
    """An explicit sub-day product is out of scope; a clock alone is not."""
    texts = [market.get("question") or "", event.get("title") or ""]
    if window:
        # Only inspect clauses that established a price endpoint, not unrelated
        # trading-hours prose or examples elsewhere in the contract.
        evidence = window["time_extraction"]
        texts.extend(entry["text"] for key in ("start_evidence", "end_evidence") for entry in evidence[key])
    for index, text in enumerate(texts):
        if index < 2:  # A rule's "1-minute candle" is a price resolution, not a horizon.
            if re.search(r"\b(?:hourly|minutely|next\s+(?:hour|minute))\b", text, re.I):
                return True
            for length, unit in re.findall(r"\b(\d+(?:\.\d+)?)\s*[- ]?(minutes?|mins?|m|hours?|hrs?|h)\b", text, re.I):
                hours = float(length) / 60 if unit.lower().startswith("m") else float(length)
                if hours < 24:
                    return True
        # Do not double-count the same date appearing in both title and question.
        # Overnight exchange sessions remain daily even when shorter than 24h.
        explicit = list(POINT.finditer(text))
        if len(explicit) == 2 and all(point["clock"] for point in explicit) and \
                not re.search(r"\bdaily\b|\b(?:open|opening|close|closing)\s+price", text, re.I):
            try:
                if pd.Timestamp(explicit[0]["date"]) == pd.Timestamp(explicit[1]["date"]):
                    return True
            except ValueError:
                pass
    return False
