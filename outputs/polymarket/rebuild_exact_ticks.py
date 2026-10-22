"""Strictly offline schema-2 recompilation of already collected price histories.

Run from the repository root:
    .venv/bin/python outputs/polymarket/rebuild_exact_ticks.py

Original side coverage windows and requests are authoritative: no new date bounds,
grains or API requests are invented. A journal per market makes an interruption
between replacing CSVs and coverage recoverable without losing before statistics.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import sys
import traceback

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from polymarket_ext import collect

SCHEMA = 2
NOTE = (
    "Offline recompilation from complete original cached API pages. Data v2 "
    "resolution_seconds=0 is an exact tick, including terminal ticks; it is "
    "retained and wins same-timestamp positive-resolution buckets. Original "
    "requested_start/requested_end, request list, source_identity and collected_at "
    "are preserved. Negative resolution is rejected. No network is permitted."
)
OUTPUT = ROOT / "outputs/polymarket"
CHECKPOINTS = OUTPUT / "exact_tick_rebuild_checkpoints"
AUDIT = OUTPUT / "exact_tick_rebuild.json"
RUN_HISTORY = OUTPUT / "exact_tick_rebuild_runs"
NETWORK_FAILURES = OUTPUT / "exact_tick_network_failures.jsonl"
NETWORK_ATTEMPTS = []


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def deny_network(method, *args, **kwargs):
    failure = {"timestamp": now(), "method": method,
               "error": "Strict offline rebuild forbids every network/client call"}
    NETWORK_ATTEMPTS.append(failure)
    with NETWORK_FAILURES.open("a") as stream:
        stream.write(json.dumps(failure) + "\n")
    raise RuntimeError(f"Network forbidden: {method}")


class OfflineClient:
    """Any accidental API call is recorded and fails; never creates an SDK client."""
    def prices(self, *args, **kwargs):
        return deny_network("client.prices", *args, **kwargs)
    def clob_prices(self, *args, **kwargs):
        return deny_network("client.clob_prices", *args, **kwargs)
    def trades(self, *args, **kwargs):
        return deny_network("client.trades", *args, **kwargs)
    def events(self, *args, **kwargs):
        return deny_network("client.events", *args, **kwargs)


def enforce_offline():
    collect.OfficialClient = lambda *a, **k: deny_network("OfficialClient.constructor", *a, **k)
    socket.create_connection = lambda *a, **k: deny_network("socket.create_connection", *a, **k)
    socket.getaddrinfo = lambda *a, **k: deny_network("socket.getaddrinfo", *a, **k)
    socket.socket.connect = lambda *a, **k: deny_network("socket.connect", *a, **k)
    socket.socket.connect_ex = lambda *a, **k: deny_network("socket.connect_ex", *a, **k)
    return OfflineClient()


def strict_json(path):
    if not path.is_file():
        raise FileNotFoundError(f"Required original cache file missing: {path}")
    before = path.stat()
    payload = json.loads(path.read_text())
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Raw cache changed while being read: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"Cached API payload must be a JSON object: {path}")
    return payload


def csv_stats(path):
    frame = pd.read_csv(path, usecols=["resolution_seconds"])
    return {"rows": len(frame), "exact_tick_rows": int(frame["resolution_seconds"].eq(0).sum()),
            "sha256": sha(path), "size": path.stat().st_size}


def compile_side(directory, side, original, settings):
    if not original.get("api_complete") or original.get("errors"):
        raise ValueError(f"Original side coverage is incomplete: {directory}/{side}")
    start, end = int(original["requested_start"]), int(original["requested_end"])
    chunk = int(float(settings["price_chunk_days"]) * 86400)
    expected = [(left, min(left + chunk, end)) for left in range(start, end, chunk)]
    windows = original.get("windows", [])
    if [(window["start"], window["end"]) for window in windows] != expected:
        raise ValueError(f"Original side windows disagree with original chunk setting: {directory}/{side}")
    all_rows, updated_windows = [], []
    raw_page_count = 0
    raw_point_count = 0
    raw_exact_tick_count = 0
    for window in windows:
        left, right = int(window["start"]), int(window["end"])
        window_dir = directory / "raw" / side / f"{left}-{right}"
        requests = []
        if not window.get("requests"):
            raise ValueError(f"Original window has no completed requests: {window_dir}")
        for request in window["requests"]:
            if not request.get("complete"):
                raise ValueError(f"Original request is incomplete: {window_dir}/{request}")
            source, grain = request["source"], int(request["grain"])
            if source == "clob":
                payload = strict_json(window_dir / f"clob_{grain}.json")
                points = payload.get("history")
                if not isinstance(points, list):
                    raise ValueError(f"CLOB cache lacks history: {window_dir}")
                raw_page_count += 1
            elif source == "data_v2":
                page_dir = window_dir / f"data_{grain}"
                manifest = strict_json(page_dir / "manifest.json")
                if not manifest.get("complete") or manifest.get("error") or manifest.get("next_cursor") is not None:
                    raise ValueError(f"Data v2 raw manifest is incomplete: {page_dir}")
                pages = int(manifest.get("pages", 0))
                if pages < 1:
                    raise ValueError(f"Data v2 complete cache lacks pages: {page_dir}")
                points = []
                for page in range(pages):
                    payload = strict_json(page_dir / f"{page:06d}.json")
                    data = payload.get("data")
                    if not isinstance(data, list):
                        raise ValueError(f"Data v2 cache lacks data list: {page_dir}/{page:06d}.json")
                    points.extend(data)
                    raw_page_count += 1
            else:
                raise ValueError(f"Unknown original source: {source}")
            raw_point_count += len(points)
            raw_exact_tick_count += sum(point.get("resolution_seconds", grain) == 0 for point in points)
            found = collect._price_rows(points, source, grain, left, right)
            all_rows.extend(found)
            requests.append({**request, "rows": len(found), "complete": True})
        updated_windows.append({**window, "requests": requests})
    frame = pd.DataFrame(all_rows, columns=collect.PRICE_COLUMNS)
    if not frame.empty:
        frame = frame.sort_values(["timestamp", "resolution_seconds", "source"])
        frame = frame.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    staged = directory / f".{side}_prices.exact-v2.csv.tmp"
    frame.to_csv(staged, index=False)
    coverage = {**original, **collect._price_coverage(frame), "windows": updated_windows,
                "normalized_price_schema_version": SCHEMA, "note": collect.PRICE_NOTE}
    after = {"rows": len(frame), "exact_tick_rows": int(frame["resolution_seconds"].eq(0).sum()),
             "sha256": sha(staged), "size": staged.stat().st_size}
    return staged, coverage, after, {"raw_pages_read": raw_page_count, "raw_points_read": raw_point_count,
                                    "raw_exact_tick_points_before_dedup_and_time_filter": raw_exact_tick_count}


def rebuild_market(market, data_dir, config_settings):
    market_id = str(market["market_id"])
    directory = data_dir / market["market_dir"]
    path = CHECKPOINTS / f"{market_id}.json"
    checkpoint = json.loads(path.read_text()) if path.exists() else None
    current = strict_json(directory / "coverage.json")
    if checkpoint and checkpoint.get("status") == "complete":
        if current.get("normalized_price_schema_version") != SCHEMA or current.get("artifacts") != collect._artifacts(directory):
            raise RuntimeError(f"Completed rebuild artifacts changed: {market_id}")
        return checkpoint, True
    if not checkpoint:
        if not current.get("api_complete"):
            raise ValueError(f"Original market collection is incomplete: {market_id}")
        if current.get("artifacts") != collect._artifacts(directory):
            raise ValueError(f"Original artifacts do not match collection coverage: {market_id}")
        checkpoint = {"market_id": market_id, "market_dir": market["market_dir"], "status": "staging",
                      "started_at": now(), "original_coverage": current,
                      "before": {side: csv_stats(directory / f"{side}_prices.csv") for side in ("yes", "no")},
                      "trades_sha256": sha(directory / "trades.jsonl")}
        dump(path, checkpoint)
    original = checkpoint["original_coverage"]
    if original.get("source_identity") != current.get("source_identity") or original.get("collected_at") != current.get("collected_at"):
        raise RuntimeError(f"Original market identity or collection timestamp changed: {market_id}")
    if sha(directory / "trades.jsonl") != checkpoint["trades_sha256"]:
        raise RuntimeError(f"Trade history changed during rebuild: {market_id}")
    settings = dict(config_settings)
    identity = original.get("source_identity", {})
    for key in ("price_chunk_days", "data_url", "clob_url"):
        if key in identity:
            settings[key] = identity[key]
    if "price_chunk_days" not in settings:
        raise ValueError(f"Original price_chunk_days unavailable: {market_id}")
    staged = {}
    for side, token_key in (("yes", "yes_token_id"), ("no", "no_token_id")):
        if str(original[side]["token_id"]) != str(identity[token_key]):
            raise RuntimeError(f"Original side token disagrees with source identity: {market_id}/{side}")
        staged[side] = compile_side(directory, side, original[side], settings)
    coverage = copy.deepcopy(original)
    details = {}
    for side, (stage_path, side_coverage, after, raw_stats) in staged.items():
        before = checkpoint["before"][side]
        if after["rows"] < before["rows"] or after["exact_tick_rows"] < before["exact_tick_rows"]:
            raise RuntimeError(f"Exact tick rebuild unexpectedly drops rows: {market_id}/{side}")
        details[side] = {"before": before, "after": after,
                         "row_increment": after["rows"] - before["rows"],
                         "exact_tick_row_increment": after["exact_tick_rows"] - before["exact_tick_rows"],
                         **raw_stats, "requested_start": side_coverage["requested_start"],
                         "requested_end": side_coverage["requested_end"]}
        coverage[side] = side_coverage
    # Only replace destinations after both sides have passed strict raw validation.
    for side, (stage_path, *_rest) in staged.items():
        stage_path.replace(directory / f"{side}_prices.csv")
    coverage["normalized_price_schema_version"] = SCHEMA
    coverage["price_normalization"] = {"schema_version": SCHEMA, "recompiled_at": now(),
                                        "offline": True, "note": NOTE}
    coverage["artifacts"] = collect._artifacts(directory)
    if coverage["source_identity"] != original["source_identity"] or coverage["collected_at"] != original["collected_at"]:
        raise AssertionError("Offline recompilation changed collection provenance")
    dump(directory / "coverage.json", coverage)
    result = {"market_id": market_id, "market_dir": market["market_dir"], "status": "complete",
              "started_at": checkpoint["started_at"], "completed_at": now(),
              "source_identity": identity, "collected_at": original["collected_at"],
              "settings": settings, "settings_provenance": "Config snapshot, with original source_identity chunk/API settings authoritative",
              "sides": details, "artifacts": coverage["artifacts"], "trades_sha256": checkpoint["trades_sha256"]}
    dump(path, result)
    return result, False


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    RUN_HISTORY.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load((ROOT / "configs/polymarket.yaml").read_text())
    data_dir = ROOT / config["data_dir"]
    markets = list(collect._json_lines(data_dir / "markets.jsonl"))
    collection_path = data_dir / "collection.json"
    untouched_collection_sha = sha(collection_path)
    offline_client = enforce_offline()
    assert isinstance(offline_client, OfflineClient)  # Never an OfficialClient/SDK instance.
    previous = json.loads(AUDIT.read_text()) if AUDIT.exists() else {}
    if previous:
        stamp = str(previous.get("updated_at", now())).replace(":", "").replace("+", "_")
        archived = RUN_HISTORY / f"{stamp}.json"
        if not archived.exists():
            dump(archived, previous)
    status = {"complete": False, "normalized_price_schema_version": SCHEMA, "note": NOTE,
              "started_at": previous.get("started_at", now()), "last_resumed_at": now(),
              "total_markets": len(markets), "processed_markets": 0, "reused_checkpoints": 0,
              "network_policy": "All client calls, OfficialClient construction, socket connections and DNS raise and append an error record",
              "network_attempts": NETWORK_ATTEMPTS, "collection_json_sha256": untouched_collection_sha,
              "execution_history": previous.get("execution_history", {}),
              "markets": [], "failures": []}
    dump(AUDIT, status)
    try:
        for market in markets:
            try:
                result, reused = rebuild_market(market, data_dir, config.get("collection", {}))
                status["markets"].append(result)
                status["reused_checkpoints"] += int(reused)
            except Exception as error:
                status["failures"].append({"market_id": str(market["market_id"]), "error": str(error),
                                           "traceback": traceback.format_exc(), "timestamp": now()})
                print(f"FAILED market {market['market_id']}: {error}", flush=True)
            status["processed_markets"] += 1
            if status["processed_markets"] % 25 == 0 or status["processed_markets"] == len(markets):
                status["updated_at"] = now()
                dump(AUDIT, status)
                print(f"Offline rebuilt {status['processed_markets']}/{len(markets)}, failures={len(status['failures'])}", flush=True)
        status["totals"] = {key: sum(side[key] for market in status["markets"] for side in market["sides"].values())
                            for key in ("row_increment", "exact_tick_row_increment", "raw_pages_read", "raw_points_read",
                                        "raw_exact_tick_points_before_dedup_and_time_filter")}
        for phase in ("before", "after"):
            status["totals"][phase] = {key: sum(side[phase][key] for market in status["markets"] for side in market["sides"].values())
                                       for key in ("rows", "exact_tick_rows", "size")}
        status["totals"]["sides"] = sum(len(market["sides"]) for market in status["markets"])
        status["complete"] = (not status["failures"] and not NETWORK_ATTEMPTS and
                              len(status["markets"]) == len(markets) and sha(collection_path) == untouched_collection_sha)
        status["collection_json_unchanged"] = sha(collection_path) == untouched_collection_sha
        status["completed_at"] = now()
    finally:
        status["updated_at"] = now()
        dump(AUDIT, status)
        stamp = status["updated_at"].replace(":", "").replace("+", "_")
        dump(RUN_HISTORY / f"{stamp}.json", status)
    print(json.dumps({key: status.get(key) for key in ("complete", "processed_markets", "reused_checkpoints", "totals", "network_attempts")}, indent=2), flush=True)
    if not status["complete"]:
        raise SystemExit("Offline rebuild incomplete; inspect exact_tick_rebuild.json and checkpoints")


if __name__ == "__main__":
    main()
