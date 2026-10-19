"""Official public data, with cached pages and explicit historical coverage.

The pinned polymarket-client SDK handles Gamma and Data API v2. Its public
client has no CLOB history method, so that documented endpoint is read through
its HTTP transport. No signing keys, account credentials or orders are used.
"""
from __future__ import annotations

import json
import hashlib
import logging
import math
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)
PRICE_COLUMNS = ["timestamp", "price", "resolution_seconds", "source"]
PRICE_NOTE = (
    "CLOB is requested at fidelity=1 minute. Data v2 explicitly probes 60, 300, "
    "1800 and 10800 second grains. Fine history expires; the documented minimum "
    "retention is 7/60/90 days for 1/5/30 minute data, with 3-hour history "
    "permanent from 2022-11-18. Actual returned coverage is recorded; empty "
    "responses are not evidence that a market never traded. Data v2 resolution=0 "
    "points may be synthetic settlement prices and are retained only in raw pages."
)
TRADE_NOTE = (
    "Public Data API v2 condition feeds expose a rolling three-year window, "
    "ignore start/end, and apply a minimum size of 0.01 tokens. taker_only=True "
    "records each fill once. API pagination completion does not establish "
    "complete lifetime trade coverage."
)


def _save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _json_lines(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _timestamp(value) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        return int(value)
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        return None
    return int(stamp.tz_localize("UTC").timestamp() if stamp.tzinfo is None else stamp.timestamp())


def _retry(operation, settings: dict):
    attempts = max(1, int(settings.get("retries", 3)))
    pause = float(settings.get("pause", 0.15))
    for attempt in range(attempts):
        try:
            result = operation()
            time.sleep(pause)
            return result
        except Exception:
            if attempt + 1 == attempts:
                raise
            time.sleep(max(pause, 1.0) * 2**attempt)


class OfficialClient:
    """Small adapter around the pinned SDK; capture unmodified API responses."""

    def __init__(self, settings: dict):
        import httpx
        from polymarket import PublicClient

        self.sdk = PublicClient()
        self.settings = settings
        self.raw = None
        # SDK 0.13 has no public timeout/response-hook constructor arguments.
        # Keep this version-specific transport customization in one place.
        for name in ("gamma", "data", "clob"):
            transport = getattr(self.sdk._ctx, name)
            transport._client.timeout = httpx.Timeout(float(settings.get("timeout", 30)))
            transport._client.event_hooks["response"].append(self._capture)
            if settings.get(f"{name}_url"):
                url = settings[f"{name}_url"].rstrip("/")
                transport._base_url = url
                transport._client.base_url = httpx.URL(url)

    def _capture(self, response):
        response.read()
        if response.is_success:
            self.raw = response.json()

    def _page(self, paginator, cursor):
        self.raw = None
        page = (paginator.from_cursor(cursor) if cursor is not None else paginator).first_page()
        if page.limit_reached:
            raise RuntimeError("Official API pagination depth limit reached")
        if page.has_more and not page.next_cursor:
            raise RuntimeError("Official API returned has_more without a cursor")
        if not isinstance(self.raw, dict):
            raise ValueError("Official API returned an invalid page")
        return self.raw, page.next_cursor if page.has_more else None

    def events(self, closed: bool, cursor=None):
        paginator = self.sdk.list_events(
            closed=closed, order="id", ascending=True,
            tag_slug=self.settings.get("event_tag_slug"),
            page_size=int(self.settings.get("event_page_size", 100)),
        )
        return self._page(paginator, cursor)

    def prices(self, token: str, start: int, end: int, grain: int, cursor=None):
        paginator = self.sdk.list_price_history(
            asset_id=token, start=start, end=end, bucket_seconds=grain,
            page_size=int(self.settings.get("price_page_size", 10000)),
        )
        return self._page(paginator, cursor)

    def clob_prices(self, token: str, start: int, end: int):
        result = self.sdk._ctx.clob.get_json("/prices-history", params={
            "market": token, "startTs": start, "endTs": end, "fidelity": 1,
        })
        if not isinstance(result, dict) or not isinstance(result.get("history"), list):
            raise ValueError("Invalid CLOB price-history response")
        return result

    def trades(self, condition: str, cursor=None):
        paginator = self.sdk.list_trades(
            condition_id=condition, taker_only=True, filter_type="TOKENS", filter_amount=0.01,
            page_size=int(self.settings.get("trade_page_size", 1000)),
        )
        return self._page(paginator, cursor)

    def close(self):
        self.sdk.close()


def _pages(fetch, directory: Path, settings: dict, refresh: bool = False,
           *, source_identity: dict | None = None) -> dict:
    """Checkpoint after each page; retrying a page replaces the same cache file."""
    manifest_path = directory / "manifest.json"
    if not refresh and source_identity is not None \
            and _load(manifest_path).get("source_identity") != source_identity:
        refresh = True
    if refresh and directory.exists():
        # An interruption during cleanup must not leave an old complete marker.
        manifest_path.unlink(missing_ok=True)
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = _load(manifest_path) or {"complete": False, "pages": 0, "next_cursor": None}
    if source_identity is not None:
        state["source_identity"] = source_identity
    if state["complete"]:
        return state
    cursor = state["next_cursor"]
    seen = set()
    try:
        while True:
            if cursor is not None:
                if cursor in seen:
                    raise RuntimeError("API repeated a pagination cursor")
                seen.add(cursor)
            payload, next_cursor = _retry(lambda: fetch(cursor), settings)
            if next_cursor is not None and next_cursor == cursor:
                raise RuntimeError("API did not advance its pagination cursor")
            _save(directory / f'{state["pages"]:06d}.json', payload)
            state.update(pages=state["pages"] + 1, next_cursor=next_cursor,
                         complete=next_cursor is None, error=None)
            _save(manifest_path, state)
            if state["complete"]:
                return state
            cursor = next_cursor
    except Exception as error:
        state.update(complete=False, error=str(error))
        _save(manifest_path, state)
        raise


def _page_rows(directory: Path, key: str):
    state = _load(directory / "manifest.json")
    for page in range(state.get("pages", 0)):
        payload = _load(directory / f"{page:06d}.json")
        rows = payload.get(key)
        if not isinstance(rows, list):
            raise ValueError(f"Missing {key} list in cached API page {directory}")
        yield from rows


def discover(config: dict, refresh: bool = False) -> dict:
    """Save all open and closed Gamma events, or import a JSON/JSONL source."""
    settings = config.get("collection", {})
    raw_dir = Path(config["data_dir"]) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path = raw_dir / "events.jsonl"
    status_path = raw_dir / "discovery.json"
    previous = _load(status_path)
    source = settings.get("events_file") or config.get("events_file")
    source_identity = {"gamma_url": settings.get("gamma_url", "https://gamma-api.polymarket.com")}
    if settings.get("event_tag_slug"):
        source_identity["event_tag_slug"] = settings["event_tag_slug"]
    if source:
        source_path = Path(source)
        with source_path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        source_identity = {"file": str(source_path.resolve()), "sha256": digest}
    if previous.get("source_identity") != source_identity:
        refresh = True
    if result_path.exists() and previous.get("complete") and not refresh:
        return previous
    status = {"complete": False, "event_count": 0, "source_identity": source_identity,
              "updated_at": datetime.now(UTC).isoformat()}
    _save(status_path, status)
    client = None
    temporary = result_path.with_suffix(".jsonl.tmp")
    try:
        if source:
            source_path = Path(source)
            if source_path.suffix.lower() == ".jsonl":
                events = _json_lines(source_path)
            else:
                payload = json.loads(source_path.read_text(encoding="utf-8"))
                events = payload if isinstance(payload, list) else payload["events"]
            status["source"] = str(source_path)
        else:
            client = OfficialClient(settings)
            directories = []
            for closed in (True, False):
                directory = raw_dir / "event_pages" / ("closed" if closed else "open")
                _pages(lambda cursor: client.events(closed, cursor), directory, settings, refresh,
                       source_identity=source_identity)
                directories.append(directory)
            events = (row for directory in directories for row in _page_rows(directory, "events"))
            status["source"] = "Gamma /events/keyset, closed=true and closed=false"
            if settings.get("event_tag_slug"):
                status["source"] += f", tag_slug={settings['event_tag_slug']}"
        seen = set()
        with temporary.open("w", encoding="utf-8") as handle:
            for event in events:
                event_id = str(event["id"])
                if event_id not in seen:
                    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                    seen.add(event_id)
        temporary.replace(result_path)
        status.update(complete=True, event_count=len(seen), path=str(result_path))
    except Exception as error:
        status["error"] = str(error)
        _save(status_path, status)
        raise
    finally:
        if client is not None:
            client.close()
    _save(status_path, status)
    return status


def _history_bounds(market: dict, now: int) -> tuple[int, int]:
    raw = market.get("raw_market", {})
    starts = [_timestamp(value) for value in (
        market.get("created_at"), market.get("metadata_available_at"), market.get("opened_at"),
        raw.get("createdAt"), raw.get("startDate"),
    ) if value]
    start = min(starts) if starts else _timestamp("2020-01-01T00:00:00Z")
    # Expiry is not necessarily the last trading time. Only actual closure bounds history.
    closed = _timestamp(market.get("closed_at") or raw.get("closedTime"))
    end = min(now, closed + 86400) if closed else now
    return start, end


def _price_rows(payload: list[dict], source: str, grain: int, start: int, end: int):
    rows = []
    for point in payload:
        stamp = _timestamp(point.get("timestamp", point.get("t")))
        price = float(point.get("price", point.get("p")))
        resolution = int(point.get("resolution_seconds", grain))
        if stamp is None or not math.isfinite(price) or not 0 <= price <= 1:
            raise ValueError("Invalid token price returned by official API")
        if resolution > 0 and start <= stamp < end:
            rows.append({"timestamp": stamp, "price": price,
                         "resolution_seconds": resolution, "source": source})
    return rows


def _price_coverage(frame: pd.DataFrame) -> dict:
    if frame.empty:
        return {"rows": 0, "first_timestamp": None, "last_timestamp": None,
                "resolution_seconds": [], "max_gap_seconds": None, "gap_count": 0}
    gaps = frame["timestamp"].diff()
    expected = frame["resolution_seconds"].shift().fillna(frame["resolution_seconds"])
    return {"rows": len(frame), "first_timestamp": int(frame["timestamp"].min()),
            "last_timestamp": int(frame["timestamp"].max()),
            "resolution_seconds": sorted(frame["resolution_seconds"].astype(int).unique().tolist()),
            "max_gap_seconds": int(gaps.max()) if len(frame) > 1 else None,
            "gap_count": int((gaps > expected * 2).sum())}


def _download_prices(client, token: str, directory: Path, side: str, start: int, end: int,
                     settings: dict, refresh: bool) -> dict:
    chunk_seconds = int(float(settings.get("price_chunk_days", 7)) * 86400)
    if not 0 < chunk_seconds <= 15 * 86400:
        raise ValueError("price_chunk_days must be in (0, 15]")
    rows, windows, errors = [], [], []
    raw_dir = directory / "raw" / side
    for left in range(start, end, chunk_seconds):
        right = min(left + chunk_seconds, end)
        window_dir = raw_dir / f"{left}-{right}"
        window_rows = []
        requests = []
        # The legacy endpoint may retain fine observations absent from Data v2.
        clob_path = window_dir / "clob_60.json"
        try:
            if refresh or not clob_path.exists():
                payload = _retry(lambda: client.clob_prices(token, left, right), settings)
                _save(clob_path, payload)
            payload = _load(clob_path)
            found = _price_rows(payload["history"], "clob", 60, left, right)
            window_rows.extend(found)
            requests.append({"source": "clob", "grain": 60, "rows": len(found), "complete": True})
        except Exception as error:
            requests.append({"source": "clob", "grain": 60, "complete": False, "error": str(error)})
            errors.append(f"CLOB {left}-{right}: {error}")
        for grain in (60, 300, 1800, 10800):
            page_dir = window_dir / f"data_{grain}"
            try:
                _pages(lambda cursor: client.prices(token, left, right, grain, cursor),
                       page_dir, settings, refresh)
                found = _price_rows(list(_page_rows(page_dir, "data")), "data_v2", grain, left, right)
                window_rows.extend(found)
                requests.append({"source": "data_v2", "grain": grain,
                                 "rows": len(found), "complete": True})
                fine = sorted({row["timestamp"] for row in window_rows if row["resolution_seconds"] <= grain})
                if fine and fine[0] <= left + grain and fine[-1] >= right - grain \
                        and all(b - a <= grain for a, b in zip(fine[:-1], fine[1:])):
                    break
            except Exception as error:
                requests.append({"source": "data_v2", "grain": grain,
                                 "complete": False, "error": str(error)})
                errors.append(f"Data v2 {left}-{right}/{grain}s: {error}")
        rows.extend(window_rows)
        windows.append({"start": left, "end": right, "requests": requests})
    frame = pd.DataFrame(rows, columns=PRICE_COLUMNS)
    if not frame.empty:
        # Preserve sparse observations; a fine quote on either side of a gap
        # does not prove fine history exists inside it. Prefer finer duplicate timestamps.
        frame = frame.sort_values(["timestamp", "resolution_seconds", "source"])
        frame = frame.drop_duplicates("timestamp", keep="first").reset_index(drop=True)
    destination = directory / f"{side}_prices.csv"
    temporary = destination.with_suffix(".csv.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(destination)
    return {"token_id": token, "api_complete": not errors, "requested_start": start,
            "requested_end": end, **_price_coverage(frame), "windows": windows,
            "errors": errors, "note": PRICE_NOTE}


def _download_trades(client, condition: str, directory: Path, settings: dict, refresh: bool) -> dict:
    raw_dir = directory / "raw" / "trades"
    error = None
    try:
        _pages(lambda cursor: client.trades(condition, cursor), raw_dir, settings, refresh)
    except Exception as caught:
        error = str(caught)
    path = directory / "trades.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    count, first, last = 0, None, None
    with temporary.open("w", encoding="utf-8") as handle:
        # Cursor pages are disjoint. Do not deduplicate identical fills: the API
        # can return separate executions with the same timestamp/price/quantity.
        for trade in _page_rows(raw_dir, "data"):
            handle.write(json.dumps(trade, ensure_ascii=False) + "\n")
            stamp = _timestamp(trade.get("timestamp"))
            if stamp is not None:
                first = min(first, stamp) if first is not None else stamp
                last = max(last, stamp) if last is not None else stamp
            count += 1
    temporary.replace(path)
    return {"api_complete": error is None, "lifetime_complete": False,
            "rows": count, "first_timestamp": first, "last_timestamp": last,
            "error": error, "note": TRADE_NOTE}


def _source_identity(market: dict, settings: dict) -> dict:
    return {
        "yes_token_id": market.get("yes_token_id") or market.get("token_id"),
        "no_token_id": market.get("no_token_id"), "condition_id": market.get("condition_id"),
        "created_at": market.get("created_at"), "opened_at": market.get("opened_at"),
        "metadata_available_at": market.get("metadata_available_at"),
        "closed_at": market.get("closed_at"), "sdk_version": "0.13.0",
        "raw_created_at": market.get("raw_market", {}).get("createdAt"),
        "raw_start_date": market.get("raw_market", {}).get("startDate"),
        "raw_closed_time": market.get("raw_market", {}).get("closedTime"),
        "data_url": settings.get("data_url", "https://data-api.polymarket.com"),
        "clob_url": settings.get("clob_url", "https://clob.polymarket.com"),
        "price_chunk_days": settings.get("price_chunk_days", 7),
    }


def _artifacts(directory: Path) -> dict:
    result = {}
    for name in ("yes_prices.csv", "no_prices.csv", "trades.jsonl"):
        path = directory / name
        if path.exists():
            stat = path.stat()
            result[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return result


def _download_market(market: dict, data_dir: Path, settings: dict, now: int,
                     refresh: bool, get_client) -> dict:
    """Collect one market independently; keep its existing cache and coverage contract."""
    directory = data_dir / market["market_dir"]
    directory.mkdir(parents=True, exist_ok=True)
    coverage_path = directory / "coverage.json"
    previous = _load(coverage_path)
    identity = _source_identity(market, settings)
    identity_changed = previous.get("source_identity") != identity
    raw_dir = directory / "raw"
    raw_identity_path = raw_dir / "source_identity.json"
    raw_identity = _load(raw_identity_path)
    raw_identity_changed = raw_identity != identity if raw_identity else identity_changed
    artifacts = _artifacts(directory)
    if previous.get("api_complete") and not refresh and not identity_changed and not raw_identity_changed \
            and len(artifacts) == 3 and previous.get("artifacts") == artifacts:
        return previous
    # Switch every raw stream before advertising the new identity in coverage.
    # An interrupt after Yes must never leave old No/trades pages reusable as the
    # new source. Rename retains old evidence atomically; new-source partial pages
    # remain resumable even if coverage has not yet been written.
    if raw_identity_changed and raw_dir.exists():
        archive = directory / "raw_archive" / str(time.time_ns())
        archive.parent.mkdir(parents=True, exist_ok=True)
        raw_dir.rename(archive)
    _save(raw_identity_path, identity)
    market_refresh = refresh
    start, end = _history_bounds(market, now)
    coverage = {"market_id": market["market_id"], "api_complete": False,
                "source_identity": identity, "collected_at": datetime.now(UTC).isoformat()}
    _save(coverage_path, coverage)
    try:
        if start >= end:
            coverage["skipped"] = "Market has no historical collection window yet"
            for side in ("yes", "no"):
                path = directory / f"{side}_prices.csv"
                if not path.exists():
                    pd.DataFrame(columns=PRICE_COLUMNS).to_csv(path, index=False)
            (directory / "trades.jsonl").touch(exist_ok=True)
        else:
            client = get_client()
            for side, key in (("yes", "yes_token_id"), ("no", "no_token_id")):
                token = market.get(key) or (market.get("token_id") if side == "yes" else None)
                if not token:
                    coverage[side] = {"api_complete": False, "errors": [f"Missing {key}"]}
                else:
                    coverage[side] = _download_prices(client, str(token), directory, side, start, end,
                                                      settings, market_refresh)
            if market.get("condition_id"):
                coverage["trades"] = _download_trades(client, market["condition_id"], directory,
                                                       settings, market_refresh)
            else:
                coverage["trades"] = {"api_complete": False, "error": "Missing condition_id"}
            coverage["api_complete"] = all(coverage[key]["api_complete"] for key in ("yes", "no", "trades"))
    except Exception as error:
        coverage["error"] = str(error)
    coverage["artifacts"] = _artifacts(directory)
    _save(coverage_path, coverage)
    return coverage


def download(config: dict, refresh: bool = False) -> dict:
    """Download classified markets; each worker owns an isolated SDK client."""
    data_dir = Path(config["data_dir"])
    markets_path = data_dir / "markets.jsonl"
    if not markets_path.exists():
        raise FileNotFoundError(f"Run classification first: {markets_path}")
    settings = config.get("collection", {})
    workers = int(settings.get("workers", 1))
    if workers < 1:
        raise ValueError("collection.workers must be at least 1")
    markets = list(_json_lines(markets_path))
    identifiers, directories = set(), set()
    for market in markets:
        market_id = str(market["market_id"])
        directory = (data_dir / market["market_dir"]).resolve()
        if market_id in identifiers:
            raise ValueError(f"Duplicate market_id in collection input: {market_id}")
        if directory in directories:
            raise ValueError(f"Duplicate market_dir in collection input: {market['market_dir']}")
        identifiers.add(market_id)
        directories.add(directory)
    now = int(datetime.now(UTC).timestamp())
    status = {"complete": False, "markets": 0, "processed": 0, "total": len(markets),
              "completed": 0, "failures": 0, "failed_markets": [], "skipped_future_markets": [],
              "workers": workers, "updated_at": datetime.now(UTC).isoformat()}
    _save(data_dir / "collection.json", status)
    local = threading.local()
    clients = []
    clients_lock = threading.Lock()

    def get_client():
        if not hasattr(local, "client"):
            local.client = OfficialClient(settings)
            with clients_lock:
                clients.append(local.client)
        return local.client

    def collect_one(market):
        try:
            return _download_market(market, data_dir, settings, now, refresh, get_client)
        except Exception as error:
            # A corrupt market/cache must not prevent independent markets from finishing.
            coverage = {"market_id": market["market_id"], "api_complete": False,
                        "error": str(error), "collected_at": datetime.now(UTC).isoformat()}
            _save(data_dir / market["market_dir"] / "coverage.json", coverage)
            return coverage

    def record(coverage):
        status["processed"] += 1
        market_id = coverage["market_id"]
        if coverage.get("skipped"):
            status["skipped_future_markets"].append(market_id)
        else:
            status["markets"] += 1
            if coverage["api_complete"]:
                status["completed"] += 1
            else:
                status["failures"] += 1
                status["failed_markets"].append(market_id)
        status["updated_at"] = datetime.now(UTC).isoformat()
        _save(data_dir / "collection.json", status)
        LOGGER.info("Markets %s/%s: completed=%s failures=%s skipped=%s; market=%s api_complete=%s",
                    status["processed"], status["total"], status["completed"], status["failures"],
                    len(status["skipped_future_markets"]), market_id, coverage["api_complete"])

    try:
        if workers == 1:
            for market in markets:
                record(collect_one(market))
        elif markets:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="polymarket") as executor:
                futures = [executor.submit(collect_one, market) for market in markets]
                for future in as_completed(futures):
                    record(future.result())
        status["complete"] = not status["failed_markets"]
    except Exception as error:
        status["error"] = str(error)
        raise
    finally:
        for client in clients:
            client.close()
        status["updated_at"] = datetime.now(UTC).isoformat()
        _save(data_dir / "collection.json", status)
    if status["failed_markets"]:
        raise RuntimeError(f'{len(status["failed_markets"])} markets incomplete; see {data_dir / "collection.json"}')
    return status
