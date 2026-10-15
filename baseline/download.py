"""Download AKShare prices and BaoStock's historical CSI 300 snapshots."""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing
import re
import socket
import time
from datetime import date
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)
BAOSTOCK_CALENDAR_START = "1990-12-19"
PRICE_COLUMNS = [
    "日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "振幅", "涨跌幅", "涨跌额", "换手率"
]
MEMBERSHIP_NOTE = (
    "BaoStock historical CSI 300 snapshots update weekly. Files are queried as of each "
    "trading date; updateDate is retained and is not an official index effective date."
)
_WORKER_RETRIES = 1
_WORKER_PAUSE = 0.0


def _write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _save_metadata(path: Path, metadata: dict) -> None:
    _write_text(path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")


def _retry(operation, retries: int, pause: float):
    """Retries means total attempts, with a finite pause between attempts."""
    for attempt in range(retries):
        try:
            return operation()
        except Exception as error:
            if attempt + 1 == retries:
                raise
            LOGGER.warning("Attempt %d/%d failed: %s", attempt + 1, retries, error)
            time.sleep(pause)


def _result_frame(result) -> pd.DataFrame:
    """Consume pagination without BaoStock's pandas.DataFrame.append-based get_data."""
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock {result.error_code}: {result.error_msg}")
    rows = []
    while result.next():
        rows.append(result.get_row_data())
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock pagination {result.error_code}: {result.error_msg}")
    return pd.DataFrame(rows, columns=result.fields)


def _baostock_login(bs) -> None:
    result = bs.login()
    if result.error_code != "0":
        raise RuntimeError(f"BaoStock login {result.error_code}: {result.error_msg}")


def _baostock_query(method, retries: int, pause: float, **kwargs) -> pd.DataFrame:
    def attempt():
        try:
            return _result_frame(method(**kwargs))
        except Exception:
            # Each process owns its own connection; a failed socket must be replaced.
            import baostock as bs
            import baostock.common.context as context

            connection = getattr(context, "default_socket", None)
            if connection is not None:
                connection.close()
            _baostock_login(bs)
            raise
    frame = _retry(attempt, retries, pause)
    time.sleep(pause)
    return frame


def _worker_logout() -> None:
    import baostock as bs

    try:
        bs.logout()
    except Exception as error:
        LOGGER.warning("BaoStock worker logout: %s", error)


def _init_membership_worker(timeout: float, retries: int, pause: float) -> None:
    import baostock as bs

    global _WORKER_RETRIES, _WORKER_PAUSE
    socket.setdefaulttimeout(timeout)
    _WORKER_RETRIES, _WORKER_PAUSE = retries, pause
    _retry(lambda: _baostock_login(bs), retries, pause)
    atexit.register(_worker_logout)


def _membership_task(day: str) -> tuple[str, list[str], str | None, str | None]:
    import baostock as bs

    try:
        frame = _baostock_query(bs.query_hs300_stocks, _WORKER_RETRIES, _WORKER_PAUSE, date=day)
        members, update_date = _members(frame, day)
        return day, members, update_date, None
    except Exception as error:
        return day, [], None, str(error)


def _parallel_memberships(days: list[str], workers: int, timeout: float, retries: int, pause: float):
    """Bound pending requests to one batch of workers; only the parent writes files."""
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_membership_worker, initargs=(timeout, retries, pause),
    )
    try:
        for offset in range(0, len(days), workers):
            futures = {
                pool.submit(_membership_task, day): day
                for day in days[offset:offset + workers]
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    result = futures[future], [], None, str(error)
                yield result
    finally:
        # On interruption, at most `workers` requests exist, not the entire history.
        pool.shutdown(wait=True, cancel_futures=True)


def _calendar_dates(frame: pd.DataFrame, start: str, end: str) -> list[str]:
    if frame.empty or not {"calendar_date", "is_trading_day"}.issubset(frame.columns):
        raise ValueError("BaoStock returned an empty or malformed calendar")
    dates = pd.to_datetime(frame["calendar_date"], errors="raise").dt.strftime("%Y-%m-%d")
    expected = pd.date_range(max(start, BAOSTOCK_CALENDAR_START), end).strftime("%Y-%m-%d").tolist()
    if dates.duplicated().any() or sorted(dates.tolist()) != expected:
        raise ValueError(f"BaoStock calendar does not completely cover {start} through {end}")
    flags = frame["is_trading_day"].astype(str)
    if not flags.isin(["0", "1"]).all():
        raise ValueError("BaoStock calendar contains invalid trading-day flags")
    return sorted(dates[flags == "1"].tolist())


def _members(frame: pd.DataFrame, query_date: str) -> tuple[list[str], str]:
    required = {"updateDate", "code", "code_name"}
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError(f"Empty or malformed CSI 300 membership for {query_date}")
    source_dates = pd.to_datetime(frame["updateDate"], errors="raise").dt.strftime("%Y-%m-%d")
    if source_dates.isna().any() or source_dates.nunique() != 1:
        raise ValueError(f"Inconsistent membership updateDate for {query_date}")
    update_date = source_dates.iloc[0]
    if update_date > query_date:
        raise ValueError(f"Future membership snapshot {update_date} returned for {query_date}")
    codes = frame["code"].astype(str)
    if not codes.str.fullmatch(r"(?:sh|sz)\.\d{6}").all() or codes.duplicated().any():
        raise ValueError(f"Invalid or duplicate membership codes for {query_date}")
    symbols = sorted(codes.str.replace(".", "", regex=False).str.upper().tolist())
    return symbols, update_date


def _price_frame(adjusted: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    if adjusted.empty or raw.empty:
        raise ValueError("AKShare returned empty adjusted or unadjusted price history")
    missing = set(PRICE_COLUMNS) - set(adjusted.columns)
    if missing or not {"日期", "收盘"}.issubset(raw.columns):
        raise ValueError(f"AKShare returned incomplete columns: {sorted(missing)}")
    adjusted = adjusted.copy()
    raw = raw[["日期", "收盘"]].copy().rename(columns={"收盘": "raw_close"})
    for frame in (adjusted, raw):
        frame["日期"] = pd.to_datetime(frame["日期"], errors="raise").dt.strftime("%Y-%m-%d")
        if frame["日期"].isna().any() or frame["日期"].duplicated().any():
            raise ValueError("AKShare history contains missing or duplicate dates")
    if set(adjusted["日期"]) != set(raw["日期"]):
        raise ValueError("Adjusted and unadjusted AKShare histories have different dates")
    result = adjusted.merge(raw, on="日期", validate="one_to_one").sort_values("日期")
    numeric = PRICE_COLUMNS[1:] + ["raw_close"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    if not (result[["开盘", "最高", "最低", "收盘", "raw_close"]] > 0).all().all():
        raise ValueError("AKShare history contains missing or nonpositive OHLC prices")
    if not (result[["成交量", "成交额"]] >= 0).all().all():
        raise ValueError("AKShare history contains missing or negative volume/amount")
    return result.reset_index(drop=True)


def download(config: dict, refresh: bool = False) -> dict:
    """Fetch all history for the union of point-in-time research constituents.

    Successful files and metadata are checkpointed immediately. Failed days or stocks
    are listed in failures.json; any failure makes the command fail after attempting
    the remaining work. A rerun resumes incomplete entries. ``refresh`` refetches all.
    """
    import akshare as ak
    import baostock as bs

    root = Path(config["data_dir"])
    for directory in (root, root / "instruments", root / "data"):
        directory.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(config.get("download_start", "1990-01-01")).strftime("%Y-%m-%d")
    end = pd.Timestamp(config.get("download_end") or date.today()).strftime("%Y-%m-%d")
    research_start = pd.Timestamp(config["research_start"]).strftime("%Y-%m-%d")
    research_end = pd.Timestamp(config["research_end"]).strftime("%Y-%m-%d")
    if not start <= research_start <= research_end <= end:
        raise ValueError("Download dates must contain the entire research period")
    settings = config.get("download", {})
    retries = int(settings.get("retries", 3))
    timeout = float(settings.get("timeout", 30))
    pause = float(settings.get("pause", 0.2))
    workers = int(settings.get("membership_workers", 4))
    if retries < 1 or timeout <= 0 or pause < 0 or workers < 1:
        raise ValueError("download requires retries/workers >= 1, timeout > 0 and pause >= 0")

    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    metadata.update({"schema_version": 1, "membership_source": "baostock", "membership_note": MEMBERSHIP_NOTE})
    metadata.setdefault("memberships", {})
    metadata.setdefault("prices", {})
    failures = []
    symbols = set()

    def checkpoint():
        _save_metadata(metadata_path, metadata)

    def failed(stage: str, item: str, error):
        failures.append({"stage": stage, "item": item, "error": str(error)})
        LOGGER.error("%s %s: %s", stage, item, error)

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    logged_in = False
    try:
        _retry(lambda: _baostock_login(bs), retries, pause)
        logged_in = True
        calendar_path = root / "calendar.txt"
        request_range = {"start": start, "end": end}
        if refresh or metadata.get("calendar") != request_range or not calendar_path.exists():
            metadata.pop("calendar", None)
            checkpoint()
            frame = _baostock_query(bs.query_trade_dates, retries, pause, start_date=start, end_date=end)
            calendar = _calendar_dates(frame, start, end)
            _write_text(calendar_path, "\n".join(calendar) + "\n")
            metadata["calendar"] = request_range
            checkpoint()
        else:
            calendar = calendar_path.read_text().splitlines()
        research_dates = [day for day in calendar if research_start <= day <= research_end]
        if not research_dates:
            raise ValueError("No trading dates in the research period")

        pending = []
        for day in research_dates:
            path = root / "instruments" / f"{day}.txt"
            if not refresh and day in metadata["memberships"] and path.exists():
                symbols.update(path.read_text().splitlines())
                continue
            metadata["memberships"].pop(day, None)
            pending.append(day)
        checkpoint()
        completed = len(research_dates) - len(pending)

        def save_membership(day, members, update_date, error):
            nonlocal completed
            if error is not None:
                failed("membership", day, error)
            else:
                _write_text(root / "instruments" / f"{day}.txt", "\n".join(members) + "\n")
                metadata["memberships"][day] = {"updateDate": update_date, "count": len(members)}
                checkpoint()
                symbols.update(members)
            completed += 1
            if completed % 50 == 0 or completed == len(research_dates):
                LOGGER.info("CSI 300 membership: %d/%d dates", completed, len(research_dates))

        if workers == 1:
            for day in pending:
                try:
                    frame = _baostock_query(bs.query_hs300_stocks, retries, pause, date=day)
                    members, update_date = _members(frame, day)
                    save_membership(day, members, update_date, None)
                except Exception as error:
                    save_membership(day, [], None, str(error))
        elif pending:
            LOGGER.info("Query %d pending membership dates with %d processes", len(pending), workers)
            for result in _parallel_memberships(pending, workers, timeout, retries, pause):
                save_membership(*result)
    except Exception as error:
        failed("calendar/login", f"{start}:{end}", error)
    finally:
        if logged_in:
            try:
                bs.logout()
            except Exception as error:
                LOGGER.warning("BaoStock logout: %s", error)
        socket.setdefaulttimeout(old_timeout)

    for position, symbol in enumerate(sorted(symbols), 1):
        path = root / "data" / f"{symbol}.csv"
        previous = metadata["prices"].get(symbol, {})
        if not refresh and path.exists() and previous.get("start") == start and previous.get("end") == end:
            continue
        metadata["prices"].pop(symbol, None)
        checkpoint()
        try:
            if not re.fullmatch(r"(?:SH|SZ)\d{6}", symbol):
                raise ValueError(f"Invalid instrument symbol: {symbol}")

            def fetch(adjust):
                def attempt():
                    frame = ak.stock_zh_a_hist(
                        symbol=symbol[2:], period="daily", start_date=start.replace("-", ""),
                        end_date=end.replace("-", ""), adjust=adjust, timeout=timeout,
                    )
                    if frame.empty:
                        raise ValueError(f"Empty AKShare history ({adjust or 'unadjusted'})")
                    return frame
                frame = _retry(attempt, retries, pause)
                time.sleep(pause)
                return frame

            frame = _price_frame(fetch("hfq"), fetch(""))
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
            metadata["prices"][symbol] = {
                "start": start, "end": end, "rows": len(frame),
                "first_date": frame["日期"].iloc[0], "last_date": frame["日期"].iloc[-1],
                "adjust": "hfq", "raw_close": "unadjusted AKShare close",
            }
            checkpoint()
        except Exception as error:
            failed("price", symbol, error)
        LOGGER.info("AKShare history: %d/%d symbols (%s)", position, len(symbols), symbol)

    _save_metadata(root / "failures.json", failures)
    if failures:
        raise RuntimeError(f"Download incomplete: {len(failures)} failures; see {root / 'failures.json'}")
    LOGGER.info("Download complete: %d historical constituents", len(symbols))
    return {
        "data_dir": str(root), "metadata": str(metadata_path),
        "calendar_first": calendar[0], "calendar_last": calendar[-1],
        "research_dates": len(research_dates), "historical_constituents": len(symbols),
        "membership_note": MEMBERSHIP_NOTE,
    }
