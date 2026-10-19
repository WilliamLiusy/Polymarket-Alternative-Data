"""Download AKShare prices and BaoStock's historical CSI 300 snapshots."""

from __future__ import annotations

import atexit
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
import json
import logging
import multiprocessing
import re
import socket
import time
from datetime import date
from pathlib import Path

import numpy as np
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
_WORKER_BLOCKED = None
_WORKER_STOP = None


class BaoStockBlockedError(RuntimeError):
    """BaoStock's explicit blacklist response must stop further requests."""

    error_code = "10001011"


def _is_baostock_blocked(error) -> bool:
    # Workers serialize their errors as strings; retain the exact source code.
    return str(error).startswith(f"BaoStock {BaoStockBlockedError.error_code}:")


def _raise_if_worker_blocked() -> None:
    if _WORKER_BLOCKED is not None:
        raise BaoStockBlockedError(_WORKER_BLOCKED)
    if _WORKER_STOP is not None and _WORKER_STOP.is_set():
        raise BaoStockBlockedError("BaoStock 10001011: another worker received a blacklist response")


def _check_baostock_result(result, stage: str = "") -> None:
    if result.error_code == "0":
        return
    detail = f"{stage}: {result.error_msg}" if stage else result.error_msg
    message = f"BaoStock {result.error_code}: {detail}"
    if result.error_code == BaoStockBlockedError.error_code:
        if _WORKER_STOP is not None:
            _WORKER_STOP.set()
        raise BaoStockBlockedError(message)
    raise RuntimeError(message)


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
        except BaoStockBlockedError:
            raise
        except Exception as error:
            if attempt + 1 == retries:
                raise
            LOGGER.warning("Attempt %d/%d failed: %s", attempt + 1, retries, error)
            time.sleep(pause)


def _result_frame(result) -> pd.DataFrame:
    """Consume pagination without BaoStock's pandas.DataFrame.append-based get_data."""
    _check_baostock_result(result)
    rows = []
    while True:
        _raise_if_worker_blocked()
        if not result.next():
            break
        rows.append(result.get_row_data())
    _check_baostock_result(result, "pagination")
    return pd.DataFrame(rows, columns=result.fields)


def _baostock_login(bs) -> None:
    _raise_if_worker_blocked()
    result = bs.login()
    _check_baostock_result(result, "login")


def _baostock_query(method, retries: int, pause: float, **kwargs) -> pd.DataFrame:
    def attempt():
        _raise_if_worker_blocked()
        try:
            return _result_frame(method(**kwargs))
        except BaoStockBlockedError:
            raise
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

    if _WORKER_BLOCKED is not None or (_WORKER_STOP is not None and _WORKER_STOP.is_set()):
        return
    try:
        bs.logout()
    except Exception as error:
        LOGGER.warning("BaoStock worker logout: %s", error)


def _init_membership_worker(timeout: float, retries: int, pause: float, stop_event=None) -> None:
    import baostock as bs

    global _WORKER_RETRIES, _WORKER_PAUSE, _WORKER_BLOCKED, _WORKER_STOP
    socket.setdefaulttimeout(timeout)
    _WORKER_RETRIES, _WORKER_PAUSE = retries, pause
    _WORKER_BLOCKED = None
    _WORKER_STOP = stop_event
    try:
        _retry(lambda: _baostock_login(bs), retries, pause)
    except BaoStockBlockedError as error:
        # Let the first task report the code, rather than losing it in BrokenProcessPool.
        _WORKER_BLOCKED = str(error)
        return
    atexit.register(_worker_logout)


def _membership_task(day: str) -> tuple[str, list[str], str | None, str | None]:
    import baostock as bs

    global _WORKER_BLOCKED
    if _WORKER_BLOCKED is not None:
        return day, [], None, _WORKER_BLOCKED
    try:
        frame = _baostock_query(bs.query_hs300_stocks, _WORKER_RETRIES, _WORKER_PAUSE, date=day)
        members, update_date = _members(frame, day)
        return day, members, update_date, None
    except BaoStockBlockedError as error:
        _WORKER_BLOCKED = str(error)
        return day, [], None, _WORKER_BLOCKED
    except Exception as error:
        return day, [], None, str(error)


def _parallel_memberships(days: list[str], workers: int, timeout: float, retries: int, pause: float):
    """Bound pending requests to one batch of workers; only the parent writes files."""
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    pool = ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=_init_membership_worker, initargs=(timeout, retries, pause, stop_event),
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
                if _is_baostock_blocked(result[3]):
                    stop_event.set()
                yield result
                if _is_baostock_blocked(result[3]):
                    return
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


def _invalid_adjusted_ohlc(frame: pd.DataFrame) -> pd.Series:
    values = frame[["开盘", "最高", "最低", "收盘"]]
    return ~(np.isfinite(values) & (values > 0)).all(axis=1)


def _sina_assignment(text: str, variable: str, description: str):
    """Parse one JSON-valued assignment, allowing only trailing block comments."""
    prefix = re.match(rf"\s*var\s+{re.escape(variable)}\s*=\s*", text)
    try:
        if prefix is None:
            raise ValueError("Wrong assignment name")
        value, end = json.JSONDecoder().raw_decode(text[prefix.end():])
        suffix = text[prefix.end() + end:]
        comments = r"(?:/\*[\s\S]*?\*/\s*)*"
        if re.fullmatch(rf"\s*{comments};?\s*{comments}", suffix) is None:
            raise ValueError("Unexpected content after assignment")
    except ValueError as error:
        raise ValueError(f"Malformed Sina {description} payload") from error
    return value


def _sina_history(symbol: str, timeout: float) -> pd.DataFrame:
    """Decode Sina's unadjusted bars without share-count requests or filling dates.

    AKShare 1.19.1 supplies the endpoint and decoder. Its public daily helper also
    merges share-count history and forward-fills, neither of which is appropriate
    when only an independently observed unadjusted close is required.
    """
    import requests
    from akshare.stock.stock_zh_a_sina import (
        hk_js_decode, py_mini_racer, zh_sina_a_stock_hist_url,
    )

    symbol = symbol.lower()
    if not re.fullmatch(r"(?:sh|sz)\d{6}", symbol):
        raise ValueError(f"Invalid Sina instrument symbol: {symbol}")
    response = requests.get(zh_sina_a_stock_hist_url.format(symbol), timeout=timeout)
    response.raise_for_status()
    encoded = _sina_assignment(response.text, f"KLC_K2_{symbol}", "history")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError(f"Malformed Sina history payload for {symbol}")
    decoder = py_mini_racer.MiniRacer()
    decoder.eval(hk_js_decode)
    raw = pd.DataFrame(decoder.call("d", encoded))
    if raw.empty or not {"date", "close"}.issubset(raw.columns):
        raise ValueError(f"Empty or incomplete Sina history for {symbol}")
    raw = raw.rename(columns={"date": "日期"})
    raw["日期"] = pd.to_datetime(raw["日期"], format="ISO8601", utc=True, errors="raise").dt.strftime("%Y-%m-%d")
    if raw["日期"].isna().any() or raw["日期"].duplicated().any():
        raise ValueError("Sina history contains missing or duplicate dates")
    raw["close"] = pd.to_numeric(raw["close"], errors="raise")
    if not (np.isfinite(raw["close"]) & (raw["close"] > 0)).all():
        raise ValueError("Sina history contains nonfinite or nonpositive unadjusted close prices")
    return raw.sort_values("日期").reset_index(drop=True)


def _sina_raw_close(symbol: str, timeout: float) -> pd.DataFrame:
    return _sina_history(symbol, timeout)[["日期", "close"]].rename(columns={"close": "收盘"})


def _sina_hfq_factors(symbol: str, timeout: float) -> pd.DataFrame:
    """Read effective-date adjustment factors, never execute remote JavaScript."""
    import requests
    from akshare.stock.stock_zh_a_sina import zh_sina_a_stock_hfq_url

    symbol = symbol.lower()
    if not re.fullmatch(r"(?:sh|sz)\d{6}", symbol):
        raise ValueError(f"Invalid Sina instrument symbol: {symbol}")
    response = requests.get(zh_sina_a_stock_hfq_url.format(symbol), timeout=timeout)
    response.raise_for_status()
    payload = _sina_assignment(response.text, f"{symbol}hfq", "hfq factor")
    if not isinstance(payload, dict) or "data" not in payload:
        raise ValueError(f"Malformed Sina hfq factor payload for {symbol}")
    factors = pd.DataFrame(payload["data"])
    if factors.empty or set(factors.columns) != {"d", "f"}:
        raise ValueError(f"Empty or incomplete Sina hfq factors for {symbol}")
    factors = factors.rename(columns={"d": "日期", "f": "hfq_factor"})
    factors["日期"] = pd.to_datetime(factors["日期"], errors="raise").dt.normalize()
    if factors["日期"].isna().any() or factors["日期"].duplicated().any():
        raise ValueError("Sina hfq factors contain missing or duplicate dates")
    factors["hfq_factor"] = pd.to_numeric(factors["hfq_factor"], errors="raise")
    if not (np.isfinite(factors["hfq_factor"]) & (factors["hfq_factor"] > 0)).all():
        raise ValueError("Sina hfq factors must be finite and positive")
    return factors.sort_values("日期").reset_index(drop=True)


def _sina_hfq_frame(history: pd.DataFrame, factors: pd.DataFrame, symbol: str,
                    start: str, end: str) -> pd.DataFrame:
    """Apply only past/effective factors to actual raw bars, with SDK price rounding."""
    required = ["日期", "open", "high", "low", "close", "volume", "amount"]
    if history.empty or not set(required).issubset(history.columns):
        raise ValueError("Sina history has incomplete OHLCV/amount columns")
    bars = history.loc[history["日期"].between(start, end), required].copy()
    if bars.empty:
        raise ValueError("Sina history has no bars in the requested interval")
    bars["日期"] = pd.to_datetime(bars["日期"], errors="raise")
    bars = bars.sort_values("日期")
    bars = pd.merge_asof(bars, factors.sort_values("日期"), on="日期", direction="backward", allow_exact_matches=True)
    if bars["hfq_factor"].isna().any():
        raise ValueError("Sina hfq factor is unavailable on or before a requested price date")
    numeric = ["open", "high", "low", "close", "volume", "amount"]
    bars[numeric] = bars[numeric].apply(pd.to_numeric, errors="raise")
    adjusted = pd.DataFrame({"日期": bars["日期"], "股票代码": symbol[2:]})
    for english, chinese in (("open", "开盘"), ("high", "最高"), ("low", "最低"), ("close", "收盘")):
        adjusted[chinese] = (bars[english] * bars["hfq_factor"]).round(2)
    adjusted["成交量"] = bars["volume"] / 100.0  # Sina shares -> AKShare Eastmoney hands.
    adjusted["成交额"] = bars["amount"]
    previous_close = adjusted["收盘"].shift(1)
    adjusted["振幅"] = (adjusted["最高"] - adjusted["最低"]) / previous_close * 100.0
    adjusted["涨跌幅"] = (adjusted["收盘"] / previous_close - 1.0) * 100.0
    adjusted["涨跌额"] = adjusted["收盘"] - previous_close
    adjusted["换手率"] = np.nan  # Historical share counts are not requested or invented.
    raw = bars[["日期", "close"]].rename(columns={"close": "收盘"})
    return _price_frame(adjusted, raw)


def _price_frame(adjusted: pd.DataFrame, raw: pd.DataFrame, *, allow_extra_raw_dates: bool = False) -> pd.DataFrame:
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
    raw["raw_close"] = pd.to_numeric(raw["raw_close"], errors="raise")
    if not (np.isfinite(raw["raw_close"]) & (raw["raw_close"] > 0)).all():
        raise ValueError("AKShare history contains nonfinite or nonpositive unadjusted close prices")
    if allow_extra_raw_dates:
        # Sina supplies all available history. Select actual observations only;
        # missing hfq dates must still fail the exact-date check below.
        raw = raw.loc[raw["日期"].isin(adjusted["日期"])]
    if set(adjusted["日期"]) != set(raw["日期"]):
        raise ValueError("Adjusted and unadjusted AKShare histories have different dates")
    result = adjusted.merge(raw, on="日期", validate="one_to_one").sort_values("日期")
    numeric = PRICE_COLUMNS[1:] + ["raw_close"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="raise")
    quantities = result[["成交量", "成交额"]]
    if not (np.isfinite(quantities) & (quantities >= 0)).all().all():
        raise ValueError("AKShare history contains nonfinite or negative volume/amount")
    # Some hfq histories contain isolated nonpositive prices from the source.
    # Keep them auditable in CSV; prepare.convert_bars masks these complete bars.
    if _invalid_adjusted_ohlc(result).all():
        raise ValueError("AKShare history has no usable adjusted OHLC bars")
    return result.reset_index(drop=True)


def download(config: dict, refresh: bool = False) -> dict:
    """Fetch all history for the union of point-in-time research constituents.

    Successful files and metadata are checkpointed immediately. Ordinary failures
    are listed in failures.json while remaining work is attempted. BaoStock code
    10001011 instead stops new requests immediately, including retries and relogins.
    Any failure makes the command fail. A rerun resumes incomplete entries;
    ``refresh`` refetches all.
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
    workers = int(settings.get("membership_workers", 1))
    allow_sina_fallback = settings.get("allow_sina_fallback", False)
    if type(allow_sina_fallback) is not bool:
        raise ValueError("download.allow_sina_fallback must be a boolean")
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
        entry = {"stage": stage, "item": item, "error": str(error)}
        if _is_baostock_blocked(error):
            entry["error_code"] = BaoStockBlockedError.error_code
        failures.append(entry)
        LOGGER.error("%s %s: %s", stage, item, error)

    old_timeout = socket.getdefaulttimeout()
    socket.setdefaulttimeout(timeout)
    logged_in = False
    blocked = None
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
            nonlocal completed, blocked
            if error is not None:
                failed("membership", day, error)
                if _is_baostock_blocked(error):
                    blocked = BaoStockBlockedError(str(error))
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
                if blocked is not None:
                    break
        elif pending:
            LOGGER.info("Query %d pending membership dates with %d processes", len(pending), workers)
            with closing(_parallel_memberships(pending, workers, timeout, retries, pause)) as results:
                for result in results:
                    save_membership(*result)
                    if blocked is not None:
                        break
    except BaoStockBlockedError as error:
        blocked = error
        failed("calendar/login", f"{start}:{end}", error)
    except Exception as error:
        failed("calendar/login", f"{start}:{end}", error)
    finally:
        if logged_in and blocked is None:
            try:
                bs.logout()
            except Exception as error:
                LOGGER.warning("BaoStock logout: %s", error)
        socket.setdefaulttimeout(old_timeout)

    if blocked is not None:
        _save_metadata(root / "failures.json", failures)
        raise BaoStockBlockedError(
            f"{blocked}; download stopped without further requests; see {root / 'failures.json'}"
        ) from blocked

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

            try:
                adjusted = fetch("hfq")
            except Exception as error:
                if not allow_sina_fallback:
                    raise
                LOGGER.warning("%s: Eastmoney hfq history failed; using Sina history and factors: %s", symbol, error)
                history = _retry(lambda: _sina_history(symbol, timeout), retries, pause)
                time.sleep(pause)
                factors = _retry(lambda: _sina_hfq_factors(symbol, timeout), retries, pause)
                time.sleep(pause)
                frame = _sina_hfq_frame(history, factors, symbol, start, end)
                sources = {
                    "price_source": "sina.hisdata_klc2", "raw_close_source": "sina.hisdata_klc2",
                    "price_fallback_reason": str(error), "adjustment_source": "sina.hfq",
                    "adjustment_method": "round(raw OHLC * effective hfq factor, 2); backward asof including effective date",
                    "volume_conversion": "Sina shares / 100 -> hands", "amount_unit": "CNY",
                    "derived_fields": {
                        "振幅": "(hfq high - hfq low) / previous observed hfq close * 100",
                        "涨跌幅": "(hfq close / previous observed hfq close - 1) * 100",
                        "涨跌额": "hfq close - previous observed hfq close",
                    },
                    "missing_fields": ["换手率"],
                }
            else:
                sources = {"price_source": "eastmoney.stock_zh_a_hist", "raw_close_source": "eastmoney.stock_zh_a_hist"}
                try:
                    raw = fetch("")
                except Exception as error:
                    if not allow_sina_fallback:
                        raise
                    LOGGER.warning("%s: Eastmoney unadjusted history failed; using Sina raw close: %s", symbol, error)
                    raw = _retry(lambda: _sina_raw_close(symbol, timeout), retries, pause)
                    time.sleep(pause)
                    sources.update(raw_close_source="sina.hisdata_klc2", raw_close_fallback_reason=str(error))
                frame = _price_frame(adjusted, raw, allow_extra_raw_dates="raw_close_fallback_reason" in sources)
            invalid_dates = frame.loc[_invalid_adjusted_ohlc(frame), "日期"].tolist()
            research_invalid_count = sum(research_start <= day <= research_end for day in invalid_dates)
            if invalid_dates:
                LOGGER.warning(
                    "%s: retaining %d nonfinite/nonpositive adjusted OHLC bars (%d in research range); "
                    "prepare will mask these bars as NaN; dates are recorded in metadata",
                    symbol, len(invalid_dates), research_invalid_count,
                )
            temporary = path.with_suffix(".csv.tmp")
            frame.to_csv(temporary, index=False)
            temporary.replace(path)
            metadata["prices"][symbol] = {
                "start": start, "end": end, "rows": len(frame),
                "first_date": frame["日期"].iloc[0], "last_date": frame["日期"].iloc[-1],
                "adjust": "hfq", "raw_close": "unadjusted AKShare close",
                **sources,
                "adjusted_ohlc_anomalies": {
                    "count": len(invalid_dates), "dates": invalid_dates,
                    "research_count": research_invalid_count,
                },
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
