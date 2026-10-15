import importlib
import json

import pandas as pd
import pytest


downloader = importlib.import_module("baseline.download")


class Result:
    def __init__(self, fields, rows, final_error="0"):
        self.fields = fields
        self.rows = iter(rows)
        self.error_code = "0"
        self.error_msg = "test failure"
        self.final_error = final_error

    def next(self):
        self.current = next(self.rows, None)
        if self.current is None:
            self.error_code = self.final_error
        return self.current is not None

    def get_row_data(self):
        return self.current


def prices(dates=("2026-09-14", "2026-09-15"), multiplier=1):
    frame = pd.DataFrame({"日期": list(dates)})
    for column in downloader.PRICE_COLUMNS[1:]:
        frame[column] = 10.0
    for column in ("开盘", "收盘", "最高", "最低"):
        frame[column] *= multiplier
    return frame


def test_members_use_query_date_but_retain_weekly_update_date():
    frame = pd.DataFrame({
        "updateDate": ["2026-09-14", "2026-09-14"],
        "code": ["sz.000001", "sh.600000"], "code_name": ["A", "B"],
    })
    assert downloader._members(frame, "2026-09-18") == (["SH600000", "SZ000001"], "2026-09-14")
    with pytest.raises(ValueError, match="Future"):
        downloader._members(frame, "2026-09-11")
    frame.loc[1, "code"] = "sz.000001"
    with pytest.raises(ValueError, match="duplicate"):
        downloader._members(frame, "2026-09-18")


def test_calendar_detects_partial_page_and_preserves_holidays():
    frame = pd.DataFrame({
        "calendar_date": ["2026-09-18", "2026-09-19", "2026-09-20", "2026-09-21"],
        "is_trading_day": ["1", "0", "0", "1"],
    })
    assert downloader._calendar_dates(frame, "2026-09-18", "2026-09-21") == ["2026-09-18", "2026-09-21"]
    with pytest.raises(ValueError, match="completely cover"):
        downloader._calendar_dates(frame.iloc[:-1], "2026-09-18", "2026-09-21")


def test_calendar_can_begin_at_baostock_inception():
    frame = pd.DataFrame({
        "calendar_date": ["1990-12-19", "1990-12-20"], "is_trading_day": ["1", "1"],
    })
    assert downloader._calendar_dates(frame, "1990-01-01", "1990-12-20") == ["1990-12-19", "1990-12-20"]


def test_pagination_error_is_not_mistaken_for_complete_result():
    result = Result(["code"], [["sh.600000"]], final_error="1000")
    with pytest.raises(RuntimeError, match="pagination"):
        downloader._result_frame(result)


def test_prices_join_by_date_and_reject_missing_raw_observation():
    adjusted = prices(multiplier=2).iloc[::-1]
    raw = prices()
    merged = downloader._price_frame(adjusted, raw)
    assert merged["日期"].tolist() == ["2026-09-14", "2026-09-15"]
    assert (merged["收盘"] / merged["raw_close"]).tolist() == [2.0, 2.0]
    with pytest.raises(ValueError, match="different dates"):
        downloader._price_frame(adjusted, raw.iloc[:1])


def test_retry_has_finite_attempts():
    calls = []

    def failure():
        calls.append(1)
        raise TimeoutError("unavailable")

    with pytest.raises(TimeoutError):
        downloader._retry(failure, retries=3, pause=0)
    assert len(calls) == 3


@pytest.fixture
def mocked_sources(monkeypatch, tmp_path):
    import akshare as ak
    import baostock as bs

    calls = {"calendar": [], "membership": [], "price": []}
    success = Result([], [])
    monkeypatch.setattr(bs, "login", lambda: success)
    monkeypatch.setattr(bs, "logout", lambda: success)

    def calendar(**kwargs):
        calls["calendar"].append(kwargs)
        return Result(["calendar_date", "is_trading_day"], [["2026-09-14", "1"], ["2026-09-15", "1"]])

    def membership(date):
        calls["membership"].append(date)
        symbol = "sh.600000" if date == "2026-09-14" else "sz.000001"
        return Result(["updateDate", "code", "code_name"], [["2026-09-14", symbol, "Name"]])

    def price(**kwargs):
        calls["price"].append(kwargs)
        return prices(multiplier=2 if kwargs["adjust"] == "hfq" else 1)

    monkeypatch.setattr(bs, "query_trade_dates", calendar)
    monkeypatch.setattr(bs, "query_hs300_stocks", membership)
    monkeypatch.setattr(ak, "stock_zh_a_hist", price)
    config = {
        "data_dir": str(tmp_path), "download_start": "2026-09-14", "download_end": "2026-09-15",
        "research_start": "2026-09-14", "research_end": "2026-09-15",
        "download": {"timeout": 1, "retries": 1, "pause": 0, "membership_workers": 1},
    }
    return config, calls


def test_download_union_resume_and_refresh(mocked_sources, tmp_path):
    config, calls = mocked_sources
    summary = downloader.download(config)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert summary["historical_constituents"] == 2
    assert sorted(metadata["prices"]) == ["SH600000", "SZ000001"]
    assert (tmp_path / "instruments/2026-09-14.txt").read_text() == "SH600000\n"
    assert metadata["memberships"]["2026-09-15"]["updateDate"] == "2026-09-14"
    assert len(calls["price"]) == 4
    frame = pd.read_csv(tmp_path / "data/SH600000.csv")
    assert frame["raw_close"].tolist() == [10.0, 10.0]
    downloader.download(config)
    assert len(calls["price"]) == 4
    assert len(calls["membership"]) == 2
    downloader.download(config, refresh=True)
    assert len(calls["price"]) == 8
    assert len(calls["membership"]) == 4


def test_failed_stock_is_recorded_and_retry_resumes_other_successes(mocked_sources, monkeypatch, tmp_path):
    import akshare as ak

    config, calls = mocked_sources
    successful_source = ak.stock_zh_a_hist

    def fail_one(**kwargs):
        if kwargs["symbol"] == "600000":
            return pd.DataFrame()
        return successful_source(**kwargs)

    monkeypatch.setattr(ak, "stock_zh_a_hist", fail_one)
    with pytest.raises(RuntimeError, match="Download incomplete"):
        downloader.download(config)
    failures = json.loads((tmp_path / "failures.json").read_text())
    assert failures[0]["item"] == "SH600000"
    assert (tmp_path / "data/SZ000001.csv").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "SH600000" not in metadata["prices"]
    monkeypatch.setattr(ak, "stock_zh_a_hist", successful_source)
    downloader.download(config)
    assert len(calls["price"]) == 4  # The previously successful stock was not downloaded again.
    assert json.loads((tmp_path / "failures.json").read_text()) == []


def test_parallel_requests_are_spawned_bounded_and_cancelable(monkeypatch):
    state = {"outstanding": 0, "maximum": 0, "shutdown": None}

    class Future:
        def __init__(self, day):
            self.day = day

        def result(self):
            state["outstanding"] -= 1
            return self.day, ["SH600000"], "2026-09-14", None

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["mp_context"].get_start_method() == "spawn"
            assert kwargs["initializer"] is downloader._init_membership_worker
            assert kwargs["initargs"] == (10, 2, 0.1)

        def submit(self, function, day):
            assert function is downloader._membership_task
            state["outstanding"] += 1
            state["maximum"] = max(state["maximum"], state["outstanding"])
            return Future(day)

        def shutdown(self, **kwargs):
            state["shutdown"] = kwargs

    monkeypatch.setattr(downloader, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(downloader, "as_completed", lambda futures: reversed(list(futures)))
    dates = [f"2026-09-{day}" for day in range(14, 21)]
    results = list(downloader._parallel_memberships(dates, workers=2, timeout=10, retries=2, pause=0.1))
    assert sorted(result[0] for result in results) == dates
    assert state["maximum"] == 2
    assert state["outstanding"] == 0
    assert state["shutdown"] == {"wait": True, "cancel_futures": True}
    # Closing the generator after a single result must not enqueue later batches.
    state["maximum"] = 0
    result_stream = downloader._parallel_memberships(dates, workers=2, timeout=10, retries=2, pause=0.1)
    next(result_stream)
    result_stream.close()
    assert state["maximum"] == 2
    assert state["outstanding"] == 1
    assert state["shutdown"] == {"wait": True, "cancel_futures": True}


def test_parallel_parent_checkpoints_success_and_records_failed_date(mocked_sources, monkeypatch, tmp_path):
    config, _ = mocked_sources
    config["download"]["membership_workers"] = 2

    def results(days, workers, timeout, retries, pause):
        assert days == ["2026-09-14", "2026-09-15"]
        yield "2026-09-15", [], None, "source unavailable"
        yield "2026-09-14", ["SH600000"], "2026-09-14", None

    monkeypatch.setattr(downloader, "_parallel_memberships", results)
    with pytest.raises(RuntimeError, match="Download incomplete"):
        downloader.download(config)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert list(metadata["memberships"]) == ["2026-09-14"]
    assert (tmp_path / "instruments/2026-09-14.txt").read_text() == "SH600000\n"
    assert (tmp_path / "data/SH600000.csv").exists()
    assert json.loads((tmp_path / "failures.json").read_text())[0]["item"] == "2026-09-15"
