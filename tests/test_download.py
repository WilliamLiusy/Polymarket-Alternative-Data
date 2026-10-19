import importlib
import json

import numpy as np
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


@pytest.mark.parametrize("bad_price", [-1.0, 0.0, np.nan, np.inf, -np.inf])
def test_prices_preserve_isolated_adjusted_anomalies(bad_price):
    adjusted, raw = prices(multiplier=2), prices()
    adjusted.loc[0, "最低"] = bad_price
    merged = downloader._price_frame(adjusted, raw)
    pd.testing.assert_frame_equal(merged[adjusted.columns], adjusted)
    assert merged["raw_close"].tolist() == [10.0, 10.0]
    assert downloader._invalid_adjusted_ohlc(merged).tolist() == [True, False]


def test_prices_reject_history_without_usable_adjusted_bars():
    adjusted = prices()
    adjusted.loc[0, "收盘"] = -0.34
    adjusted.loc[1, "最低"] = np.nan
    with pytest.raises(ValueError, match="no usable adjusted OHLC"):
        downloader._price_frame(adjusted, prices())


@pytest.mark.parametrize("bad_price", [-1.0, 0.0, np.nan, np.inf, -np.inf])
def test_prices_still_reject_invalid_unadjusted_close(bad_price):
    raw = prices()
    raw.loc[0, "收盘"] = bad_price
    with pytest.raises(ValueError, match="unadjusted close"):
        downloader._price_frame(prices(multiplier=2), raw)


@pytest.mark.parametrize("column", ["成交量", "成交额"])
@pytest.mark.parametrize("bad_quantity", [-1.0, np.nan, np.inf, -np.inf])
def test_prices_still_reject_invalid_volume_or_amount(column, bad_quantity):
    adjusted = prices(multiplier=2)
    adjusted.loc[0, column] = bad_quantity
    with pytest.raises(ValueError, match="volume/amount"):
        downloader._price_frame(adjusted, prices())


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


def test_download_union_resume_and_refresh(mocked_sources, monkeypatch, tmp_path):
    config, calls = mocked_sources
    monkeypatch.setattr(downloader, "_sina_raw_close", lambda *args: pytest.fail("successful raw fetch must not use Sina"))
    summary = downloader.download(config)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert summary["historical_constituents"] == 2
    assert sorted(metadata["prices"]) == ["SH600000", "SZ000001"]
    assert (tmp_path / "instruments/2026-09-14.txt").read_text() == "SH600000\n"
    assert metadata["memberships"]["2026-09-15"]["updateDate"] == "2026-09-14"
    assert len(calls["price"]) == 4
    frame = pd.read_csv(tmp_path / "data/SH600000.csv")
    assert frame["raw_close"].tolist() == [10.0, 10.0]
    assert all(row["raw_close_source"] == "eastmoney.stock_zh_a_hist" for row in metadata["prices"].values())
    # Existing completion records have no source field; they remain reusable.
    for row in metadata["prices"].values():
        row.pop("raw_close_source")
        row.pop("price_source")
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    before = {path: path.stat().st_mtime_ns for path in (tmp_path / "data").glob("*.csv")}
    downloader.download(config)
    assert len(calls["price"]) == 4
    assert len(calls["membership"]) == 2
    assert before == {path: path.stat().st_mtime_ns for path in before}
    assert json.loads((tmp_path / "metadata.json").read_text())["prices"] == metadata["prices"]
    downloader.download(config, refresh=True)
    assert len(calls["price"]) == 8
    assert len(calls["membership"]) == 4


@pytest.mark.parametrize("invalid_date", ["2026-09-14", "2026-09-15"])
def test_download_records_adjusted_anomalies_without_discarding_history(
        mocked_sources, monkeypatch, tmp_path, caplog, invalid_date):
    import akshare as ak

    config, _ = mocked_sources
    config["research_start"] = "2026-09-15"
    source = ak.stock_zh_a_hist

    def anomalous_source(**kwargs):
        frame = source(**kwargs)
        if kwargs["adjust"] == "hfq":
            frame.loc[frame["日期"] == invalid_date, "收盘"] = -0.34
        return frame

    monkeypatch.setattr(ak, "stock_zh_a_hist", anomalous_source)
    downloader.download(config)
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["prices"]["SZ000001"]["adjusted_ohlc_anomalies"] == {
        "count": 1, "dates": [invalid_date], "research_count": int(invalid_date == "2026-09-15"),
    }
    saved = pd.read_csv(tmp_path / "data/SZ000001.csv", index_col="日期")
    assert saved.loc[invalid_date, "收盘"] == -0.34
    assert saved.loc[invalid_date, "raw_close"] == 10
    assert len(saved) == 2
    assert json.loads((tmp_path / "failures.json").read_text()) == []
    assert "SZ000001: retaining 1" in caplog.text


def test_failed_stock_is_recorded_and_retry_resumes_other_successes(mocked_sources, monkeypatch, tmp_path):
    import akshare as ak

    config, calls = mocked_sources
    monkeypatch.setattr(downloader, "_sina_raw_close", lambda *args: pytest.fail("hfq failure must not use Sina"))
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


@pytest.fixture
def mocked_sina_history(monkeypatch):
    import requests

    sina = importlib.import_module("akshare.stock.stock_zh_a_sina")
    state = {
        "requests": [], "checked_status": 0, "eval": [], "decode": [],
        "factors": [{"d": "2026-09-15", "f": "3"}, {"d": "2026-09-14", "f": "2"},
                    {"d": "1900-01-01", "f": "1"}],
        "rows": [
            {"date": "2026-09-15T00:00:00.000Z", "close": 11.5, "volume": 99999},
            {"date": "2026-09-13T00:00:00.000Z", "close": 7.0, "volume": 99999},
            {"date": "2026-09-14T00:00:00.000Z", "close": 11.5, "volume": 99999},
        ],
    }

    class Response:
        def __init__(self, symbol, factors=False):
            if factors:
                self.text = state.get("factor_payload", f'var {symbol}hfq={json.dumps({"data": state["factors"]})};\n')
            else:
                self.text = state.get("payload", f'var KLC_K2_{symbol}="ENCODED_FIXTURE";\n')

        def raise_for_status(self):
            state["checked_status"] += 1
            if "http_error" in state:
                raise state["http_error"]

    class Decoder:
        def eval(self, script):
            assert script == sina.hk_js_decode
            state["eval"].append(script)

        def call(self, function, encoded):
            assert (function, encoded) == ("d", "ENCODED_FIXTURE")
            state["decode"].append(encoded)
            return state["rows"]

    def get(url, *, timeout):
        factors = url.endswith("/hfq.js")
        symbol = url.split("/")[-2 if factors else -3]
        expected_url = sina.zh_sina_a_stock_hfq_url if factors else sina.zh_sina_a_stock_hist_url
        assert url == expected_url.format(symbol)
        state["requests"].append((url, timeout))
        return Response(symbol, factors)

    monkeypatch.setattr(requests, "get", get)
    monkeypatch.setattr(sina.py_mini_racer, "MiniRacer", Decoder)
    return state


def fail_eastmoney_raw(monkeypatch, *, empty=False):
    import akshare as ak

    original = ak.stock_zh_a_hist
    raw_calls = []

    def fetch(**kwargs):
        if kwargs["symbol"] == "600000" and kwargs["adjust"] == "":
            raw_calls.append(kwargs)
            if empty:
                return pd.DataFrame()
            raise TimeoutError("mocked Eastmoney raw request failed")
        return original(**kwargs)

    monkeypatch.setattr(ak, "stock_zh_a_hist", fetch)
    return raw_calls


@pytest.mark.parametrize("empty", [False, True])
def test_raw_failure_retries_then_uses_only_sina_close_aligned_to_hfq(
        mocked_sources, mocked_sina_history, monkeypatch, tmp_path, empty):
    config, _ = mocked_sources
    config["download"]["allow_sina_fallback"] = True
    config["download"]["retries"] = 2
    raw_calls = fail_eastmoney_raw(monkeypatch, empty=empty)
    downloader.download(config)
    assert len(raw_calls) == 2
    assert mocked_sina_history["requests"] == [(
        "https://finance.sina.com.cn/realstock/company/sh600000/hisdata_klc2/klc_kl.js", 1.0)]
    assert mocked_sina_history["checked_status"] == 1
    assert len(mocked_sina_history["eval"]) == len(mocked_sina_history["decode"]) == 1
    frame = pd.read_csv(tmp_path / "data/SH600000.csv")
    pd.testing.assert_frame_equal(frame[downloader.PRICE_COLUMNS], prices(multiplier=2))
    # Equal prices on distinct dates are independent observations, never deduplicated.
    assert frame["raw_close"].tolist() == [11.5, 11.5]
    metadata = json.loads((tmp_path / "metadata.json").read_text())["prices"]
    assert metadata["SH600000"]["raw_close_source"] == "sina.hisdata_klc2"
    assert metadata["SH600000"]["raw_close_fallback_reason"]
    assert metadata["SZ000001"]["raw_close_source"] == "eastmoney.stock_zh_a_hist"
    assert "raw_close_fallback_reason" not in metadata["SZ000001"]


@pytest.mark.parametrize("rows, message", [
    ([{"date": "2026-09-14", "close": 10}], "different dates"),
    ([{"date": "2026-09-14", "close": 10}, {"date": "2026-09-14", "close": 12},
      {"date": "2026-09-15", "close": 11}], "duplicate dates"),
])
def test_missing_or_duplicate_sina_dates_fail_without_filling_and_resume(
        mocked_sources, mocked_sina_history, monkeypatch, tmp_path, rows, message):
    config, _ = mocked_sources
    config["download"]["allow_sina_fallback"] = True
    fail_eastmoney_raw(monkeypatch)
    mocked_sina_history["rows"] = rows
    with pytest.raises(RuntimeError, match="Download incomplete"):
        downloader.download(config)
    assert not (tmp_path / "data/SH600000.csv").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "SH600000" not in metadata["prices"]
    assert "SZ000001" in metadata["prices"]
    failure, = json.loads((tmp_path / "failures.json").read_text())
    assert failure["item"] == "SH600000" and message in failure["error"]
    other = tmp_path / "data/SZ000001.csv"
    before = other.stat().st_mtime_ns
    mocked_sina_history["rows"] = [{"date": day, "close": 10} for day in ("2026-09-14", "2026-09-15")]
    downloader.download(config)
    assert other.stat().st_mtime_ns == before
    assert json.loads((tmp_path / "failures.json").read_text()) == []
    assert pd.read_csv(tmp_path / "data/SH600000.csv")["raw_close"].tolist() == [10, 10]


@pytest.mark.parametrize("bad_close", [0, -1, np.nan, np.inf, -np.inf, "not-a-price"])
def test_sina_decoder_rejects_invalid_close(mocked_sina_history, bad_close):
    mocked_sina_history["rows"][0]["close"] = bad_close
    with pytest.raises(ValueError):
        downloader._sina_raw_close("SH600000", timeout=7)


@pytest.mark.parametrize("rows", [[], [{"date": "2026-09-14"}],
    [{"date": None, "close": 10}], [{"date": "not-a-date", "close": 10}]])
def test_sina_decoder_rejects_missing_or_invalid_observations(mocked_sina_history, rows):
    mocked_sina_history["rows"] = rows
    with pytest.raises(ValueError):
        downloader._sina_raw_close("SH600000", timeout=7)


def test_sina_decoder_checks_http_status_before_decoding(mocked_sina_history):
    import requests

    mocked_sina_history["http_error"] = requests.HTTPError("503 unavailable")
    with pytest.raises(requests.HTTPError):
        downloader._sina_raw_close("SH600000", timeout=7)
    assert not mocked_sina_history["eval"]
    assert mocked_sina_history["requests"][0][1] == 7


@pytest.mark.parametrize("payload", ['var KLC_K2_sh600001="ENCODED_FIXTURE";', '<html>Unavailable</html>',
    'var KLC_K2_sh600000="ENCODED_FIXTURE"; malicious();'])
def test_sina_decoder_rejects_wrong_symbol_or_malformed_payload(mocked_sina_history, payload):
    mocked_sina_history["payload"] = payload
    with pytest.raises(ValueError, match="Malformed Sina history payload"):
        downloader._sina_raw_close("SH600000", timeout=7)
    assert not mocked_sina_history["eval"]


def test_sina_payloads_allow_real_endpoint_trailing_comments_without_executing_them(mocked_sina_history):
    # Both real endpoints append long block comments, potentially containing JSON punctuation.
    trailing = '/* arbitrary trailer {"x": "}"}; malicious();\n' + "random-text" * 100 + ' */\n'
    mocked_sina_history["payload"] = 'var KLC_K2_sh600000="ENCODED_FIXTURE";\n' + trailing
    mocked_sina_history["factor_payload"] = 'var sh600000hfq=' + json.dumps({"data": mocked_sina_history["factors"]}) + ';\n' + trailing
    raw = downloader._sina_raw_close("SH600000", 7)
    factors = downloader._sina_hfq_factors("SH600000", 7)
    assert len(raw) == 3 and len(factors) == 3
    assert len(mocked_sina_history["eval"]) == 1  # Only the SDK's trusted local decoder.


@pytest.mark.parametrize("trailer", ['/* comment */ malicious();', '/* unterminated', '; alert(1);'])
def test_sina_factor_payload_rejects_trailing_remote_code(mocked_sina_history, trailer):
    mocked_sina_history["factor_payload"] = 'var sh600000hfq=' + json.dumps({"data": mocked_sina_history["factors"]}) + ';' + trailer
    with pytest.raises(ValueError, match="Malformed Sina hfq factor payload"):
        downloader._sina_hfq_factors("SH600000", 7)


@pytest.mark.parametrize("enabled", [None, False])
@pytest.mark.parametrize("failure_adjust", ["", "hfq"])
def test_sina_fallback_is_disabled_by_default(mocked_sources, monkeypatch, tmp_path, enabled, failure_adjust):
    import akshare as ak

    config, _ = mocked_sources
    if enabled is not None:
        config["download"]["allow_sina_fallback"] = enabled
    original = ak.stock_zh_a_hist

    def fetch(**kwargs):
        if kwargs["symbol"] == "600000" and kwargs["adjust"] == failure_adjust:
            raise TimeoutError("mocked Eastmoney failure")
        return original(**kwargs)

    monkeypatch.setattr(ak, "stock_zh_a_hist", fetch)
    monkeypatch.setattr(downloader, "_sina_history", lambda *args: pytest.fail("Sina fallback is disabled"))
    with pytest.raises(RuntimeError, match="Download incomplete"):
        downloader.download(config)
    assert "SH600000" not in json.loads((tmp_path / "metadata.json").read_text())["prices"]


def sina_bars():
    return [
        {"date": "2026-09-15T00:00:00.000Z", "open": 10, "high": 13, "low": 9,
         "close": 12, "volume": 5000, "amount": 60000, "prevclose": np.nan, "postVol": np.nan},
        {"date": "2026-09-14T00:00:00.000Z", "open": 8, "high": 11, "low": 7,
         "close": 10, "volume": 1000, "amount": 10000, "prevclose": np.nan, "postVol": np.nan},
    ]


def test_hfq_failure_uses_sina_prices_factors_and_same_source_raw_close(
        mocked_sources, mocked_sina_history, monkeypatch, tmp_path):
    import akshare as ak
    from baseline.prepare import convert_bars

    config, calls = mocked_sources
    config["download"].update(allow_sina_fallback=True, retries=2)
    mocked_sina_history["rows"] = sina_bars()
    original = ak.stock_zh_a_hist
    failures = []

    def fetch(**kwargs):
        if kwargs["symbol"] == "600000":
            assert kwargs["adjust"] == "hfq", "full Sina fallback must not fetch Eastmoney raw prices"
            failures.append(kwargs)
            raise TimeoutError("mocked Eastmoney hfq failure")
        return original(**kwargs)

    monkeypatch.setattr(ak, "stock_zh_a_hist", fetch)
    downloader.download(config)
    assert len(failures) == 2
    assert mocked_sina_history["requests"] == [
        ("https://finance.sina.com.cn/realstock/company/sh600000/hisdata_klc2/klc_kl.js", 1.0),
        ("https://finance.sina.com.cn/realstock/company/sh600000/hfq.js", 1.0),
    ]
    frame = pd.read_csv(tmp_path / "data/SH600000.csv")
    np.testing.assert_allclose(frame[["开盘", "最高", "最低", "收盘"]], [[16, 22, 14, 20], [30, 39, 27, 36]])
    np.testing.assert_allclose(frame["raw_close"], [10, 12])
    np.testing.assert_allclose(frame["成交量"], [10, 50])
    np.testing.assert_allclose(frame["成交额"], [10000, 60000])
    assert frame["换手率"].isna().all()
    assert frame.loc[0, ["振幅", "涨跌幅", "涨跌额"]].isna().all()
    np.testing.assert_allclose(frame.loc[1, ["振幅", "涨跌幅", "涨跌额"]].astype(float), [60, 80, 16])
    converted = convert_bars(frame)
    np.testing.assert_allclose(converted["factor"], [2, 3])
    np.testing.assert_allclose(converted["volume"], [500, 5000 / 3], rtol=1e-6)
    np.testing.assert_allclose(converted["vwap"], [20, 36])
    np.testing.assert_allclose(converted["amount"], [10000, 60000])
    metadata = json.loads((tmp_path / "metadata.json").read_text())["prices"]["SH600000"]
    assert metadata["adjust"] == "hfq"
    assert metadata["price_source"] == metadata["raw_close_source"] == "sina.hisdata_klc2"
    assert metadata["adjustment_source"] == "sina.hfq"
    assert metadata["price_fallback_reason"] == "mocked Eastmoney hfq failure"
    assert metadata["missing_fields"] == ["换手率"]
    assert set(metadata["derived_fields"]) == {"振幅", "涨跌幅", "涨跌额"}
    before = len(mocked_sina_history["requests"])
    downloader.download(config)
    assert len(mocked_sina_history["requests"]) == before


def test_sina_factors_apply_only_from_effective_date_and_never_create_price_rows(mocked_sina_history):
    mocked_sina_history["rows"] = sina_bars()
    mocked_sina_history["rows"][0]["date"] = "2026-09-16"
    mocked_sina_history["factors"] = [
        {"d": "2026-09-18", "f": "9"}, {"d": "2026-09-15", "f": "3"},
        {"d": "2026-09-13", "f": "2"},  # Weekend predecessor must survive range filtering.
    ]
    history = downloader._sina_history("SH600000", 7)
    factors = downloader._sina_hfq_factors("SH600000", 7)
    frame = downloader._sina_hfq_frame(history, factors, "SH600000", "2026-09-14", "2026-09-16")
    assert frame["日期"].tolist() == ["2026-09-14", "2026-09-16"]
    assert frame["收盘"].tolist() == [20, 36]
    assert frame["raw_close"].tolist() == [10, 12]


@pytest.mark.parametrize("bad_factor", [0, -1, np.nan, np.inf, -np.inf, "bad", None])
def test_sina_rejects_nonfinite_or_nonpositive_factors(mocked_sina_history, bad_factor):
    mocked_sina_history["factors"][0]["f"] = bad_factor
    with pytest.raises(ValueError):
        downloader._sina_hfq_factors("SH600000", 7)


@pytest.mark.parametrize("factors", [[], [{"d": None, "f": "1"}], [{"d": "bad-date", "f": "1"}],
    [{"d": "2026-09-14", "f": "1"}, {"d": "2026-09-14", "f": "2"}]])
def test_sina_rejects_missing_or_duplicate_factor_dates(mocked_sina_history, factors):
    mocked_sina_history["factors"] = factors
    with pytest.raises(ValueError):
        downloader._sina_hfq_factors("SH600000", 7)


def test_sina_never_backfills_a_missing_initial_factor(mocked_sina_history):
    mocked_sina_history["rows"] = sina_bars()
    mocked_sina_history["factors"] = [{"d": "2026-09-15", "f": "3"}]
    history = downloader._sina_history("SH600000", 7)
    factors = downloader._sina_hfq_factors("SH600000", 7)
    with pytest.raises(ValueError, match="unavailable on or before"):
        downloader._sina_hfq_frame(history, factors, "SH600000", "2026-09-14", "2026-09-15")


@pytest.mark.parametrize("factors", [[], [{"d": "1900-01-01", "f": "0"}], [{"d": "2026-09-15", "f": "3"}]])
def test_hfq_fallback_with_invalid_or_missing_factors_never_checkpoints_success(
        mocked_sources, mocked_sina_history, monkeypatch, tmp_path, factors):
    import akshare as ak

    config, _ = mocked_sources
    config["download"]["allow_sina_fallback"] = True
    mocked_sina_history["rows"] = sina_bars()
    mocked_sina_history["factors"] = factors
    original = ak.stock_zh_a_hist

    def fetch(**kwargs):
        if kwargs["symbol"] == "600000":
            raise TimeoutError("mocked Eastmoney hfq failure")
        return original(**kwargs)

    monkeypatch.setattr(ak, "stock_zh_a_hist", fetch)
    with pytest.raises(RuntimeError, match="Download incomplete"):
        downloader.download(config)
    assert not (tmp_path / "data/SH600000.csv").exists()
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert "SH600000" not in metadata["prices"] and "SZ000001" in metadata["prices"]
    failure, = json.loads((tmp_path / "failures.json").read_text())
    assert failure["item"] == "SH600000" and "factor" in failure["error"].lower()


def test_sina_hfq_rounds_prices_like_the_sdk(mocked_sina_history):
    mocked_sina_history["rows"] = sina_bars()
    mocked_sina_history["factors"] = [{"d": "1900-01-01", "f": "2.34567"}]
    history = downloader._sina_history("SH600000", 7)
    factors = downloader._sina_hfq_factors("SH600000", 7)
    frame = downloader._sina_hfq_frame(history, factors, "SH600000", "2026-09-14", "2026-09-15")
    np.testing.assert_allclose(frame.loc[0, ["开盘", "最高", "最低", "收盘"]].astype(float), [18.77, 25.8, 16.42, 23.46])


@pytest.mark.parametrize("bad_open", [-1, 0, np.nan, np.inf, -np.inf])
def test_sina_hfq_preserves_isolated_adjusted_ohlc_anomalies(mocked_sina_history, bad_open):
    mocked_sina_history["rows"] = sina_bars()
    mocked_sina_history["rows"][1]["open"] = bad_open
    history = downloader._sina_history("SH600000", 7)
    factors = downloader._sina_hfq_factors("SH600000", 7)
    frame = downloader._sina_hfq_frame(history, factors, "SH600000", "2026-09-14", "2026-09-15")
    assert downloader._invalid_adjusted_ohlc(frame).tolist() == [True, False]
    assert frame["raw_close"].tolist() == [10, 12]


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
            assert kwargs["initargs"][:3] == (10, 2, 0.1)
            assert not kwargs["initargs"][3].is_set()

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


def blocked_result():
    result = Result([], [])
    result.error_code = "10001011"
    result.error_msg = "mocked blacklist response"
    return result


@pytest.mark.parametrize("pagination", [False, True])
def test_blacklist_response_does_not_retry_or_relogin(monkeypatch, pagination):
    import baostock as bs

    calls = []
    monkeypatch.setattr(bs, "login", lambda: pytest.fail("blacklisted query must not log in again"))

    def query():
        calls.append(1)
        return Result(["code"], [["sh.600000"]], final_error="10001011") if pagination else blocked_result()

    with pytest.raises(downloader.BaoStockBlockedError, match="10001011") as caught:
        downloader._baostock_query(query, retries=3, pause=0)
    assert caught.value.error_code == "10001011"
    assert calls == [1]


def test_transient_baostock_error_still_reconnects_and_retries(monkeypatch):
    import baostock as bs
    import baostock.common.context as context

    queries, logins = [], []
    monkeypatch.setattr(context, "default_socket", None, raising=False)
    monkeypatch.setattr(bs, "login", lambda: logins.append(1) or Result([], []))

    def query():
        queries.append(1)
        if len(queries) == 1:
            raise TimeoutError("temporary socket failure")
        return Result(["code"], [["sh.600000"]])

    frame = downloader._baostock_query(query, retries=3, pause=0)
    assert frame["code"].tolist() == ["sh.600000"]
    assert len(queries) == 2
    assert len(logins) == 1


def test_blacklisted_login_stops_download_and_preserves_checkpoints(
        mocked_sources, monkeypatch, tmp_path):
    import baostock as bs

    config, calls = mocked_sources
    config["download"]["retries"] = 3
    old_metadata = {"prices": {"SH600000": {"rows": 123}}, "memberships": {"older": {"count": 300}}}
    (tmp_path / "metadata.json").write_text(json.dumps(old_metadata))
    logins = []
    monkeypatch.setattr(bs, "login", lambda: logins.append(1) or blocked_result())
    monkeypatch.setattr(bs, "logout", lambda: pytest.fail("no API calls after blacklisting"))
    with pytest.raises(downloader.BaoStockBlockedError, match="download stopped"):
        downloader.download(config)
    assert logins == [1]
    assert calls == {"calendar": [], "membership": [], "price": []}
    assert json.loads((tmp_path / "metadata.json").read_text()) == old_metadata
    failure, = json.loads((tmp_path / "failures.json").read_text())
    assert failure["stage"] == "calendar/login"
    assert failure["error_code"] == "10001011"


def test_serial_blacklist_stops_later_dates_and_price_downloads(mocked_sources, monkeypatch, tmp_path):
    import baostock as bs

    config, calls = mocked_sources
    config["download_end"] = config["research_end"] = "2026-09-16"
    config["download"]["retries"] = 3
    monkeypatch.setattr(bs, "query_trade_dates", lambda **kwargs: Result(
        ["calendar_date", "is_trading_day"], [[f"2026-09-{day}", "1"] for day in (14, 15, 16)]))
    source = bs.query_hs300_stocks
    queried = []

    def membership(date):
        queried.append(date)
        return blocked_result() if date == "2026-09-15" else source(date)

    monkeypatch.setattr(bs, "query_hs300_stocks", membership)
    monkeypatch.setattr(bs, "logout", lambda: pytest.fail("no API calls after blacklisting"))
    with pytest.raises(downloader.BaoStockBlockedError, match="10001011"):
        downloader.download(config)
    assert queried == ["2026-09-14", "2026-09-15"]
    assert not calls["price"]
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert list(metadata["memberships"]) == ["2026-09-14"]
    assert (tmp_path / "instruments/2026-09-14.txt").read_text() == "SH600000\n"
    failure, = json.loads((tmp_path / "failures.json").read_text())
    assert failure["stage"] == "membership" and failure["item"] == "2026-09-15"
    assert failure["error_code"] == "10001011"


def test_parallel_parent_stops_on_serialized_blacklist(mocked_sources, monkeypatch, tmp_path):
    import baostock as bs

    config, calls = mocked_sources
    config["download"]["membership_workers"] = 2
    closed = []

    def results(*args):
        try:
            yield "2026-09-14", ["SH600000"], "2026-09-14", None
            yield "2026-09-15", [], None, "BaoStock 10001011: mocked blacklist"
            pytest.fail("parent must stop consuming work after blacklisting")
        finally:
            closed.append(True)

    monkeypatch.setattr(downloader, "_parallel_memberships", results)
    monkeypatch.setattr(bs, "logout", lambda: pytest.fail("no API calls after blacklisting"))
    with pytest.raises(downloader.BaoStockBlockedError, match="10001011"):
        downloader.download(config)
    assert closed == [True]
    assert not calls["price"]
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert list(metadata["memberships"]) == ["2026-09-14"]
    failure, = json.loads((tmp_path / "failures.json").read_text())
    assert failure["error_code"] == "10001011"


def test_parallel_blacklist_never_submits_another_batch(monkeypatch):
    submitted, shutdown = [], []

    class Future:
        def __init__(self, day):
            self.day = day

        def result(self):
            return self.day, [], None, "BaoStock 10001011: mocked blacklist"

    class Executor:
        def __init__(self, **kwargs):
            pass

        def submit(self, function, day):
            submitted.append(day)
            return Future(day)

        def shutdown(self, **kwargs):
            shutdown.append(kwargs)

    monkeypatch.setattr(downloader, "ProcessPoolExecutor", Executor)
    monkeypatch.setattr(downloader, "as_completed", lambda futures: list(futures))
    days = [f"2026-09-{day}" for day in (14, 15, 16, 17)]
    results = list(downloader._parallel_memberships(days, workers=2, timeout=1, retries=3, pause=0))
    assert len(results) == 1
    assert submitted == days[:2]
    assert shutdown == [{"wait": True, "cancel_futures": True}]


def test_worker_reports_blacklisted_initial_login_without_more_requests(monkeypatch):
    import baostock as bs
    from threading import Event

    logins = []
    monkeypatch.setattr(downloader, "_WORKER_BLOCKED", None)
    monkeypatch.setattr(downloader, "_WORKER_STOP", None)
    monkeypatch.setattr(downloader, "_WORKER_RETRIES", 1)
    monkeypatch.setattr(downloader, "_WORKER_PAUSE", 0.0)
    monkeypatch.setattr(downloader.socket, "setdefaulttimeout", lambda value: None)
    monkeypatch.setattr(bs, "login", lambda: logins.append(1) or blocked_result())
    monkeypatch.setattr(bs, "query_hs300_stocks", lambda **kwargs: pytest.fail("worker must not query"))
    monkeypatch.setattr(bs, "logout", lambda: pytest.fail("worker must not call logout after blacklisting"))
    stop = Event()
    downloader._init_membership_worker(timeout=1, retries=3, pause=0, stop_event=stop)
    for day in ("2026-09-14", "2026-09-15"):
        result = downloader._membership_task(day)
        assert result[:3] == (day, [], None)
        assert downloader._is_baostock_blocked(result[3])
    downloader._worker_logout()
    assert logins == [1]
    assert stop.is_set()


def test_peer_blacklist_stops_worker_queries_relogin_and_logout(monkeypatch):
    import baostock as bs
    from threading import Event

    stop = Event()
    stop.set()
    monkeypatch.setattr(downloader, "_WORKER_BLOCKED", None)
    monkeypatch.setattr(downloader, "_WORKER_STOP", stop)
    monkeypatch.setattr(bs, "login", lambda: pytest.fail("peer blacklist must stop login"))
    monkeypatch.setattr(bs, "logout", lambda: pytest.fail("peer blacklist must stop logout"))
    with pytest.raises(downloader.BaoStockBlockedError, match="another worker"):
        downloader._baostock_query(lambda: pytest.fail("peer blacklist must stop queries"), retries=3, pause=0)
    with pytest.raises(downloader.BaoStockBlockedError, match="another worker"):
        downloader._baostock_login(bs)
    downloader._worker_logout()
