"""Collection checks use small fake API pages, not the live market universe."""
import json
import threading

import pandas as pd
import pytest

from polymarket_ext import collect


SETTINGS = {"retries": 1, "pause": 0, "price_chunk_days": 1}


@pytest.mark.parametrize("tag_slug", [None, "crude-oil"])
def test_official_events_forwards_optional_tag_and_pagination(tag_slug):
    calls = []
    paginator = object()
    class SDK:
        def list_events(self, **kwargs):
            calls.append(kwargs)
            return paginator
    client = collect.OfficialClient.__new__(collect.OfficialClient)
    client.sdk = SDK()
    client.settings = {"event_page_size": 25}
    if tag_slug is not None:
        client.settings["event_tag_slug"] = tag_slug
    def page(actual_paginator, cursor):
        assert actual_paginator is paginator
        assert cursor == "next-page"
        return {"events": [{"id": "1"}]}, None
    client._page = page

    assert client.events(True, "next-page") == ({"events": [{"id": "1"}]}, None)
    assert calls == [{"closed": True, "order": "id", "ascending": True,
                      "tag_slug": tag_slug, "page_size": 25}]


def test_cursor_checkpoint_resumes_after_failure(tmp_path):
    seen = []
    def fail_on_second(cursor):
        seen.append(cursor)
        if cursor == "second":
            raise RuntimeError("offline")
        return {"data": [{"id": 1}]}, "second"
    with pytest.raises(RuntimeError, match="offline"):
        collect._pages(fail_on_second, tmp_path, SETTINGS)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert not manifest["complete"]
    assert manifest["next_cursor"] == "second"
    def resume(cursor):
        seen.append(cursor)
        assert cursor == "second"
        return {"data": [{"id": 2}]}, None
    collect._pages(resume, tmp_path, SETTINGS)
    assert seen == [None, "second", "second"]
    assert list(collect._page_rows(tmp_path, "data")) == [{"id": 1}, {"id": 2}]
    collect._pages(lambda _: pytest.fail("completed pages should be cached"), tmp_path, SETTINGS)


def test_repeating_cursor_is_incomplete(tmp_path):
    with pytest.raises(RuntimeError, match="advance"):
        collect._pages(lambda cursor: ({"data": []}, "same"), tmp_path, SETTINGS)
    assert not json.loads((tmp_path / "manifest.json").read_text())["complete"]


def test_explicit_page_refresh_replaces_corrupt_source_manifest(tmp_path):
    (tmp_path / "manifest.json").write_text("interrupted invalid JSON")
    identity = {"event_tag_slug": "oil"}
    result = collect._pages(lambda cursor: ({"events": [{"id": "fresh"}]}, None),
                            tmp_path, SETTINGS, True, source_identity=identity)
    assert result["complete"] and result["source_identity"] == identity
    assert list(collect._page_rows(tmp_path, "events")) == [{"id": "fresh"}]


def test_discover_includes_closed_events_and_preserves_raw(monkeypatch, tmp_path):
    calls = []
    class Client:
        def __init__(self, settings): pass
        def events(self, closed, cursor):
            calls.append((closed, cursor))
            event = {"id": "old" if closed else "new", "unknownRawField": 12,
                     "markets": [{"id": "market", "clobTokenIds": '["1", "2"]'}]}
            return {"events": [event]}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)
    result = collect.discover({"data_dir": str(tmp_path), "collection": SETTINGS})
    assert result["event_count"] == 2
    assert calls == [(True, None), (False, None)]
    events = list(collect._json_lines(tmp_path / "raw" / "events.jsonl"))
    assert events[0]["unknownRawField"] == 12
    assert events[0]["markets"][0]["clobTokenIds"] == '["1", "2"]'


@pytest.mark.parametrize("initial_tag,next_tag", [
    (None, "crude-oil"),
    ("crude-oil", "natural-gas"),
    ("crude-oil", None),
])
def test_discover_tag_change_invalidates_event_pages_and_result(
        monkeypatch, tmp_path, initial_tag, next_tag):
    calls = []
    class Client:
        def __init__(self, settings):
            self.tag = settings.get("event_tag_slug")
        def events(self, closed, cursor):
            calls.append((self.tag, closed, cursor))
            prefix = self.tag or "all"
            return {"events": [{"id": f"{prefix}-{closed}"}]}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)

    def config(tag):
        settings = dict(SETTINGS)
        if tag is not None:
            settings["event_tag_slug"] = tag
        return {"data_dir": str(tmp_path), "collection": settings}

    first = collect.discover(config(initial_tag))
    expected_identity = {"gamma_url": "https://gamma-api.polymarket.com"}
    if initial_tag is not None:
        expected_identity["event_tag_slug"] = initial_tag
    assert first["source_identity"] == expected_identity
    assert collect.discover(config(initial_tag)) == first
    assert calls == [(initial_tag, True, None), (initial_tag, False, None)]

    second = collect.discover(config(next_tag))
    expected_identity = {"gamma_url": "https://gamma-api.polymarket.com"}
    if next_tag is not None:
        expected_identity["event_tag_slug"] = next_tag
    assert second["source_identity"] == expected_identity
    assert second["event_count"] == 2
    assert calls == [(initial_tag, True, None), (initial_tag, False, None),
                     (next_tag, True, None), (next_tag, False, None)]
    prefix = next_tag or "all"
    assert list(collect._json_lines(tmp_path / "raw" / "events.jsonl")) == [
        {"id": f"{prefix}-True"}, {"id": f"{prefix}-False"},
    ]
    # The replacement result must also be cacheable under its own source identity.
    assert collect.discover(config(next_tag)) == second
    assert len(calls) == 4


def test_discover_source_change_resumes_without_reusing_old_open_pages(monkeypatch, tmp_path):
    calls = []
    offline = True
    class Client:
        def __init__(self, settings):
            self.tag = settings.get("event_tag_slug") or "all"
        def events(self, closed, cursor):
            calls.append((self.tag, closed, cursor))
            if self.tag == "oil" and closed:
                if cursor is None:
                    return {"events": [{"id": "oil-closed-1"}]}, "second"
                if offline:
                    raise RuntimeError("offline during source change")
                return {"events": [{"id": "oil-closed-2"}]}, None
            return {"events": [{"id": f"{self.tag}-{'closed' if closed else 'open'}"}]}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)
    config = {"data_dir": str(tmp_path), "collection": dict(SETTINGS)}
    collect.discover(config)
    config["collection"]["event_tag_slug"] = "oil"

    with pytest.raises(RuntimeError, match="offline during source change"):
        collect.discover(config)
    # Closed pages already belong to oil, while the old all-site open cache is
    # still complete. The top-level status has already recorded the new source.
    raw = tmp_path / "raw"
    assert collect._load(raw / "discovery.json")["source_identity"]["event_tag_slug"] == "oil"
    assert collect._load(raw / "event_pages" / "open" / "manifest.json")["complete"]
    assert list(collect._page_rows(raw / "event_pages" / "open", "events")) == [{"id": "all-open"}]

    offline = False
    result = collect.discover(config)
    assert result["complete"] and result["event_count"] == 3
    assert list(collect._json_lines(raw / "events.jsonl")) == [
        {"id": "oil-closed-1"}, {"id": "oil-closed-2"}, {"id": "oil-open"},
    ]
    assert calls == [("all", True, None), ("all", False, None),
                     ("oil", True, None), ("oil", True, "second"),
                     ("oil", True, "second"), ("oil", False, None)]
    for state in ("closed", "open"):
        manifest = collect._load(raw / "event_pages" / state / "manifest.json")
        assert manifest["source_identity"] == result["source_identity"]


def test_price_retention_fallback_keeps_old_data_and_excludes_settlement(tmp_path):
    queried = []
    class Client:
        def clob_prices(self, token, start, end):
            return {"history": [{"t": 240, "p": 0.6}]}
        def prices(self, token, start, end, grain, cursor):
            queried.append(grain)
            points = []
            if grain == 60:
                points = [{"timestamp": 240, "price": 0.6, "resolution_seconds": 60}]
            if grain == 300:
                points = [{"timestamp": 0, "price": 0.4, "resolution_seconds": 300},
                          {"timestamp": 240, "price": 0.5, "resolution_seconds": 300},
                          {"timestamp": 299, "price": 1.0, "resolution_seconds": 0}]
            return {"data": points}, None
    result = collect._download_prices(Client(), "1", tmp_path, "yes", 0, 300, SETTINGS, False)
    frame = pd.read_csv(tmp_path / "yes_prices.csv")
    assert queried == [60, 300]
    assert frame["timestamp"].tolist() == [0, 240]
    assert frame["price"].tolist() == [0.4, 0.6]
    assert frame["resolution_seconds"].tolist() == [300, 60]
    assert result["api_complete"]
    assert result["resolution_seconds"] == [60, 300]
    assert result["max_gap_seconds"] == 240
    # Raw settlement evidence is retained, but never enters prediction quotes.
    raw = collect._load(tmp_path / "raw" / "yes" / "0-300" / "data_300" / "000000.json")
    assert raw["data"][-1]["resolution_seconds"] == 0


def test_trade_pages_keep_identical_distinct_fills(tmp_path):
    trade = {"timestamp": 100, "price": 0.7, "size": 2}
    class Client:
        def trades(self, condition, cursor):
            return ({"data": [trade]}, "second") if cursor is None else ({"data": [trade]}, None)
    result = collect._download_trades(Client(), "condition", tmp_path, SETTINGS, False)
    assert result["rows"] == 2
    assert result["api_complete"]
    assert not result["lifetime_complete"]
    assert len(list(collect._json_lines(tmp_path / "trades.jsonl"))) == 2


def test_download_records_failures_and_keeps_partial_data(monkeypatch, tmp_path):
    market = {"market_id": "m1", "market_dir": "up_down/e/m", "token_id": "1",
              "no_token_id": "2", "condition_id": "condition",
              "created_at": "2026-01-01", "closed_at": "2026-01-01"}
    (tmp_path / "markets.jsonl").write_text(json.dumps(market) + "\n")
    class Client:
        def __init__(self, settings): pass
        def clob_prices(self, *args): raise RuntimeError("offline")
        def prices(self, *args): return {"data": []}, None
        def trades(self, *args): return {"data": []}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)
    with pytest.raises(RuntimeError, match="markets incomplete"):
        collect.download({"data_dir": str(tmp_path), "collection": SETTINGS})
    status = collect._load(tmp_path / "collection.json")
    assert status["failed_markets"] == ["m1"]
    assert not status["complete"]
    coverage = collect._load(tmp_path / market["market_dir"] / "coverage.json")
    assert not coverage["yes"]["api_complete"]
    assert coverage["yes"]["rows"] == 0
    assert (tmp_path / market["market_dir"] / "yes_prices.csv").exists()


def test_history_bounds_use_full_life_not_research_window():
    start, end = collect._history_bounds({"created_at": "2023-01-01", "opened_at": "2023-01-03",
                                          "expires_at": "2023-02-01"}, 1800000000)
    assert start == int(pd.Timestamp("2023-01-01", tz="UTC").timestamp())
    assert end == 1800000000  # expiry alone does not prove trading has closed


def test_local_event_source_changes_are_not_silently_cached(tmp_path):
    source = tmp_path / "source.json"
    destination = tmp_path / "data"
    source.write_text(json.dumps([{"id": "1", "title": "before"}]))
    config = {"data_dir": str(destination), "collection": {"events_file": str(source)}}
    collect.discover(config)
    source.write_text(json.dumps([{"id": "1", "title": "after"}]))
    collect.discover(config)
    assert list(collect._json_lines(destination / "raw" / "events.jsonl"))[0]["title"] == "after"


def test_changed_token_invalidates_price_cache(monkeypatch, tmp_path):
    market = {"market_id": "m1", "market_dir": "up_down/e/m", "token_id": "1",
              "no_token_id": "2", "condition_id": "condition",
              "created_at": "2026-01-01", "closed_at": "2026-01-01"}
    path = tmp_path / "markets.jsonl"
    path.write_text(json.dumps(market) + "\n")
    calls = []
    class Client:
        def __init__(self, settings): pass
        def clob_prices(self, token, start, end):
            calls.append(token)
            return {"history": [{"t": start, "p": float(token) / 10}]}
        def prices(self, *args): return {"data": []}, None
        def trades(self, *args): return {"data": []}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)
    config = {"data_dir": str(tmp_path), "collection": SETTINGS}
    collect.download(config)
    assert calls == ["1", "2"]
    collect.download(config)
    assert calls == ["1", "2"]
    market["token_id"] = "3"
    path.write_text(json.dumps(market) + "\n")
    collect.download(config)
    assert calls == ["1", "2", "3", "2"]
    assert pd.read_csv(tmp_path / market["market_dir"] / "yes_prices.csv")["price"].iloc[0] == 0.3
    (tmp_path / market["market_dir"] / "yes_prices.csv").unlink()
    collect.download(config)
    assert (tmp_path / market["market_dir"] / "yes_prices.csv").exists()
    assert calls == ["1", "2", "3", "2"]  # rebuilt from matching raw pages


def _write_markets(directory, count):
    markets = [{"market_id": f"m{index}", "market_dir": f"up_down/e/m{index}",
                "token_id": f"yes{index}", "no_token_id": f"no{index}",
                "condition_id": f"condition{index}",
                "created_at": "2026-01-01", "closed_at": "2026-01-01"}
               for index in range(count)]
    (directory / "markets.jsonl").write_text(
        "".join(json.dumps(market) + "\n" for market in markets))
    return markets


def test_parallel_download_isolates_clients_reports_progress_and_reuses_cache(monkeypatch, tmp_path):
    markets = _write_markets(tmp_path, 4)
    barrier = threading.Barrier(2)
    clients, calls, progress = [], [], []
    main_thread = threading.get_ident()
    class Client:
        def __init__(self, settings):
            self.owner = threading.get_ident()
            self.started = False
            self.closed = False
            clients.append(self)
        def clob_prices(self, token, start, end):
            assert threading.get_ident() == self.owner
            calls.append(token)
            if not self.started:
                self.started = True
                barrier.wait(timeout=5)
            return {"history": [{"t": start, "p": 0.6}]}
        def prices(self, *args):
            assert threading.get_ident() == self.owner
            return {"data": []}, None
        def trades(self, *args):
            assert threading.get_ident() == self.owner
            return {"data": [{"timestamp": 100, "price": 0.6}]}, None
        def close(self):
            self.closed = True
    save = collect._save
    def track_progress(path, value):
        if path.name == "collection.json":
            assert threading.get_ident() == main_thread
            progress.append((value["processed"], value["completed"], value["failures"]))
        save(path, value)
    monkeypatch.setattr(collect, "OfficialClient", Client)
    monkeypatch.setattr(collect, "_save", track_progress)
    config = {"data_dir": str(tmp_path), "collection": {**SETTINGS, "workers": 2}}

    status = collect.download(config)
    assert len(clients) == 2
    assert len({client.owner for client in clients}) == 2
    assert all(client.closed for client in clients)
    assert len(calls) == 8
    assert status["complete"]
    assert (status["processed"], status["total"], status["completed"], status["failures"]) == (4, 4, 4, 0)
    assert progress == [(0, 0, 0), (1, 1, 0), (2, 2, 0), (3, 3, 0), (4, 4, 0), (4, 4, 0)]
    for market in markets:
        directory = tmp_path / market["market_dir"]
        assert collect._load(directory / "coverage.json")["api_complete"]
        assert pd.read_csv(directory / "yes_prices.csv")["price"].tolist() == [0.6]
        assert pd.read_csv(directory / "no_prices.csv")["price"].tolist() == [0.6]
        assert len(list(collect._json_lines(directory / "trades.jsonl"))) == 1

    def reject_client(settings):
        pytest.fail("completed market caches must not need a network client")
    monkeypatch.setattr(collect, "OfficialClient", reject_client)
    assert collect.download(config)["completed"] == 4
    assert len(calls) == 8


def test_parallel_download_finishes_other_markets_and_resumes_failed_requests(monkeypatch, tmp_path):
    markets = _write_markets(tmp_path, 3)
    calls = []
    offline = True
    class Client:
        def __init__(self, settings): pass
        def clob_prices(self, token, start, end):
            calls.append(token)
            if offline and token == "yes1":
                raise RuntimeError("offline")
            return {"history": [{"t": start, "p": 0.4}]}
        def prices(self, *args): return {"data": []}, None
        def trades(self, *args): return {"data": []}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)
    config = {"data_dir": str(tmp_path), "collection": {**SETTINGS, "workers": 2}}

    with pytest.raises(RuntimeError, match="1 markets incomplete"):
        collect.download(config)
    status = collect._load(tmp_path / "collection.json")
    assert not status["complete"]
    assert (status["processed"], status["total"], status["completed"], status["failures"]) == (3, 3, 2, 1)
    assert status["failed_markets"] == ["m1"]
    coverage = collect._load(tmp_path / markets[1]["market_dir"] / "coverage.json")
    assert not coverage["yes"]["api_complete"]
    assert "offline" in coverage["yes"]["errors"][0]
    assert coverage["no"]["api_complete"]
    assert coverage["yes"]["rows"] == 0
    assert coverage["no"]["rows"] == 1
    assert len(calls) == 6

    offline = False
    assert collect.download(config)["completed"] == 3
    # Successful markets and successful raw requests inside the failed market survive.
    assert len(calls) == 7
    assert calls[-1] == "yes1"


@pytest.mark.parametrize("workers", [1, 2])
def test_empty_market_download_completes_without_client(monkeypatch, tmp_path, workers):
    _write_markets(tmp_path, 0)
    def reject_client(settings):
        pytest.fail("an empty market set must not create a network client")
    monkeypatch.setattr(collect, "OfficialClient", reject_client)
    status = collect.download({"data_dir": str(tmp_path), "collection": {"workers": workers}})
    assert status["complete"]
    assert (status["processed"], status["total"], status["completed"], status["failures"]) == (0, 0, 0, 0)
    assert collect._load(tmp_path / "collection.json") == status


@pytest.mark.parametrize("duplicate", ["market_id", "market_dir", "market_dir_alias"])
def test_download_rejects_duplicate_markets_before_creating_clients(monkeypatch, tmp_path, duplicate):
    markets = _write_markets(tmp_path, 2)
    key = "market_dir" if duplicate == "market_dir_alias" else duplicate
    markets[1][key] = markets[0][key]
    if duplicate == "market_dir_alias":
        markets[1][key] += "/../m0"
    (tmp_path / "markets.jsonl").write_text("".join(json.dumps(market) + "\n" for market in markets))
    def reject_client(settings):
        pytest.fail("duplicate input must be rejected before creating network clients")
    monkeypatch.setattr(collect, "OfficialClient", reject_client)
    with pytest.raises(ValueError, match=f"Duplicate {key}"):
        collect.download({"data_dir": str(tmp_path), "collection": {"workers": 2}})


@pytest.mark.parametrize("interrupt_before", ["no", "trades"])
def test_identity_switch_interrupt_keeps_new_pages_and_never_reuses_old_streams(
        monkeypatch, tmp_path, interrupt_before):
    market = _write_markets(tmp_path, 1)[0]
    calls = []
    class Client:
        def __init__(self, settings): pass
        def clob_prices(self, token, start, end):
            calls.append(token)
            price = 0.8 if token.startswith("new-") else 0.2
            return {"history": [{"t": start, "p": price}]}
        def prices(self, *args): return {"data": []}, None
        def trades(self, condition, cursor):
            calls.append(condition)
            return {"data": [{"timestamp": 100, "condition": condition}]}, None
        def close(self): pass
    monkeypatch.setattr(collect, "OfficialClient", Client)
    config = {"data_dir": str(tmp_path), "collection": SETTINGS}
    collect.download(config)
    directory = tmp_path / market["market_dir"]
    raw = directory / "raw"
    old_raw = {path.relative_to(raw): path.read_bytes() for path in raw.rglob("*") if path.is_file()}

    market.update(token_id="new-yes", no_token_id="new-no", condition_id="new-condition")
    (tmp_path / "markets.jsonl").write_text(json.dumps(market) + "\n")
    prices = collect._download_prices
    trades = collect._download_trades
    interrupted = False
    def interrupt_prices(client, token, directory, side, *args):
        nonlocal interrupted
        if interrupt_before == side and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated process interruption")
        return prices(client, token, directory, side, *args)
    def interrupt_trades(*args):
        nonlocal interrupted
        if interrupt_before == "trades" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("simulated process interruption")
        return trades(*args)
    monkeypatch.setattr(collect, "_download_prices", interrupt_prices)
    monkeypatch.setattr(collect, "_download_trades", interrupt_trades)

    with pytest.raises(KeyboardInterrupt, match="simulated process interruption"):
        collect.download(config)
    assert not collect._load(tmp_path / "collection.json")["complete"]
    assert collect._load(directory / "coverage.json")["source_identity"]["no_token_id"] == "new-no"
    assert calls.count("new-yes") == 1
    assert collect.download(config)["complete"]
    # The new Yes raw pages survive interruption; untouched No/trades fetch the new source.
    assert calls == ["yes0", "no0", "condition0", "new-yes", "new-no", "new-condition"]
    assert pd.read_csv(directory / "yes_prices.csv")["price"].tolist() == [0.8]
    assert pd.read_csv(directory / "no_prices.csv")["price"].tolist() == [0.8]
    assert list(collect._json_lines(directory / "trades.jsonl"))[0]["condition"] == "new-condition"
    archives = list((directory / "raw_archive").iterdir())
    assert len(archives) == 1
    archived = {path.relative_to(archives[0]): path.read_bytes()
                for path in archives[0].rglob("*") if path.is_file()}
    assert archived == old_raw
