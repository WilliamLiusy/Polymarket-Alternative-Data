"""Convert complete historical membership and AKShare bars to Qlib Alpha158."""
from __future__ import annotations

import gc
import hashlib
import json
import logging
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
LABEL = "Ref($close, -2)/Ref($close, -1) - 1"
FIELDS = ("open", "close", "high", "low", "volume", "amount", "factor", "vwap")
VERSION = 1


def read_inputs(config: dict) -> tuple[pd.DatetimeIndex, dict[pd.Timestamp, set[str]]]:
    """Require a membership snapshot for every research trading day."""
    data_dir = Path(config["data_dir"])
    failures = data_dir / "failures.json"
    if failures.is_file() and json.loads(failures.read_text()):
        raise ValueError(f"Download is incomplete; resolve failures before prepare: {failures}")
    metadata_path = data_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else None
    calendar = pd.DatetimeIndex(pd.to_datetime((data_dir / "calendar.txt").read_text().splitlines()))
    if calendar.empty or calendar.has_duplicates or not calendar.is_monotonic_increasing:
        raise ValueError("calendar.txt must contain unique trading days in ascending order")
    start, end = pd.Timestamp(config["research_start"]), pd.Timestamp(config["research_end"])
    if metadata is not None:
        coverage = metadata.get("calendar", {})
        if not coverage or pd.Timestamp(coverage["start"]) > start or pd.Timestamp(coverage["end"]) < end:
            raise ValueError("Download metadata has no completed calendar covering the research range")
    if calendar.min() > start or calendar.max() < end:
        raise ValueError("calendar.txt must cover the complete research range, including warmup")
    research_days = calendar[(calendar >= start) & (calendar <= end)]
    if research_days.empty:
        raise ValueError("No trading days in the research range")
    membership = {}
    for day in research_days:
        path = data_dir / "instruments" / f"{day:%Y-%m-%d}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing historical membership snapshot: {path}")
        codes = path.read_text().splitlines()
        if not codes or any(not re.fullmatch(r"(?:SH|SZ)\d{6}", code) for code in codes):
            raise ValueError(f"Expected nonempty SH/SZ stock codes in {path}")
        if len(codes) != len(set(codes)):
            raise ValueError(f"Duplicate instruments in {path}")
        if metadata is not None and metadata.get("memberships", {}).get(f"{day:%Y-%m-%d}", {}).get("count") != len(codes):
            raise ValueError(f"Missing or stale completed membership metadata: {path}")
        membership[day] = set(codes)
    for code in sorted(set.union(*membership.values())):
        path = data_dir / "data" / f"{code}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing historical member's bars (cannot omit survivors): {path}")
        if metadata is not None:
            price = metadata.get("prices", {}).get(code, {})
            if (not price or price.get("adjust") != "hfq" or pd.Timestamp(price["start"]) > start
                    or pd.Timestamp(price["end"]) < end):
                raise ValueError(f"Missing or stale completed price metadata: {path}")
    return calendar[calendar <= end], membership


def membership_intervals(membership: dict[pd.Timestamp, set[str]]) -> list[tuple[str, str, str]]:
    """Compress daily membership into closed intervals, preserving exits/reentries."""
    rows, active = [], {}
    previous = None
    for day, members in sorted(membership.items()):
        for code in sorted(active.keys() - members):
            rows.append((code, active.pop(code).strftime("%Y-%m-%d"), previous.strftime("%Y-%m-%d")))
        for code in members - active.keys():
            active[code] = day
        previous = day
    for code, start in active.items():
        rows.append((code, start.strftime("%Y-%m-%d"), previous.strftime("%Y-%m-%d")))
    return sorted(rows)


def convert_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep adjusted prices and inversely adjusted shares consistent with turnover."""
    columns = {"日期": "datetime", "开盘": "open", "收盘": "close", "最高": "high",
               "最低": "low", "成交量": "volume", "成交额": "amount", "raw_close": "raw_close"}
    missing = columns.keys() - set(frame.columns)
    if missing:
        raise ValueError(f"AKShare bars missing required columns: {sorted(missing)}")
    bars = frame[list(columns)].rename(columns=columns).copy()
    bars["datetime"] = pd.to_datetime(bars["datetime"])
    bars = bars.set_index("datetime").sort_index()
    if bars.index.has_duplicates or bars.index.hasnans:
        raise ValueError("Bars must have unique, nonmissing dates")
    bars = bars.apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    shares = bars["volume"] * 100.0  # AKShare stock_zh_a_hist volume unit is hands.
    valid = (bars[["open", "close", "high", "low", "raw_close"]] > 0).all(axis=1) & (shares > 0)
    bars["factor"] = bars["close"] / bars["raw_close"]
    bars["vwap"] = bars["amount"] / shares * bars["factor"]
    bars["volume"] = shares / bars["factor"]
    bars.loc[bars["amount"] <= 0, ["amount", "vwap"]] = np.nan
    bars.loc[~valid, :] = np.nan  # Suspensions must not turn into tradable bars.
    return bars[list(FIELDS)].replace([np.inf, -np.inf], np.nan).astype("float32")


def write_binary(bars: pd.DataFrame, calendar: pd.DatetimeIndex, directory: Path) -> int:
    """Qlib day binary: little-endian float32 calendar offset, followed by values."""
    bars = bars.loc[bars.index.intersection(calendar)]
    if bars.empty or bars["close"].notna().sum() == 0:
        raise ValueError(f"No usable research/warmup bars for {directory.name}")
    first, last = calendar.get_indexer([bars.index.min(), bars.index.max()])
    aligned = bars.reindex(calendar[first:last + 1])
    directory.mkdir(parents=True, exist_ok=True)
    for field in FIELDS:
        values = np.r_[float(first), aligned[field].to_numpy()].astype("<f4")
        values.tofile(directory / f"{field}.day.bin")
    return len(aligned)


def source_fingerprint(config: dict, membership: dict[pd.Timestamp, set[str]]) -> str:
    root = Path(config["data_dir"])
    files = [root / "calendar.txt"]
    if (root / "metadata.json").is_file():
        files.append(root / "metadata.json")
    files += [root / "instruments" / f"{day:%Y-%m-%d}.txt" for day in sorted(membership)]
    files += [root / "data" / f"{code}.csv" for code in sorted(set.union(*membership.values()))]
    digest = hashlib.sha256(json.dumps({"version": VERSION, "research_start": config["research_start"],
                                      "research_end": config["research_end"]}, sort_keys=True).encode())
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_manifest(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}


def _provider_complete(path: Path, manifest: dict, fingerprint: str) -> bool:
    if manifest.get("fingerprint") != fingerprint or manifest.get("version") != VERSION:
        return False
    if not all((path / name).is_file() for name in ("calendars/day.txt", "instruments/csi300.txt", "instruments/all.txt")):
        return False
    lengths = manifest.get("stock_lengths", {})
    return bool(lengths) and all(
        (path / "features" / code.lower() / f"{field}.day.bin").is_file()
        and (path / "features" / code.lower() / f"{field}.day.bin").stat().st_size == 4 * (length + 1)
        for code, length in lengths.items() for field in FIELDS
    )


def build_provider(config: dict, calendar: pd.DatetimeIndex, membership: dict,
                   fingerprint: str, refresh: bool = False) -> Path:
    destination = Path(config["provider_dir"])
    existing = read_manifest(destination / "manifest.json")
    if not refresh and _provider_complete(destination, existing, fingerprint):
        LOG.info("Reuse Qlib provider %s", destination)
        return destination
    if destination.exists() and any(destination.iterdir()) and existing.get("producer") != "baseline.prepare":
        raise ValueError(f"Refuse to replace an unrecognized provider directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent))
    try:
        (staging / "calendars").mkdir()
        (staging / "calendars/day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")) + "\n")
        (staging / "instruments").mkdir()
        instrument_text = "".join("\t".join(row) + "\n" for row in membership_intervals(membership))
        for name in ("all", "csi300"):
            (staging / "instruments" / f"{name}.txt").write_text(instrument_text)
        stock_lengths = {}
        for code in sorted(set.union(*membership.values())):
            bars = convert_bars(pd.read_csv(Path(config["data_dir"]) / "data" / f"{code}.csv"))
            member_days = pd.DatetimeIndex([day for day, members in membership.items() if code in members])
            if not bars["close"].reindex(member_days).notna().any():
                raise ValueError(f"Historical member {code} has no usable bars while it belongs to the index")
            stock_lengths[code] = write_binary(bars, calendar, staging / "features" / code.lower())
        manifest = {"producer": "baseline.prepare", "version": VERSION,
                    "fingerprint": fingerprint, "stock_lengths": stock_lengths}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def _cache_complete(path: Path, manifest: dict, fingerprint: str, years: list[int]) -> bool:
    if manifest.get("fingerprint") != fingerprint or manifest.get("years") != years:
        return False
    import pyarrow.parquet as pq
    for year in years:
        file = path / f"{year}.parquet"
        expected = manifest.get("year_files", {}).get(str(year), {})
        if not file.is_file() or file.stat().st_size != expected.get("bytes"):
            return False
        try:
            metadata = pq.read_metadata(file)
        except (ValueError, OSError):
            return False
        if metadata.num_rows != expected.get("rows") or metadata.num_columns != 162:
            return False
    return True


def prepare(config: dict, refresh: bool = False) -> Path:
    """Build a local provider and year-sized, cross-section-normalized Alpha158 cache."""
    import qlib
    from qlib.contrib.data.handler import Alpha158
    from qlib.data.cache import H
    from qlib.data.dataset.handler import DataHandlerLP

    calendar, membership = read_inputs(config)
    fingerprint = source_fingerprint(config, membership)
    provider = build_provider(config, calendar, membership, fingerprint, refresh)
    start, end = pd.Timestamp(config["research_start"]), pd.Timestamp(config["research_end"])
    years = list(range(start.year, end.year + 1))
    cache = Path(config["cache_dir"])
    cache_fingerprint = hashlib.sha256(f"{fingerprint}|{qlib.__version__}|{LABEL}|Alpha158|zscore-ddof1-v{VERSION}".encode()).hexdigest()
    previous = read_manifest(cache / "manifest.json")
    if not refresh and _cache_complete(cache, previous, cache_fingerprint, years):
        LOG.info("Reuse Alpha158 cache %s", cache)
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    # No fitted feature normalization: LightGBM handles missing values natively.
    qlib.init(provider_uri=str(provider.resolve()), region="cn", kernels=1,
              expression_cache=None, dataset_cache=None)
    manifest = {"version": VERSION, "fingerprint": cache_fingerprint,
                "provider_fingerprint": fingerprint, "research_start": str(start.date()),
                "research_end": str(end.date()), "years": years, "label": LABEL,
                "feature_count": 158, "year_files": {}}
    source_metadata = read_manifest(Path(config["data_dir"]) / "metadata.json")
    manifest["membership_source"] = source_metadata.get("membership_source", "local")
    manifest["membership_note"] = source_metadata.get("membership_note", "Local input without source metadata")
    # Remove the completion marker before writing any new chunks.
    (cache / "manifest.json").unlink(missing_ok=True)
    for year in years:
        begin, finish = max(start, pd.Timestamp(year, 1, 1)), min(end, pd.Timestamp(year, 12, 31))
        LOG.info("Compute Alpha158 %s to %s", begin.date(), finish.date())
        handler = Alpha158(
            instruments="csi300", start_time=begin, end_time=finish,
            infer_processors=[],
            learn_processors=[{"class": "DropnaLabel"},
                              {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label", "method": "zscore"}},
                              {"class": "DropnaLabel"}],
            label=([LABEL], ["LABEL0"]),
        )
        features = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
        raw = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)
        label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L).reindex(features.index)
        if len(features.columns) != 158:
            raise ValueError(f"Expected 158 Alpha158 features, got {len(features.columns)}")
        panel = pd.concat({"feature": features, "label": label, "raw_label": raw.reindex(features.index)}, axis=1)
        panel = panel.replace([np.inf, -np.inf], np.nan).astype("float32").sort_index()
        if panel.index.names != ["datetime", "instrument"]:
            panel = panel.reorder_levels(["datetime", "instrument"]).sort_index()
        path = cache / f"{year}.parquet"
        temporary = cache / f".{year}.parquet.tmp"
        panel.to_parquet(temporary)
        temporary.replace(path)
        manifest["year_files"][str(year)] = {"bytes": path.stat().st_size, "rows": len(panel)}
        LOG.info("Cached %s rows (%s MiB) for %s", len(panel), round(panel.memory_usage(deep=True).sum() / 2**20, 1), year)
        del handler, features, raw, label, panel
        H.clear()
        gc.collect()
    (cache / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return cache
