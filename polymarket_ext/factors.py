"""Align token histories to A-share closes and calculate the five oil factor sets."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
VERSION = 4
# Fixed elapsed-hour lookbacks. Each limit applies to both resolution and age
# at each endpoint; hourly age is measured from availability, not bucket start.
HOURLY_LIMITS = {1: 3600, 3: 10800, 6: 10800}
HOURLY_FACTORS = {
    category: tuple(f"{prefix}H{hours}" for prefix in prefixes for hours in HOURLY_LIMITS)
    for category, prefixes in {
        "direction": ("",), "threshold": ("",), "touch": ("HIGH_", "LOW_"),
        "range": ("",), "record": ("HIGH_",),
    }.items()
}
CATEGORY_FACTORS = {
    "direction": ("PROB", "ENTROPY", "DIRECTION_SCORE", "DIRECTION_CERTAINTY",
                  "WINDOW_LENGTH", "TIME_TO_END"),
    "threshold": ("PROB", "ENTROPY", "THRESHOLD_REL_POSITION",
                  "THRESHOLD_ADJACENT_MASS", "THRESHOLD_REL_DENSITY"),
    "touch": ("HIGH_PROB", "HIGH_D1", "HIGH_REL_POSITION", "LOW_PROB", "LOW_D1",
              "LOW_REL_POSITION", "ENTROPY"),
    "range": ("PROB", "D1", "D5", "RANGE_REL_WIDTH", "RANGE_REL_DENSITY"),
    "record": ("HIGH_PROB", "HIGH_D1", "HIGH_D5", "ENTROPY"),
}
CATEGORY_FACTORS = {category: factors + HOURLY_FACTORS[category]
                    for category, factors in CATEGORY_FACTORS.items()}
FACTOR_COLUMNS = tuple(dict.fromkeys(name for names in CATEGORY_FACTORS.values() for name in names))
TIME_COLUMNS = ("opened_at", "closed_at", "metadata_available_at", "observation_start",
                "observation_end", "expires_at")
GROUP_COLUMNS = ("event_id", "category", "underlying", "price_basis",
                 "observation_start", "observation_end", "expires_at", "side")


def _calendar(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    days = pd.DatetimeIndex(calendar)
    if days.tz is not None:
        days = days.tz_convert("Asia/Shanghai").tz_localize(None)
    days = days.normalize()
    if days.hasnans or days.has_duplicates or not days.is_monotonic_increasing:
        raise ValueError("Calendar must contain unique trading days in ascending order")
    return days


def _timestamps(values: pd.Series) -> pd.Series:
    """Official history timestamps are Unix seconds; metadata uses ISO timestamps."""
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="s", utc=True, errors="coerce")
    return pd.to_datetime(values, format="mixed", utc=True, errors="coerce")


def _active_at(row: dict, cutoffs) -> np.ndarray:
    active = np.asarray((cutoffs >= row["opened_at"]) & (cutoffs >= row["metadata_available_at"]))
    for column in ("observation_end", "expires_at", "closed_at"):
        if pd.notna(row[column]):
            active &= np.asarray(cutoffs < row[column])
    return active


def _match_quotes(targets: pd.DataFrame, quotes: pd.DataFrame | None) -> pd.DataFrame:
    """Backward as-of selection, before any price/age/resolution validation."""
    if quotes is not None and not quotes.empty:
        return pd.merge_asof(targets, quotes, left_on="cutoff", right_on="quote_available_at",
                             direction="backward")
    frame = targets.copy()
    for column in ("quote_time", "quote_available_at"):
        frame[column] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    frame["price"] = np.nan
    frame["quote_resolution_seconds"] = np.nan
    frame["quote_source"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    return frame


def _hourly_endpoints(targets: pd.DataFrame, quotes: pd.DataFrame | None, row: dict,
                      current: pd.DataFrame) -> pd.DataFrame:
    """Keep raw endpoint evidence, including invalid prices and inactive lags."""
    columns = {}
    for hours in (0, *HOURLY_LIMITS):
        shifted = targets.copy()
        shifted["cutoff"] -= pd.Timedelta(hours=hours)
        endpoint = current if hours == 0 else _match_quotes(shifted, quotes)
        for name in ("cutoff", "price", "quote_time", "quote_available_at",
                     "quote_resolution_seconds", "quote_source"):
            columns[f"hourly_{hours}h_{name}"] = endpoint[name]
        columns[f"hourly_{hours}h_age_seconds"] = (
            endpoint["cutoff"] - endpoint["quote_available_at"]).dt.total_seconds()
        columns[f"hourly_{hours}h_market_active"] = _active_at(row, endpoint["cutoff"])
    return pd.DataFrame(columns, index=targets.index)


def _hourly_probability(panel: pd.DataFrame, hours: int, limit: int) -> pd.Series:
    prefix = f"hourly_{hours}h_"
    price = panel[prefix + "price"]
    resolution = panel[prefix + "quote_resolution_seconds"]
    valid = (panel[prefix + "market_active"] & price.between(0, 1)
             & resolution.between(0, limit) & panel[prefix + "age_seconds"].between(0, limit))
    return price.where(valid)


def align_prices(markets: pd.DataFrame, prices: pd.DataFrame, calendar: pd.DatetimeIndex,
                 max_quote_age=None) -> pd.DataFrame:
    """Use the latest available observation at 15:00 Beijing, retaining missing quotes.

    A market becomes available at both its opening and metadata creation times.
    Quotes remain usable until the earliest observation end, expiry, or actual
    close. Forecasts before an observation window begins are allowed. The most
    recent invalid price is never replaced with an older, valid observation.
    Data v2 aggregate timestamps mark bucket starts, so those observations only
    become available at bucket end. Exact ticks, CLOB observations and local
    histories without a source retain their original observation timestamps.
    ``max_quote_age`` accepts a timedelta string or a number of seconds and
    affects daily quotes only. Hourly endpoints retain raw evidence for their
    independent resolution/availability-age limits in ``compute_factors``.
    """
    days = _calendar(calendar)
    cutoffs = (days.tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).tz_convert("UTC")
    metadata = markets.copy()
    for column in TIME_COLUMNS:
        metadata[column] = _timestamps(metadata.get(column, pd.Series(pd.NaT, index=metadata.index)))
    for column in ("threshold", "lower", "upper", "record_threshold"):
        metadata[column] = pd.to_numeric(metadata.get(column, np.nan), errors="coerce")
    for column in GROUP_COLUMNS:
        if column not in metadata:
            metadata[column] = None
    # Side is a grouping key only for touch markets.
    metadata.loc[metadata["category"] != "touch", "side"] = None
    for column in ("event_id", "market_id", "token_id"):
        if column not in metadata:
            metadata[column] = pd.Series(dtype=str)
        metadata[column] = metadata[column].astype(str)
    if metadata["market_id"].duplicated().any():
        raise ValueError("Normalized metadata must have one row per market_id")

    history = prices.copy()
    for column in ("timestamp", "token_id", "price"):
        if column not in history:
            history[column] = pd.Series(dtype=object)
    history["timestamp"] = _timestamps(history["timestamp"])
    history["token_id"] = history["token_id"].astype(str)
    history["price"] = pd.to_numeric(history["price"], errors="coerce")
    history["resolution_seconds"] = pd.to_numeric(history.get("resolution_seconds", np.nan), errors="coerce")
    history["source"] = history.get("source", pd.Series(index=history.index, dtype=object)).astype("string")
    resolution = history["resolution_seconds"]
    delay = resolution.where(np.isfinite(resolution) & resolution.ge(0)).where(history["source"].eq("data_v2"), 0)
    history["quote_available_at"] = history["timestamp"] + pd.to_timedelta(delay, unit="s")
    history = history.dropna(subset=["timestamp", "quote_available_at"]).sort_values("timestamp", kind="stable")
    histories = {token: frame for token, frame in history.groupby("token_id", sort=False)}
    age = None
    if max_quote_age is not None:
        age = (pd.Timedelta(seconds=max_quote_age) if isinstance(max_quote_age, (int, float))
               else pd.Timedelta(max_quote_age))
        if age < pd.Timedelta(0):
            raise ValueError("max_quote_age cannot be negative")

    frames = []
    for row in metadata.to_dict("records"):
        opened, available = row["opened_at"], row["metadata_available_at"]
        if pd.isna(opened) or pd.isna(available):
            continue
        active = _active_at(row, cutoffs)
        if not active.any():
            continue
        targets = pd.DataFrame({"datetime": days[active], "cutoff": cutoffs[active]})
        token_history = histories.get(row["token_id"])
        quotes = None
        if token_history is not None and not token_history.empty:
            quotes = token_history[["timestamp", "quote_available_at", "price", "resolution_seconds", "source"]].rename(
                columns={"timestamp": "quote_time", "resolution_seconds": "quote_resolution_seconds",
                         "source": "quote_source"})
            quotes = quotes.loc[quotes["quote_time"] >= opened].sort_values(
                ["quote_available_at", "quote_time"], kind="stable")
            # A delayed older bucket must never displace a newer observation.
            # Keep only updates to the latest observation known at each arrival.
            quotes = quotes.loc[quotes["quote_time"].eq(quotes["quote_time"].cummax())]
            quotes = quotes.drop_duplicates("quote_available_at", keep="last")
        frame = _match_quotes(targets, quotes)
        hourly = _hourly_endpoints(targets, quotes, row, frame)
        frame["quote_age_seconds"] = (frame["cutoff"] - frame["quote_time"]).dt.total_seconds()
        valid = frame["price"].between(0, 1) & frame["quote_time"].notna()
        if age is not None:
            valid &= frame["quote_age_seconds"] <= age.total_seconds()
        frame["probability"] = frame["price"].where(valid)
        frame["quote_valid"] = valid
        frame = pd.concat([frame, hourly], axis=1)
        for name, value in row.items():
            # Preserve list/dict metadata as objects, should callers supply any.
            frame[name] = [value] * len(frame)
        frames.append(frame)
    if not frames:
        empty = metadata.iloc[:0].copy()
        empty["datetime"] = pd.Series(dtype="datetime64[ns]")
        for column in ("cutoff", "quote_time", "quote_available_at"):
            empty[column] = pd.Series(dtype="datetime64[ns, UTC]")
        for column in ("price", "probability", "quote_age_seconds", "quote_resolution_seconds"):
            empty[column] = pd.Series(dtype=float)
        empty["quote_valid"] = pd.Series(dtype=bool)
        empty["quote_source"] = pd.Series(dtype="string")
        row = dict.fromkeys(TIME_COLUMNS, pd.NaT)
        hourly = _hourly_endpoints(empty[["datetime", "cutoff"]], None, row, empty)
        return pd.concat([empty, hourly], axis=1)
    return pd.concat(frames, ignore_index=True).sort_values(["datetime", "market_id"]).reset_index(drop=True)


def _differences(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> tuple[pd.Series, pd.Series]:
    d1 = pd.Series(np.nan, index=panel.index)
    d5 = pd.Series(np.nan, index=panel.index)
    for _, token in panel.groupby("token_id", sort=False):
        if token["datetime"].duplicated().any():
            raise ValueError("Each token must have only one row per trading day")
        values = token.set_index("datetime")["probability"].reindex(calendar)
        one = values.diff()
        five = values.diff(5).where(values.notna().rolling(6, min_periods=6).sum() == 6)
        d1.loc[token.index] = one.reindex(token["datetime"]).to_numpy()
        d5.loc[token.index] = five.reindex(token["datetime"]).to_numpy()
    return d1, d5


def compute_factors(aligned: pd.DataFrame, calendar: pd.DatetimeIndex,
                    negative_mass_tolerance: float = 1e-8) -> pd.DataFrame:
    """Calculate the AGENTS.md factors without crossing token or calendar gaps.

    Adjacent threshold mass is attached to its lower threshold. Duplicate
    thresholds are ambiguous and invalidate both adjacent intervals. Group
    boundaries include open members with missing prices, so missing quotes never
    cause a jump to a more distant threshold or alter the group scale.
    """
    if negative_mass_tolerance < 0:
        raise ValueError("negative_mass_tolerance cannot be negative")
    calendar = _calendar(calendar)
    result = aligned.copy().reset_index(drop=True)
    for column in FACTOR_COLUMNS:
        result[column] = np.nan
    result["group_center"] = np.nan
    result["group_scale"] = np.nan
    result["group_boundary_count"] = 0
    if result.empty:
        return result
    result["datetime"] = pd.to_datetime(result["datetime"]).dt.normalize()
    if not result["datetime"].isin(calendar).all():
        raise ValueError("Aligned rows contain dates outside the trading calendar")
    probability = pd.to_numeric(result["probability"], errors="coerce")
    result["probability"] = probability.where(probability.between(0, 1))
    p = result["probability"]
    entropy = pd.Series(np.nan, index=result.index)
    entropy.loc[p.isin([0, 1])] = 0.0
    interior = p.between(0, 1, inclusive="neither")
    entropy.loc[interior] = -p[interior] * np.log(p[interior]) - (1 - p[interior]) * np.log1p(-p[interior])
    d1, d5 = _differences(result, calendar)
    category = result["category"]
    result.loc[category.isin(["direction", "threshold", "range"]), "PROB"] = p
    result.loc[category != "range", "ENTROPY"] = entropy
    direction = category == "direction"
    result.loc[direction, "DIRECTION_SCORE"] = 2 * p[direction] - 1
    result.loc[direction, "DIRECTION_CERTAINTY"] = abs(2 * p[direction] - 1)
    # Rule-defined comparison times are independent of quotes and market creation.
    # UTC seconds preserve weekends and daylight-saving transitions.
    starts = _timestamps(result.loc[direction, "observation_start"])
    ends = _timestamps(result.loc[direction, "observation_end"])
    cutoffs = _timestamps(result.loc[direction, "cutoff"])
    window_days = (ends - starts).dt.total_seconds() / 86400
    remaining_days = (ends - cutoffs).dt.total_seconds() / 86400
    result.loc[direction, "WINDOW_LENGTH"] = np.log1p(window_days.where(window_days > 0))
    result.loc[direction, "TIME_TO_END"] = np.log1p(remaining_days.where(remaining_days > 0))
    ranges = category == "range"
    result.loc[ranges, "D1"] = d1
    result.loc[ranges, "D5"] = d5
    highs = (category == "record") | ((category == "touch") & (result["side"] == "high"))
    lows = (category == "touch") & (result["side"] == "low")
    for mask, prefix in ((highs, "HIGH"), (lows, "LOW")):
        result.loc[mask, f"{prefix}_PROB"] = p
        result.loc[mask, f"{prefix}_D1"] = d1
    result.loc[category == "record", "HIGH_D5"] = d5

    for hours, limit in HOURLY_LIMITS.items():
        change = _hourly_probability(result, 0, limit) - _hourly_probability(result, hours, limit)
        result.loc[category.isin(["direction", "threshold", "range"]), f"H{hours}"] = change
        result.loc[highs, f"HIGH_H{hours}"] = change
        result.loc[lows, f"LOW_H{hours}"] = change

    grouped = result.groupby(["datetime", *GROUP_COLUMNS], sort=False, dropna=False)
    for _, group in grouped:
        kind = group["category"].iloc[0]
        if kind not in ("threshold", "touch", "range"):
            continue
        boundary_columns = ["lower", "upper"] if kind == "range" else ["threshold"]
        boundaries = group[boundary_columns].to_numpy(dtype=float).ravel()
        boundaries = np.unique(boundaries[np.isfinite(boundaries)])
        result.loc[group.index, "group_boundary_count"] = len(boundaries)
        if not len(boundaries):
            continue
        center, scale = np.median(boundaries), np.median(abs(boundaries))
        result.loc[group.index, "group_center"] = center
        if not np.isfinite(scale) or scale <= 0:
            continue
        result.loc[group.index, "group_scale"] = scale
        if kind == "range":
            a, b = group["lower"], group["upper"]
            width = (np.arctan((b - center) / scale) - np.arctan((a - center) / scale)) / np.pi
            valid = a.notna() & b.notna() & (a < b) & width.gt(0) & group["probability"].notna()
            result.loc[group.index[valid], "RANGE_REL_WIDTH"] = width[valid]
            result.loc[group.index[valid], "RANGE_REL_DENSITY"] = group.loc[valid, "probability"] / width[valid]
            continue
        if len(boundaries) < 2:
            continue
        position_column = ("THRESHOLD_REL_POSITION" if kind == "threshold"
                           else f"{str(group['side'].iloc[0]).upper()}_REL_POSITION")
        if position_column in FACTOR_COLUMNS:
            positions = (group["threshold"] - center) / scale
            result.loc[group.index, position_column] = positions.where(np.isfinite(positions))
        if kind != "threshold":
            continue
        by_threshold = {value: rows for value, rows in group.groupby("threshold", sort=True)}
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            lower, upper = by_threshold[left], by_threshold[right]
            if len(lower) != 1 or len(upper) != 1:
                continue
            mass = lower["probability"].iloc[0] - upper["probability"].iloc[0]
            if not np.isfinite(mass) or mass < -negative_mass_tolerance:
                continue
            mass = max(0.0, mass)
            result.loc[lower.index, "THRESHOLD_ADJACENT_MASS"] = mass
            result.loc[lower.index, "THRESHOLD_REL_DENSITY"] = mass / ((right - left) / scale)
    return result


def prepare(config: dict, refresh: bool = False) -> Path:
    """Read downloaded Yes/Up histories and write auditable market-level panels."""
    data_dir, panel_dir = Path(config["data_dir"]), Path(config["panel_dir"])
    calendar_path = Path(config["calendar_path"])
    days = _calendar(pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines())))
    start, end = pd.Timestamp(config["research_start"]), pd.Timestamp(config["research_end"])
    if days.empty or days.min() > start or days.max() < end:
        raise ValueError("Calendar must cover the complete research range, including warmup")
    research_days = days[(days >= start) & (days <= end)]
    if research_days.empty:
        raise ValueError("Calendar has no trading days in the research range")
    records_path = data_dir / "markets.jsonl"
    records = [json.loads(line) for line in records_path.read_text().splitlines() if line.strip()]
    if any(record.get("category") == "direction" and record.get("time_extraction_version") != 1
           for record in records):
        raise ValueError("Direction comparison times use an outdated catalog; rerun classify before prepare")
    catalog_markets = len(records)
    last_cutoff = (research_days[-1].tz_localize("Asia/Shanghai") + pd.Timedelta(hours=15)).tz_convert("UTC")
    # Markets opening after the research period cannot contribute even metadata.
    records = [record for record in records if pd.to_datetime(record.get("opened_at") or pd.NaT, utc=True) <= last_cutoff]
    markets = pd.DataFrame(records)
    status_files, source_warnings = [], []
    for path in (data_dir / "raw/discovery.json", data_dir / "collection.json"):
        if path.is_file():
            status = json.loads(path.read_text())
            if not status.get("complete") or status.get("failed_markets"):
                raise ValueError(f"Collection is incomplete; resolve failures before prepare: {path}")
            status_files.append(path)
    files, missing_coverage = [], []
    for record in records:
        directory = Path(record["market_dir"])
        directory = directory if directory.is_absolute() else data_dir / directory
        files.append(directory / "yes_prices.csv")
        coverage_path = directory / "coverage.json"
        if coverage_path.is_file():
            coverage = json.loads(coverage_path.read_text())
            if not coverage.get("api_complete") or not coverage.get("yes", {}).get("api_complete"):
                raise ValueError(f"Market collection is incomplete; resolve failures before prepare: {coverage_path}")
            if coverage.get("yes", {}).get("token_id", str(record["token_id"])) != str(record["token_id"]):
                raise ValueError(f"Collected token differs from normalized metadata: {coverage_path}")
            identity = coverage.get("source_identity")
            if identity:
                expected = {"yes_token_id": record.get("yes_token_id") or record["token_id"],
                            **{key: record.get(key) for key in ("no_token_id", "condition_id", "opened_at", "closed_at")}}
                if any(identity.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Collection no longer matches normalized metadata: {coverage_path}")
            for name, expected in coverage.get("artifacts", {}).items():
                path = directory / name
                if not path.is_file() or {"size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} != expected:
                    raise ValueError(f"Collected file no longer matches completed coverage: {path}")
            status_files.append(coverage_path)
        elif (data_dir / "collection.json").is_file():
            raise ValueError(f"Collection has no completion metadata for market {record['market_id']}: {coverage_path}")
        else:
            missing_coverage.append(str(record["market_id"]))
    if missing_coverage:
        source_warnings.append(f"{len(missing_coverage)} markets use local CSV inputs without official collection coverage metadata")
        LOG.warning(source_warnings[-1])
    digest = hashlib.sha256(json.dumps({
        "version": VERSION, "start": str(start), "end": str(end),
        "max_quote_age": config.get("max_quote_age"),
        "negative_mass_tolerance": config.get("negative_mass_tolerance", 1e-8),
        "hourly_limits_seconds": HOURLY_LIMITS, "hourly_age_basis": "quote_available_at",
    }, sort_keys=True).encode())
    for path in [calendar_path, records_path, *status_files, *files]:
        digest.update(str(path).encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    fingerprint = digest.hexdigest()
    manifest_path = panel_dir / "manifest.json"
    if not refresh and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("fingerprint") == fingerprint and all(
                (panel_dir / name).is_file() and (panel_dir / name).stat().st_size == size
                for name, size in manifest.get("files", {}).items()
        ) and len(manifest.get("files", {})) == 2:
            LOG.info("Reuse Polymarket factor panel %s", panel_dir)
            return panel_dir
    # Include earlier calendar days for valid D1/D5 at the research boundary.
    alignment_days = days[days <= end]
    # Raw quotes can contain tens of millions of rows. Align one token at a time
    # and retain only its daily cutoffs, keeping memory bounded by one history.
    aligned_parts = []
    for index, (record, path) in enumerate(zip(records, files)):
        history = pd.read_csv(path, usecols=lambda column: column in {
            "timestamp", "price", "resolution_seconds", "source"})
        history["token_id"] = str(record["token_id"])
        aligned_parts.append(align_prices(
            markets.iloc[[index]], history, alignment_days, config.get("max_quote_age")))
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            LOG.info("Aligned %s/%s markets", index + 1, len(records))
    aligned = (pd.concat(aligned_parts, ignore_index=True).sort_values(["datetime", "market_id"])
               .reset_index(drop=True) if aligned_parts else
               align_prices(markets, pd.DataFrame(), alignment_days, config.get("max_quote_age")))
    factors = compute_factors(aligned, alignment_days, config.get("negative_mass_tolerance", 1e-8))
    aligned = aligned.loc[aligned["datetime"] >= start].reset_index(drop=True)
    factors = factors.loc[factors["datetime"] >= start].reset_index(drop=True)
    panel_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.unlink(missing_ok=True)
    sizes = {}
    for name, frame in (("aligned.parquet", aligned), ("factors.parquet", factors)):
        temporary = panel_dir / f".{name}.tmp"
        frame.to_parquet(temporary, index=False)
        temporary.replace(panel_dir / name)
        sizes[name] = (panel_dir / name).stat().st_size
    manifest = {
        "producer": "polymarket_ext.factors", "version": VERSION, "fingerprint": fingerprint,
        "research_start": str(start.date()), "research_end": str(end.date()),
        "cutoff": "15:00 Asia/Shanghai", "max_quote_age": config.get("max_quote_age"),
        "catalog_markets": catalog_markets, "markets": len(markets),
        "rows": len(factors), "valid_quotes": int(aligned["quote_valid"].sum()),
        "factor_columns": list(FACTOR_COLUMNS), "files": sizes,
        "hourly_limits_seconds": HOURLY_LIMITS, "hourly_age_basis": "quote_available_at",
        "category_rows": factors["category"].value_counts().to_dict(),
        "source_warnings": source_warnings, "markets_without_collection_metadata": missing_coverage,
        "metadata_vintage_limitation": (
            "Gamma does not supply historical contract metadata versions. Current normalized terms are "
            "assumed unchanged since metadata_available_at; timestamps exclude future-created markets "
            "but cannot prove that historical contract terms were never amended."),
    }
    coverage = {}
    for category in CATEGORY_FACTORS:
        quoted = factors.loc[(factors["category"] == category) & factors["quote_valid"]]
        coverage[category] = {
            "first_valid_date": quoted["datetime"].min().strftime("%Y-%m-%d") if len(quoted) else None,
            "last_valid_date": quoted["datetime"].max().strftime("%Y-%m-%d") if len(quoted) else None,
            "valid_rows": len(quoted), "quoted_markets": int(quoted["market_id"].nunique()),
            "trading_days_with_quotes": int(quoted["datetime"].nunique()),
        }
    manifest["category_coverage"] = coverage
    manifest["hourly_coverage"] = {
        category: {
            factor: {"valid_rows": len(valid), "markets": int(valid["market_id"].nunique()),
                     "trading_days": int(valid["datetime"].nunique())}
            for factor in HOURLY_FACTORS[category]
            for valid in [factors.loc[(factors["category"] == category) & np.isfinite(factors[factor])]]
        }
        for category in CATEGORY_FACTORS
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    LOG.info("Prepared %s market/day rows from %s markets", len(factors), len(markets))
    return panel_dir
