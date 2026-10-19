"""Monthly fixed-epoch MLP training on all available historical labels."""

from __future__ import annotations

import gc
import hashlib
import json
import resource
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


def label_cutoff(calendar: pd.DatetimeIndex, end: str | pd.Timestamp) -> pd.Timestamp:
    """Last feature date whose T+2 close is available by ``end``."""
    dates = calendar[calendar <= pd.Timestamp(end)]
    if len(dates) < 3:
        raise ValueError(f"Not enough trading dates before {end} for a T+2 label")
    return dates[-3]


def monthly_windows(config: dict, calendar: pd.DatetimeIndex) -> list[dict]:
    train_start = pd.Timestamp(config["train_start"])
    research_end = pd.Timestamp(config["research_end"])
    first_month = pd.Timestamp(config["first_test_month"])
    if first_month.day != 1 or train_start >= first_month:
        raise ValueError("first_test_month must start a month after train_start")
    result = []
    for start in pd.date_range(first_month, research_end, freq="MS"):
        available_end = start - pd.Timedelta(days=1)
        test_end = min(start + pd.offsets.MonthEnd(0), research_end)
        result.append(
            {
                "month": start.strftime("%Y-%m"),
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": label_cutoff(calendar, available_end).strftime("%Y-%m-%d"),
                "label_available_end": available_end.strftime("%Y-%m-%d"),
                "test_start": start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "evaluation_end": label_cutoff(calendar, research_end).strftime("%Y-%m-%d"),
            }
        )
    if not result:
        raise ValueError("No test months within the research period")
    return result


def assemble_month(cache_dir: Path, window: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Load the expanding training history and test rows from cached factors."""
    parts, raw_labels = [], []
    years = range(pd.Timestamp(window["train_start"]).year, pd.Timestamp(window["test_end"]).year + 1)
    for year in years:
        frame = pd.read_parquet(cache_dir / f"{year}.parquet")
        dates = frame.index.get_level_values("datetime")
        train = (dates >= window["train_start"]) & (dates <= window["train_end"])
        test = (dates >= window["test_start"]) & (dates <= window["test_end"])
        usable_label = np.isfinite(frame[("label", "LABEL0")].to_numpy())
        keep = (train & usable_label) | test
        selected = frame.loc[keep]
        selected_dates = selected.index.get_level_values("datetime")
        is_test = selected_dates >= window["test_start"]
        raw = selected.loc[is_test, ("raw_label", "LABEL0")].copy()
        raw.loc[raw.index.get_level_values("datetime") > window["evaluation_end"]] = np.nan
        raw_labels.append(raw)
        selected = selected.loc[:, ["feature", "label"]].astype("float32", copy=False)
        selected.loc[selected_dates > window["evaluation_end"], ("label", "LABEL0")] = np.nan
        parts.append(selected)
    frame = pd.concat(parts)
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Annual factor caches must have unique, sorted (datetime, instrument) rows")
    train_rows = frame.index.get_level_values("datetime") <= window["train_end"]
    test_rows = frame.index.get_level_values("datetime") >= window["test_start"]
    if not train_rows.any() or not test_rows.any():
        raise ValueError(f"Empty training or test data for {window['month']}")
    return frame, pd.concat(raw_labels).rename("raw_label")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _number(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def summarize_ic(daily: pd.DataFrame) -> dict:
    """Aggregate observations equally by trading day, never by month."""
    valid = daily["ic"].dropna()
    mean, std = valid.mean(), valid.std()
    return {
        "ic_mean": _number(mean),
        "ic_std": _number(std),
        "ic_ir": _number(mean / std) if std > 0 else None,
        "rank_ic_mean": _number(daily["rank_ic"].mean()),
        "prediction_days": len(daily),
        "evaluated_days": len(valid),
        "evaluated_samples": int(daily["n"].sum()),
    }


def _date_range(index: pd.MultiIndex) -> dict:
    dates = index.get_level_values("datetime")
    return {
        "first_date": dates.min().strftime("%Y-%m-%d"),
        "last_date": dates.max().strftime("%Y-%m-%d"),
        "samples": len(index),
        "days": dates.nunique(),
    }


def _verified_month(path: Path, fingerprint: str) -> bool:
    marker = path / "completed.json"
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    if saved.get("fingerprint") != fingerprint:
        raise ValueError(f"{path} was generated from different inputs/config; use --refresh")
    for name, expected in saved["artifacts"].items():
        artifact = path / name
        if not artifact.exists() or _sha256(artifact) != expected:
            raise ValueError(f"Incomplete or modified result {artifact}; use --refresh")
    return True


def train(config: dict, refresh: bool = False) -> dict:
    """Train a fresh Qlib MLP per month and export cross-sectional IC."""
    from .__main__ import validate_training_config

    validate_training_config(config)
    import qlib
    import torch
    from qlib.constant import REG_CN
    from qlib.contrib.eva.alpha import calc_ic
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from qlib.workflow import R
    from qlib.workflow.record_temp import SigAnaRecord, SignalRecord
    from mlflow.tracking import MlflowClient

    from .model import MLPModel
    from .prepare import CACHE_INDEX_POLICY, CACHE_INDEX_VERSION

    cache_dir = Path(config["cache_dir"])
    output_dir = Path(config["output_dir"])
    manifest_path = cache_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("fingerprint"):
        raise ValueError("Factor-cache manifest has no fingerprint; run prepare first")
    if (manifest.get("index_policy") != CACHE_INDEX_POLICY
            or manifest.get("index_version") != CACHE_INDEX_VERSION):
        raise ValueError("Factor cache must use the complete historical membership index; run prepare first")
    provider_manifest = json.loads(
        (Path(config["provider_dir"]) / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("provider_fingerprint") != provider_manifest.get("fingerprint"):
        raise ValueError("Factor cache does not match the provider; run prepare first")
    for name in ("research_start", "research_end"):
        if str(manifest[name]) != str(config[name]):
            raise ValueError(f"Factor cache {name} does not match config; run prepare first")
    calendar_path = Path(config["provider_dir"]) / "calendars" / "day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    if not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Provider calendar must contain unique, sorted trading dates")
    windows = monthly_windows(config, calendar)
    cache_files = []
    for year in range(pd.Timestamp(config["train_start"]).year, pd.Timestamp(config["research_end"]).year + 1):
        path = cache_dir / f"{year}.parquet"
        stat = path.stat()
        if year not in manifest["years"] or stat.st_size != manifest["year_files"][str(year)]["bytes"]:
            raise ValueError(f"Incomplete or modified factor cache {path}; run prepare first")
        cache_files.append((path.name, stat.st_size, stat.st_mtime_ns))
    scheme = "expanding_monthly_fixed_epochs"
    identity = {"model": "alpha158_mlp_v3", "training_scheme": scheme, "config": config, "cache": manifest,
                "files": cache_files, "calendar": _sha256(calendar_path)}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    output_dir.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri=str(Path(config["provider_dir"]).resolve()),
        region=REG_CN,
        kernels=config.get("qlib", {}).get("kernels", 1),
        expression_cache=None,
        dataset_cache=None,
    )
    tracking_uri = "sqlite:///" + str((output_dir / "mlflow.db").resolve())
    client = MlflowClient(tracking_uri=tracking_uri)
    if client.get_experiment_by_name("baseline_monthly") is None:
        client.create_experiment(
            "baseline_monthly", artifact_location=(output_dir / "artifacts").resolve().as_uri()
        )
    R.set_uri(tracking_uri)
    monthly, daily_parts, prediction_parts = [], [], []
    for window in windows:
        month_dir = output_dir / window["month"]
        if refresh or not _verified_month(month_dir, fingerprint):
            started = time.monotonic()
            stage = output_dir / f".{window['month']}-{uuid.uuid4().hex[:8]}"
            stage.mkdir()
            # Include all historical rows whose T+2 labels are known before this month.
            frame, raw_label = assemble_month(cache_dir, window)
            segments = {"train": (window["train_start"], window["train_end"]),
                        "test": (window["test_start"], window["test_end"])}
            dates = frame.index.get_level_values("datetime")
            actual = {name: _date_range(frame.index[(dates >= start) & (dates <= end)])
                      for name, (start, end) in segments.items()}
            handler = DataHandlerLP.from_df(frame)
            dataset = DatasetH(handler=handler, segments=segments)
            with R.start(
                experiment_name="baseline_monthly",
                recorder_name=window["month"],
                uri=tracking_uri,
            ):
                recorder = R.get_recorder()
                R.log_params(month=window["month"], training_scheme=scheme,
                             model_class="baseline.model.MLPModel",
                             fingerprint=fingerprint, **config["model"])
                model = MLPModel(**config["model"])
                print(f"Training MLP {window['month']}: {segments}, "
                      f"epochs={model.epochs}", flush=True)
                model.fit(dataset)
                R.save_objects(**{"model.pkl": model, "raw_label.pkl": raw_label.to_frame()})
                SignalRecord(model, dataset, recorder).generate()
                SigAnaRecord(recorder, ana_long_short=False).generate()
                pred = recorder.load_object("pred.pkl").iloc[:, 0].rename("score")
                label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
                predictions = pd.concat([pred, label.rename("label"), raw_label], axis=1)
                ic, rank_ic = calc_ic(predictions["score"], predictions["label"])
                pairs = np.isfinite(predictions[["score", "label"]]).all(axis=1)
                daily = pd.DataFrame({"ic": ic, "rank_ic": rank_ic, "n": pairs.groupby(level="datetime").sum()})
                metrics = summarize_ic(daily)
                metrics.update({"month": window["month"], "prediction_samples": len(predictions)})
                metadata = {
                    "training_scheme": scheme,
                    "model_class": "baseline.model.MLPModel",
                    "architecture": "Alpha158 + MLP",
                    "fingerprint": fingerprint,
                    "cache_fingerprint": manifest["fingerprint"],
                    "membership_source": manifest.get("membership_source"),
                    "membership_note": manifest.get("membership_note"),
                    "window": window,
                    "segments": segments,
                    "actual": actual,
                    "model": config["model"],
                    "trained_epochs": model.trained_epochs,
                    "recorder_id": recorder.id,
                    "seconds": time.monotonic() - started,
                    "process_peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                }
                model.to_pickle(stage / "model.pkl")
                torch.save(model.network.state_dict(), stage / "model.pt")
                pd.DataFrame(model.history).to_csv(stage / "training_history.csv", index=False)
                predictions.to_parquet(stage / "predictions.parquet")
                daily.to_csv(stage / "daily_ic.csv", index_label="datetime")
                _write_json(stage / "metrics.json", metrics)
                _write_json(stage / "run.json", metadata)
                _write_json(stage / "config.json", config)
                R.save_objects(**{"run.pkl": metadata})
            artifacts = {path.name: _sha256(path) for path in stage.iterdir() if path.is_file()}
            _write_json(stage / "completed.json", {"fingerprint": fingerprint, "artifacts": artifacts})
            # A failed refresh leaves the previously completed month intact.
            if month_dir.exists():
                shutil.rmtree(month_dir)
            stage.rename(month_dir)
            del frame, handler, dataset, model, predictions, daily, raw_label, label, pred
            gc.collect()
        else:
            print(f"Reusing verified results for {window['month']}", flush=True)
        monthly.append(json.loads((month_dir / "metrics.json").read_text(encoding="utf-8")))
        daily_parts.append(pd.read_csv(month_dir / "daily_ic.csv", index_col="datetime", parse_dates=True))
        prediction_parts.append(pd.read_parquet(month_dir / "predictions.parquet"))
    daily = pd.concat(daily_parts)
    predictions = pd.concat(prediction_parts)
    summary = summarize_ic(daily)
    summary.update({"training_scheme": scheme, "model_class": "baseline.model.MLPModel", "fingerprint": fingerprint,
                    "months": len(monthly), "prediction_samples": len(predictions),
                    "membership_source": manifest.get("membership_source"),
                    "membership_note": manifest.get("membership_note")})
    daily.to_csv(output_dir / "daily_ic.csv", index_label="datetime")
    pd.DataFrame(monthly).to_csv(output_dir / "monthly_ic.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet")
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary
