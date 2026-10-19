"""Monthly Qlib evaluation using the existing historical CSI 300 Alpha158 cache."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from baseline.__main__ import load_config as load_baseline_config
from baseline.train import (
    _sha256, _verified_month, _write_json, assemble_month,
    monthly_windows, summarize_ic,
)

from .__main__ import validate_training_config


def evaluate(prediction: pd.Series, label: pd.Series) -> pd.DataFrame:
    from qlib.contrib.eva.alpha import calc_ic

    paired = pd.concat([prediction.rename("score"), label.rename("label")], axis=1)
    ic, rank_ic = calc_ic(paired["score"], paired["label"])
    count = np.isfinite(paired).all(axis=1).groupby(level="datetime").sum()
    return pd.DataFrame({"ic": ic, "rank_ic": rank_ic, "n": count})


def train(config: dict, refresh: bool = False) -> dict:
    """Fit independent monthly models, exporting sensitivities and daily IC."""
    import qlib
    from qlib.constant import REG_CN
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    import baseline.model as shared_mlp
    from . import model as extension_mlp
    from .factors import VERSION as FACTOR_VERSION
    from .model import PolymarketModel, REBUILD_INSTRUCTIONS

    validate_training_config(config)
    baseline = load_baseline_config(config["baseline_config"])
    model_config = {**baseline["model"], **config.get("model", {})}
    cache_dir = Path(baseline["cache_dir"])
    output_dir = Path(config["output_dir"])
    panel_path = Path(config["panel_dir"]) / "factors.parquet"
    panel = pd.read_parquet(panel_path)
    panel_manifest = json.loads((Path(config["panel_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    if panel_manifest.get("version") != FACTOR_VERSION:
        raise ValueError("Incompatible Polymarket factor panel version. " + REBUILD_INSTRUCTIONS)
    manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
    provider_manifest = json.loads((Path(baseline["provider_dir"]) / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("fingerprint") or manifest.get("provider_fingerprint") != provider_manifest.get("fingerprint"):
        raise ValueError("Alpha158 cache does not match its provider; run baseline prepare first")
    for key in ("research_start", "research_end"):
        if any(str(value) != str(baseline[key]) for value in (manifest[key], config[key], panel_manifest[key])):
            raise ValueError(f"{key} must agree in baseline config, extension config and both factor manifests")
    calendar_path = Path(baseline["provider_dir"]) / "calendars" / "day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    if calendar.hasnans or not calendar.is_unique or not calendar.is_monotonic_increasing:
        raise ValueError("Provider calendar must contain unique, sorted trading dates")
    cache_files = []
    for year in range(pd.Timestamp(baseline["train_start"]).year, pd.Timestamp(baseline["research_end"]).year + 1):
        path = cache_dir / f"{year}.parquet"
        stat = path.stat()
        if year not in manifest["years"] or stat.st_size != manifest["year_files"][str(year)]["bytes"]:
            raise ValueError(f"Incomplete or modified Alpha158 cache {path}; run baseline prepare first")
        cache_files.append((path.name, stat.st_size, stat.st_mtime_ns))
    identity = {"config": config, "baseline": baseline, "cache": manifest["fingerprint"],
                "cache_files": cache_files, "panel": _sha256(panel_path), "calendar": _sha256(calendar_path),
                "model": "joint_gated_mlp_v5", "model_config": model_config,
                "input_schema_version": extension_mlp.INPUT_SCHEMA_VERSION,
                "training_scheme": "expanding_monthly_fixed_epochs",
                "model_sources": {"shared": _sha256(Path(shared_mlp.__file__)),
                                  "extension": _sha256(Path(extension_mlp.__file__))}}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    qlib.init(provider_uri=baseline["provider_dir"], region=REG_CN,
              kernels=baseline.get("qlib", {}).get("kernels", 1),
              expression_cache=None, dataset_cache=None)
    output_dir.mkdir(parents=True, exist_ok=True)
    monthly, daily_parts, prediction_parts = [], [], []
    for window in monthly_windows(baseline, calendar):
        month_dir = output_dir / window["month"]
        if refresh or not _verified_month(month_dir, fingerprint):
            started = time.monotonic()
            stage = output_dir / f".{window['month']}-{uuid.uuid4().hex[:8]}"
            stage.mkdir()
            frame, raw_label = assemble_month(cache_dir, window)
            segments = {"train": (window["train_start"], window["train_end"]),
                        "test": (window["test_start"], window["test_end"])}
            dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
            print(f"Training Polymarket {window['month']}: {segments}", flush=True)
            model = PolymarketModel(panel, **model_config)
            model.fit(dataset)
            prediction = model.predict(dataset)
            label = dataset.prepare("test", col_set="label", data_key=DataHandlerLP.DK_L).iloc[:, 0]
            predictions = pd.concat([prediction, label.rename("label"), raw_label], axis=1)
            daily = evaluate(prediction, label)
            metrics = {**summarize_ic(daily), "month": window["month"], "prediction_samples": len(predictions)}
            model.to_pickle(stage / "model.pkl")
            model.sensitivity_frame().to_csv(stage / "sensitivities.csv")
            predictions.to_parquet(stage / "predictions.parquet")
            daily.to_csv(stage / "daily_ic.csv", index_label="datetime")
            pd.DataFrame(model.history).to_csv(stage / "training_history.csv", index=False)
            _write_json(stage / "metrics.json", metrics)
            _write_json(stage / "run.json", {
                "window": window, "segments": segments, "trained_epochs": model.trained_epochs,
                "model": model_config, "training_scheme": "expanding_monthly_fixed_epochs",
                "input_schema_version": model.input_schema_version,
                "fingerprint": fingerprint, "seconds": time.monotonic() - started,
                "architecture": "Alpha158 + five stock gates + category market-set encoders + MLP",
                "pooling": "mean embeddings of all active market rows per category, plus log(1 + count) and category missing flag",
                "normalization": "finite values in train segment only; unseen stock sensitivities = 1",
                "missing_semantics": {
                    "factor": "1 for non-finite values; 0 for finite values, including zero",
                    "category": "1 if no real market row has any finite factor; 0 otherwise",
                    "padding": "excluded from pooling, counts and category availability",
                    "values": "missing values filled with 0 after normalization; only values scaled by stock sensitivities",
                },
                "training_polymarket_coverage": model.training_coverage,
                "label": "cross-sectional zscore of close(T+2)/close(T+1)-1",
                "membership_source": manifest.get("membership_source"),
                "membership_note": manifest.get("membership_note"),
            })
            _write_json(stage / "config.json", config)
            artifacts = {path.name: _sha256(path) for path in stage.iterdir() if path.is_file()}
            _write_json(stage / "completed.json", {"fingerprint": fingerprint, "artifacts": artifacts})
            if month_dir.exists():
                shutil.rmtree(month_dir)
            stage.rename(month_dir)
            del frame, dataset, model, predictions, daily
            gc.collect()
        else:
            print(f"Reusing verified Polymarket results for {window['month']}", flush=True)
        monthly.append(json.loads((month_dir / "metrics.json").read_text(encoding="utf-8")))
        daily_parts.append(pd.read_csv(month_dir / "daily_ic.csv", index_col="datetime", parse_dates=True))
        prediction_parts.append(pd.read_parquet(month_dir / "predictions.parquet"))
    daily, predictions = pd.concat(daily_parts), pd.concat(prediction_parts)
    summary = {**summarize_ic(daily), "training_scheme": "expanding_monthly_fixed_epochs",
               "fingerprint": fingerprint, "months": len(monthly), "prediction_samples": len(predictions),
               "membership_source": manifest.get("membership_source"),
               "membership_note": manifest.get("membership_note")}
    daily.to_csv(output_dir / "daily_ic.csv", index_label="datetime")
    predictions.to_parquet(output_dir / "predictions.parquet")
    pd.DataFrame(monthly).to_csv(output_dir / "monthly_ic.csv", index=False)
    _write_json(output_dir / "summary.json", summary)
    return summary
