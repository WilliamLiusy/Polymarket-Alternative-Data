"""Audit completed monthly results without loading any expanding training set.

Run after training:
    .venv/bin/python outputs/polymarket/audit_run.py
Inspect completed months while training:
    .venv/bin/python outputs/polymarket/audit_run.py --allow-partial --output outputs/polymarket/run_audit_partial.json

Only the requested audit JSON is written. Model replay reads one test month at
a time using Parquet date filters and the saved Qlib model's public predict().
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
EXPECTED_ENDS = {
    "2026-04": "2026-03-27", "2026-05": "2026-04-28", "2026-06": "2026-05-27",
    "2026-07": "2026-06-26", "2026-08": "2026-07-29", "2026-09": "2026-08-27",
}
ARTIFACTS = {
    "model.pkl", "sensitivities.csv", "predictions.parquet", "daily_ic.csv",
    "training_history.csv", "metrics.json", "run.json", "config.json",
}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def equal_values(left, right):
    return left.shape == right.shape and np.array_equal(left, right, equal_nan=True)


def recompute_daily(predictions):
    rows = []
    for date, day in predictions.groupby(level="datetime", sort=True):
        valid = np.isfinite(day[["score", "label"]]).all(axis=1)
        pairs = day.loc[valid]
        rows.append({"datetime": date, "ic": pairs["score"].corr(pairs["label"]),
                     "rank_ic": pairs["score"].corr(pairs["label"], method="spearman"),
                     "n": int(valid.sum())})
    return pd.DataFrame(rows).set_index("datetime")


def summarize(daily):
    valid = daily["ic"].dropna()
    mean, std = valid.mean(), valid.std()
    number = lambda value: float(value) if pd.notna(value) and np.isfinite(value) else None
    return {"ic_mean": number(mean), "ic_std": number(std),
            "ic_ir": number(mean / std) if std > 0 else None,
            "rank_ic_mean": number(daily["rank_ic"].mean()),
            "prediction_days": len(daily), "evaluated_days": len(valid),
            "evaluated_samples": int(daily["n"].sum())}


class Audit:
    def __init__(self):
        self.checks = []

    def check(self, name, passed, **details):
        self.checks.append({"check": name, "passed": bool(passed), **details})

    def metrics(self, name, saved, expected):
        mismatches = {}
        for key, value in expected.items():
            actual = saved.get(key)
            matches = actual is None if value is None else (
                actual is not None and np.isclose(actual, value, rtol=1e-10, atol=1e-12))
            if not matches:
                mismatches[key] = {"saved": actual, "recomputed": value}
        self.check(name, not mismatches, mismatches=mismatches)

    def daily(self, name, saved, expected):
        self.check(name + ": dates", saved.index.equals(expected.index))
        if saved.index.equals(expected.index):
            for column in ("ic", "rank_ic", "n"):
                left, right = saved[column].to_numpy(), expected[column].to_numpy()
                self.check(name + ": " + column,
                           np.allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=True))


def replay_month(audit, directory, predictions, cache_dir, window, model_config):
    import torch
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP
    from polymarket_ext.model import CATEGORIES, INPUT_SCHEMA_VERSION, PolymarketModel

    name = directory.name
    torch.set_num_threads(model_config["num_threads"])
    model = PolymarketModel.load(directory / "model.pkl")
    audit.check(name + ": loaded model schema", model.input_schema_version == INPUT_SCHEMA_VERSION)
    audit.check(name + ": loaded model trained epochs", model.trained_epochs == 10)
    audit.check(name + ": finite model parameters",
                all(bool(torch.isfinite(value).all()) for value in model.network.parameters()))
    neutral = model.network.sensitivities.weight.detach().cpu().numpy()[0]
    audit.check(name + ": unseen-stock sensitivity row is exactly one", np.array_equal(neutral, np.ones(5)))
    sensitivity = pd.read_csv(directory / "sensitivities.csv", index_col="instrument")
    loaded_sensitivity = model.sensitivity_frame()
    same_sensitivity = (sensitivity.index.equals(loaded_sensitivity.index)
                        and list(sensitivity.columns) == list(CATEGORIES)
                        and np.allclose(sensitivity, loaded_sensitivity, rtol=1e-7, atol=1e-8))
    audit.check(name + ": sensitivity CSV equals loaded model", same_sensitivity)
    for category, rows in model.training_coverage["category_active_rows"].items():
        if rows == 0:
            audit.check(name + ": unobserved category stays neutral: " + category,
                        np.array_equal(loaded_sensitivity[category], np.ones(len(loaded_sensitivity))))

    # Reading just the test dates avoids the 1.3-million-row expanding history.
    cache = pd.read_parquet(cache_dir / "2026.parquet", filters=[
        ("datetime", ">=", pd.Timestamp(window["test_start"])),
        ("datetime", "<=", pd.Timestamp(window["test_end"])),
    ])
    audit.check(name + ": prediction index equals filtered Alpha158 test rows", cache.index.equals(predictions.index))
    for group, column in (("label", "label"), ("raw_label", "raw_label")):
        expected = cache[(group, "LABEL0")].copy()
        expected.loc[expected.index.get_level_values("datetime") > window["evaluation_end"]] = np.nan
        audit.check(name + ": " + column + " equals cached label including NaN",
                    cache.index.equals(predictions.index)
                    and equal_values(predictions[column].to_numpy(), expected.to_numpy()))
    features = cache.loc[:, ["feature"]].astype("float32", copy=False)
    dataset = DatasetH(handler=DataHandlerLP.from_df(features), segments={
        "test": (window["test_start"], window["test_end"]),
    })
    replay = model.predict(dataset)
    expected = predictions["score"]
    max_difference = float(np.max(np.abs(replay.to_numpy() - expected.to_numpy())))
    audit.check(name + ": reloaded model prediction replay",
                replay.index.equals(expected.index)
                and np.allclose(replay, expected, rtol=1e-6, atol=1e-7),
                samples=len(replay), max_abs_difference=max_difference,
                bitwise_equal=equal_values(replay.to_numpy(), expected.to_numpy()),
                feature_rows_loaded=len(cache))
    stocks = set(predictions.index.get_level_values("instrument").astype(str))
    unseen = sorted(stocks - set(model.stock_ids))
    # Also exercise the mapping when every real test stock has appeared before.
    probe = cache["feature"].iloc[[0]].copy()
    probe.index = pd.MultiIndex.from_tuples(
        [(probe.index[0][0], "__AUDIT_UNSEEN_STOCK__")], names=probe.index.names)
    ids = model._arrays(probe)[1]
    audit.check(name + ": unknown stock maps to neutral embedding", ids.tolist() == [0],
                actual_unseen_test_stocks=unseen)
    detail = {"month": name, "test_samples": len(predictions),
              "training_stocks": len(model.stock_ids), "unseen_test_stocks": unseen,
              "replay_max_abs_difference": max_difference,
              "replay_bitwise_equal": equal_values(replay.to_numpy(), expected.to_numpy())}
    del model, cache, features, dataset, replay, probe
    gc.collect()
    return detail


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--no-replay", action="store_true", help="Skip model loading and test-cache replay")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/polymarket/run_audit.json")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "configs/polymarket.yaml").read_text())
    baseline = yaml.safe_load((ROOT / config["baseline_config"]).read_text())
    output, baseline_output = ROOT / config["output_dir"], ROOT / baseline["output_dir"]
    cache_dir = ROOT / baseline["cache_dir"]
    model_config = {**baseline["model"], **config.get("model", {})}
    calendar_path = ROOT / baseline["provider_dir"] / "calendars/day.txt"
    calendar = pd.DatetimeIndex(pd.to_datetime(calendar_path.read_text().splitlines()))
    months = list(EXPECTED_ENDS)
    completed = [month for month in months if (output / month / "completed.json").is_file()]
    missing = [month for month in months if month not in completed]
    audit, details, fingerprints, parts, daily_parts = Audit(), [], [], [], []
    audit.check("factor panel uses availability-safe version 3",
                read_json(ROOT / config["panel_dir"] / "manifest.json")["version"] == 3)
    audit.check("configured fixed 10 epochs", model_config["epochs"] == 10)
    if not args.allow_partial:
        audit.check("six completed months", completed == months, missing_months=missing)
    if not args.no_replay and completed:
        import qlib
        qlib.init(provider_uri=str(ROOT / baseline["provider_dir"]), region="cn", kernels=1,
                  expression_cache=None, dataset_cache=None)

    for month in completed:
        directory = output / month
        print("Auditing " + month, flush=True)
        try:
            marker = read_json(directory / "completed.json")
            absent = [name for name in ARTIFACTS if not (directory / name).is_file()]
            audit.check(month + ": required artifacts present", not absent, missing=absent)
            if absent:
                continue
            bad_hashes = [name for name, digest in marker["artifacts"].items()
                          if not (directory / name).is_file() or sha256(directory / name) != digest]
            audit.check(month + ": completion checksums", not bad_hashes, mismatched=bad_hashes)
            audit.check(month + ": completion marker covers every artifact", ARTIFACTS <= set(marker["artifacts"]))
            run, metrics = read_json(directory / "run.json"), read_json(directory / "metrics.json")
            window = run["window"]
            fingerprints.append(run["fingerprint"])
            audit.check(month + ": marker/run fingerprints agree", marker["fingerprint"] == run["fingerprint"])
            test_start = pd.Timestamp(month + "-01")
            known = calendar[calendar < test_start]
            evaluation_end = calendar[calendar <= pd.Timestamp("2026-09-30")][-3]
            expected_window = {
                "month": month, "train_start": "2008-01-01", "train_end": EXPECTED_ENDS[month],
                "label_available_end": (test_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": (test_start + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d"),
                "evaluation_end": "2026-09-28",
            }
            audit.check(month + ": expanding T+2 window", window == expected_window
                        and known[-3].strftime("%Y-%m-%d") == EXPECTED_ENDS[month]
                        and evaluation_end.strftime("%Y-%m-%d") == "2026-09-28", window=window)
            expected_segments = {"train": ["2008-01-01", EXPECTED_ENDS[month]],
                                 "test": [expected_window["test_start"], expected_window["test_end"]]}
            audit.check(month + ": train/test only with no validation", run["segments"] == expected_segments)
            audit.check(month + ": fixed-epoch model configuration", run["trained_epochs"] == 10
                        and run["model"] == model_config
                        and run["training_scheme"] == "expanding_monthly_fixed_epochs")
            history = pd.read_csv(directory / "training_history.csv")
            audit.check(month + ": ten finite training epochs", history["epoch"].tolist() == list(range(1, 11))
                        and bool(np.isfinite(history["train_mse"]).all()))
            predictions = pd.read_parquet(directory / "predictions.parquet")
            reference = pd.read_parquet(baseline_output / month / "predictions.parquet")
            audit.check(month + ": unique sorted prediction index", predictions.index.is_unique
                        and predictions.index.is_monotonic_increasing)
            audit.check(month + ": baseline prediction index identical", predictions.index.equals(reference.index))
            audit.check(month + ": all predictions finite", bool(np.isfinite(predictions["score"]).all()))
            for column in ("label", "raw_label"):
                audit.check(month + ": " + column + " exactly equals baseline including NaN",
                            predictions.index.equals(reference.index)
                            and equal_values(predictions[column].to_numpy(), reference[column].to_numpy()))
            after_end = predictions.index.get_level_values("datetime") > "2026-09-28"
            audit.check(month + ": beyond-research T+2 labels missing",
                        bool(predictions.loc[after_end, ["label", "raw_label"]].isna().all().all()))
            sensitivity = pd.read_csv(directory / "sensitivities.csv", index_col="instrument")
            audit.check(month + ": finite five-category sensitivities", sensitivity.shape[1] == 5
                        and sensitivity.index.is_unique and bool(np.isfinite(sensitivity.to_numpy()).all()))
            daily = recompute_daily(predictions)
            stored_daily = pd.read_csv(directory / "daily_ic.csv", index_col="datetime", parse_dates=True)
            audit.daily(month + ": independently recomputed daily IC", stored_daily, daily)
            audit.metrics(month + ": monthly metrics", metrics, {**summarize(daily),
                          "prediction_samples": len(predictions)})
            detail = replay_month(audit, directory, predictions, cache_dir, window, model_config) \
                if not args.no_replay else {"month": month, "test_samples": len(predictions), "replay": "skipped"}
            detail.update({"train_end": window["train_end"], "seconds": run["seconds"],
                           "ic_mean": metrics["ic_mean"], "rank_ic_mean": metrics["rank_ic_mean"]})
            details.append(detail)
            parts.append(predictions)
            daily_parts.append(daily)
            del reference, sensitivity
            gc.collect()
        except Exception as error:
            audit.check(month + ": audit completed without exception", False, error=repr(error))

    if fingerprints:
        audit.check("all completed months share one fingerprint", len(set(fingerprints)) == 1)
    complete = completed == months
    if complete and len(parts) == 6:
        predictions, daily = pd.concat(parts), pd.concat(daily_parts)
        audit.check("exactly 37500 predictions", len(predictions) == 37500, samples=len(predictions))
        audit.check("125 prediction days and 123 evaluated days",
                    len(daily) == 125 and int(daily["ic"].notna().sum()) == 123)
        overall = pd.read_parquet(output / "predictions.parquet")
        audit.check("aggregate predictions equal monthly concatenation", overall.equals(predictions))
        audit.daily("aggregate daily IC", pd.read_csv(output / "daily_ic.csv", index_col="datetime", parse_dates=True), daily)
        summary = read_json(output / "summary.json")
        audit.metrics("aggregate summary metrics", summary, {**summarize(daily), "months": 6,
                      "prediction_samples": 37500})
        audit.check("aggregate fingerprint equals monthly fingerprints", summary["fingerprint"] == fingerprints[0])
        monthly = pd.read_csv(output / "monthly_ic.csv")
        audit.check("aggregate monthly table has six ordered months", monthly["month"].tolist() == months)
        for _, row in monthly.iterrows():
            expected = read_json(output / row["month"] / "metrics.json")
            audit.metrics("aggregate monthly metrics: " + row["month"], row.to_dict(),
                          {key: value for key, value in expected.items() if key != "month"})
    failures = [item for item in audit.checks if not item["passed"]]
    status = "failed" if failures else ("passed" if complete else "partial_passed")
    report = {"audited_at": datetime.now(timezone.utc).isoformat(), "status": status,
              "complete": complete, "model_replay": not args.no_replay,
              "completed_months": completed, "missing_months": missing,
              "fingerprint": fingerprints[0] if fingerprints and len(set(fingerprints)) == 1 else None,
              "audited_prediction_samples": sum(len(part) for part in parts),
              "checks_passed": len(audit.checks) - len(failures), "checks_failed": len(failures),
              "months": details, "failures": failures, "checks": audit.checks,
              "memory_policy": "Only filtered monthly test Alpha158 rows and one saved model are loaded; no training set is assembled."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "checks_passed", "checks_failed",
                     "audited_prediction_samples", "missing_months")}, ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
