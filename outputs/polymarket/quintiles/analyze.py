"""Reproduce prediction-quintile plots from saved predictions; no model fitting."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from baseline.quintiles import GROUPS, analyze

OUTPUT = Path(__file__).resolve().parent
COLORS = ["#c23b35", "#e58c35", "#85858b", "#329a91", "#3569b0"]
NAMES = ["Q1: highest 20%", "Q2: 20-40%", "Q3: 40-60%", "Q4: 60-80%", "Q5: lowest 20%"]


def plot(values, directory, model_name, cumulative=False):
    period = "cumulative" if cumulative else "daily"
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")
    for column, color, name in zip(GROUPS, COLORS, NAMES):
        ax.plot(values.index, values[column], label=name, color=color,
                linewidth=1.9 if cumulative else 1.2, alpha=.94 if cumulative else .82)
        if cumulative:
            valid = values[column].dropna()
            ax.scatter(valid.index[-1], valid.iloc[-1], color=color, s=22, zorder=3)
    ax.axhline(0, color="#353e4c", linewidth=.8, alpha=.65)
    ax.grid(axis="y", alpha=.18)
    ax.grid(axis="x", alpha=.07)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8cdd5")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(values.index.min(), values.index.max())
    ax.set_xlabel("Prediction date (T)", labelpad=10)
    ax.set_ylabel("Cumulative sum of group means (z-score units)" if cumulative
                  else "Group mean realized return (cross-sectional z-score)", labelpad=10)
    fig.suptitle(f"{model_name}: {period} returns by prediction quintile",
                 fontsize=16, x=.08, ha="left", y=.96)
    fig.text(.08, .91, "Q1 = highest predicted return. 60 stocks per group, re-ranked daily; 123 evaluated dates.",
             fontsize=10, color="#596375")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=5, frameon=False, fontsize=10)
    note = ("Target: cross-sectional z-score of close(T+2) / close(T+1) - 1. "
            + ("Arithmetic sum through each date, including that date." if cumulative
               else "Equal-weighted mean of available labels within each group."))
    fig.text(.08, .045, note, fontsize=9, color="#596375")
    fig.text(.08, .018, "Labels unavailable on 2026-09-29 and 2026-09-30; left missing. "
             + ("The vertical axis is not a percentage return or compounded wealth." if cumulative
                else "All 300 predictions are ranked before unavailable labels are excluded from the mean."),
             fontsize=9, color="#596375")
    fig.subplots_adjust(left=.08, right=.98, bottom=.15, top=.82)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(directory / f"{period}_quintile_means.{extension}", facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    reports, summaries, reference = {}, [], None
    for key, title, directory in (
        ("polymarket", "Polymarket extension", OUTPUT),
        ("baseline", "Baseline", OUTPUT / "baseline_reference"),
    ):
        source = ROOT / "outputs" / key / "predictions.parquet"
        predictions = pd.read_parquet(source)
        if reference is None:
            reference = predictions[["label", "raw_label"]]
        else:
            pd.testing.assert_frame_equal(reference, predictions[["label", "raw_label"]], check_exact=True)
        members, daily, counts, sizes, summary = analyze(predictions)
        cumulative = daily.cumsum(skipna=True)
        assert (sizes == 60).all().all()
        assert len(daily) == 125 and daily.notna().any(axis=1).sum() == 123
        assert daily.loc["2026-09-29":].isna().all().all()
        assert cumulative.loc["2026-09-29":].isna().all().all()
        np.testing.assert_allclose((daily * counts).sum(axis=1) / counts.sum(axis=1),
                                   predictions.label.groupby(level="datetime").mean(),
                                   rtol=0, atol=1e-7, equal_nan=True)
        summary["cumulative_final"] = cumulative.dropna(how="all").iloc[-1]
        summary["model"] = key
        summaries.append(summary.reset_index())
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in (
            ("daily_quintile_means", daily), ("cumulative_quintile_means", cumulative),
            ("daily_quintile_valid_counts", counts), ("daily_quintile_sizes", sizes),
        ):
            frame.to_csv(directory / f"{name}.csv", index_label="datetime")
        summary.to_csv(directory / "quintile_summary.csv")
        members.to_parquet(directory / "quintile_membership.parquet")
        plot(daily, directory, title)
        plot(cumulative, directory, title, cumulative=True)
        reports[key] = {
            "source": str(source.relative_to(ROOT)),
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "prediction_rows": len(predictions), "prediction_days": len(daily),
            "evaluated_days": int(daily.notna().any(axis=1).sum()),
            "last_evaluated_date": str(daily.dropna(how="all").index[-1].date()),
            "assigned_stocks_per_group": 60,
            "missing_label_stock_days": int(predictions.label.isna().sum()),
            "daily_mean_by_group": daily.mean().to_dict(),
            "cumulative_final_by_group": summary.cumulative_final.to_dict(),
            "daily_mean_q1_minus_q5": float((daily.Q1 - daily.Q5).mean()),
            "cumulative_q1_minus_q5": float((daily.Q1 - daily.Q5).sum()),
        }
    pd.concat(summaries, ignore_index=True).to_csv(OUTPUT / "model_comparison.csv", index=False)
    metadata = {
        "calculation": "baseline.quintiles.analyze",
        "ranking": "Daily score descending, instrument ascending for ties; full prediction universe before label filtering",
        "groups": "Q1 highest 20%, Q5 lowest 20%; equal weight over valid group labels",
        "target": "Saved label: cross-sectional z-score of close(T+2)/close(T+1)-1 on prediction date T",
        "date_axis": "prediction date T",
        "cumulative": "Arithmetic sum of group daily means, including current date; missing dates stay NaN",
        "models": reports,
    }
    (OUTPUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(reports, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
