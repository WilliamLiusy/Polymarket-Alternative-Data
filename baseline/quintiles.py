"""Plot daily and cumulative mean standardized labels by prediction quintile."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


GROUPS = [f"Q{i}" for i in range(1, 6)]


def analyze(predictions: pd.DataFrame):
    """Assign the full prediction universe before ignoring unavailable labels."""
    if predictions.index.names != ["datetime", "instrument"]:
        raise ValueError("Expected a (datetime, instrument) MultiIndex")
    if predictions.empty or not predictions.index.is_unique:
        raise ValueError("Predictions must have nonempty, unique stock/date rows")
    if not {"score", "label"}.issubset(predictions.columns):
        raise ValueError("Predictions must contain score and standardized label")
    if not np.isfinite(predictions["score"]).all():
        raise ValueError("Every stock must have a finite prediction for ranking")

    pieces = []
    for day, frame in predictions.reset_index().groupby("datetime", sort=True):
        if len(frame) < 5:
            raise ValueError(f"Fewer than five predictions on {day}")
        frame = frame.sort_values(["score", "instrument"], ascending=[False, True]).copy()
        frame["quintile"] = np.arange(len(frame)) * 5 // len(frame) + 1
        pieces.append(frame)
    members = pd.concat(pieces).set_index(["datetime", "instrument"]).sort_index()
    # Aggregate in float64; unavailable labels never change a stock's group.
    valid_label = members["label"].astype("float64").where(np.isfinite(members["label"]))
    evaluation = members[["quintile"]].assign(valid_label=valid_label)
    grouped = evaluation.groupby([pd.Grouper(level="datetime"), "quintile"])["valid_label"]
    means = grouped.mean().unstack("quintile").reindex(columns=range(1, 6))
    counts = grouped.count().unstack("quintile").reindex(columns=range(1, 6))
    sizes = grouped.size().unstack("quintile").reindex(columns=range(1, 6))
    for frame in (means, counts, sizes):
        frame.columns = GROUPS
        frame.columns.name = "quintile"
    summary = pd.DataFrame({
        "mean_label": means.mean(),
        "std_daily_mean_label": means.std(ddof=1),
        "valid_days": means.count(),
        "assigned_stock_days": sizes.sum(),
        "valid_label_stock_days": counts.sum(),
    })
    summary.index.name = "quintile"
    return members, means, counts, sizes, summary


def plot_quintiles(values: pd.DataFrame, output: Path, *, cumulative: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    colors = ["#c23b35", "#e58c35", "#8d8d91", "#329a91", "#3569b0"]
    names = ["Q1: highest 20%", "Q2: 20-40%", "Q3: 40-60%", "Q4: 60-80%", "Q5: lowest 20%"]
    fig, ax = plt.subplots(figsize=(14, 7), dpi=180)
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")
    for column, color, name in zip(GROUPS, colors, names):
        ax.plot(values.index, values[column], label=name, color=color, linewidth=1.6 if cumulative else 1.25, alpha=.88)
    ax.axhline(0, color="#353e4c", linewidth=.9, alpha=.7)
    ax.grid(axis="y", alpha=.18)
    ax.grid(axis="x", alpha=.07)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c8cdd5")
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_xlim(values.index.min(), values.index.max())
    ax.set_xlabel("Prediction date (T)", labelpad=10)
    ylabel = "Cumulative sum of daily mean returns (z-score units)" if cumulative else "Mean realized return (cross-sectional z-score)"
    ax.set_ylabel(ylabel, labelpad=10)
    period = "cumulative" if cumulative else "daily"
    fig.suptitle(f"Baseline: {period} realized returns by prediction quintile", fontsize=16, x=.08, ha="left", y=.96)
    evaluated = int(values.notna().any(axis=1).sum())
    fig.text(.08, .91, f"Rank daily by score; Q1 is highest. Equal-weighted valid labels. {evaluated} evaluated trading days.",
             fontsize=10, color="#596375")
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.015), ncol=5, frameon=False, fontsize=10)
    note = (
        "Target: cross-sectional z-score of T+1 to T+2 return. Arithmetic sum through each date, including that date; not a percentage return."
        if cumulative else
        "Target: close(T+2) / close(T+1) - 1, standardized across stocks on T. Daily means; no accumulation or smoothing."
    )
    fig.text(.08, .045, note,
             fontsize=9, color="#596375")
    missing = values.index[values.isna().all(axis=1)]
    if len(missing):
        dates = ", ".join(day.strftime("%m-%d") for day in missing)
        fig.text(.08, .018, f"Unavailable labels: {dates}; kept as missing rather than zero.", fontsize=9, color="#596375")
    fig.subplots_adjust(left=.08, right=.98, bottom=.15, top=.82)
    fig.savefig(output / f"{period}_quintile_means.png", dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(output / f"{period}_quintile_means.svg", facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/baseline/predictions.parquet"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baseline/quintiles"))
    args = parser.parse_args()
    predictions = pd.read_parquet(args.predictions)
    members, means, counts, sizes, summary = analyze(predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    means.to_csv(args.output_dir / "daily_quintile_means.csv", index_label="datetime")
    counts.to_csv(args.output_dir / "daily_quintile_valid_counts.csv", index_label="datetime")
    sizes.to_csv(args.output_dir / "daily_quintile_sizes.csv", index_label="datetime")
    summary.to_csv(args.output_dir / "quintile_summary.csv")
    members.to_parquet(args.output_dir / "quintile_membership.parquet")
    plot_quintiles(means, args.output_dir)
    # Include the current date in the arithmetic sum. Missing dates stay NaN;
    # a later valid date resumes accumulation over observed group means.
    cumulative = means.cumsum(skipna=True)
    cumulative.to_csv(args.output_dir / "cumulative_quintile_means.csv", index_label="datetime")
    plot_quintiles(cumulative, args.output_dir, cumulative=True)
    metadata = {
        "source": str(args.predictions.resolve()),
        "source_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "score_column": "score", "target_column": "label",
        "target": "T+1 to T+2 return, cross-sectionally z-scored on prediction date T",
        "ranking": "descending score; ascending instrument for ties; group before filtering labels",
        "groups": "Q1 highest 20%, Q5 lowest 20%; equal-weighted valid labels within each day/group",
        "date_axis": "prediction date T", "cumulative": False, "smoothed": False,
        "cumulative_plot": {
            "file": "cumulative_quintile_means.png",
            "data": "cumulative_quintile_means.csv",
            "method": "arithmetic sum of daily group mean labels from first date through current date inclusive",
            "missing": "missing dates remain NaN; later valid dates resume the sum of observed values",
            "units": "sum of cross-sectional z-scores; not percentage returns or compounded wealth",
        },
        "prediction_days": len(means), "evaluated_days": int(means.notna().any(axis=1).sum()),
        "prediction_rows": len(members), "missing_target_rows": int((~np.isfinite(members["label"])).sum()),
        "group_size_min": int(sizes.min().min()), "group_size_max": int(sizes.max().max()),
        "mean_daily_q1_minus_q5": float((means["Q1"] - means["Q5"]).mean()),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(summary.to_string())
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
