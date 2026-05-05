"""Cross-model comparison.

Pick the best basin (by per-basin NSE on the test holdout) for each anchor
model in {TFT, Vanilla, iTransformer}, then overlay all four models'
predictions (TFT, LSTM baseline, Vanilla transformer, iTransformer) for those
basins. Also produces a grouped-bar NSE comparison and a per-basin metrics CSV.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from visualize_tft_predictions import (  # noqa: E402
    PLOT_FONT_SIZE,
    PLOT_LEGEND_FONT_SIZE,
    apply_plot_style,
)

DEFAULT_TFT = PROJECT_DIR / "outputs" / "simple_tft" / "test_tft_predictions.csv"
DEFAULT_LSTM = PROJECT_DIR / "outputs" / "lstm_baseline" / "test_tft_predictions.csv"
DEFAULT_VANILLA = (
    PROJECT_DIR.parent
    / "CAMELS_data_load-xs-gpu_run"
    / "model_artifacts" / "raven_default" / "figures"
    / "test_vanilla_predictions.csv"
)
DEFAULT_ITRANSFORMER = (
    PROJECT_DIR.parent
    / "CAMELS_data_load-itransformer"
    / "model_artifacts" / "itransformer_default" / "figures"
    / "test_itransformer_predictions.csv"
)

ANCHOR_MODELS: Tuple[str, ...] = ("TFT", "Vanilla", "iTransformer")
ALL_MODELS: Tuple[str, ...] = ("TFT", "LSTM", "Vanilla", "iTransformer")
MODEL_COLORS: Dict[str, str] = {
    "Actual": "#1f1f1f",
    "TFT": "#4C78A8",
    "LSTM": "#54A24B",
    "Vanilla": "#E45756",
    "iTransformer": "#9D6FB1",
}


def load_predictions(name: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} predictions CSV not found: {path}")
    df = pd.read_csv(path, usecols=["basin_id", "target_date", "actual", "predicted"])
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    df["target_date"] = pd.to_datetime(df["target_date"])
    return df.rename(columns={"predicted": f"pred_{name}"})


def join_predictions(paths: Dict[str, Path]) -> pd.DataFrame:
    frames = {name: load_predictions(name, path) for name, path in paths.items()}
    anchor_name = next(iter(frames))
    base = frames[anchor_name][["basin_id", "target_date", "actual"]].copy()
    for name, df in frames.items():
        base = base.merge(
            df[["basin_id", "target_date", f"pred_{name}"]],
            on=["basin_id", "target_date"],
            how="inner",
        )
    return base.sort_values(["basin_id", "target_date"]).reset_index(drop=True)


def per_basin_metrics(
    joined: pd.DataFrame,
    model_names: Sequence[str],
    min_samples: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for basin_id, sub in joined.groupby("basin_id"):
        if len(sub) < min_samples:
            continue
        y = sub["actual"].to_numpy()
        sst = float(np.sum((y - float(y.mean())) ** 2))
        if sst <= 1e-12:
            continue
        row: Dict[str, object] = {"basin_id": basin_id, "n_samples": int(len(sub))}
        for name in model_names:
            pred = sub[f"pred_{name}"].to_numpy()
            diff = pred - y
            sse = float(np.sum(diff * diff))
            row[f"nse_{name}"] = 1.0 - sse / sst
            row[f"rmse_{name}"] = float(np.sqrt(sse / len(sub)))
            row[f"mae_{name}"] = float(np.mean(np.abs(diff)))
        rows.append(row)
    return pd.DataFrame(rows)


def select_best_basins(
    metrics: pd.DataFrame,
    anchors: Sequence[str],
    n_each: int,
) -> List[Tuple[str, str]]:
    seen: set = set()
    selected: List[Tuple[str, str]] = []
    for anchor in anchors:
        col = f"nse_{anchor}"
        finite = metrics[np.isfinite(metrics[col])]
        ranked = finite[["basin_id", col]].sort_values(col, ascending=False).head(n_each)
        for basin_id in ranked["basin_id"]:
            key = str(basin_id)
            if key in seen:
                continue
            seen.add(key)
            selected.append((anchor, key))
    return selected


def plot_basin_overlay(
    output_path: Path,
    joined: pd.DataFrame,
    metrics_row: pd.Series,
    basin_id: str,
    anchor_label: str,
    model_names: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style(plt)
    sub = joined[joined["basin_id"] == basin_id].sort_values("target_date")
    fig, ax = plt.subplots(figsize=(14, 5.0))
    ax.plot(
        sub["target_date"],
        sub["actual"],
        color=MODEL_COLORS["Actual"],
        linewidth=1.4,
        label="Actual",
    )
    for name in model_names:
        nse = float(metrics_row[f"nse_{name}"])
        ax.plot(
            sub["target_date"],
            sub[f"pred_{name}"],
            color=MODEL_COLORS[name],
            linewidth=1.0,
            alpha=0.85,
            label=f"{name} (NSE={nse:.2f})",
        )
    ax.set_xlabel("Target date")
    ax.set_ylabel("Runoff (mm/day)")
    ax.set_title(
        f"Basin {basin_id} — best by {anchor_label} "
        f"(n={int(metrics_row['n_samples'])})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=PLOT_LEGEND_FONT_SIZE)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-basin overlay: {output_path}", flush=True)


def plot_nse_bars(
    output_path: Path,
    metrics: pd.DataFrame,
    selected: Sequence[Tuple[str, str]],
    model_names: Sequence[str],
) -> None:
    import matplotlib.pyplot as plt

    apply_plot_style(plt)
    basin_ids = [bid for _, bid in selected]
    labels = [f"{bid}\n(best by {anchor})" for anchor, bid in selected]
    n_basins = len(basin_ids)
    n_models = len(model_names)
    bar_w = 0.8 / n_models
    x = np.arange(n_basins)
    fig, ax = plt.subplots(figsize=(max(7.0, 1.8 * n_basins + 5.0), 5.5))
    for i, name in enumerate(model_names):
        vals = [
            float(metrics.loc[metrics["basin_id"] == bid, f"nse_{name}"].iloc[0])
            for bid in basin_ids
        ]
        offsets = (i - (n_models - 1) / 2) * bar_w
        bars = ax.bar(
            x + offsets,
            vals,
            width=bar_w,
            color=MODEL_COLORS[name],
            label=name,
        )
        for bar, value in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + (0.02 if value >= 0 else -0.05),
                f"{value:.2f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=PLOT_FONT_SIZE - 2,
            )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Per-basin NSE (test holdout)")
    ax.set_title("Best-basin NSE comparison across models")
    ax.legend(loc="lower right", fontsize=PLOT_LEGEND_FONT_SIZE)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NSE comparison bar chart: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tft", type=Path, default=DEFAULT_TFT)
    parser.add_argument("--lstm", type=Path, default=DEFAULT_LSTM)
    parser.add_argument("--vanilla", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--itransformer", type=Path, default=DEFAULT_ITRANSFORMER)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "outputs" / "model_comparison",
    )
    parser.add_argument(
        "--n-each",
        type=int,
        default=1,
        help="Best basins to pick per anchor model.",
    )
    parser.add_argument("--min-samples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "TFT": args.tft,
        "LSTM": args.lstm,
        "Vanilla": args.vanilla,
        "iTransformer": args.itransformer,
    }
    print("Prediction sources:", flush=True)
    for name, path in paths.items():
        print(f"  {name:<13s} {path}", flush=True)

    joined = join_predictions(paths)
    print(
        f"Common (basin_id, target_date) rows: {len(joined):,} | "
        f"basins: {joined['basin_id'].nunique()} | "
        f"date range: {joined['target_date'].min().date()} → {joined['target_date'].max().date()}",
        flush=True,
    )

    metrics = per_basin_metrics(joined, ALL_MODELS, args.min_samples)
    metrics_path = args.output_dir / "per_basin_metrics.csv"
    metrics.to_csv(metrics_path, index=False)
    print(f"Saved per-basin metrics: {metrics_path} ({len(metrics)} basins)", flush=True)

    print("Per-model summary on intersection:", flush=True)
    for name in ALL_MODELS:
        col = metrics[f"nse_{name}"]
        finite = col[np.isfinite(col)]
        print(
            f"  {name:<13s} mean_NSE={finite.mean():.3f}  "
            f"median_NSE={finite.median():.3f}  "
            f"frac_NSE<0={(finite < 0).mean():.2%}",
            flush=True,
        )

    selected = select_best_basins(metrics, ANCHOR_MODELS, args.n_each)
    print("Selected basins (anchor → basin_id):", flush=True)
    for anchor, basin_id in selected:
        print(f"  {anchor:<13s} → {basin_id}", flush=True)

    summary_rows: List[Dict[str, object]] = []
    for anchor, basin_id in selected:
        row = metrics.loc[metrics["basin_id"] == basin_id].iloc[0]
        plot_basin_overlay(
            args.output_dir / f"timeseries_{basin_id}_best_by_{anchor}.png",
            joined,
            row,
            basin_id,
            anchor,
            ALL_MODELS,
        )
        entry: Dict[str, object] = {
            "anchor_model": anchor,
            "basin_id": basin_id,
            "n_samples": int(row["n_samples"]),
        }
        for name in ALL_MODELS:
            entry[f"nse_{name}"] = float(row[f"nse_{name}"])
            entry[f"rmse_{name}"] = float(row[f"rmse_{name}"])
            entry[f"mae_{name}"] = float(row[f"mae_{name}"])
        summary_rows.append(entry)

    plot_nse_bars(
        args.output_dir / "best_basin_nse_comparison.png",
        metrics,
        selected,
        ALL_MODELS,
    )

    summary_path = args.output_dir / "best_basin_summary.json"
    summary_path.write_text(
        json.dumps({"basins": summary_rows, "min_samples": args.min_samples}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
