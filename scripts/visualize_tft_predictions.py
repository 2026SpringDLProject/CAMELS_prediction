import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from train import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    CamelsWindowDataset,
    build_config_from_sources,
    build_model,
    compute_train_normalization_stats,
    denormalize_targets,
    format_targets,
    make_loader,
    normalize_features,
    resolve_device,
    temporal_split_indices,
)

PER_BASIN_NSE_YLIM = (-1.0, 1.5)
PLOT_FONT_SIZE = 12
PLOT_SMALL_FONT_SIZE = 11
PLOT_LEGEND_FONT_SIZE = 12


def apply_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.size": PLOT_FONT_SIZE,
            "axes.labelsize": PLOT_FONT_SIZE,
            "xtick.labelsize": PLOT_FONT_SIZE,
            "ytick.labelsize": PLOT_FONT_SIZE,
            "legend.fontsize": PLOT_LEGEND_FONT_SIZE,
            "figure.titlesize": PLOT_FONT_SIZE,
        }
    )


def resolve_project_path(path_value: Path) -> Path:
    return path_value if path_value.is_absolute() else PROJECT_DIR / path_value


def global_metrics(true: np.ndarray, pred: np.ndarray) -> Dict[str, float]:
    diff = pred - true
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(max(mse, 0.0))
    mae = float(np.mean(np.abs(diff)))
    sse = float(np.sum(diff * diff))
    centered = true - float(np.mean(true))
    sst = float(np.sum(centered * centered))
    nse = 1.0 - sse / sst if sst > 1e-12 else float("nan")
    r2 = nse
    return {"rmse": rmse, "mae": mae, "r2": r2, "nse": nse}


def per_basin_nse(
    true: np.ndarray,
    pred: np.ndarray,
    basin_slots: np.ndarray,
    basin_ids: Optional[Sequence[str]],
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    n_basins = int(np.max(basin_slots)) + 1 if basin_slots.size else 0
    scores = np.full(n_basins, np.nan, dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for slot in range(n_basins):
        mask = basin_slots == slot
        if np.count_nonzero(mask) < 2:
            continue
        y = true[mask]
        y_hat = pred[mask]
        sst = float(np.sum((y - float(np.mean(y))) ** 2))
        if sst <= 1e-12:
            continue
        score = 1.0 - float(np.sum((y - y_hat) ** 2)) / sst
        scores[slot] = score
        basin_id = str(basin_ids[slot]) if basin_ids is not None and slot < len(basin_ids) else str(slot)
        rows.append({"basin_slot": slot, "basin_id": basin_id, "nse": score})
    rows.sort(key=lambda row: row["nse"])
    return scores, rows


def collect_predictions(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    future_feature_mean: Optional[torch.Tensor],
    future_feature_std: Optional[torch.Tensor],
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> Dict[str, np.ndarray]:
    model.eval()
    pred_parts: List[np.ndarray] = []
    true_parts: List[np.ndarray] = []
    basin_slot_parts: List[np.ndarray] = []
    date_parts: List[str] = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y_raw = batch["y"].to(device)
            future_known = batch.get("future_known")
            static_features = batch.get("static")
            basin_slot = batch.get("basin_slot")
            if basin_slot is None:
                raise ValueError("Prediction visualization requires basin_slot in the dataset.")
            basin_slot = basin_slot.to(device)
            if static_features is not None:
                static_features = static_features.to(device)

            x = normalize_features(x, basin_slot, feature_mean, feature_std)
            if future_known is not None:
                future_known = future_known.to(device)
                future_known = normalize_features(
                    future_known,
                    basin_slot,
                    future_feature_mean,
                    future_feature_std,
                )

            pred, _ = model(x, future_known=future_known, static_features=static_features)
            target = format_targets(y_raw, model.prediction_length)
            pred_phys = denormalize_targets(pred, basin_slot, target_mean, target_std)

            if pred_phys.ndim == 1:
                pred_phys = pred_phys.unsqueeze(-1)
                target = target.unsqueeze(-1)

            horizon = pred_phys.size(-1)
            pred_parts.append(pred_phys.cpu().numpy().reshape(-1))
            true_parts.append(target.cpu().numpy().reshape(-1))
            basin_slot_parts.append(
                basin_slot.unsqueeze(-1).expand(-1, horizon).cpu().numpy().reshape(-1)
            )

            batch_dates = batch.get("target_date")
            if batch_dates is not None:
                for date_value in batch_dates:
                    date_parts.extend([str(date_value)] * horizon)

    payload = {
        "pred": np.concatenate(pred_parts),
        "true": np.concatenate(true_parts),
        "basin_slot": np.concatenate(basin_slot_parts).astype(np.int64),
    }
    if date_parts:
        payload["target_date"] = np.array(date_parts)
    return payload


def save_predictions_csv(
    path: Path,
    true: np.ndarray,
    pred: np.ndarray,
    basin_slots: np.ndarray,
    dates: Optional[np.ndarray],
    basin_ids: Optional[Sequence[str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample", "target_date", "basin_slot", "basin_id", "actual", "predicted", "residual"],
        )
        writer.writeheader()
        for i, (actual, predicted, slot) in enumerate(zip(true, pred, basin_slots)):
            basin_id = (
                str(basin_ids[int(slot)])
                if basin_ids is not None and int(slot) < len(basin_ids)
                else str(int(slot))
            )
            writer.writerow(
                {
                    "sample": i,
                    "target_date": str(dates[i]) if dates is not None and i < len(dates) else "",
                    "basin_slot": int(slot),
                    "basin_id": basin_id,
                    "actual": float(actual),
                    "predicted": float(predicted),
                    "residual": float(predicted - actual),
                }
            )


def static_attribute_values(
    dataset: CamelsWindowDataset,
    processed_dir: Path,
) -> Tuple[np.ndarray, List[str], str]:
    columns = list(dataset.static_feature_columns)
    values = np.asarray(dataset.static_features, dtype=np.float64)
    scale_label = "z-score"
    if values.size == 0 or not columns:
        return values, columns, scale_label

    metadata_path = processed_dir / "camels_transformer_metadata.json"
    if not metadata_path.exists():
        return values, columns, scale_label

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        static_metadata = metadata.get("static_attributes", {})
        metadata_columns = list(static_metadata.get("columns", []))
        mean = np.asarray(static_metadata.get("mean", []), dtype=np.float64)
        std = np.asarray(static_metadata.get("std", []), dtype=np.float64)
    except Exception as exc:
        print(f"Could not read static attribute metadata ({exc}); using z-scored attributes.", flush=True)
        return values, columns, scale_label

    if metadata_columns == columns and mean.shape[0] == values.shape[1] and std.shape[0] == values.shape[1]:
        return values * std + mean, columns, "raw"

    print(
        "Static attribute metadata did not match dataset columns; using z-scored attributes.",
        flush=True,
    )
    return values, columns, scale_label


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return float("nan")
    return pearson_corr(average_ranks(x), average_ranks(y))


def static_attribute_relationships(
    per_basin_scores: np.ndarray,
    static_values: np.ndarray,
    static_columns: Sequence[str],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if static_values.size == 0 or not static_columns:
        return rows

    n = min(len(per_basin_scores), static_values.shape[0])
    nse = per_basin_scores[:n].astype(np.float64)
    for col_idx, col_name in enumerate(static_columns):
        attr = static_values[:n, col_idx].astype(np.float64)
        mask = np.isfinite(nse) & np.isfinite(attr)
        if np.count_nonzero(mask) < 3:
            continue
        x = attr[mask]
        y = nse[mask]
        rows.append(
            {
                "attribute": str(col_name),
                "n": int(np.count_nonzero(mask)),
                "pearson_r": pearson_corr(x, y),
                "spearman_r": spearman_corr(x, y),
                "attribute_min": float(np.min(x)),
                "attribute_median": float(np.median(x)),
                "attribute_max": float(np.max(x)),
                "nse_median": float(np.median(y)),
            }
        )

    rows.sort(
        key=lambda row: max(
            abs(float(row["spearman_r"])) if math.isfinite(float(row["spearman_r"])) else -1.0,
            abs(float(row["pearson_r"])) if math.isfinite(float(row["pearson_r"])) else -1.0,
        ),
        reverse=True,
    )
    return rows


def save_static_attribute_analysis(
    output_base: Path,
    per_basin_scores: np.ndarray,
    basin_ids: Optional[Sequence[str]],
    static_values: np.ndarray,
    static_columns: Sequence[str],
    relationship_rows: Sequence[Dict[str, object]],
) -> None:
    if static_values.size == 0 or not static_columns:
        print("No static attributes available; skipped static attribute analysis.", flush=True)
        return

    output_base.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(per_basin_scores), static_values.shape[0])
    basin_path = output_base.with_name(f"{output_base.name}_basin_static_attributes.csv")
    with basin_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["basin_slot", "basin_id", "nse"] + [str(col) for col in static_columns]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for slot in range(n):
            basin_id = (
                str(basin_ids[slot])
                if basin_ids is not None and slot < len(basin_ids)
                else str(slot)
            )
            row = {
                "basin_slot": slot,
                "basin_id": basin_id,
                "nse": float(per_basin_scores[slot]),
            }
            for col_idx, col_name in enumerate(static_columns):
                row[str(col_name)] = float(static_values[slot, col_idx])
            writer.writerow(row)
    print(f"Saved per-basin NSE + static attributes to: {basin_path}", flush=True)

    corr_path = output_base.with_name(f"{output_base.name}_static_attribute_correlations.csv")
    with corr_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "attribute",
            "n",
            "pearson_r",
            "spearman_r",
            "attribute_min",
            "attribute_median",
            "attribute_max",
            "nse_median",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in relationship_rows:
            writer.writerow(row)
    print(f"Saved static attribute correlations to: {corr_path}", flush=True)


def plot_static_attribute_relationships(
    output_path: Path,
    per_basin_scores: np.ndarray,
    static_values: np.ndarray,
    static_columns: Sequence[str],
    relationship_rows: Sequence[Dict[str, object]],
    title: str,
    scale_label: str,
    top_n: int,
) -> None:
    if static_values.size == 0 or not static_columns:
        return
    selected = list(relationship_rows[: max(1, top_n)])
    if not selected:
        print("No finite static attribute relationships; skipped NSE vs attribute plot.", flush=True)
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped NSE vs static attribute plot.", flush=True)
        return
    apply_plot_style(plt)

    col_index = {str(name): idx for idx, name in enumerate(static_columns)}
    n_plots = len(selected)
    n_cols = min(3, n_plots)
    n_rows = int(math.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.0 * n_cols, 4.0 * n_rows), squeeze=False)

    for ax_idx, ax in enumerate(axes.reshape(-1)):
        if ax_idx >= n_plots:
            ax.axis("off")
            continue
        row = selected[ax_idx]
        attr_name = str(row["attribute"])
        attr = static_values[: len(per_basin_scores), col_index[attr_name]].astype(np.float64)
        nse = per_basin_scores[: len(attr)].astype(np.float64)
        mask = np.isfinite(attr) & np.isfinite(nse)
        x = attr[mask]
        y = nse[mask]
        ax.scatter(x, y, s=18, alpha=0.7, color="#4C78A8", edgecolor="none")
        if len(x) >= 2 and float(np.std(x)) > 1e-12:
            slope, intercept = np.polyfit(x, y, deg=1)
            x_line = np.linspace(float(np.min(x)), float(np.max(x)), 100)
            ax.plot(x_line, slope * x_line + intercept, color="#B54A4A", linewidth=1.5)
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_ylim(*PER_BASIN_NSE_YLIM)
        ax.set_xlabel(f"{attr_name} ({scale_label})")
        ax.set_ylabel("Per-basin NSE")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved NSE vs static attribute plot to: {output_path}", flush=True)


def plot_predictions(
    output_path: Path,
    true: np.ndarray,
    pred: np.ndarray,
    per_basin_scores: np.ndarray,
    title: str,
    n_plot: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped prediction visualization.", flush=True)
        return
    apply_plot_style(plt)

    metrics = global_metrics(true, pred)
    residuals = pred - true
    finite_scores = per_basin_scores[np.isfinite(per_basin_scores)]
    n_plot = min(n_plot, len(true))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    lo = float(min(np.min(true), np.min(pred)))
    hi = float(max(np.max(true), np.max(pred)))
    axes[0, 0].scatter(true, pred, s=6, alpha=0.25)
    axes[0, 0].plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1)
    axes[0, 0].set_xlabel("Actual runoff (mm/day)")
    axes[0, 0].set_ylabel("Predicted runoff (mm/day)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].scatter(true, residuals, s=6, alpha=0.25)
    axes[0, 1].axhline(0, linestyle="--", color="black", linewidth=1)
    axes[0, 1].set_xlabel("Actual runoff (mm/day)")
    axes[0, 1].set_ylabel("Residual = Predicted - Actual")
    axes[0, 1].grid(True, alpha=0.3)

    bins = min(100, max(20, int(np.sqrt(len(true)))))
    axes[1, 0].hist(true, bins=bins, alpha=0.55, label="Actual")
    axes[1, 0].hist(pred, bins=bins, alpha=0.55, label="Predicted")
    axes[1, 0].set_xlabel("Runoff (mm/day)")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(true[:n_plot], label="Actual", linewidth=1.5)
    axes[1, 1].plot(pred[:n_plot], label="Predicted", linewidth=1.5)
    axes[1, 1].set_xlabel("Sample index")
    axes[1, 1].set_ylabel("Runoff (mm/day)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    score_text = "Per-basin NSE: n/a"
    if finite_scores.size:
        score_text = (
            f"Per-basin NSE median={np.median(finite_scores):.3f}, "
            f"mean={np.mean(finite_scores):.3f}, p10={np.percentile(finite_scores, 10):.3f}"
        )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved prediction visualization to: {output_path}", flush=True)


def sort_basin_series(
    true: np.ndarray,
    pred: np.ndarray,
    dates: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    if dates is None:
        return true, pred, None
    date_values = dates.astype("datetime64[D]")
    order = np.argsort(date_values, kind="stable")
    return true[order], pred[order], date_values[order]


def plot_single_basin_timeseries(
    output_path: Path,
    true: np.ndarray,
    pred: np.ndarray,
    basin_slots: np.ndarray,
    dates: Optional[np.ndarray],
    basin_slot: int,
    basin_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped single-basin visualization.", flush=True)
        return
    apply_plot_style(plt)

    mask = basin_slots == basin_slot
    if np.count_nonzero(mask) == 0:
        print(f"No samples found for basin {basin_label}; skipped single-basin plot.", flush=True)
        return

    basin_true, basin_pred, basin_dates = sort_basin_series(true[mask], pred[mask], dates[mask] if dates is not None else None)
    x = basin_dates if basin_dates is not None else np.arange(len(basin_true))
    metrics = global_metrics(basin_true, basin_pred)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(x, basin_true, label="Actual", linewidth=1.5)
    ax.plot(x, basin_pred, label="Predicted", linewidth=1.5)
    ax.set_xlabel("Target date" if basin_dates is not None else "Sample index")
    ax.set_ylabel("Runoff (mm/day)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved single-basin time series to: {output_path}", flush=True)


def plot_best_worst_basin_panels(
    output_path: Path,
    true: np.ndarray,
    pred: np.ndarray,
    basin_slots: np.ndarray,
    dates: Optional[np.ndarray],
    per_basin_rows: Sequence[Dict[str, object]],
    n_each: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped best/worst basin panels.", flush=True)
        return
    apply_plot_style(plt)

    finite_rows = [row for row in per_basin_rows if math.isfinite(float(row["nse"]))]
    if not finite_rows:
        print("No finite per-basin NSE values; skipped best/worst basin panels.", flush=True)
        return

    worst = finite_rows[:n_each]
    best = list(reversed(finite_rows[-n_each:]))
    selected = worst + best
    n_rows = len(selected)
    fig, axes = plt.subplots(n_rows, 1, figsize=(14, max(3.0 * n_rows, 4.5)), sharex=False)
    if n_rows == 1:
        axes = [axes]

    for panel_idx, (ax, row) in enumerate(zip(axes, selected)):
        slot = int(row["basin_slot"])
        label = str(row["basin_id"])
        nse = float(row["nse"])
        mask = basin_slots == slot
        basin_true, basin_pred, basin_dates = sort_basin_series(
            true[mask],
            pred[mask],
            dates[mask] if dates is not None else None,
        )
        x = basin_dates if basin_dates is not None else np.arange(len(basin_true))
        tag = "Worst" if row in worst else "Best"
        ax.plot(x, basin_true, label="Actual", linewidth=1.25)
        ax.plot(x, basin_pred, label="Predicted", linewidth=1.25)
        panel_label = f"({chr(ord('a') + panel_idx)})"
        ax.text(
            0.01,
            0.95,
            f"{panel_label} {tag} basin {label} | NSE={nse:.3f}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=PLOT_FONT_SIZE,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none", "pad": 2},
        )
        ax.set_ylabel("Runoff")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Target date" if dates is not None else "Sample index")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved best/worst basin panels to: {output_path}", flush=True)


def plot_per_basin_nse_distribution(
    output_path: Path,
    per_basin_scores: np.ndarray,
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped per-basin NSE distribution.", flush=True)
        return
    apply_plot_style(plt)

    finite_scores = per_basin_scores[np.isfinite(per_basin_scores)]
    if finite_scores.size == 0:
        print("No finite per-basin NSE values; skipped NSE distribution plot.", flush=True)
        return

    # NSE can have very large negative outliers. Show the useful skill range while
    # preserving how many basins fall below the plotted lower edge.
    x_min = max(float(np.percentile(finite_scores, 2)), -1.0)
    x_max = min(float(np.percentile(finite_scores, 98)), 1.0)
    if x_max <= x_min:
        x_min = min(float(np.min(finite_scores)), -1.0)
        x_max = max(float(np.max(finite_scores)), 1.0)
    clipped_scores = np.clip(finite_scores, x_min, x_max)
    n_below = int(np.sum(finite_scores < x_min))
    n_above = int(np.sum(finite_scores > x_max))

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 5.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 4]},
    )
    bins = min(60, max(15, int(np.sqrt(finite_scores.size))))
    axes[1].hist(clipped_scores, bins=bins, range=(x_min, x_max), color="#4C78A8", alpha=0.8)
    median_score = float(np.median(finite_scores))
    mean_score = float(np.mean(finite_scores))
    if x_min <= median_score <= x_max:
        axes[1].axvline(median_score, color="black", linestyle="--", linewidth=1, label="Median")
    if x_min <= mean_score <= x_max:
        axes[1].axvline(mean_score, color="#B54A4A", linestyle=":", linewidth=1.5, label="Mean")
    if n_below or n_above:
        axes[1].text(
            0.02,
            0.95,
            f"clipped: {n_below} below, {n_above} above",
            transform=axes[1].transAxes,
            va="top",
            ha="left",
            fontsize=PLOT_SMALL_FONT_SIZE,
        )
    axes[1].set_xlim(x_min, x_max)
    axes[1].set_xlabel("Per-basin NSE")
    axes[1].set_ylabel("Basin count")
    if axes[1].get_legend_handles_labels()[0]:
        axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[0].boxplot(finite_scores, vert=False, showfliers=False)
    axes[0].set_yticks([1])
    axes[0].set_yticklabels(["Basins"])
    axes[0].grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-basin NSE distribution to: {output_path}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize TFT predictions from a trained checkpoint.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--n-plot", type=int, default=500)
    parser.add_argument("--basin-id", type=str, default=None)
    parser.add_argument("--basin-slot", type=int, default=None)
    parser.add_argument("--n-basin-panels", type=int, default=3)
    parser.add_argument("--n-static-attribute-plots", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def loss_label_from_config(cfg) -> str:
    loss_name_lower = cfg.loss_name.lower()
    if loss_name_lower == "nse":
        return "Basin-averaged NSE* Loss"
    if loss_name_lower in {"global_nse", "global-nse"}:
        return "Global NSE Loss"
    if loss_name_lower == "mixed":
        return f"Mixed Loss ({cfg.mse_weight:.2f} MSE + {1.0 - cfg.mse_weight:.2f} NSE*)"
    return "MSE Loss"


def plot_cv_training_curves(
    cv_histories: Sequence[Dict[str, object]],
    output_path: Path,
    loss_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipped saving CV training curve.", flush=True)
        return
    apply_plot_style(plt)

    if not cv_histories:
        return

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cv_histories), 1)))
    handles = []

    for color, fold_payload in zip(colors, cv_histories):
        fold_id = fold_payload.get("fold_id", "?")
        history = fold_payload.get("history", [])
        if not history:
            continue
        epochs = [item["epoch"] for item in history]
        train_loss = [item["train_loss"] for item in history]
        val_loss = [item["val_loss"] for item in history]
        (train_line,) = ax.plot(
            epochs,
            train_loss,
            color=color,
            linewidth=2,
            marker="o",
            markersize=3,
            label=f"Fold {fold_id} train",
        )
        (val_line,) = ax.plot(
            epochs,
            val_loss,
            color=color,
            linestyle="--",
            linewidth=2,
            marker="s",
            markersize=3,
            label=f"Fold {fold_id} val",
        )
        handles.extend([train_line, val_line])

    ax.set_xlabel("Epoch")
    ax.set_ylabel(loss_label)
    ax.grid(True, alpha=0.3)
    if handles:
        ax.legend(handles=handles, loc="best", fontsize=PLOT_LEGEND_FONT_SIZE)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CV training curve to: {output_path}", flush=True)


def plot_cv_history_from_json(history_path: Path, output_dir: Path, cfg) -> bool:
    if not history_path.exists():
        print(f"History file not found; skipped CV training curve: {history_path}", flush=True)
        return False

    try:
        payload = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Could not read history file ({exc}); skipped CV training curve.", flush=True)
        return False

    cv_histories = payload.get("cv_histories")
    if not cv_histories:
        print("history.json does not contain cv_histories; skipped CV training curve.", flush=True)
        return False

    plot_cv_training_curves(
        cv_histories,
        output_dir / "cv_training_curve.png",
        loss_label=loss_label_from_config(cfg),
    )
    return True


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(args.config)
    cfg = build_config_from_sources(config_path, argparse.Namespace())
    checkpoint_path = (
        resolve_project_path(args.checkpoint)
        if args.checkpoint is not None
        else cfg.output_dir / "best_model.pt"
    )
    output_dir = checkpoint_path.parent
    plot_cv_history_from_json(output_dir / "history.json", output_dir, cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = resolve_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_config = checkpoint.get("config") or {}
    for key in (
        "model_name",
        "d_model",
        "lstm_hidden",
        "n_heads",
        "dropout",
        "prediction_length",
    ):
        if key in checkpoint_config:
            setattr(cfg, key, checkpoint_config[key])

    dataset = CamelsWindowDataset(cfg.data_path)
    train_idx, val_idx, test_idx = temporal_split_indices(dataset, cfg)
    feature_mean_np, feature_std_np, target_mean_np, target_std_np, _ = compute_train_normalization_stats(
        dataset, train_idx
    )

    feature_mean = torch.from_numpy(feature_mean_np).to(device)
    feature_std = torch.from_numpy(feature_std_np).to(device)
    target_mean = torch.from_numpy(target_mean_np).to(device)
    target_std = torch.from_numpy(target_std_np).to(device)
    if dataset.future_known_feature_indices is not None:
        future_idx = torch.from_numpy(dataset.future_known_feature_indices).to(device=device, dtype=torch.long)
        future_feature_mean = torch.index_select(feature_mean, dim=1, index=future_idx)
        future_feature_std = torch.index_select(feature_std, dim=1, index=future_idx)
    else:
        future_feature_mean = None
        future_feature_std = None

    model = build_model(dataset, cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_label = cfg.model_name.upper()
    print(f"Loaded checkpoint: {checkpoint_path}", flush=True)
    print(f"Using device: {device}", flush=True)

    split_map = {"val": val_idx, "test": test_idx}
    batch_size = args.batch_size or cfg.batch_size
    basin_ids = dataset.basin_ids_in_model.tolist() if dataset.basin_ids_in_model is not None else None
    static_values, static_columns, static_scale_label = static_attribute_values(dataset, cfg.processed_dir)

    for split_name in args.splits:
        loader = make_loader(dataset, split_map[split_name], batch_size, False, cfg.num_workers, device)
        predictions = collect_predictions(
            model,
            loader,
            device,
            feature_mean,
            feature_std,
            future_feature_mean,
            future_feature_std,
            target_mean,
            target_std,
        )
        true = predictions["true"]
        pred = predictions["pred"]
        basin_slots = predictions["basin_slot"]
        dates = predictions.get("target_date")
        scores, rows = per_basin_nse(true, pred, basin_slots, basin_ids)
        metrics = global_metrics(true, pred)
        finite_scores = scores[np.isfinite(scores)]
        mean_basin_nse = float(np.mean(finite_scores)) if finite_scores.size else float("nan")
        median_basin_nse = float(np.median(finite_scores)) if finite_scores.size else float("nan")
        print(
            f"{split_name.upper()} performance | "
            f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} "
            f"R2={metrics['r2']:.4f} NSE={metrics['nse']:.4f} "
            f"mean_basin_NSE={mean_basin_nse:.4f} median_basin_NSE={median_basin_nse:.4f}",
            flush=True,
        )

        output_base = output_dir / f"{split_name}_tft_predictions"
        save_predictions_csv(output_base.with_suffix(".csv"), true, pred, basin_slots, dates, basin_ids)
        plot_predictions(
            output_base.with_suffix(".png"),
            true,
            pred,
            scores,
            title=f"{model_label} {split_name.upper()} Predictions",
            n_plot=args.n_plot,
        )

        selected_basin_slot: Optional[int] = None
        if args.basin_slot is not None:
            selected_basin_slot = args.basin_slot
        elif args.basin_id is not None and basin_ids is not None:
            basin_id_lookup = {str(basin_id): i for i, basin_id in enumerate(basin_ids)}
            selected_basin_slot = basin_id_lookup.get(str(args.basin_id))
            if selected_basin_slot is None:
                print(f"Basin ID {args.basin_id} not found; skipped requested basin plot.", flush=True)
        elif rows:
            selected_basin_slot = int(rows[0]["basin_slot"])

        if selected_basin_slot is not None:
            selected_basin_label = (
                str(basin_ids[selected_basin_slot])
                if basin_ids is not None and selected_basin_slot < len(basin_ids)
                else str(selected_basin_slot)
            )
            plot_single_basin_timeseries(
                output_base.with_name(f"{output_base.name}_basin_{selected_basin_label}.png"),
                true,
                pred,
                basin_slots,
                dates,
                selected_basin_slot,
                selected_basin_label,
            )

        plot_best_worst_basin_panels(
            output_base.with_name(f"{output_base.name}_best_worst_basins.png"),
            true,
            pred,
            basin_slots,
            dates,
            rows,
            max(1, args.n_basin_panels),
        )
        plot_per_basin_nse_distribution(
            output_base.with_name(f"{output_base.name}_per_basin_nse_distribution.png"),
            scores,
            title=f"{model_label} {split_name.upper()} Per-Basin NSE",
        )
        relationship_rows = static_attribute_relationships(
            scores,
            static_values,
            static_columns,
        )
        save_static_attribute_analysis(
            output_base,
            scores,
            basin_ids,
            static_values,
            static_columns,
            relationship_rows,
        )
        plot_static_attribute_relationships(
            output_base.with_name(f"{output_base.name}_nse_vs_static_attributes.png"),
            scores,
            static_values,
            static_columns,
            relationship_rows,
            title=f"{model_label} {split_name.upper()} Per-Basin NSE vs Static Attributes",
            scale_label=static_scale_label,
            top_n=args.n_static_attribute_plots,
        )

        if rows:
            save_json_path = output_base.with_name(f"{output_base.name}_per_basin_nse.json")
            save_json_path.write_text(json.dumps({"basins": rows}, indent=2) + "\n", encoding="utf-8")
            print(f"Saved per-basin NSE to: {save_json_path}", flush=True)


if __name__ == "__main__":
    main()
