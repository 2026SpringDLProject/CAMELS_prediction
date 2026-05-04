from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[0]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from visualize_tft_predictions import (  # noqa: E402
    clipped_per_basin_nse,
    global_metrics,
    per_basin_nse,
    plot_best_worst_basin_panels,
    plot_per_basin_nse_distribution,
    plot_predictions,
    plot_single_basin_timeseries,
    plot_static_attribute_relationships,
    save_predictions_csv,
    save_static_attribute_analysis,
    static_attribute_relationships,
)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class VanillaTransformerRegressor(nn.Module):
    def __init__(
        self,
        num_features: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        hidden_dim = max(1, d_model // 2)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        return self.head(x[:, -1, :]).squeeze(-1)


class TemporalSummaryiTransformer(nn.Module):
    def __init__(
        self,
        seq_len: int,
        num_dynamic_features: int,
        num_basins: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_dynamic_features = num_dynamic_features
        self.token_input_len = seq_len + 5
        self.variate_embedding = nn.Sequential(
            nn.LayerNorm(self.token_input_len),
            nn.Linear(self.token_input_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.variable_embedding = nn.Parameter(
            torch.randn(1, num_dynamic_features, d_model) * 0.02
        )
        self.basin_embedding = nn.Embedding(num_basins, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, max(1, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, d_model // 2), 1),
        )

    def forward(self, x_dynamic: torch.Tensor, basin_code: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_features = x_dynamic.shape

        x = x_dynamic.transpose(1, 2)
        x_aug = torch.cat(
            [
                x,
                x[:, :, -1:],
                x[:, :, -7:].mean(dim=-1, keepdim=True),
                x[:, :, -30:].mean(dim=-1, keepdim=True),
                x[:, :, -90:].mean(dim=-1, keepdim=True),
                x.mean(dim=-1, keepdim=True),
            ],
            dim=-1,
        )
        variable_tokens = self.variate_embedding(x_aug) + self.variable_embedding
        basin_token = self.basin_embedding(basin_code).unsqueeze(1)
        encoded = self.encoder(torch.cat([basin_token, variable_tokens], dim=1))
        return self.head(encoded[:, 0, :]).squeeze(-1)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def localize_config_paths(config: Dict, checkpoint_path: Path, data_dir: Optional[Path]) -> Dict:
    cfg = dict(config)
    if data_dir is not None:
        cfg["DATA_DIR"] = str(data_dir)
        return cfg

    if Path(str(cfg.get("DATA_DIR", ""))).exists():
        return cfg

    metadata_filename = cfg.get("METADATA_FILENAME", "camels_transformer_metadata.json")
    target_column = str(cfg.get("TARGET_COLUMN", "")).lower()
    project_root = PROJECT_DIR.parent

    candidates: List[Path] = [checkpoint_path.parent / "processed"]
    if len(checkpoint_path.parents) > 1:
        candidates.append(checkpoint_path.parents[1] / "processed")
    if len(checkpoint_path.parents) > 2:
        candidates.append(checkpoint_path.parents[2] / "processed")
    if project_root.exists():
        run_name = cfg.get("RUN_NAME", "raven_default")
        for sibling in project_root.iterdir():
            if not sibling.is_dir():
                continue
            candidates.append(sibling / "processed")
            candidates.append(sibling / "model_artifacts" / run_name / "processed")
    if "qobs" in target_column:
        candidates.extend([
            PROJECT_DIR / "processed_with_qobs",
            PROJECT_DIR / "processed",
        ])
    else:
        candidates.extend([
            PROJECT_DIR / "processed",
            PROJECT_DIR / "processed_no_qobs",
        ])

    seen: set = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if (candidate / metadata_filename).exists():
            cfg["DATA_DIR"] = str(candidate)
            return cfg
    return cfg


def read_joined_table(data_dir: Path, config: Dict, feature_columns: Sequence[str], target_column: str) -> pd.DataFrame:
    import pandas as pd

    preferences = config.get(
        "JOINED_FILENAME_PREFERENCE",
        ["camels_transformer_joined.parquet", "camels_transformer_joined.csv"],
    )
    joined_path = next(data_dir / name for name in preferences if (data_dir / name).exists())

    columns = ["basin_id", "date"] + list(feature_columns) + [target_column]
    if joined_path.suffix == ".parquet":
        df = pd.read_parquet(joined_path, columns=columns)
    else:
        df = pd.read_csv(joined_path, usecols=columns, parse_dates=["date"])
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    df["date"] = pd.to_datetime(df["date"])
    df = df.replace([-999, -999.0, -99, -99.0], np.nan)
    df = df.dropna(subset=list(feature_columns) + [target_column]).copy()
    df = df[df[target_column] >= 0].copy()
    return df.sort_values(["basin_id", "date"]).reset_index(drop=True)


def basin_store_from_joined(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    model_type: str,
    dynamic_feature_columns: Sequence[str],
    basin_code_column: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[np.ndarray], List[str], Optional[np.ndarray], Optional[List[str]], str]:
    store = {}
    basin_ids = []
    for basin_id, basin_df in df.groupby("basin_id", sort=True):
        basin_df = basin_df.sort_values("date").reset_index(drop=True)
        basin_ids.append(str(basin_id))
        if model_type == "itransformer":
            basin_code_values = basin_df[basin_code_column].unique()
            store[str(basin_id)] = {
                "features": basin_df[list(dynamic_feature_columns)].to_numpy(dtype=np.float32),
                "basin_code": int(basin_code_values[0]),
                "target": basin_df[target_column].to_numpy(dtype=np.float32),
                "dates": basin_df["date"].to_numpy(),
            }
        else:
            store[str(basin_id)] = {
                "features": basin_df[list(feature_columns)].to_numpy(dtype=np.float32),
                "target": basin_df[target_column].to_numpy(dtype=np.float32),
                "dates": basin_df["date"].to_numpy(),
            }
    return store, None, basin_ids, None, None, "z-score"


def basin_store_from_npz(
    npz_path: Path,
    feature_columns: Sequence[str],
    target_column: str,
    model_type: str,
    dynamic_feature_columns: Sequence[str],
    basin_code_column: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Optional[np.ndarray], List[str], Optional[np.ndarray], Optional[List[str]], str]:
    data = np.load(npz_path, allow_pickle=True)
    all_columns = [str(col) for col in data["feature_columns"].tolist()]
    column_index = {name: idx for idx, name in enumerate(all_columns)}
    selected_columns = list(dynamic_feature_columns) if model_type == "itransformer" else list(feature_columns)
    selected_idx = [column_index[name] for name in selected_columns]
    basin_code_idx = column_index.get(basin_code_column)

    features = np.asarray(data["features"], dtype=np.float32)
    targets = np.asarray(data["targets"], dtype=np.float32)
    dates = np.asarray(data["row_date"]).astype("datetime64[D]")
    basin_slots = np.asarray(data["row_basin_index"], dtype=np.int64)
    basin_ids = [str(x) for x in data["basin_ids_in_model"].tolist()]

    store = {}
    for slot, basin_id in enumerate(basin_ids):
        mask = basin_slots == slot
        order = np.argsort(dates[mask], kind="stable")
        basin_features = features[mask][:, selected_idx][order]
        basin_targets = targets[mask][order]
        basin_dates = dates[mask][order]
        valid = np.isfinite(basin_targets) & (basin_targets >= 0)
        payload = {
            "features": basin_features[valid],
            "target": basin_targets[valid],
            "dates": basin_dates[valid],
        }
        if model_type == "itransformer":
            payload["basin_code"] = int(features[mask][:, basin_code_idx][order][valid][0])
        store[basin_id] = payload

    static_values = None
    static_columns = None
    static_scale = "z-score"
    if "static_features" in data.files and "static_feature_columns" in data.files:
        static_values = np.asarray(data["static_features"], dtype=np.float64)
        static_columns = [str(col) for col in data["static_feature_columns"].tolist()]
        metadata_path = npz_path.parent / "camels_transformer_metadata.json"
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            static_metadata = metadata.get("static_attributes", {})
            if list(static_metadata.get("columns", [])) == static_columns:
                mean = np.asarray(static_metadata.get("mean", []), dtype=np.float64)
                std = np.asarray(static_metadata.get("std", []), dtype=np.float64)
                if len(mean) == static_values.shape[1] and len(std) == static_values.shape[1]:
                    static_values = static_values * std + mean
                    static_scale = "raw"
    return store, static_values, basin_ids, None, static_columns, static_scale


def eligible_target_indices(n_rows: int, lookback_days: int, horizon_days: int) -> np.ndarray:
    first_target_idx = lookback_days + horizon_days - 1
    if n_rows <= first_target_idx:
        return np.array([], dtype=np.int64)
    return np.arange(first_target_idx, n_rows, dtype=np.int64)


def build_holdout_target_map(
    basin_store: Dict[str, Dict[str, np.ndarray]],
    lookback_days: int,
    horizon_days: int,
    holdout_start: str,
    holdout_end: str,
) -> Dict[str, np.ndarray]:
    start = np.datetime64(holdout_start)
    end = np.datetime64(holdout_end)
    target_map = {}
    for basin_id, basin_data in basin_store.items():
        target_idx = eligible_target_indices(len(basin_data["dates"]), lookback_days, horizon_days)
        target_dates = basin_data["dates"][target_idx].astype("datetime64[D]")
        mask = (target_dates >= start) & (target_dates <= end)
        if np.any(mask):
            target_map[basin_id] = target_idx[mask]
    return target_map


class ExternalWindowDataset(Dataset):
    def __init__(
        self,
        basin_store: Dict[str, Dict[str, np.ndarray]],
        target_map: Dict[str, np.ndarray],
        lookback_days: int,
        horizon_days: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        basin_id_to_slot: Dict[str, int],
        model_type: str,
    ):
        self.basin_store = basin_store
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
        self.basin_id_to_slot = basin_id_to_slot
        self.model_type = model_type
        self.entries = []
        self.cum_counts = []
        running = 0
        for basin_id in sorted(target_map):
            target_indices = np.asarray(target_map[basin_id], dtype=np.int64)
            if len(target_indices) == 0:
                continue
            self.entries.append((basin_id, target_indices))
            running += len(target_indices)
            self.cum_counts.append(running)
        self.length = running

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, object]:
        basin_pos = bisect_right(self.cum_counts, idx)
        basin_id, target_indices = self.entries[basin_pos]
        prev_count = 0 if basin_pos == 0 else self.cum_counts[basin_pos - 1]
        target_idx = int(target_indices[idx - prev_count])
        series = self.basin_store[basin_id]
        sequence_start = target_idx - self.horizon_days - self.lookback_days + 1
        sequence_end = sequence_start + self.lookback_days
        x = series["features"][sequence_start:sequence_end]
        x = (x - self.feature_mean) / self.feature_std
        item = {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(float(series["target"][target_idx]), dtype=torch.float32),
            "basin_slot": torch.tensor(self.basin_id_to_slot[basin_id], dtype=torch.long),
            "target_date": str(np.datetime_as_string(series["dates"][target_idx], unit="D")),
        }
        if self.model_type == "itransformer":
            item["basin_code"] = torch.tensor(int(series["basin_code"]), dtype=torch.long)
        return item


def collate(batch: Sequence[Dict[str, object]]) -> Dict[str, object]:
    output = {
        "x": torch.stack([item["x"] for item in batch]),
        "y": torch.stack([item["y"] for item in batch]),
        "basin_slot": torch.stack([item["basin_slot"] for item in batch]),
        "target_date": [str(item["target_date"]) for item in batch],
    }
    if "basin_code" in batch[0]:
        output["basin_code"] = torch.stack([item["basin_code"] for item in batch])
    return output


def collect_predictions(model: nn.Module, loader: DataLoader, device: torch.device, model_type: str) -> Dict[str, np.ndarray]:
    model.eval()
    pred_parts, true_parts, slot_parts, date_parts = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            if model_type == "itransformer":
                pred = model(x, batch["basin_code"].to(device))
            else:
                pred = model(x)
            pred_parts.append(pred.detach().cpu().numpy().reshape(-1))
            true_parts.append(batch["y"].cpu().numpy().reshape(-1))
            slot_parts.append(batch["basin_slot"].cpu().numpy().reshape(-1))
            date_parts.extend(batch["target_date"])
    return {
        "pred": np.concatenate(pred_parts),
        "true": np.concatenate(true_parts),
        "basin_slot": np.concatenate(slot_parts).astype(np.int64),
        "target_date": np.asarray(date_parts),
    }


def build_model(model_type: str, checkpoint: Dict, lookback_days: int) -> nn.Module:
    hparams = checkpoint["best_hparams"]
    if model_type == "itransformer":
        return TemporalSummaryiTransformer(
            seq_len=lookback_days,
            num_dynamic_features=len(checkpoint["dynamic_feature_columns"]),
            num_basins=int(checkpoint["num_basins"]),
            d_model=int(hparams["d_model"]),
            nhead=int(hparams["nhead"]),
            num_layers=int(hparams["num_layers"]),
            dim_feedforward=int(hparams["dim_feedforward"]),
            dropout=float(hparams["dropout"]),
        )
    return VanillaTransformerRegressor(
        num_features=len(checkpoint["feature_columns"]),
        d_model=int(hparams["d_model"]),
        nhead=int(hparams["nhead"]),
        num_layers=int(hparams["num_layers"]),
        dim_feedforward=int(hparams["dim_feedforward"]),
        dropout=float(hparams["dropout"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply TFT-style prediction analysis plots to iTransformer or vanilla Transformer checkpoints."
    )
    parser.add_argument("--model-type", choices=["itransformer", "vanilla"], required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-npz", type=Path, default=None, help="Prepared indexed .npz data.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed directory with joined csv/parquet + metadata.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-name", type=str, default="test")
    parser.add_argument("--n-plot", type=int, default=500)
    parser.add_argument("--n-basin-panels", type=int, default=3)
    parser.add_argument("--n-static-attribute-plots", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--basin-id", type=str, default=None)
    parser.add_argument("--basin-slot", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = localize_config_paths(checkpoint.get("config", {}), checkpoint_path, args.data_dir)
    output_dir = args.output_dir or (checkpoint_path.parent / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)

    lookback_days = int(config.get("LOOKBACK_DAYS", 365))
    horizon_days = int(config.get("PREDICTION_HORIZON_DAYS", 1))
    feature_columns = list(checkpoint["feature_columns"])
    target_column = str(checkpoint["target_column"])
    dynamic_feature_columns = list(checkpoint.get("dynamic_feature_columns", feature_columns[:-1]))
    basin_code_column = str(checkpoint.get("basin_code_column", "basin_code"))

    if args.data_npz is not None:
        print(f"Reading prepared windows from: {args.data_npz}", flush=True)
        basin_store, static_values, basin_ids, _, static_columns, static_scale = basin_store_from_npz(
            args.data_npz,
            feature_columns=feature_columns,
            target_column=target_column,
            model_type=args.model_type,
            dynamic_feature_columns=dynamic_feature_columns,
            basin_code_column=basin_code_column,
        )
    else:
        data_dir = Path(config["DATA_DIR"])
        print(f"Reading joined table from: {data_dir}", flush=True)
        df = read_joined_table(data_dir, config, feature_columns, target_column)
        basin_store, static_values, basin_ids, _, static_columns, static_scale = basin_store_from_joined(
            df,
            feature_columns=feature_columns,
            target_column=target_column,
            model_type=args.model_type,
            dynamic_feature_columns=dynamic_feature_columns,
            basin_code_column=basin_code_column,
        )

    target_map = build_holdout_target_map(
        basin_store,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        holdout_start=config.get("INDEPENDENT_VAL_START", "2008-01-01"),
        holdout_end=config.get("INDEPENDENT_VAL_END", "2014-12-31"),
    )
    basin_id_to_slot = {basin_id: idx for idx, basin_id in enumerate(basin_ids)}
    dataset = ExternalWindowDataset(
        basin_store=basin_store,
        target_map=target_map,
        lookback_days=lookback_days,
        horizon_days=horizon_days,
        feature_mean=np.asarray(checkpoint["feature_mean"], dtype=np.float32),
        feature_std=np.asarray(checkpoint["feature_std"], dtype=np.float32),
        basin_id_to_slot=basin_id_to_slot,
        model_type=args.model_type,
    )
    device = resolve_device()
    model = build_model(args.model_type, checkpoint, lookback_days).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size or int(checkpoint["best_hparams"].get("batch_size", 128)),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    predictions = collect_predictions(model, loader, device, args.model_type)
    true = predictions["true"]
    pred = predictions["pred"]
    basin_slots = predictions["basin_slot"]
    dates = predictions["target_date"]
    scores, rows = per_basin_nse(true, pred, basin_slots, basin_ids)
    metrics = global_metrics(true, pred)
    clipped_scores, _, _, _, _ = clipped_per_basin_nse(scores)
    print(
        f"{args.model_type} {args.split_name.upper()} | "
        f"RMSE={metrics['rmse']:.4f} MAE={metrics['mae']:.4f} "
        f"Global_NSE={metrics['nse']:.4f} "
        f"clipped_mean_basin_NSE={float(np.mean(clipped_scores)) if clipped_scores.size else float('nan'):.4f} "
        f"clipped_median_basin_NSE={float(np.median(clipped_scores)) if clipped_scores.size else float('nan'):.4f}",
        flush=True,
    )

    output_base = output_dir / f"{args.split_name}_{args.model_type}_predictions"
    save_predictions_csv(output_base.with_suffix(".csv"), true, pred, basin_slots, dates, basin_ids)
    plot_predictions(output_base.with_suffix(".png"), true, pred, scores, title="", n_plot=args.n_plot)

    selected_basin_slot: Optional[int] = args.basin_slot
    if selected_basin_slot is None and args.basin_id is not None:
        selected_basin_slot = basin_id_to_slot.get(str(args.basin_id).zfill(8))
    if selected_basin_slot is None and rows:
        selected_basin_slot = int(rows[0]["basin_slot"])
    if selected_basin_slot is not None:
        selected_basin_label = (
            str(basin_ids[selected_basin_slot])
            if selected_basin_slot < len(basin_ids)
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
        title="",
    )

    if static_values is not None and static_columns:
        relationship_rows = static_attribute_relationships(scores, static_values, static_columns)
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
            title="",
            scale_label=static_scale,
            top_n=args.n_static_attribute_plots,
        )
    else:
        print("No static attributes found; skipped basin-attribute analysis.", flush=True)

    per_basin_path = output_base.with_name(f"{output_base.name}_per_basin_nse.json")
    per_basin_path.write_text(json.dumps({"basins": rows}, indent=2) + "\n", encoding="utf-8")
    metrics_path = output_base.with_name(f"{output_base.name}_metrics.json")
    metrics_payload = {
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "global_nse": metrics["nse"],
        "clipped_mean_basin_nse": float(np.mean(clipped_scores)) if clipped_scores.size else float("nan"),
        "clipped_median_basin_nse": float(np.median(clipped_scores)) if clipped_scores.size else float("nan"),
        "n_valid_basins": int(clipped_scores.size),
    }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved per-basin NSE to: {per_basin_path}", flush=True)
    print(f"Saved metrics to: {metrics_path}", flush=True)


if __name__ == "__main__":
    main()
