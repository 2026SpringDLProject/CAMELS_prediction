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
        if seq_len != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {seq_len}")
        if num_features != self.num_dynamic_features:
            raise ValueError(
                f"Expected num_dynamic_features={self.num_dynamic_features}, got {num_features}"
            )

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


class _AutoformerMovingAverage(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) — pad with first/last row replication on time axis.
        pad = (self.kernel_size - 1) // 2
        front = x[:, 0:1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        x_pad = torch.cat([front, x, end], dim=1)
        return self.avg(x_pad.transpose(1, 2)).transpose(1, 2)


class _AutoformerSeriesDecomposition(nn.Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = _AutoformerMovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        return x - trend, trend


class _AutoformerLiteEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, moving_avg_kernel: int):
        super().__init__()
        self.decomp1 = _AutoformerSeriesDecomposition(moving_avg_kernel)
        self.decomp2 = _AutoformerSeriesDecomposition(moving_avg_kernel)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + self.dropout(attn_out)
        seasonal, trend1 = self.decomp1(x)
        ffn_out = self.ffn(seasonal)
        x = seasonal + self.dropout(ffn_out)
        seasonal, trend2 = self.decomp2(x)
        return seasonal + trend1 + trend2


class AutoformerRegressor(nn.Module):
    """TemporalSummaryAutoformer — verbatim from `train_autotransformer.ipynb`.

    Forward: input_projection(x_dynamic) + position_embedding (no basin add)
    → N × AutoformerLite encoder (attn → decomp1 → ffn(seasonal) → decomp2 →
    seasonal+trend1+trend2) → temporal summary
    `[last, mean_7, mean_30, mean_90, mean_365]` (5·d_model) → concat with
    `basin_embedding(basin_code)` (1·d_model) → head."""

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
        moving_avg_kernel: int = 25,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.num_dynamic_features = num_dynamic_features
        self.input_projection = nn.Linear(num_dynamic_features, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        self.basin_embedding = nn.Embedding(num_basins, d_model)
        self.layers = nn.ModuleList(
            [
                _AutoformerLiteEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    moving_avg_kernel=moving_avg_kernel,
                )
                for _ in range(num_layers)
            ]
        )
        head_in = d_model * 6  # 5-pool summary + basin embedding
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max(1, d_model // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, d_model // 2), 1),
        )

    def forward(self, x_dynamic: torch.Tensor, basin_code: torch.Tensor) -> torch.Tensor:
        b, l, c = x_dynamic.shape
        if l != self.seq_len:
            raise ValueError(f"Expected seq_len={self.seq_len}, got {l}")
        if c != self.num_dynamic_features:
            raise ValueError(
                f"Expected num_dynamic_features={self.num_dynamic_features}, got {c}"
            )

        basin_code = basin_code.long().view(-1)
        x = self.input_projection(x_dynamic) + self.position_embedding
        for layer in self.layers:
            x = layer(x)

        last_value = x[:, -1, :]
        mean_7 = x[:, -7:, :].mean(dim=1)
        mean_30 = x[:, -30:, :].mean(dim=1)
        mean_90 = x[:, -90:, :].mean(dim=1)
        mean_365 = x.mean(dim=1)

        temporal_summary = torch.cat([last_value, mean_7, mean_30, mean_90, mean_365], dim=-1)
        basin_repr = self.basin_embedding(basin_code)
        out = torch.cat([basin_repr, temporal_summary], dim=-1)
        return self.head(out).squeeze(-1)


BASIN_CODE_MODELS = {"itransformer", "autoformer"}


class PrewindowedNpzDataset(Dataset):
    """Dataset for the auto/processed/camels_transformer_windows.npz format.

    The npz holds (X, y, basin_id, target_date) — already pre-windowed, with
    X of shape (N, T, F). Filters to a holdout date range, drops invalid
    targets, and applies feature normalization computed from the pre-holdout
    subset of the *same* npz (so train/test stats are consistent with what the
    autoformer actually saw).
    """

    def __init__(
        self,
        npz_path: Path,
        holdout_start: str,
        holdout_end: str,
        basin_id_to_slot: Dict[str, int],
    ):
        data = np.load(npz_path, allow_pickle=True)
        if "X" not in data.files or "y" not in data.files:
            raise ValueError(
                f"{npz_path} is not in the prewindowed (X, y, basin_id, target_date) "
                "format expected by PrewindowedNpzDataset."
            )
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.float32)
        basin_ids = np.asarray(data["basin_id"]).astype(str)
        dates = np.asarray(data["target_date"]).astype("datetime64[D]")
        feature_columns = [str(c) for c in np.asarray(data["feature_columns"]).tolist()]

        start = np.datetime64(holdout_start)
        end = np.datetime64(holdout_end)
        train_mask = dates < start
        if not train_mask.any():
            raise ValueError(
                f"No pre-holdout samples (target_date < {holdout_start}) in {npz_path}; "
                "cannot compute normalization stats."
            )
        train_flat = X[train_mask].reshape(-1, X.shape[-1])
        feature_mean = train_flat.mean(axis=0).astype(np.float32)
        feature_std = train_flat.std(axis=0).astype(np.float32)
        feature_std = np.where(feature_std < 1e-6, 1.0, feature_std).astype(np.float32)

        hold_mask = (dates >= start) & (dates <= end) & np.isfinite(y) & (y >= 0)
        if not hold_mask.any():
            raise ValueError(
                f"No valid holdout samples in [{holdout_start}, {holdout_end}] in {npz_path}."
            )

        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.feature_columns = feature_columns
        self.X = X[hold_mask]
        self.y = y[hold_mask]
        self.basin_ids = basin_ids[hold_mask]
        self.dates = dates[hold_mask]
        self.basin_id_to_slot = dict(basin_id_to_slot)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        x = (self.X[idx] - self.feature_mean) / self.feature_std
        bid = str(self.basin_ids[idx]).zfill(8)
        slot = self.basin_id_to_slot.get(bid)
        if slot is None:
            slot = self.basin_id_to_slot.setdefault(bid, len(self.basin_id_to_slot))
        return {
            "x": torch.from_numpy(x).float(),
            "y": torch.tensor(float(self.y[idx]), dtype=torch.float32),
            "basin_slot": torch.tensor(int(slot), dtype=torch.long),
            "target_date": str(self.dates[idx]),
        }


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_checkpoint(path: Path) -> Dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _infer_autoformer_hparams_from_state_dict(state_dict: Dict, nhead: int) -> Dict[str, int]:
    d_model = state_dict["seasonal_proj.weight"].shape[0]
    num_features = state_dict["seasonal_proj.weight"].shape[1]
    dim_ff = state_dict["encoder.layers.0.linear1.weight"].shape[0]
    num_layers = sum(
        1 for k in state_dict
        if k.startswith("encoder.layers.") and k.endswith(".self_attn.in_proj_weight")
    )
    if d_model % nhead != 0:
        raise ValueError(
            f"d_model={d_model} is not divisible by nhead={nhead}; pass --autoformer-nhead "
            f"with a value that divides {d_model}."
        )
    return {
        "d_model": int(d_model),
        "nhead": int(nhead),
        "num_layers": int(num_layers),
        "dim_feedforward": int(dim_ff),
        "num_features": int(num_features),
    }


def merge_bare_state_dict_with_metadata(
    state_dict_path: Path,
    metadata_checkpoint_path: Path,
    autoformer_nhead: int,
    autoformer_dropout: float,
    autoformer_moving_avg_kernel: int,
) -> Dict:
    """Bare-state_dict autoformer files in `auto/processed/` carry no metadata.
    Source feature_columns / feature_mean / feature_std / num_basins from a
    co-trained full checkpoint (e.g. the vanilla transformer's ckpt) and infer
    architecture sizes from the state_dict shapes."""
    raw = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected an OrderedDict-like state_dict at {state_dict_path}")
    if "seasonal_proj.weight" not in raw:
        raise ValueError(
            f"State_dict at {state_dict_path} doesn't look like an autoformer "
            "(missing 'seasonal_proj.weight')."
        )

    meta = torch.load(metadata_checkpoint_path, map_location="cpu", weights_only=False)
    if "feature_mean" not in meta or "feature_std" not in meta:
        raise ValueError(
            f"Metadata checkpoint at {metadata_checkpoint_path} is missing "
            "'feature_mean'/'feature_std' — pass a different --autoformer-meta."
        )

    inferred = _infer_autoformer_hparams_from_state_dict(raw, nhead=autoformer_nhead)
    if inferred["num_features"] != len(meta["feature_columns"]):
        raise ValueError(
            f"State_dict expects {inferred['num_features']} input features but "
            f"metadata checkpoint has {len(meta['feature_columns'])} feature_columns."
        )
    if inferred["num_features"] != len(np.asarray(meta["feature_mean"]).reshape(-1)):
        raise ValueError(
            f"State_dict expects {inferred['num_features']} input features but "
            f"feature_mean has shape {np.asarray(meta['feature_mean']).shape}. "
            "Use a metadata checkpoint whose normalization matches (e.g. the vanilla "
            "transformer ckpt)."
        )

    return {
        "model_state_dict": raw,
        "feature_columns": list(meta["feature_columns"]),
        "target_column": meta["target_column"],
        "feature_mean": np.asarray(meta["feature_mean"], dtype=np.float32),
        "feature_std": np.asarray(meta["feature_std"], dtype=np.float32),
        "dynamic_feature_columns": list(
            meta.get("dynamic_feature_columns", meta["feature_columns"][:-1])
        ),
        "basin_code_column": str(meta.get("basin_code_column", "basin_code")),
        "num_basins": int(meta.get("num_basins") or 671),
        "config": dict(meta.get("config", {})),
        "best_hparams": {
            "d_model": inferred["d_model"],
            "nhead": inferred["nhead"],
            "num_layers": inferred["num_layers"],
            "dim_feedforward": inferred["dim_feedforward"],
            "dropout": float(autoformer_dropout),
            "moving_avg_kernel": int(autoformer_moving_avg_kernel),
            "batch_size": int(meta.get("best_hparams", {}).get("batch_size", 128)),
        },
    }


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
    joined_path = next((data_dir / name for name in preferences if (data_dir / name).exists()), None)
    if joined_path is None:
        raise FileNotFoundError(
            f"Could not find joined CAMELS table in {data_dir}. "
            "Pass --data-npz for the indexed .npz data or --data-dir for joined csv/parquet data."
        )

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
        if model_type in BASIN_CODE_MODELS:
            basin_code_values = basin_df[basin_code_column].unique()
            if len(basin_code_values) != 1:
                raise ValueError(f"Basin {basin_id} has multiple basin_code values.")
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
    selected_columns = list(dynamic_feature_columns) if model_type in BASIN_CODE_MODELS else list(feature_columns)
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
        if model_type in BASIN_CODE_MODELS:
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
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                static_metadata = metadata.get("static_attributes", {})
                if list(static_metadata.get("columns", [])) == static_columns:
                    mean = np.asarray(static_metadata.get("mean", []), dtype=np.float64)
                    std = np.asarray(static_metadata.get("std", []), dtype=np.float64)
                    if len(mean) == static_values.shape[1] and len(std) == static_values.shape[1]:
                        static_values = static_values * std + mean
                        static_scale = "raw"
            except Exception:
                pass
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
        if self.model_type in BASIN_CODE_MODELS:
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
            if model_type in BASIN_CODE_MODELS:
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
    if model_type == "autoformer":
        return AutoformerRegressor(
            seq_len=lookback_days,
            num_dynamic_features=len(checkpoint["dynamic_feature_columns"]),
            num_basins=int(checkpoint["num_basins"]),
            d_model=int(hparams["d_model"]),
            nhead=int(hparams["nhead"]),
            num_layers=int(hparams["num_layers"]),
            dim_feedforward=int(hparams["dim_feedforward"]),
            dropout=float(hparams["dropout"]),
            moving_avg_kernel=int(hparams.get("moving_avg_kernel", 25)),
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
        description="Apply TFT-style prediction analysis plots to iTransformer, Autoformer, or vanilla Transformer checkpoints."
    )
    parser.add_argument(
        "--model-type",
        choices=["itransformer", "vanilla", "autoformer"],
        required=True,
    )
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
    parser.add_argument(
        "--autoformer-meta",
        type=Path,
        default=None,
        help=(
            "For --model-type=autoformer when --checkpoint is a bare state_dict "
            "(e.g. files under CAMELS_data_load-auto/processed/). Path to a full "
            "checkpoint to source feature_columns / feature_mean / feature_std / "
            "num_basins from. Defaults to the vanilla transformer checkpoint."
        ),
    )
    parser.add_argument(
        "--autoformer-nhead",
        type=int,
        default=4,
        help="Attention heads for the autoformer (cannot be inferred from state_dict).",
    )
    parser.add_argument(
        "--autoformer-dropout",
        type=float,
        default=0.1,
        help="Dropout used at training time (cannot be inferred from state_dict).",
    )
    parser.add_argument(
        "--autoformer-moving-avg-kernel",
        type=int,
        default=25,
        help="Moving-average kernel for the seasonal/trend decomposition.",
    )
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

    holdout_start = config.get("INDEPENDENT_VAL_START", "2008-01-01")
    holdout_end = config.get("INDEPENDENT_VAL_END", "2014-12-31")

    prewindowed = False
    if args.data_npz is not None:
        npz_files = set(np.load(args.data_npz, allow_pickle=True).files)
        prewindowed = "X" in npz_files and "y" in npz_files

    if prewindowed:
        print(f"Reading prewindowed (X, y) dataset from: {args.data_npz}", flush=True)
        npz = np.load(args.data_npz, allow_pickle=True)
        basin_ids = sorted({str(b).zfill(8) for b in np.asarray(npz["basin_id"]).tolist()})
        basin_id_to_slot = {bid: i for i, bid in enumerate(basin_ids)}
        dataset = PrewindowedNpzDataset(
            args.data_npz,
            holdout_start=holdout_start,
            holdout_end=holdout_end,
            basin_id_to_slot=basin_id_to_slot,
        )
        static_values = None
        static_columns = []
        static_scale = "z-score"
        print(
            f"  feature_mean (computed from pre-{holdout_start} subset of npz): "
            f"{np.asarray(dataset.feature_mean).round(3).tolist()}",
            flush=True,
        )
    else:
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
            holdout_start=holdout_start,
            holdout_end=holdout_end,
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
    if len(dataset) == 0:
        raise ValueError("No holdout samples found for the requested data/config.")

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
