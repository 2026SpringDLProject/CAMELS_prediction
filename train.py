import argparse
import copy
import json
import math
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from data.prepare_camels_data import (
    DataPrepConfig,
    config_to_dict as data_prep_config_to_dict,
    prepare_camels_training_data,
)
from model.lstm_baseline import LSTMBaseline
from model.tftransformer import SimpleTFT

CONFIG_DIR = PROJECT_DIR / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "train.json"
DECODER_FUTURE_KNOWN_FEATURES = ("Dayl(s)", "doy_sin", "doy_cos")
EXPECTED_DATASET_FORMAT = "flat_indexed_v3"
EXPECTED_TARGET_TRANSFORM = "identity"


@dataclass
class TrainConfig:
    data_dir: Path
    processed_dir: Path
    data_path: Path
    output_dir: Path
    forcing_product: str = "nldas"
    allow_daymet_model_output_fallback: bool = True
    basin_ids: List[str] = field(default_factory=list)
    start_date: Optional[str] = "1980-01-01"
    end_date: Optional[str] = "2014-12-31"
    target_unit: str = "mm/day"
    include_past_streamflow: bool = False
    lookback_days: int = 365
    forecast_horizon_days: int = 1
    stride_days: int = 1
    max_windows: Optional[int] = None
    write_flat_files: bool = False
    rebuild_dataset: bool = False
    loss_name: str = "mse"
    mse_warmup_epochs: int = 0
    mse_weight: float = 0.5
    batch_size: int = 64
    epochs: int = 30
    early_stopping_patience: int = 8
    early_stopping_min_delta: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    warmup_start_factor: float = 0.1
    split_strategy: str = "ratio"
    cv_num_folds: int = 4
    cv_train_ratio: float = 0.8
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    random_seed: int = 42
    num_workers: int = 0
    model_name: str = "tft"
    d_model: int = 64
    lstm_hidden: int = 64
    n_heads: int = 4
    dropout: float = 0.1
    prediction_length: int = 1


class CamelsWindowDataset(Dataset):
    def __init__(self, npz_path: Path):
        data = np.load(npz_path, allow_pickle=True)
        self.dataset_format = (
            str(data["dataset_format"][0]) if "dataset_format" in data.files else "dense_windows_v1"
        )
        self.target_transform = (
            str(data["target_transform"][0])
            if "target_transform" in data.files
            else "identity"
        )
        self.feature_columns = (
            data["feature_columns"].tolist() if "feature_columns" in data.files else None
        )
        self.future_known_feature_columns: List[str] = []
        self.future_known_feature_indices: Optional[np.ndarray] = None
        self.target_date = data["target_date"] if "target_date" in data.files else None
        self.basin_id = data["basin_id"] if "basin_id" in data.files else None
        self.row_date = data["row_date"] if "row_date" in data.files else None
        self.row_basin_index = (
            np.asarray(data["row_basin_index"], dtype=np.int64)
            if "row_basin_index" in data.files
            else None
        )
        self.normalize_feature_indices = (
            np.asarray(data["normalize_feature_indices"], dtype=np.int64)
            if "normalize_feature_indices" in data.files
            else np.empty(0, dtype=np.int64)
        )
        self.static_features = (
            np.asarray(data["static_features"], dtype=np.float32)
            if "static_features" in data.files
            else np.empty((0, 0), dtype=np.float32)
        )
        self.static_feature_columns = (
            data["static_feature_columns"].tolist()
            if "static_feature_columns" in data.files
            else []
        )
        self.num_static_features = (
            self.static_features.shape[1] if self.static_features.ndim == 2 else 0
        )

        if self.feature_columns is not None:
            feature_index = {name: idx for idx, name in enumerate(self.feature_columns)}
            self.future_known_feature_columns = [
                name for name in DECODER_FUTURE_KNOWN_FEATURES if name in feature_index
            ]
            if self.future_known_feature_columns:
                self.future_known_feature_indices = np.asarray(
                    [feature_index[name] for name in self.future_known_feature_columns],
                    dtype=np.int64,
                )

        if "X" in data.files:
            self.mode = "dense"
            self.future_known_feature_columns = []
            self.future_known_feature_indices = None
            self.X = torch.from_numpy(data["X"]).float()
            y = np.asarray(data["y"], dtype=np.float32)
            if y.ndim == 1:
                y = y[:, None]
            self.y = torch.from_numpy(y).float()
            self.num_features = self.X.shape[-1]
            self.prediction_length = self.y.shape[-1]

            if self.X.ndim != 3:
                raise ValueError(f"Expected X to have shape [N, T, F], got {tuple(self.X.shape)}")
            if self.y.ndim != 2:
                raise ValueError(f"Expected y to have shape [N, H], got {tuple(self.y.shape)}")
            if len(self.X) != len(self.y):
                raise ValueError("X and y must contain the same number of samples")
            return

        if "features" in data.files and "window_start" in data.files:
            self.mode = "indexed"
            self.features = np.asarray(data["features"], dtype=np.float32)
            self.targets = np.asarray(data["targets"], dtype=np.float32)
            self.window_start = np.asarray(data["window_start"], dtype=np.int64)
            self.lookback_days = int(np.asarray(data["lookback_days"]).reshape(-1)[0])
            self.forecast_horizon_days = int(
                np.asarray(data["forecast_horizon_days"]).reshape(-1)[0]
            )
            self.prediction_length = int(np.asarray(data["prediction_length"]).reshape(-1)[0])
            self.window_basin_index = (
                np.asarray(data["window_basin_index"], dtype=np.int64)
                if "window_basin_index" in data.files
                else None
            )
            self.basin_ids_in_model = (
                data["basin_ids_in_model"] if "basin_ids_in_model" in data.files else None
            )
            self.num_features = self.features.shape[-1]
            if (
                self.num_static_features > 0
                and self.basin_ids_in_model is not None
                and self.static_features.shape[0] != len(self.basin_ids_in_model)
            ):
                raise ValueError(
                    "static_features row count must match basin_ids_in_model length; "
                    f"got {self.static_features.shape[0]} and {len(self.basin_ids_in_model)}"
                )

            if self.features.ndim != 2:
                raise ValueError(
                    f"Expected flat features to have shape [num_rows, num_features], got {tuple(self.features.shape)}"
                )
            if self.targets.ndim != 1:
                raise ValueError(
                    f"Expected flat targets to have shape [num_rows], got {tuple(self.targets.shape)}"
                )
            if len(self.window_start) == 0:
                raise ValueError("Indexed dataset contains zero windows")
            return

        raise ValueError(
            f"Unsupported dataset format in {npz_path}. Expected keys for dense or indexed dataset."
        )

    def __len__(self) -> int:
        if self.mode == "dense":
            return len(self.y)
        return len(self.window_start)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "dense":
            item = {"x": self.X[idx], "y": self.y[idx]}
            if self.basin_id is not None:
                item["basin_id"] = self.basin_id[idx]
            if self.target_date is not None:
                item["target_date"] = self.target_date[idx]
            return item

        start_idx = int(self.window_start[idx])
        end_idx = start_idx + self.lookback_days
        target_start_idx = end_idx + self.forecast_horizon_days - 1
        target_end_idx = target_start_idx + self.prediction_length

        x = torch.from_numpy(self.features[start_idx:end_idx]).float()
        y = torch.from_numpy(self.targets[target_start_idx:target_end_idx]).float()
        item = {"x": x, "y": y}
        if self.future_known_feature_indices is not None:
            future_known = self.features[
                target_start_idx:target_end_idx, self.future_known_feature_indices
            ]
            item["future_known"] = torch.from_numpy(future_known).float()
        if self.window_basin_index is not None:
            basin_idx = int(self.window_basin_index[idx])
            item["basin_slot"] = torch.tensor(basin_idx, dtype=torch.long)
            if self.num_static_features > 0:
                item["static"] = torch.from_numpy(self.static_features[basin_idx]).float()
            if self.basin_ids_in_model is not None:
                item["basin_id"] = self.basin_ids_in_model[basin_idx]
        if self.target_date is not None:
            item["target_date"] = self.target_date[idx]
        return item


class NSELoss(nn.Module):
    """Kratzert et al. (2019) basin-averaged NSE* loss.

    Reference:
        Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G.
        (2019). "Towards learning universal, regional, and local hydrological behaviors
        via machine learning applied to large-sample datasets." HESS, 23, 5089-5110.

    Implements (Eq. 1 in the paper):
        L = (1/B) * sum_b (1/N_b) * sum_n (y_n - y_hat_n)^2 / (s_b + eps)^2
    where:
        y, y_hat are observed and predicted streamflow in physical mm/day,
        s_b is the std of training observations for basin b (mm/day) — precomputed
            from the training split only, so it is a fixed, basin-specific constant
            (this is what makes the loss stable under mini-batch shuffling),
        eps is a small constant (paper default: 0.1).

    Because this codebase trains the model in z-scored log1p space for optimization
    stability, we denormalize predictions to mm/day inside the loss and apply NSE*
    in physical units exactly as the paper specifies. A clamp on the pre-expm1 value
    prevents runaway gradients early in training when predictions are far off, and
    the denormalized prediction is clamped to non-negative runoff.
    """

    def __init__(self, eps: float = 0.1, log_clip: float = 10.0):
        super().__init__()
        self.eps = eps
        self.log_clip = log_clip

    def forward(
        self,
        pred: torch.Tensor,
        raw_target: torch.Tensor,
        basin_slot: Optional[torch.Tensor] = None,
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
        target_std_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if (
            basin_slot is None
            or target_mean is None
            or target_std is None
            or target_std_raw is None
        ):
            # No per-basin info; fall back to MSE on whatever space pred is in.
            return ((pred - raw_target) ** 2).mean()

        sigma_log = target_std[basin_slot]
        mu_log = target_mean[basin_slot]
        sigma_raw = target_std_raw[basin_slot]
        if pred.ndim == 2:
            sigma_log = sigma_log.unsqueeze(-1)
            mu_log = mu_log.unsqueeze(-1)
            sigma_raw = sigma_raw.unsqueeze(-1)

        pred_log = pred * sigma_log + mu_log
        pred_log = torch.clamp(pred_log, max=self.log_clip, min=-20.0)
        pred_phys = torch.clamp(torch.expm1(pred_log), min=0.0)

        squared_error = (pred_phys - raw_target) ** 2
        denom = (sigma_raw + self.eps) ** 2
        element_loss = squared_error / denom

        basin_losses = []
        for slot in torch.unique(basin_slot):
            basin_losses.append(element_loss[basin_slot == slot].mean())
        return torch.stack(basin_losses).mean()


class GlobalNSELoss(nn.Module):
    """Vanilla-transformer style global NSE loss over the current mini-batch.

    This uses the same form as the baseline script:
        sum((pred - target)^2) / sum((target - target.mean())^2)

    In this training script pred and target are in the normalized log1p target
    space, so this option matches the vanilla loss shape while leaving the
    physical-unit NSE metrics unchanged for reporting.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        numerator = torch.sum((pred_flat - target_flat) ** 2)
        denominator = torch.sum((target_flat - torch.mean(target_flat)) ** 2) + self.eps
        return numerator / denominator


class MixedLoss(nn.Module):
    """Weighted MSE (z-scored log1p space) + Kratzert NSE* (physical mm/day).

    L = mse_weight * MSE(pred, target_z) + (1 - mse_weight) * NSE*(pred, raw_target)

    MSE provides a smooth, well-conditioned signal early in training; NSE*
    aligns with the hydrology evaluation metric. With LR warmup, MSE-heavy
    mixing in early epochs followed by an NSE-heavy schedule is a common
    recipe — for the simple version here we use a fixed weight.
    """

    def __init__(self, mse_weight: float = 0.5, eps: float = 0.1, log_clip: float = 10.0):
        super().__init__()
        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError(f"mse_weight must be in [0, 1], got {mse_weight}")
        self.mse_weight = float(mse_weight)
        self.nse = NSELoss(eps=eps, log_clip=log_clip)

    def forward(
        self,
        pred: torch.Tensor,
        target_z: torch.Tensor,
        raw_target: torch.Tensor,
        basin_slot: Optional[torch.Tensor] = None,
        target_mean: Optional[torch.Tensor] = None,
        target_std: Optional[torch.Tensor] = None,
        target_std_raw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mse_term = ((pred - target_z) ** 2).mean()
        if (
            basin_slot is None
            or target_mean is None
            or target_std is None
            or target_std_raw is None
        ):
            return mse_term
        nse_term = self.nse(
            pred,
            raw_target,
            basin_slot=basin_slot,
            target_mean=target_mean,
            target_std=target_std,
            target_std_raw=target_std_raw,
        )
        return self.mse_weight * mse_term + (1.0 - self.mse_weight) * nse_term


def denormalize_targets(
    values: torch.Tensor,
    basin_slot: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    log_clip: Optional[float] = None,
) -> torch.Tensor:
    """Invert log1p+z-score: q = expm1(z * std + mean), clipped to 0."""
    mean = target_mean[basin_slot]
    std = target_std[basin_slot]
    if values.ndim == 2:
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    log_values = values * std + mean
    if log_clip is not None:
        log_values = torch.clamp(log_values, max=log_clip)
    q = torch.expm1(log_values)
    return torch.clamp(q, min=0.0)


def normalize_targets(
    values: torch.Tensor,
    basin_slot: torch.Tensor,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
) -> torch.Tensor:
    mean = target_mean[basin_slot]
    std = target_std[basin_slot]
    if values.ndim == 2:
        mean = mean.unsqueeze(-1)
        std = std.unsqueeze(-1)
    log_values = torch.log1p(torch.clamp(values, min=0.0))
    return (log_values - mean) / std


def normalize_features(
    values: Optional[torch.Tensor],
    basin_slot: Optional[torch.Tensor],
    feature_mean: Optional[torch.Tensor],
    feature_std: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if values is None or basin_slot is None or feature_mean is None or feature_std is None:
        return values
    mean = feature_mean[basin_slot]
    std = feature_std[basin_slot]
    if values.ndim == 3:
        mean = mean.unsqueeze(1)
        std = std.unsqueeze(1)
    return (values - mean) / std


def compute_train_normalization_stats(
    dataset: CamelsWindowDataset,
    train_idx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-basin train-only stats.

    Returns (feature_mean, feature_std, target_mean_log, target_std_log, target_std_raw).
    target_*_log are stats of log1p(target) used for input normalization.
    target_std_raw is std of raw mm/day target — used as the NSE* denominator.
    """
    if dataset.mode != "indexed":
        raise ValueError("Train-only normalization stats require an indexed dataset.")
    if dataset.window_basin_index is None or dataset.row_basin_index is None or dataset.row_date is None:
        raise ValueError("Dataset is missing row/window basin indices required for train-only normalization.")
    if dataset.basin_ids_in_model is None:
        raise ValueError("Dataset is missing basin_ids_in_model required for train-only normalization.")

    n_basins = len(dataset.basin_ids_in_model)
    n_features = dataset.num_features
    feature_mean = np.zeros((n_basins, n_features), dtype=np.float32)
    feature_std = np.ones((n_basins, n_features), dtype=np.float32)
    target_mean = np.zeros(n_basins, dtype=np.float32)
    target_std = np.ones(n_basins, dtype=np.float32)
    target_std_raw = np.ones(n_basins, dtype=np.float32)

    row_dates = np.asarray(dataset.row_date).astype("datetime64[D]")
    target_dates = np.asarray(dataset.target_date).astype("datetime64[D]")
    train_target_dates = target_dates[train_idx]
    if len(train_target_dates) == 0:
        raise ValueError("Training split is empty; cannot compute normalization statistics.")

    global_cutoff = train_target_dates.max()
    train_window_basin = dataset.window_basin_index[train_idx]
    basin_cutoffs = np.full(n_basins, global_cutoff, dtype="datetime64[D]")

    for basin_slot in range(n_basins):
        basin_mask = train_window_basin == basin_slot
        if np.any(basin_mask):
            basin_cutoffs[basin_slot] = train_target_dates[basin_mask].max()
        else:
            raise ValueError(
                f"Basin slot {basin_slot} has no training windows; cannot compute "
                "train-only normalization stats without leaking val/test data."
            )

    for basin_slot in range(n_basins):
        row_mask = (dataset.row_basin_index == basin_slot) & (row_dates <= basin_cutoffs[basin_slot])
        if not np.any(row_mask):
            raise ValueError(
                f"Basin slot {basin_slot} has no training rows up to its cutoff "
                f"{basin_cutoffs[basin_slot]}; cannot compute normalization stats."
            )

        basin_features = dataset.features[row_mask]
        if dataset.normalize_feature_indices.size > 0:
            block = basin_features[:, dataset.normalize_feature_indices]
            mu = block.mean(axis=0)
            sigma = block.std(axis=0)
            sigma = np.where(sigma < 1e-6, 1.0, sigma).astype(np.float32)
            feature_mean[basin_slot, dataset.normalize_feature_indices] = mu.astype(np.float32)
            feature_std[basin_slot, dataset.normalize_feature_indices] = sigma

        basin_targets = dataset.targets[row_mask]
        log_targets = np.log1p(np.maximum(basin_targets, 0.0))
        tgt_mean = float(log_targets.mean())
        tgt_std = float(log_targets.std())
        if tgt_std < 1e-6:
            tgt_std = 1.0
        target_mean[basin_slot] = tgt_mean
        target_std[basin_slot] = tgt_std

        # Raw mm/day std for Kratzert NSE* denominator.
        raw_std = float(np.maximum(basin_targets, 0.0).std())
        if raw_std < 1e-6:
            raw_std = 1.0
        target_std_raw[basin_slot] = raw_std

    return feature_mean, feature_std, target_mean, target_std, target_std_raw


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_num_basins(X: torch.Tensor, basin_idx: int) -> int:
    basin_codes = X[:, :, basin_idx].round().long()
    first_step_codes = basin_codes[:, 0]
    if not torch.equal(basin_codes, first_step_codes.unsqueeze(1).expand_as(basin_codes)):
        raise ValueError("Found a window whose basin_code changes across time steps")
    if torch.any(first_step_codes < 0):
        raise ValueError("basin_code must be non-negative")
    return int(first_step_codes.max().item()) + 1


def infer_num_basins_from_dataset(dataset: CamelsWindowDataset, basin_idx: int) -> int:
    if dataset.mode == "dense":
        return infer_num_basins(dataset.X, basin_idx)

    basin_codes = np.rint(dataset.features[:, basin_idx]).astype(np.int64)
    if np.any(basin_codes < 0):
        raise ValueError("basin_code must be non-negative")
    return int(basin_codes.max()) + 1


def temporal_split_indices(
    dataset: CamelsWindowDataset, cfg: TrainConfig
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_ratio = cfg.train_ratio + cfg.val_ratio + cfg.test_ratio
    if not math.isclose(total_ratio, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"train/val/test ratios must sum to 1.0, got {cfg.train_ratio}, {cfg.val_ratio}, {cfg.test_ratio}"
        )

    n_samples = len(dataset)
    if n_samples < 3:
        raise ValueError("Need at least 3 samples to build train/val/test splits")

    if dataset.target_date is not None:
        dates = np.asarray(dataset.target_date).astype("datetime64[D]")
        order = np.argsort(dates, kind="stable")
    else:
        rng = np.random.default_rng(cfg.random_seed)
        order = rng.permutation(n_samples)

    train_end = max(1, int(n_samples * cfg.train_ratio))
    val_end = min(n_samples - 1, train_end + max(1, int(n_samples * cfg.val_ratio)))

    train_idx = order[:train_end]
    val_idx = order[train_end:val_end]
    test_idx = order[val_end:]

    if len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError("Validation and test splits must both contain at least one sample")

    return train_idx, val_idx, test_idx


def expanding_cv_split_indices(
    dataset: CamelsWindowDataset,
    cfg: TrainConfig,
) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
    if dataset.target_date is None:
        raise ValueError("Expanding-window CV requires target_date in the prepared dataset.")
    if dataset.window_basin_index is None:
        raise ValueError(
            "Per-basin expanding-window CV requires window_basin_index in the prepared dataset."
        )
    if cfg.cv_num_folds < 1:
        raise ValueError(f"cv_num_folds must be >= 1, got {cfg.cv_num_folds}")
    if not 0.0 < cfg.cv_train_ratio < 1.0:
        raise ValueError(f"cv_train_ratio must be in (0, 1), got {cfg.cv_train_ratio}")

    dates = np.asarray(dataset.target_date).astype("datetime64[D]")
    basin_slots = np.asarray(dataset.window_basin_index, dtype=np.int64)
    if len(dates) != len(basin_slots):
        raise ValueError("target_date and window_basin_index must have the same length.")
    if len(dates) < cfg.cv_num_folds + 2:
        raise ValueError("Not enough samples to build expanding-window CV splits.")

    basin_ids = (
        [str(value) for value in dataset.basin_ids_in_model]
        if dataset.basin_ids_in_model is not None
        else None
    )
    basin_chunks: Dict[int, List[np.ndarray]] = {}
    pre_holdout_parts: List[np.ndarray] = []
    holdout_parts: List[np.ndarray] = []
    skipped: List[str] = []

    for basin_slot in np.unique(basin_slots):
        basin_idx = np.flatnonzero(basin_slots == basin_slot)
        basin_order = basin_idx[np.argsort(dates[basin_idx], kind="stable")]
        min_needed = cfg.cv_num_folds + 2
        label = (
            basin_ids[int(basin_slot)]
            if basin_ids is not None and int(basin_slot) < len(basin_ids)
            else str(int(basin_slot))
        )
        if len(basin_order) < min_needed:
            skipped.append(f"{label}({len(basin_order)} windows)")
            continue

        pre_end = int(len(basin_order) * cfg.cv_train_ratio)
        pre_end = max(cfg.cv_num_folds + 1, min(len(basin_order) - 1, pre_end))
        basin_pre = basin_order[:pre_end]
        basin_holdout = basin_order[pre_end:]
        chunks = [chunk for chunk in np.array_split(basin_pre, cfg.cv_num_folds + 1)]
        if len(basin_holdout) == 0 or any(len(chunk) == 0 for chunk in chunks):
            skipped.append(f"{label}({len(basin_order)} windows)")
            continue

        basin_chunks[int(basin_slot)] = chunks
        pre_holdout_parts.append(basin_pre)
        holdout_parts.append(basin_holdout)

    if skipped:
        preview = ", ".join(skipped[:10])
        suffix = "..." if len(skipped) > 10 else ""
        raise ValueError(
            "Some basins do not have enough windows for per-basin expanding-window CV: "
            f"{preview}{suffix}. Reduce cv_num_folds or use a denser window stride."
        )
    if not basin_chunks:
        raise ValueError("No basins had enough samples to build expanding-window CV splits.")

    folds: List[Dict[str, Any]] = []
    for fold_id in range(cfg.cv_num_folds):
        train_parts = []
        val_parts = []
        for chunks in basin_chunks.values():
            train_parts.append(np.concatenate(chunks[: fold_id + 1]))
            val_parts.append(chunks[fold_id + 1])
        train_idx = np.concatenate(train_parts).astype(np.int64)
        val_idx = np.concatenate(val_parts).astype(np.int64)
        train_idx = train_idx[np.argsort(dates[train_idx], kind="stable")]
        val_idx = val_idx[np.argsort(dates[val_idx], kind="stable")]
        folds.append(
            {
                "fold_id": fold_id,
                "train_idx": train_idx,
                "val_idx": val_idx,
                "n_train_basins": len(basin_chunks),
                "n_val_basins": len(basin_chunks),
                "split_unit": "per_basin_window_order",
            }
        )

    pre_holdout_idx = np.concatenate(pre_holdout_parts).astype(np.int64)
    holdout_idx = np.concatenate(holdout_parts).astype(np.int64)
    pre_holdout_idx = pre_holdout_idx[np.argsort(dates[pre_holdout_idx], kind="stable")]
    holdout_idx = holdout_idx[np.argsort(dates[holdout_idx], kind="stable")]
    return folds, pre_holdout_idx, holdout_idx


def collate_batch(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    x = torch.stack([item["x"] for item in batch], dim=0)
    y = torch.stack([item["y"] for item in batch], dim=0)
    output = {"x": x, "y": y}
    if "future_known" in batch[0]:
        output["future_known"] = torch.stack([item["future_known"] for item in batch], dim=0)
    if "basin_slot" in batch[0]:
        output["basin_slot"] = torch.stack([item["basin_slot"] for item in batch], dim=0)
    if "static" in batch[0]:
        output["static"] = torch.stack([item["static"] for item in batch], dim=0)

    if "basin_id" in batch[0]:
        output["basin_id"] = [item["basin_id"] for item in batch]
    if "target_date" in batch[0]:
        output["target_date"] = [item["target_date"] for item in batch]
    return output


def make_loader(
    dataset: CamelsWindowDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )


def format_targets(y: torch.Tensor, prediction_length: int) -> torch.Tensor:
    if prediction_length == 1:
        return y.squeeze(-1)
    return y


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 0.0,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
    future_feature_mean: Optional[torch.Tensor] = None,
    future_feature_std: Optional[torch.Tensor] = None,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
    target_std_raw: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    loss_sum = 0.0
    mae_sum = 0.0
    mse_sum = 0.0
    sample_count = 0
    sq_err_sum = 0.0
    abs_err_sum = 0.0
    element_count = 0
    target_sum = 0.0
    target_sum_sq = 0.0

    # Per-basin NSE accumulators (in original mm/day units).
    n_basins = target_mean.numel() if target_mean is not None else 0
    track_per_basin = n_basins > 0
    per_basin_sse = torch.zeros(max(n_basins, 1), dtype=torch.float64, device=device)
    per_basin_sum = torch.zeros(max(n_basins, 1), dtype=torch.float64, device=device)
    per_basin_sum_sq = torch.zeros(max(n_basins, 1), dtype=torch.float64, device=device)
    per_basin_count = torch.zeros(max(n_basins, 1), dtype=torch.float64, device=device)
    track_epoch_nse_loss = (
        isinstance(criterion, (NSELoss, MixedLoss))
        and track_per_basin
        and target_std_raw is not None
    )
    per_basin_nse_loss_sum = torch.zeros(max(n_basins, 1), dtype=torch.float64, device=device)
    per_basin_nse_loss_count = torch.zeros(max(n_basins, 1), dtype=torch.float64, device=device)
    z_mse_sum = 0.0
    z_mse_count = 0

    for batch in loader:
        x = batch["x"].to(device)
        y_raw = batch["y"].to(device)
        future_known = batch.get("future_known")
        static_features = batch.get("static")
        basin_slot = batch.get("basin_slot")
        if basin_slot is not None:
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

        raw_target = format_targets(y_raw, model.prediction_length)
        if target_mean is not None and target_std is not None and basin_slot is not None:
            target = normalize_targets(raw_target, basin_slot, target_mean, target_std)
        else:
            target = raw_target

        with torch.set_grad_enabled(is_train):
            pred, _ = model(x, future_known=future_known, static_features=static_features)
            if isinstance(criterion, MixedLoss):
                loss = criterion(
                    pred,
                    target,
                    raw_target,
                    basin_slot=basin_slot,
                    target_mean=target_mean,
                    target_std=target_std,
                    target_std_raw=target_std_raw,
                )
            elif isinstance(criterion, NSELoss):
                loss = criterion(
                    pred,
                    raw_target,
                    basin_slot=basin_slot,
                    target_mean=target_mean,
                    target_std=target_std,
                    target_std_raw=target_std_raw,
                )
            else:
                loss = criterion(pred, target)

        if isinstance(criterion, MixedLoss):
            z_diff = (pred.detach() - target.detach()).reshape(-1).double()
            z_mse_sum += torch.sum(z_diff * z_diff).item()
            z_mse_count += z_diff.numel()

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = x.size(0)
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

        pred_detached = pred.detach()
        raw_target_detached = raw_target.detach()

        loss_log_clip = None
        if isinstance(criterion, MixedLoss):
            loss_log_clip = criterion.nse.log_clip
        elif isinstance(criterion, NSELoss):
            loss_log_clip = criterion.log_clip

        if target_mean is not None and target_std is not None and basin_slot is not None:
            pred_phys = denormalize_targets(pred_detached, basin_slot, target_mean, target_std)
            pred_phys_for_loss = denormalize_targets(
                pred_detached,
                basin_slot,
                target_mean,
                target_std,
                log_clip=loss_log_clip,
            )
            target_phys = raw_target_detached
        else:
            pred_phys = pred_detached
            pred_phys_for_loss = pred_detached
            target_phys = raw_target_detached

        if pred_phys.ndim == 1:
            pred_phys = pred_phys.unsqueeze(-1)
            pred_phys_for_loss = pred_phys_for_loss.unsqueeze(-1)
            target_phys = target_phys.unsqueeze(-1)

        horizon = pred_phys.size(-1)
        pred_flat = pred_phys.reshape(-1).double()
        pred_loss_flat = pred_phys_for_loss.reshape(-1).double()
        target_flat = target_phys.reshape(-1).double()
        diff = pred_flat - target_flat
        loss_diff = pred_loss_flat - target_flat

        abs_err_sum += torch.sum(torch.abs(diff)).item()
        sq_err_sum += torch.sum(diff * diff).item()
        element_count += target_flat.numel()
        target_sum += torch.sum(target_flat).item()
        target_sum_sq += torch.sum(target_flat * target_flat).item()

        mae_sum += torch.mean(torch.abs(diff)).item() * batch_size
        mse_sum += torch.mean(diff * diff).item() * batch_size

        if track_per_basin and basin_slot is not None:
            slot_per_elem = basin_slot.unsqueeze(-1).expand(-1, horizon).reshape(-1)
            per_basin_sse.scatter_add_(0, slot_per_elem, diff * diff)
            per_basin_sum.scatter_add_(0, slot_per_elem, target_flat)
            per_basin_sum_sq.scatter_add_(0, slot_per_elem, target_flat * target_flat)
            per_basin_count.scatter_add_(0, slot_per_elem, torch.ones_like(target_flat))
            if track_epoch_nse_loss:
                if isinstance(criterion, MixedLoss):
                    nse_eps = criterion.nse.eps
                else:
                    nse_eps = criterion.eps
                denom = (target_std_raw[slot_per_elem].double() + nse_eps) ** 2
                element_nse_loss = (loss_diff * loss_diff) / denom
                per_basin_nse_loss_sum.scatter_add_(0, slot_per_elem, element_nse_loss)
                per_basin_nse_loss_count.scatter_add_(
                    0, slot_per_elem, torch.ones_like(element_nse_loss)
                )

    mean_loss = loss_sum / sample_count
    if track_epoch_nse_loss:
        valid_loss = per_basin_nse_loss_count > 0
        if bool(valid_loss.any()):
            epoch_nse_loss = (
                per_basin_nse_loss_sum[valid_loss] / per_basin_nse_loss_count[valid_loss]
            ).mean().item()
            if isinstance(criterion, MixedLoss):
                if z_mse_count > 0:
                    epoch_mse_loss = z_mse_sum / z_mse_count
                    mean_loss = (
                        criterion.mse_weight * epoch_mse_loss
                        + (1.0 - criterion.mse_weight) * epoch_nse_loss
                    )
            else:
                mean_loss = epoch_nse_loss
    overall_mae = abs_err_sum / max(element_count, 1)
    overall_rmse = math.sqrt(max(sq_err_sum / max(element_count, 1), 0.0))
    global_sst = target_sum_sq - (target_sum * target_sum / max(element_count, 1))
    global_nse = 1.0 - sq_err_sum / global_sst if global_sst > 1e-8 else float("nan")

    per_basin_nse_arr: Optional[np.ndarray] = None
    median_nse = float("nan")
    if track_per_basin:
        counts = per_basin_count.clamp(min=1.0)
        basin_means = per_basin_sum / counts
        basin_sst = per_basin_sum_sq - counts * basin_means * basin_means
        valid = (per_basin_count > 1) & (basin_sst > 1e-8)
        _arr = np.full(n_basins, float("nan"), dtype=np.float32)
        if bool(valid.any()):
            per_basin_nse_vals = 1.0 - per_basin_sse[valid] / basin_sst[valid]
            nse = per_basin_nse_vals.mean().item()
            median_nse = per_basin_nse_vals.median().item()
            _arr[valid.cpu().numpy()] = per_basin_nse_vals.cpu().float().numpy()
        else:
            nse = float("nan")
        per_basin_nse_arr = _arr
    else:
        nse = float("nan")

    result: Dict[str, Any] = {
        "loss": mean_loss,
        "mae": overall_mae,
        "rmse": overall_rmse,
        "nse": nse,
        "median_nse": median_nse,
        "global_nse": global_nse,
        "mean_nse_loss": 1.0 - nse if not math.isnan(nse) else float("nan"),
        "median_nse_loss": 1.0 - median_nse if not math.isnan(median_nse) else float("nan"),
        "global_nse_loss": 1.0 - global_nse if not math.isnan(global_nse) else float("nan"),
    }
    if per_basin_nse_arr is not None:
        result["per_basin_nse"] = per_basin_nse_arr
    return result


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    feature_mean: Optional[torch.Tensor] = None,
    feature_std: Optional[torch.Tensor] = None,
    future_feature_mean: Optional[torch.Tensor] = None,
    future_feature_std: Optional[torch.Tensor] = None,
    target_mean: Optional[torch.Tensor] = None,
    target_std: Optional[torch.Tensor] = None,
    target_std_raw: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    with torch.no_grad():
        return run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer=None,
            feature_mean=feature_mean,
            feature_std=feature_std,
            future_feature_mean=future_feature_mean,
            future_feature_std=future_feature_std,
            target_mean=target_mean,
            target_std=target_std,
            target_std_raw=target_std_raw,
        )


def build_model(dataset: CamelsWindowDataset, cfg: TrainConfig) -> nn.Module:
    num_features = dataset.num_features
    basin_idx = num_features - 1

    num_real_features = num_features - 1
    num_basins = infer_num_basins_from_dataset(dataset, basin_idx)

    if dataset.prediction_length != cfg.prediction_length:
        raise ValueError(
            f"prediction_length={cfg.prediction_length} but dataset was prepared with prediction_length={dataset.prediction_length}"
        )

    model_name = cfg.model_name.lower()
    common_kwargs = {
        "num_real_features": num_real_features,
        "num_basins": num_basins,
        "prediction_length": cfg.prediction_length,
        "num_future_known_features": len(dataset.future_known_feature_columns),
        "num_static_features": dataset.num_static_features,
        "d_model": cfg.d_model,
        "lstm_hidden": cfg.lstm_hidden,
        "dropout": cfg.dropout,
    }
    if model_name in {"tft", "simple_tft", "simple-tft"}:
        return SimpleTFT(
            **common_kwargs,
            n_heads=cfg.n_heads,
        )
    if model_name in {"lstm", "lstm_baseline", "lstm-baseline"}:
        return LSTMBaseline(**common_kwargs)
    raise ValueError(
        f"Unsupported model_name={cfg.model_name!r}. Expected 'tft' or 'lstm'."
    )


def build_criterion(cfg: TrainConfig) -> nn.Module:
    loss_name = cfg.loss_name.lower()
    if loss_name == "mse":
        return nn.MSELoss()
    if loss_name == "nse":
        return NSELoss()
    if loss_name in {"global_nse", "global-nse"}:
        return GlobalNSELoss()
    if loss_name == "mixed":
        return MixedLoss(mse_weight=cfg.mse_weight)
    raise ValueError(
        f"Unsupported loss_name={cfg.loss_name!r}. Expected 'mse', 'nse', 'global_nse', or 'mixed'."
    )


def normalization_tensors_for_split(
    dataset: CamelsWindowDataset,
    train_idx: np.ndarray,
    device: torch.device,
) -> Dict[str, Optional[torch.Tensor]]:
    (
        feature_mean_np,
        feature_std_np,
        target_mean_np,
        target_std_np,
        target_std_raw_np,
    ) = compute_train_normalization_stats(dataset, train_idx)
    feature_mean = torch.from_numpy(feature_mean_np).to(device)
    feature_std = torch.from_numpy(feature_std_np).to(device)
    target_mean = torch.from_numpy(target_mean_np).to(device)
    target_std = torch.from_numpy(target_std_np).to(device)
    target_std_raw = torch.from_numpy(target_std_raw_np).to(device)

    if dataset.future_known_feature_indices is not None:
        future_idx = torch.from_numpy(dataset.future_known_feature_indices).to(
            device=device, dtype=torch.long
        )
        future_feature_mean = torch.index_select(feature_mean, dim=1, index=future_idx)
        future_feature_std = torch.index_select(feature_std, dim=1, index=future_idx)
    else:
        future_feature_mean = None
        future_feature_std = None

    return {
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "future_feature_mean": future_feature_mean,
        "future_feature_std": future_feature_std,
        "target_mean": target_mean,
        "target_std": target_std,
        "target_std_raw": target_std_raw,
    }


def metrics_without_arrays(metrics: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "per_basin_nse"}


def train_one_split(
    dataset: CamelsWindowDataset,
    cfg: TrainConfig,
    device: torch.device,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    checkpoint_path: Optional[Path] = None,
    label: str = "split",
) -> Dict[str, Any]:
    model = build_model(dataset, cfg).to(device)
    criterion = build_criterion(cfg)
    mse_warmup_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    if cfg.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.warmup_start_factor,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
    else:
        warmup_scheduler = None

    train_loader = make_loader(dataset, train_idx, cfg.batch_size, True, cfg.num_workers, device)
    val_loader = make_loader(dataset, val_idx, cfg.batch_size, False, cfg.num_workers, device)
    stats = normalization_tensors_for_split(dataset, train_idx, device)

    history: List[Dict[str, float]] = []
    best_val_median_nse = float("-inf")
    best_val_nse = float("-inf")
    best_val_global_nse = float("-inf")
    best_val_loss = float("inf")
    best_val_metrics: Optional[Dict[str, float]] = None
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, cfg.epochs + 1):
        active_criterion = (
            mse_warmup_criterion
            if cfg.loss_name.lower() != "mse" and epoch <= cfg.mse_warmup_epochs
            else criterion
        )
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=active_criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
            **stats,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            **stats,
        )
        if warmup_scheduler is not None and epoch <= cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            "loss_name": "mse_warmup" if active_criterion is mse_warmup_criterion else cfg.loss_name,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_nse": train_metrics["nse"],
            "train_median_nse": train_metrics["median_nse"],
            "train_global_nse": train_metrics["global_nse"],
            "train_mean_nse_loss": train_metrics["mean_nse_loss"],
            "train_median_nse_loss": train_metrics["median_nse_loss"],
            "train_global_nse_loss": train_metrics["global_nse_loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_nse": val_metrics["nse"],
            "val_median_nse": val_metrics["median_nse"],
            "val_global_nse": val_metrics["global_nse"],
            "val_mean_nse_loss": val_metrics["mean_nse_loss"],
            "val_median_nse_loss": val_metrics["median_nse_loss"],
            "val_global_nse_loss": val_metrics["global_nse_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)
        print(
            f"{label} Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} train_nse={train_metrics['nse']:.6f} "
            f"train_global_nse={train_metrics['global_nse']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} val_nse={val_metrics['nse']:.6f} "
            f"val_median_nse={val_metrics['median_nse']:.6f} "
            f"val_global_nse={val_metrics['global_nse']:.6f}",
            flush=True,
        )

        val_median_nse_cur = val_metrics.get("median_nse", float("nan"))
        val_nse_cur = val_metrics.get("nse", float("nan"))
        val_global_nse_cur = val_metrics.get("global_nse", float("nan"))
        val_loss_cur = val_metrics["loss"]
        if not math.isnan(val_median_nse_cur):
            is_improved = val_median_nse_cur > best_val_median_nse + cfg.early_stopping_min_delta
        elif not math.isnan(val_nse_cur):
            is_improved = val_nse_cur > best_val_nse + cfg.early_stopping_min_delta
        else:
            is_improved = val_loss_cur < best_val_loss - cfg.early_stopping_min_delta

        if is_improved:
            if not math.isnan(val_median_nse_cur):
                best_val_median_nse = val_median_nse_cur
            if not math.isnan(val_nse_cur):
                best_val_nse = val_nse_cur
            if not math.isnan(val_global_nse_cur):
                best_val_global_nse = val_global_nse_cur
            best_val_loss = val_loss_cur
            best_val_metrics = metrics_without_arrays(val_metrics)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "config": config_to_dict(cfg),
                        "feature_columns": dataset.feature_columns,
                        "static_feature_columns": dataset.static_feature_columns,
                        "best_val_median_nse": best_val_median_nse,
                        "best_val_nse": best_val_nse,
                        "best_val_global_nse": best_val_global_nse,
                        "best_val_loss": best_val_loss,
                    },
                    checkpoint_path,
                )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(f"{label} early stopping at epoch {epoch:03d}.", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "model": model,
        "criterion": criterion,
        "stats": stats,
        "history": history,
        "best_val_median_nse": best_val_median_nse,
        "best_val_nse": best_val_nse,
        "best_val_global_nse": best_val_global_nse,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "best_epoch": best_epoch,
    }


def save_per_basin_nse(
    path: Path,
    metrics: Dict[str, Any],
    basin_ids: Optional[Sequence[str]],
    metric_name: str = "test_nse",
) -> Optional[Dict[str, Any]]:
    if "per_basin_nse" not in metrics or basin_ids is None:
        return None
    per_basin_nse_np: np.ndarray = metrics["per_basin_nse"]
    rows = [
        {"basin_id": str(basin_ids[i]), metric_name: float(per_basin_nse_np[i])}
        for i in range(len(basin_ids))
    ]
    rows.sort(key=lambda row: (math.isnan(row[metric_name]), row[metric_name]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    valid_nse_list = [row[metric_name] for row in rows if not math.isnan(row[metric_name])]
    summary: Dict[str, Any] = {
        "n_basins": len(basin_ids),
        "n_valid": len(valid_nse_list),
    }
    if valid_nse_list:
        arr_nse = np.asarray(valid_nse_list, dtype=np.float64)
        summary.update(
            {
                "median_nse": float(np.median(arr_nse)),
                "mean_nse": float(np.mean(arr_nse)),
                "p10_nse": float(np.percentile(arr_nse, 10)),
                "p25_nse": float(np.percentile(arr_nse, 25)),
                "p75_nse": float(np.percentile(arr_nse, 75)),
                "n_below_0": int(np.sum(arr_nse < 0)),
                "n_below_0_5": int(np.sum(arr_nse < 0.5)),
                "n_above_0_7": int(np.sum(arr_nse >= 0.7)),
            }
        )
    save_json(path, {"summary": summary, "basins": rows})
    return summary


def train_expanding_cv(cfg: TrainConfig, dataset: CamelsWindowDataset, device: torch.device) -> Dict[str, object]:
    folds, pre_holdout_idx, holdout_idx = expanding_cv_split_indices(dataset, cfg)
    print(
        f"Split strategy: expanding_cv | pre-holdout={len(pre_holdout_idx)} "
        f"holdout={len(holdout_idx)} folds={len(folds)}",
        flush=True,
    )
    for fold in folds:
        print(
            f"  Fold {fold['fold_id']}: per-basin expanding split "
            f"| train={len(fold['train_idx'])} val={len(fold['val_idx'])} "
            f"| train_basins={fold['n_train_basins']} val_basins={fold['n_val_basins']}",
            flush=True,
        )

    cv_results: List[Dict[str, Any]] = []
    cv_histories: List[Dict[str, Any]] = []
    for fold in folds:
        set_seed(cfg.random_seed + int(fold["fold_id"]))
        result = train_one_split(
            dataset=dataset,
            cfg=cfg,
            device=device,
            train_idx=fold["train_idx"],
            val_idx=fold["val_idx"],
            checkpoint_path=cfg.output_dir / f"cv_fold_{fold['fold_id']}_best.pt",
            label=f"CV fold {fold['fold_id']}",
        )
        fold_summary = {
            "fold_id": fold["fold_id"],
            "split_unit": fold["split_unit"],
            "n_train_windows": int(len(fold["train_idx"])),
            "n_val_windows": int(len(fold["val_idx"])),
            "n_train_basins": int(fold["n_train_basins"]),
            "n_val_basins": int(fold["n_val_basins"]),
            "best_epoch": result["best_epoch"],
            "best_val_loss": result["best_val_loss"],
            "best_val_nse": result["best_val_nse"],
            "best_val_median_nse": result["best_val_median_nse"],
            "best_val_global_nse": result["best_val_global_nse"],
            "best_val_metrics": result["best_val_metrics"],
        }
        cv_results.append(fold_summary)
        cv_histories.append({"fold_id": fold["fold_id"], "history": result["history"]})

    best_epochs = [row["best_epoch"] for row in cv_results if int(row["best_epoch"]) > 0]
    final_epochs = max(1, min(cfg.epochs, int(round(float(np.mean(best_epochs)))))) if best_epochs else cfg.epochs
    print(f"Final training epochs chosen from CV: {final_epochs}", flush=True)

    set_seed(cfg.random_seed)
    model = build_model(dataset, cfg).to(device)
    criterion = build_criterion(cfg)
    mse_warmup_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler]
    if cfg.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.warmup_start_factor,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs,
        )
    else:
        warmup_scheduler = None

    final_train_loader = make_loader(
        dataset, pre_holdout_idx, cfg.batch_size, True, cfg.num_workers, device
    )
    holdout_loader = make_loader(
        dataset, holdout_idx, cfg.batch_size, False, cfg.num_workers, device
    )
    stats = normalization_tensors_for_split(dataset, pre_holdout_idx, device)

    final_history: List[Dict[str, float]] = []
    for epoch in range(1, final_epochs + 1):
        active_criterion = (
            mse_warmup_criterion
            if cfg.loss_name.lower() != "mse" and epoch <= cfg.mse_warmup_epochs
            else criterion
        )
        train_metrics = run_epoch(
            model=model,
            loader=final_train_loader,
            criterion=active_criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
            **stats,
        )
        if warmup_scheduler is not None and epoch <= cfg.warmup_epochs:
            warmup_scheduler.step()
        row = {
            "epoch": epoch,
            "loss_name": "mse_warmup" if active_criterion is mse_warmup_criterion else cfg.loss_name,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_nse": train_metrics["nse"],
            "train_median_nse": train_metrics["median_nse"],
            "train_global_nse": train_metrics["global_nse"],
            "train_mean_nse_loss": train_metrics["mean_nse_loss"],
            "train_median_nse_loss": train_metrics["median_nse_loss"],
            "train_global_nse_loss": train_metrics["global_nse_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        final_history.append(row)
        print(
            f"Final Epoch {epoch:03d} | train_loss={train_metrics['loss']:.6f} "
            f"train_nse={train_metrics['nse']:.6f} "
            f"train_global_nse={train_metrics['global_nse']:.6f}",
            flush=True,
        )

    holdout_metrics = evaluate(
        model,
        holdout_loader,
        criterion,
        device,
        **stats,
    )
    print(
        f"Expanding-CV holdout metrics | loss={holdout_metrics['loss']:.6f} "
        f"mae={holdout_metrics['mae']:.6f} rmse={holdout_metrics['rmse']:.6f} "
        f"mean_basin_nse={holdout_metrics['nse']:.6f} "
        f"median_basin_nse={holdout_metrics['median_nse']:.6f} "
        f"global_nse={holdout_metrics['global_nse']:.6f}",
        flush=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config_to_dict(cfg),
            "feature_columns": dataset.feature_columns,
            "static_feature_columns": dataset.static_feature_columns,
            "final_epochs_from_cv": final_epochs,
            "cv_results": cv_results,
            "holdout_metrics": metrics_without_arrays(holdout_metrics),
        },
        cfg.output_dir / "best_model.pt",
    )

    basin_ids = list(dataset.basin_ids_in_model) if dataset.basin_ids_in_model is not None else None
    basin_summary = save_per_basin_nse(
        cfg.output_dir / "basin_holdout_nse.json",
        holdout_metrics,
        basin_ids,
        metric_name="holdout_nse",
    )

    save_json(
        cfg.output_dir / "history.json",
        {
            "cv_histories": cv_histories,
            "final_history": final_history,
        },
    )
    save_json(
        cfg.output_dir / "metrics.json",
        {
            "split_strategy": "expanding_cv",
            "cv_num_folds": cfg.cv_num_folds,
            "cv_train_ratio": cfg.cv_train_ratio,
            "cv_results": cv_results,
            "cv_mean_val_nse": float(np.mean([r["best_val_nse"] for r in cv_results])),
            "cv_mean_val_median_nse": float(np.mean([r["best_val_median_nse"] for r in cv_results])),
            "cv_mean_val_global_nse": float(np.mean([r["best_val_global_nse"] for r in cv_results])),
            "final_epochs_from_cv": final_epochs,
            "holdout_metrics": metrics_without_arrays(holdout_metrics),
            "holdout_basin_summary": basin_summary,
            "config": config_to_dict(cfg),
            "device": str(device),
        },
    )
    return {
        "split_strategy": "expanding_cv",
        "cv_results": cv_results,
        "final_epochs_from_cv": final_epochs,
        "test_metrics": holdout_metrics,
        "history": final_history,
        "output_dir": str(cfg.output_dir),
        "device": str(device),
    }


def save_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def plot_training_curves(
    history: Sequence[Dict[str, float]],
    output_path: Path,
    loss_label: str = "Loss",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipped saving training curve.",
            flush=True,
        )
        return

    epochs = [item["epoch"] for item in history]
    train_loss = [item["train_loss"] for item in history]
    val_loss = [item["val_loss"] for item in history]
    train_mae = [item["train_mae"] for item in history]
    val_mae = [item["val_mae"] for item in history]
    train_rmse = [item["train_rmse"] for item in history]
    val_rmse = [item["val_rmse"] for item in history]
    train_nse = [item["train_nse"] for item in history]
    val_nse = [item["val_nse"] for item in history]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title(loss_label)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel(loss_label)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_mae, label="Train MAE", linewidth=2)
    axes[1].plot(epochs, val_mae, label="Val MAE", linewidth=2)
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(epochs, train_rmse, label="Train RMSE", linewidth=2)
    axes[2].plot(epochs, val_rmse, label="Val RMSE", linewidth=2)
    axes[2].set_title("RMSE")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("RMSE")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    axes[3].plot(epochs, train_nse, label="Train NSE", linewidth=2)
    axes[3].plot(epochs, val_nse, label="Val NSE", linewidth=2)
    axes[3].set_title("NSE")
    axes[3].set_xlabel("Epoch")
    axes[3].set_ylabel("NSE")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curve to: {output_path}", flush=True)


def config_to_dict(cfg: TrainConfig) -> Dict:
    payload = asdict(cfg)
    payload["data_dir"] = str(cfg.data_dir)
    payload["processed_dir"] = str(cfg.processed_dir)
    payload["data_path"] = str(cfg.data_path)
    payload["output_dir"] = str(cfg.output_dir)
    return payload


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def load_config_file(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must contain a JSON object, got {type(payload).__name__}")
    return payload


def build_data_prep_config(cfg: TrainConfig) -> DataPrepConfig:
    return DataPrepConfig(
        data_dir=cfg.data_dir,
        output_dir=cfg.processed_dir,
        dataset_path=cfg.data_path,
        forcing_product=cfg.forcing_product,
        allow_daymet_model_output_fallback=cfg.allow_daymet_model_output_fallback,
        basin_ids=cfg.basin_ids,
        start_date=cfg.start_date,
        end_date=cfg.end_date,
        target_unit=cfg.target_unit,
        include_past_streamflow=cfg.include_past_streamflow,
        lookback_days=cfg.lookback_days,
        forecast_horizon_days=cfg.forecast_horizon_days,
        prediction_length=cfg.prediction_length,
        stride_days=cfg.stride_days,
        max_windows=cfg.max_windows,
        random_seed=cfg.random_seed,
        write_flat_files=cfg.write_flat_files,
    )


def normalize_path_value(value) -> Optional[str]:
    if value is None:
        return None
    return str(Path(value).expanduser().resolve(strict=False))


def normalize_basin_ids(value) -> List[str]:
    if value is None:
        return []
    return [str(item).strip().zfill(8) for item in value]


def normalized_data_prep_signature(payload: Dict) -> Dict:
    normalized = dict(payload)
    path_keys = [
        "data_dir",
        "timeseries_zip",
        "model_output_nldas_zip",
        "model_output_daymet_zip",
    ]
    for key in path_keys:
        normalized[key] = normalize_path_value(normalized.get(key))

    normalized["basin_ids"] = normalize_basin_ids(normalized.get("basin_ids"))
    normalized["include_past_streamflow"] = bool(normalized.get("include_past_streamflow", False))
    return normalized


def dataset_matches_config(cfg: TrainConfig) -> Tuple[bool, str]:
    if not cfg.data_path.exists():
        return False, "dataset file does not exist"

    try:
        with np.load(cfg.data_path, allow_pickle=True) as data:
            dataset_format = (
                str(data["dataset_format"][0]) if "dataset_format" in data.files else ""
            )
            if dataset_format != EXPECTED_DATASET_FORMAT:
                return False, f"dataset_format={dataset_format!r} does not match {EXPECTED_DATASET_FORMAT!r}"
            target_transform = (
                str(data["target_transform"][0]) if "target_transform" in data.files else ""
            )
            if target_transform != EXPECTED_TARGET_TRANSFORM:
                return False, (
                    f"target_transform={target_transform!r} does not match "
                    f"{EXPECTED_TARGET_TRANSFORM!r}"
                )
            required_keys = (
                "row_date",
                "row_basin_index",
                "window_basin_index",
                "normalize_feature_indices",
            )
            missing_keys = [k for k in required_keys if k not in data.files]
            if missing_keys:
                return False, f"dataset missing required keys: {missing_keys}"
            if "X" in data.files:
                x = np.asarray(data["X"])
                if x.ndim != 3:
                    return False, f"expected X to have 3 dimensions, got {x.ndim}"
                if x.shape[1] != cfg.lookback_days:
                    return False, "lookback_days changed"

                y = np.asarray(data["y"])
                prediction_length = y.shape[1] if y.ndim > 1 else 1
            elif "features" in data.files and "window_start" in data.files:
                x = np.asarray(data["features"])
                if x.ndim != 2:
                    return False, f"expected flat features to have 2 dimensions, got {x.ndim}"
                lookback_days = int(np.asarray(data["lookback_days"]).reshape(-1)[0])
                if lookback_days != cfg.lookback_days:
                    return False, "lookback_days changed"
                prediction_length = int(np.asarray(data["prediction_length"]).reshape(-1)[0])
                forecast_horizon_days = int(
                    np.asarray(data["forecast_horizon_days"]).reshape(-1)[0]
                )
                if forecast_horizon_days != cfg.forecast_horizon_days:
                    return False, "forecast_horizon_days changed"
            else:
                return False, "dataset format is not recognized"

            if prediction_length != cfg.prediction_length:
                return False, "prediction_length changed"

            feature_columns = data["feature_columns"].tolist() if "feature_columns" in data.files else []
            if "basin_code" not in feature_columns:
                return False, "dataset is missing basin_code feature"
            if "doy_sin" not in feature_columns or "doy_cos" not in feature_columns:
                return False, "dataset is missing day-of-year features"
            has_past_streamflow = "past_streamflow" in feature_columns
            if cfg.include_past_streamflow and not has_past_streamflow:
                return False, "dataset is missing past_streamflow feature"
            if not cfg.include_past_streamflow and has_past_streamflow:
                return False, "dataset has past_streamflow feature but config disables it"
            if "static_features" not in data.files or "static_feature_columns" not in data.files:
                return False, "dataset is missing static attribute features"
            static_features = np.asarray(data["static_features"])
            static_feature_columns = data["static_feature_columns"].tolist()
            if static_features.ndim != 2:
                return False, f"expected static_features to have 2 dimensions, got {static_features.ndim}"
            if static_features.shape[1] != len(static_feature_columns):
                return False, "static_features width does not match static_feature_columns"
            if len(static_feature_columns) == 0:
                return False, "dataset has no static attribute columns"
    except Exception as exc:
        return False, f"failed to inspect dataset: {exc}"

    metadata_path = cfg.processed_dir / "camels_transformer_metadata.json"
    if not metadata_path.exists():
        return False, "metadata file does not exist"

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        previous = metadata.get("data_prep_config", {})
    except Exception as exc:
        return False, f"failed to read metadata: {exc}"

    current = normalized_data_prep_signature(
        data_prep_config_to_dict(build_data_prep_config(cfg))
    )
    previous = normalized_data_prep_signature(previous)
    compare_keys = [
        "data_dir",
        "timeseries_zip",
        "model_output_nldas_zip",
        "model_output_daymet_zip",
        "forcing_product",
        "allow_daymet_model_output_fallback",
        "basin_ids",
        "start_date",
        "end_date",
        "target_unit",
        "include_past_streamflow",
        "lookback_days",
        "forecast_horizon_days",
        "prediction_length",
        "stride_days",
        "max_windows",
    ]
    if current.get("max_windows") is not None:
        compare_keys.append("random_seed")

    for key in compare_keys:
        if previous.get(key) != current.get(key):
            return False, f"data prep config changed: {key}"

    return True, "existing dataset matches current config"


def ensure_dataset_prepared(cfg: TrainConfig) -> None:
    should_rebuild = cfg.rebuild_dataset
    reason = "rebuild_dataset=True"

    if not should_rebuild:
        matches, reason = dataset_matches_config(cfg)
        should_rebuild = not matches

    if should_rebuild:
        print(f"Preparing training data because {reason}.", flush=True)
        prepare_camels_training_data(build_data_prep_config(cfg))
    else:
        print(f"Using existing prepared dataset: {cfg.data_path}", flush=True)


def build_config_from_sources(config_path: Path, cli_args: argparse.Namespace) -> TrainConfig:
    file_config = load_config_file(config_path)
    merged = {
        "data_dir": resolve_project_path(file_config.get("data_dir", "../15529996")),
        "processed_dir": resolve_project_path(file_config.get("processed_dir", "processed")),
        "data_path": resolve_project_path(
            file_config.get("data_path", "processed/camels_transformer_windows.npz")
        ),
        "output_dir": resolve_project_path(file_config.get("output_dir", "outputs/simple_tft")),
        "forcing_product": file_config.get("forcing_product", "nldas"),
        "allow_daymet_model_output_fallback": file_config.get(
            "allow_daymet_model_output_fallback", True
        ),
        "basin_ids": file_config.get("basin_ids", []),
        "start_date": file_config.get("start_date", "1980-01-01"),
        "end_date": file_config.get("end_date", "2014-12-31"),
        "target_unit": file_config.get("target_unit", "mm/day"),
        "include_past_streamflow": file_config.get("include_past_streamflow", False),
        "lookback_days": file_config.get("lookback_days", 365),
        "forecast_horizon_days": file_config.get("forecast_horizon_days", 1),
        "stride_days": file_config.get("stride_days", 1),
        "max_windows": file_config.get("max_windows", None),
        "write_flat_files": file_config.get("write_flat_files", False),
        "rebuild_dataset": file_config.get("rebuild_dataset", False),
        "loss_name": file_config.get("loss_name", "mse"),
        "mse_warmup_epochs": file_config.get("mse_warmup_epochs", 0),
        "mse_weight": file_config.get("mse_weight", 0.5),
        "batch_size": file_config.get("batch_size", 64),
        "epochs": file_config.get("epochs", 30),
        "early_stopping_patience": file_config.get("early_stopping_patience", 8),
        "early_stopping_min_delta": file_config.get("early_stopping_min_delta", 0.0),
        "lr": file_config.get("lr", 1e-3),
        "weight_decay": file_config.get("weight_decay", 1e-4),
        "grad_clip": file_config.get("grad_clip", 1.0),
        "warmup_epochs": file_config.get("warmup_epochs", 0),
        "warmup_start_factor": file_config.get("warmup_start_factor", 0.1),
        "split_strategy": file_config.get("split_strategy", "ratio"),
        "cv_num_folds": file_config.get("cv_num_folds", 4),
        "cv_train_ratio": file_config.get("cv_train_ratio", 0.8),
        "train_ratio": file_config.get("train_ratio", 0.7),
        "val_ratio": file_config.get("val_ratio", 0.15),
        "test_ratio": file_config.get("test_ratio", 0.15),
        "random_seed": file_config.get("random_seed", 42),
        "num_workers": file_config.get("num_workers", 0),
        "model_name": file_config.get("model_name", "tft"),
        "d_model": file_config.get("d_model", 64),
        "lstm_hidden": file_config.get("lstm_hidden", 64),
        "n_heads": file_config.get("n_heads", 4),
        "dropout": file_config.get("dropout", 0.1),
        "prediction_length": file_config.get("prediction_length", 1),
    }

    for key, value in vars(cli_args).items():
        if key == "config" or value is None:
            continue
        if key in {"data_dir", "processed_dir", "data_path", "output_dir"}:
            merged[key] = resolve_project_path(value)
        else:
            merged[key] = value

    return TrainConfig(**merged)


def train(cfg: TrainConfig, trial: Optional[Any] = None) -> Dict[str, object]:
    set_seed(cfg.random_seed)
    ensure_dataset_prepared(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    dataset = CamelsWindowDataset(cfg.data_path)
    if dataset.dataset_format != EXPECTED_DATASET_FORMAT:
        raise ValueError(
            f"Dataset format {dataset.dataset_format!r} is not supported; expected "
            f"{EXPECTED_DATASET_FORMAT!r}. Re-run data prep or set rebuild_dataset=true."
        )
    if dataset.target_transform != EXPECTED_TARGET_TRANSFORM:
        raise ValueError(
            f"Dataset target_transform {dataset.target_transform!r} does not match "
            f"{EXPECTED_TARGET_TRANSFORM!r}; rebuild is required."
        )
    if cfg.split_strategy.lower() == "expanding_cv":
        print(f"Using device: {device}", flush=True)
        print(f"Loaded dataset from: {cfg.data_path}", flush=True)
        print(f"Model: {cfg.model_name}", flush=True)
        print(f"Training loss: {cfg.loss_name.upper()}", flush=True)
        print(
            f"Decoder future-known features: {dataset.future_known_feature_columns or 'none'}",
            flush=True,
        )
        print(
            f"Static attributes ({dataset.num_static_features}): {dataset.static_feature_columns or 'none'}",
            flush=True,
        )
        print(
            f"Dataset shapes: features={tuple(dataset.features.shape)}, targets={tuple(dataset.targets.shape)}, "
            f"windows={len(dataset)} features_per_step={dataset.num_features}",
            flush=True,
        )
        return train_expanding_cv(cfg, dataset, device)
    if cfg.split_strategy.lower() != "ratio":
        raise ValueError(
            f"Unsupported split_strategy={cfg.split_strategy!r}. Expected 'ratio' or 'expanding_cv'."
        )

    model = build_model(dataset, cfg).to(device)
    criterion = build_criterion(cfg)
    mse_warmup_criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
    )
    if cfg.warmup_epochs > 0:
        if not 0.0 < cfg.warmup_start_factor <= 1.0:
            raise ValueError(
                f"warmup_start_factor must be in (0, 1], got {cfg.warmup_start_factor}"
            )
        warmup_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = (
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=cfg.warmup_start_factor,
                end_factor=1.0,
                total_iters=cfg.warmup_epochs,
            )
        )
        print(
            f"LR warmup: linear from {cfg.lr * cfg.warmup_start_factor:.2e} to "
            f"{cfg.lr:.2e} over {cfg.warmup_epochs} epochs.",
            flush=True,
        )
    else:
        warmup_scheduler = None

    train_idx, val_idx, test_idx = temporal_split_indices(dataset, cfg)
    train_loader = make_loader(
        dataset, train_idx, cfg.batch_size, True, cfg.num_workers, device
    )
    val_loader = make_loader(
        dataset, val_idx, cfg.batch_size, False, cfg.num_workers, device
    )
    test_loader = make_loader(
        dataset, test_idx, cfg.batch_size, False, cfg.num_workers, device
    )

    print(f"Using device: {device}", flush=True)
    print(f"Loaded dataset from: {cfg.data_path}", flush=True)
    print(f"Model: {cfg.model_name}", flush=True)
    print(f"Training loss: {cfg.loss_name.upper()}", flush=True)
    if cfg.mse_warmup_epochs > 0 and cfg.loss_name.lower() != "mse":
        print(
            f"Loss warmup: using MSE for the first {cfg.mse_warmup_epochs} epochs, "
            f"then {cfg.loss_name.upper()}.",
            flush=True,
        )
    print(
        f"Decoder future-known features: {dataset.future_known_feature_columns or 'none'}",
        flush=True,
    )
    print(
        f"Static attributes ({dataset.num_static_features}): {dataset.static_feature_columns or 'none'}",
        flush=True,
    )
    print(
        f"Dataset format: {dataset.dataset_format}",
        flush=True,
    )
    if dataset.mode == "dense":
        print(
            f"Dataset shapes: X={tuple(dataset.X.shape)}, y={tuple(dataset.y.shape)}, "
            f"features={dataset.num_features}",
            flush=True,
        )
    else:
        print(
            f"Dataset shapes: features={tuple(dataset.features.shape)}, targets={tuple(dataset.targets.shape)}, "
            f"windows={len(dataset)} features_per_step={dataset.num_features}",
            flush=True,
        )
    print(
        f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}",
        flush=True,
    )

    (
        feature_mean_np,
        feature_std_np,
        target_mean_np,
        target_std_np,
        target_std_raw_np,
    ) = compute_train_normalization_stats(dataset, train_idx)
    feature_mean_tensor = torch.from_numpy(feature_mean_np).to(device)
    feature_std_tensor = torch.from_numpy(feature_std_np).to(device)
    target_mean_tensor = torch.from_numpy(target_mean_np).to(device)
    target_std_tensor = torch.from_numpy(target_std_np).to(device)
    target_std_raw_tensor = torch.from_numpy(target_std_raw_np).to(device)
    if dataset.future_known_feature_indices is not None:
        future_idx = torch.from_numpy(dataset.future_known_feature_indices).to(device=device, dtype=torch.long)
        future_feature_mean_tensor = torch.index_select(feature_mean_tensor, dim=1, index=future_idx)
        future_feature_std_tensor = torch.index_select(feature_std_tensor, dim=1, index=future_idx)
    else:
        future_feature_mean_tensor = None
        future_feature_std_tensor = None

    history: List[Dict[str, float]] = []
    best_val_median_nse = float("-inf")  # Kratzert-style selection metric
    best_val_nse = float("-inf")  # mean basin NSE, kept for reporting/fallback
    best_val_loss = float("inf")  # kept for scheduler and final fallback
    best_val_metrics: Optional[Dict[str, float]] = None
    epochs_without_improvement = 0
    best_checkpoint_path = cfg.output_dir / "best_model.pt"

    for epoch in range(1, cfg.epochs + 1):
        active_criterion = (
            mse_warmup_criterion
            if cfg.loss_name.lower() != "mse" and epoch <= cfg.mse_warmup_epochs
            else criterion
        )
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=active_criterion,
            device=device,
            optimizer=optimizer,
            grad_clip=cfg.grad_clip,
            feature_mean=feature_mean_tensor,
            feature_std=feature_std_tensor,
            future_feature_mean=future_feature_mean_tensor,
            future_feature_std=future_feature_std_tensor,
            target_mean=target_mean_tensor,
            target_std=target_std_tensor,
            target_std_raw=target_std_raw_tensor,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
            feature_mean=feature_mean_tensor,
            feature_std=feature_std_tensor,
            future_feature_mean=future_feature_mean_tensor,
            future_feature_std=future_feature_std_tensor,
            target_mean=target_mean_tensor,
            target_std=target_std_tensor,
            target_std_raw=target_std_raw_tensor,
        )
        if warmup_scheduler is not None and epoch <= cfg.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step(val_metrics["loss"])

        epoch_metrics = {
            "epoch": epoch,
            "loss_name": "mse_warmup" if active_criterion is mse_warmup_criterion else cfg.loss_name,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "train_nse": train_metrics["nse"],
            "train_median_nse": train_metrics["median_nse"],
            "train_global_nse": train_metrics["global_nse"],
            "train_mean_nse_loss": train_metrics["mean_nse_loss"],
            "train_median_nse_loss": train_metrics["median_nse_loss"],
            "train_global_nse_loss": train_metrics["global_nse_loss"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_nse": val_metrics["nse"],
            "val_median_nse": val_metrics["median_nse"],
            "val_global_nse": val_metrics["global_nse"],
            "val_mean_nse_loss": val_metrics["mean_nse_loss"],
            "val_median_nse_loss": val_metrics["median_nse_loss"],
            "val_global_nse_loss": val_metrics["global_nse_loss"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_metrics)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_metrics['loss']:.6f} train_mae={train_metrics['mae']:.6f} train_rmse={train_metrics['rmse']:.6f} train_nse={train_metrics['nse']:.6f} train_median_nse={train_metrics['median_nse']:.6f} train_global_nse={train_metrics['global_nse']:.6f} | "
            f"val_loss={val_metrics['loss']:.6f} val_mae={val_metrics['mae']:.6f} val_rmse={val_metrics['rmse']:.6f} val_nse={val_metrics['nse']:.6f} val_median_nse={val_metrics['median_nse']:.6f} val_global_nse={val_metrics['global_nse']:.6f}",
            flush=True,
        )

        if trial is not None:
            reported_value = val_metrics["median_nse"]
            if math.isnan(reported_value):
                reported_value = val_metrics["nse"]
            if math.isnan(reported_value):
                reported_value = -val_metrics["loss"]
            trial.report(reported_value, step=epoch)
            if trial.should_prune():
                save_json(cfg.output_dir / "history.json", {"history": history})
                save_json(
                    cfg.output_dir / "metrics.json",
                    {
                        "status": "pruned",
                        "pruned_epoch": epoch,
                        "last_train_metrics": train_metrics,
                        "last_val_metrics": val_metrics,
                        "config": config_to_dict(cfg),
                        "device": str(device),
                    },
                )
                print(
                    f"Trial pruned at epoch {epoch:03d} with val_median_nse={val_metrics['median_nse']:.6f}",
                    flush=True,
                )
                try:
                    import optuna
                except ImportError as exc:  # pragma: no cover - runtime dependency
                    raise RuntimeError(
                        "Optuna trial pruning was requested but Optuna is not installed."
                    ) from exc
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch} with val_median_nse={val_metrics['median_nse']:.6f}"
                )

        val_median_nse_cur = val_metrics.get("median_nse", float("nan"))
        val_nse_cur = val_metrics.get("nse", float("nan"))
        val_loss_cur = val_metrics["loss"]
        # Prefer median basin-wise validation NSE (Kratzert-style model selection).
        if not math.isnan(val_median_nse_cur):
            is_improved = val_median_nse_cur > best_val_median_nse + cfg.early_stopping_min_delta
        elif not math.isnan(val_nse_cur):
            is_improved = val_nse_cur > best_val_nse + cfg.early_stopping_min_delta
        else:
            is_improved = val_loss_cur < best_val_loss - cfg.early_stopping_min_delta

        if is_improved:
            if not math.isnan(val_median_nse_cur):
                best_val_median_nse = val_median_nse_cur
            if not math.isnan(val_nse_cur):
                best_val_nse = val_nse_cur
            best_val_loss = val_loss_cur
            best_val_metrics = metrics_without_arrays(val_metrics)
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config_to_dict(cfg),
                    "feature_columns": dataset.feature_columns,
                    "static_feature_columns": dataset.static_feature_columns,
                    "best_val_median_nse": best_val_median_nse,
                    "best_val_nse": best_val_nse,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= cfg.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch:03d} after "
                f"{cfg.early_stopping_patience} epochs without validation improvement.",
                flush=True,
            )
            break

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        device,
        feature_mean=feature_mean_tensor,
        feature_std=feature_std_tensor,
        future_feature_mean=future_feature_mean_tensor,
        future_feature_std=future_feature_std_tensor,
        target_mean=target_mean_tensor,
        target_std=target_std_tensor,
        target_std_raw=target_std_raw_tensor,
    )

    print(
        f"Best checkpoint test metrics | "
            f"test_loss={test_metrics['loss']:.6f} test_mae={test_metrics['mae']:.6f} "
            f"test_rmse={test_metrics['rmse']:.6f} test_nse={test_metrics['nse']:.6f} "
            f"test_median_nse={test_metrics['median_nse']:.6f} "
            f"test_global_nse={test_metrics['global_nse']:.6f}",
            flush=True,
        )

    # Per-basin test NSE analysis — useful for spotting outlier basins dragging the average.
    if "per_basin_nse" in test_metrics and dataset.basin_ids_in_model is not None:
        per_basin_nse_np: np.ndarray = test_metrics["per_basin_nse"]
        basin_id_list = list(dataset.basin_ids_in_model)
        rows = [
            {"basin_id": str(basin_id_list[i]), "test_nse": float(per_basin_nse_np[i])}
            for i in range(len(basin_id_list))
        ]
        # Sort ascending by NSE (worst first); NaN basins go to the end.
        rows.sort(key=lambda r: (math.isnan(r["test_nse"]), r["test_nse"]))
        for rank, row in enumerate(rows, 1):
            row["rank"] = rank
        valid_nse_list = [r["test_nse"] for r in rows if not math.isnan(r["test_nse"])]
        basin_summary: Dict[str, Any] = {
            "n_basins": len(basin_id_list),
            "n_valid": len(valid_nse_list),
        }
        if valid_nse_list:
            arr_nse = np.array(valid_nse_list)
            basin_summary.update({
                "median_nse": float(np.median(arr_nse)),
                "mean_nse": float(np.mean(arr_nse)),
                "p10_nse": float(np.percentile(arr_nse, 10)),
                "p25_nse": float(np.percentile(arr_nse, 25)),
                "p75_nse": float(np.percentile(arr_nse, 75)),
                "n_below_0": int(np.sum(arr_nse < 0)),
                "n_below_0_5": int(np.sum(arr_nse < 0.5)),
                "n_above_0_7": int(np.sum(arr_nse >= 0.7)),
            })
        basin_nse_path = cfg.output_dir / "basin_test_nse.json"
        save_json(basin_nse_path, {"summary": basin_summary, "basins": rows})
        print(f"Saved per-basin test NSE to: {basin_nse_path}", flush=True)
        if valid_nse_list:
            print(
                f"  median={basin_summary['median_nse']:.3f}  "
                f"p10={basin_summary['p10_nse']:.3f}  "
                f"n_below_0.5={basin_summary['n_below_0_5']}/{len(valid_nse_list)}",
                flush=True,
            )

    save_json(cfg.output_dir / "history.json", {"history": history})
    save_json(
        cfg.output_dir / "metrics.json",
        {
            "best_val_nse": best_val_nse,
            "best_val_median_nse": best_val_median_nse,
            "best_val_loss": best_val_loss,
            "best_val_metrics": best_val_metrics,
            "test_loss": test_metrics["loss"],
            "test_mae": test_metrics["mae"],
            "test_rmse": test_metrics["rmse"],
            "test_nse": test_metrics["nse"],
            "test_median_nse": test_metrics["median_nse"],
            "test_global_nse": test_metrics["global_nse"],
            "config": config_to_dict(cfg),
            "device": str(device),
        },
    )
    loss_name_lower = cfg.loss_name.lower()
    if loss_name_lower == "nse":
        loss_label = "Basin-averaged NSE* Loss"
    elif loss_name_lower in {"global_nse", "global-nse"}:
        loss_label = "Global NSE Loss"
    elif loss_name_lower == "mixed":
        loss_label = f"Mixed Loss ({cfg.mse_weight:.2f}·MSE + {1.0 - cfg.mse_weight:.2f}·NSE*)"
    else:
        loss_label = "MSE Loss"
    plot_training_curves(history, cfg.output_dir / "training_curve.png", loss_label=loss_label)
    return {
        "best_val_nse": best_val_nse,
        "best_val_median_nse": best_val_median_nse,
        "best_val_loss": best_val_loss,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "output_dir": str(cfg.output_dir),
        "device": str(device),
    }


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train the simplified CAMELS TFT model.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training config JSON file.",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--processed-dir", type=Path, default=None)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to the prepared CAMELS window dataset (.npz).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save checkpoints and metrics.",
    )
    parser.add_argument("--forcing-product", type=str, default=None)
    parser.add_argument("--basin-id", action="append", default=None)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--target-unit", type=str, default=None)
    parser.add_argument(
        "--include-past-streamflow",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--forecast-horizon-days", type=int, default=None)
    parser.add_argument("--stride-days", type=int, default=None)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--loss-name", type=str, default=None)
    parser.add_argument("--mse-warmup-epochs", type=int, default=None)
    parser.add_argument("--mse-weight", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--warmup-start-factor", type=float, default=None)
    parser.add_argument("--split-strategy", type=str, choices=["ratio", "expanding_cv"], default=None)
    parser.add_argument("--cv-num-folds", type=int, default=None)
    parser.add_argument("--cv-train-ratio", type=float, default=None)
    parser.add_argument("--train-ratio", type=float, default=None)
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--model-name", type=str, choices=["tft", "lstm"], default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--lstm-hidden", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--prediction-length", type=int, default=None)
    parser.add_argument(
        "--allow-daymet-model-output-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--write-flat-files",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--rebuild-dataset",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    args = parser.parse_args()
    if args.basin_id is not None:
        args.basin_ids = args.basin_id
        delattr(args, "basin_id")
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path
    return build_config_from_sources(config_path, args)


if __name__ == "__main__":
    train(parse_args())
