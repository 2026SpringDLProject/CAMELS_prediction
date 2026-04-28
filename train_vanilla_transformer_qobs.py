from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import signal
import sys
import time
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

DEFAULT_CONFIG = {
    'DATA_DIR': '/Users/xshan/Research/GT/cs7643_DL/courseProject/CAMELS_data_load/processed',
    'JOINED_FILENAME_PREFERENCE': ['camels_transformer_joined.parquet', 'camels_transformer_joined.csv'],
    'METADATA_FILENAME': 'camels_transformer_metadata.json',
    'OUTPUT_DIR': '/Users/xshan/Research/GT/cs7643_DL/courseProject/CAMELS_data_load/model_artifacts',
    'RUN_NAME': 'vanilla_transformer_qobs',
    'BASIN_IDS': [],
    'LOOKBACK_DAYS': 365,
    'PREDICTION_HORIZON_DAYS': 1,
    'USE_ALL_FEATURES': True,
    'INDEPENDENT_VAL_START': '2008-01-01',
    'INDEPENDENT_VAL_END': '2014-12-31',
    'CV_NUM_FOLDS': 4,
    'SEED': 7643,
    'DEVICE': 'cuda' if torch.cuda.is_available() else 'cpu',
    'NUM_WORKERS': 0,
    'PIN_MEMORY': False,
    'SEARCH_TRIALS': 8,
    'SEARCH_EPOCHS': 6,
    'SEARCH_PATIENCE': 2,
    'FINAL_MAX_EPOCHS': 20,
    'TUNING_MAX_TRAIN_WINDOWS': 50000,
    'TUNING_MAX_VAL_WINDOWS': 15000,
    'SAVE_CHECKPOINTS': True,
}

SEARCH_SPACE = {
    'd_model': [64, 96, 128, 160],
    'nhead': [4, 8],
    'num_layers': [2, 3, 4],
    'dim_feedforward': [128, 256, 384, 512],
    'dropout': [0.05, 0.10, 0.20],
    'learning_rate': [1e-4, 2e-4, 3e-4, 5e-4],
    'weight_decay': [0.0, 1e-5, 1e-4, 5e-4],
    'batch_size': [32, 64, 96, 128],
}

STOP_REQUESTED = False


def request_stop(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print(f'Received signal {signum}. Will stop cleanly after the current epoch/fold.', flush=True)


def install_signal_handlers() -> None:
    for sig_name in ('SIGUSR1', 'SIGTERM', 'SIGINT'):
        if hasattr(signal, sig_name):
            signal.signal(getattr(signal, sig_name), request_stop)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train a shared vanilla transformer for CAMELS QObs prediction.')
    parser.add_argument('--config-json', type=str, default=None, help='Optional JSON file with config overrides.')
    parser.add_argument('--data-dir', type=str, default=None, help='Processed data directory.')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for artifacts.')
    parser.add_argument('--run-name', type=str, default=None, help='Name used for the run subdirectory inside output-dir.')
    parser.add_argument('--device', type=str, default=None, help='Device, e.g. cpu or cuda.')
    parser.add_argument('--basin-ids', type=str, default=None, help='Comma-separated basin IDs. Empty means all basins.')
    parser.add_argument('--lookback-days', type=int, default=None)
    parser.add_argument('--prediction-horizon-days', type=int, default=None)
    parser.add_argument('--independent-val-start', type=str, default=None)
    parser.add_argument('--independent-val-end', type=str, default=None)
    parser.add_argument('--cv-num-folds', type=int, default=None)
    parser.add_argument('--search-trials', type=int, default=None)
    parser.add_argument('--search-epochs', type=int, default=None)
    parser.add_argument('--search-patience', type=int, default=None)
    parser.add_argument('--final-max-epochs', type=int, default=None)
    parser.add_argument('--tuning-max-train-windows', type=int, default=None)
    parser.add_argument('--tuning-max-val-windows', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--pin-memory', action='store_true', help='Enable DataLoader pin_memory.')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--save-checkpoints', action='store_true', default=None, help='Save model checkpoints.')
    parser.add_argument('--no-save-checkpoints', action='store_true', help='Do not save model checkpoints.')
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict:
    config = dict(DEFAULT_CONFIG)
    if args.config_json:
        config.update(json.loads(Path(args.config_json).read_text(encoding='utf-8')))

    overrides = {
        'DATA_DIR': args.data_dir,
        'OUTPUT_DIR': args.output_dir,
        'RUN_NAME': args.run_name,
        'DEVICE': args.device,
        'LOOKBACK_DAYS': args.lookback_days,
        'PREDICTION_HORIZON_DAYS': args.prediction_horizon_days,
        'INDEPENDENT_VAL_START': args.independent_val_start,
        'INDEPENDENT_VAL_END': args.independent_val_end,
        'CV_NUM_FOLDS': args.cv_num_folds,
        'SEARCH_TRIALS': args.search_trials,
        'SEARCH_EPOCHS': args.search_epochs,
        'SEARCH_PATIENCE': args.search_patience,
        'FINAL_MAX_EPOCHS': args.final_max_epochs,
        'TUNING_MAX_TRAIN_WINDOWS': args.tuning_max_train_windows,
        'TUNING_MAX_VAL_WINDOWS': args.tuning_max_val_windows,
        'NUM_WORKERS': args.num_workers,
        'SEED': args.seed,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if args.basin_ids is not None:
        config['BASIN_IDS'] = [x.strip().zfill(8) for x in args.basin_ids.split(',') if x.strip()]
    if args.pin_memory:
        config['PIN_MEMORY'] = True
    if args.save_checkpoints:
        config['SAVE_CHECKPOINTS'] = True
    if args.no_save_checkpoints:
        config['SAVE_CHECKPOINTS'] = False

    base_output_dir = Path(config['OUTPUT_DIR'])
    run_dir = base_output_dir / config['RUN_NAME']
    run_dir.mkdir(parents=True, exist_ok=True)
    config['RUN_DIR'] = str(run_dir)
    return config


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f'Object of type {type(obj)} is not JSON serializable')


def atomic_write_json(path: Path, data: Dict) -> None:
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    tmp_path.write_text(json.dumps(data, indent=2, default=json_default), encoding='utf-8')
    tmp_path.replace(path)


def load_json_if_exists(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding='utf-8'))


def run_paths(config: Dict) -> Dict[str, Path]:
    run_dir = Path(config['RUN_DIR'])
    return {
        'run_dir': run_dir,
        'state_json': run_dir / 'resume_state.json',
        'done_flag': run_dir / 'DONE',
        'search_results_csv': run_dir / 'vanilla_transformer_random_search_results.csv',
        'cv_folds_csv': run_dir / 'vanilla_transformer_cv_fold_results.csv',
        'final_history_csv': run_dir / 'vanilla_transformer_final_history.csv',
        'holdout_metrics_json': run_dir / 'vanilla_transformer_holdout_metrics.json',
        'config_json': run_dir / 'vanilla_transformer_config.json',
        'best_checkpoint': run_dir / 'vanilla_transformer_best.pt',
        'plot_png': run_dir / 'vanilla_transformer_training_plots.png',
        'final_resume_ckpt': run_dir / 'final_training_resume.pt',
    }


def default_resume_state(config: Dict) -> Dict:
    return {
        'status': 'search',
        'search_results': [],
        'cv_fold_results': [],
        'completed_trial_signatures': [],
        'best_hparams': None,
        'final_epochs': None,
        'final_training_completed_epochs': 0,
        'final_training_history': [],
        'holdout_metrics': None,
        'stopped_early': False,
        'config_snapshot': {
            'RUN_NAME': config['RUN_NAME'],
            'LOOKBACK_DAYS': config['LOOKBACK_DAYS'],
            'PREDICTION_HORIZON_DAYS': config['PREDICTION_HORIZON_DAYS'],
            'INDEPENDENT_VAL_START': config['INDEPENDENT_VAL_START'],
            'INDEPENDENT_VAL_END': config['INDEPENDENT_VAL_END'],
            'CV_NUM_FOLDS': config['CV_NUM_FOLDS'],
            'BASIN_IDS': config['BASIN_IDS'],
            'SEARCH_TRIALS': config['SEARCH_TRIALS'],
        },
    }


def load_or_init_state(config: Dict, paths: Dict[str, Path]) -> Dict:
    state = load_json_if_exists(paths['state_json'])
    if state is None:
        state = default_resume_state(config)
        atomic_write_json(paths['state_json'], state)
    return state


def persist_state(state: Dict, paths: Dict[str, Path]) -> None:
    atomic_write_json(paths['state_json'], state)
    if state.get('search_results'):
        pd.DataFrame(state['search_results']).sort_values('cv_mean_val_nse', ascending=False).to_csv(paths['search_results_csv'], index=False)
    if state.get('cv_fold_results'):
        pd.DataFrame(state['cv_fold_results']).to_csv(paths['cv_folds_csv'], index=False)
    if state.get('final_training_history'):
        pd.DataFrame(state['final_training_history']).to_csv(paths['final_history_csv'], index=False)
    if state.get('holdout_metrics') is not None:
        atomic_write_json(paths['holdout_metrics_json'], state['holdout_metrics'])


def load_metadata(config: Dict) -> Dict:
    metadata_path = Path(config['DATA_DIR']) / config['METADATA_FILENAME']
    return json.loads(metadata_path.read_text(encoding='utf-8'))


def choose_joined_path(config: Dict) -> Path:
    data_dir = Path(config['DATA_DIR'])
    for filename in config['JOINED_FILENAME_PREFERENCE']:
        path = data_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError('Could not find a joined CAMELS file in processed/.')


def read_joined_table(config: Dict, metadata: Dict) -> pd.DataFrame:
    joined_path = choose_joined_path(config)
    feature_columns = metadata['feature_columns']
    target_column = metadata['target_column_in_joined_file']
    columns = ['basin_id', 'date'] + feature_columns + [target_column]

    if joined_path.suffix == '.parquet':
        df = pd.read_parquet(joined_path, columns=columns)
    else:
        df = pd.read_csv(joined_path, usecols=columns, parse_dates=['date'])

    df['basin_id'] = df['basin_id'].astype(str).str.zfill(8)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values(['basin_id', 'date']).reset_index(drop=True)


def maybe_filter_basins(df: pd.DataFrame, basin_ids: Sequence[str]) -> pd.DataFrame:
    if not basin_ids:
        return df
    basin_ids = [str(x).zfill(8) for x in basin_ids]
    return df[df['basin_id'].isin(basin_ids)].copy()


def infer_feature_columns(metadata: Dict, config: Dict) -> List[str]:
    if not config['USE_ALL_FEATURES']:
        raise ValueError('This script is configured to use all features. Set USE_ALL_FEATURES=True.')
    return list(metadata['feature_columns'])


def global_nse(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6) -> float:
    numerator = np.sum((pred - target) ** 2)
    denominator = np.sum((target - target.mean()) ** 2) + eps
    return float(1.0 - numerator / denominator)


def nse_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    numerator = torch.sum((pred - target) ** 2)
    denominator = torch.sum((target - torch.mean(target)) ** 2) + eps
    return numerator / denominator


def regression_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    mae = float(np.mean(np.abs(pred - target)))
    nse = global_nse(pred, target)
    return {'rmse': rmse, 'mae': mae, 'nse': nse, 'loss': 1.0 - nse}


def sample_config(search_space: Dict[str, List], seen: set, rng: random.Random) -> Dict:
    keys = sorted(search_space)
    while True:
        candidate = {key: rng.choice(search_space[key]) for key in keys}
        if candidate['d_model'] % candidate['nhead'] != 0:
            continue
        signature = tuple((key, candidate[key]) for key in keys)
        if signature not in seen:
            seen.add(signature)
            return candidate


def trial_signature(trial_config: Dict) -> List[Tuple[str, object]]:
    return [(key, trial_config[key]) for key in sorted(trial_config)]


def build_basin_store(df: pd.DataFrame, feature_columns: List[str], target_column: str) -> Dict[str, Dict[str, np.ndarray]]:
    store = {}
    for basin_id, basin_df in df.groupby('basin_id', sort=True):
        basin_df = basin_df.sort_values('date').reset_index(drop=True)
        store[basin_id] = {
            'features': basin_df[feature_columns].to_numpy(dtype=np.float32),
            'target': basin_df[target_column].to_numpy(dtype=np.float32),
            'dates': basin_df['date'].to_numpy(),
        }
    return store


def eligible_target_indices(n_rows: int, lookback_days: int, horizon_days: int) -> np.ndarray:
    first_target_idx = lookback_days + horizon_days - 1
    if n_rows <= first_target_idx:
        return np.array([], dtype=np.int64)
    return np.arange(first_target_idx, n_rows, dtype=np.int64)


def year_chunks_for_expanding_cv(train_years: List[int], cv_num_folds: int) -> List[List[int]]:
    chunks = np.array_split(np.array(train_years), cv_num_folds + 1)
    return [chunk.astype(int).tolist() for chunk in chunks if len(chunk) > 0]


def build_time_protocol(
    basin_store: Dict[str, Dict[str, np.ndarray]],
    lookback_days: int,
    horizon_days: int,
    independent_val_start: str,
    independent_val_end: str,
    cv_num_folds: int,
) -> Dict:
    holdout_start = np.datetime64(independent_val_start)
    holdout_end = np.datetime64(independent_val_end)
    pre_holdout_years = sorted({
        int(pd.Timestamp(date_value).year)
        for basin_data in basin_store.values()
        for date_value in basin_data['dates']
        if date_value < holdout_start
    })
    chunks = year_chunks_for_expanding_cv(pre_holdout_years, cv_num_folds)
    if len(chunks) < 2:
        raise ValueError('Not enough pre-holdout years to build expanding-window CV folds.')

    cv_folds = []
    for fold_idx in range(len(chunks) - 1):
        train_years = sorted([year for chunk in chunks[: fold_idx + 1] for year in chunk])
        val_years = sorted(chunks[fold_idx + 1])
        if not train_years or not val_years:
            continue
        cv_folds.append({'fold_id': fold_idx, 'train_years': train_years, 'val_years': val_years})

    basin_protocol = {}
    usable_basins = []
    for basin_id, basin_data in basin_store.items():
        dates = basin_data['dates']
        target_indices = eligible_target_indices(len(dates), lookback_days, horizon_days)
        if len(target_indices) == 0:
            continue
        target_dates = dates[target_indices]

        pre_holdout_mask = target_dates < holdout_start
        holdout_mask = (target_dates >= holdout_start) & (target_dates <= holdout_end)
        pre_holdout_targets = target_indices[pre_holdout_mask]
        holdout_targets = target_indices[holdout_mask]
        if len(holdout_targets) == 0:
            continue

        fold_targets = []
        valid_basin = True
        for fold in cv_folds:
            train_year_set = set(fold['train_years'])
            val_year_set = set(fold['val_years'])
            train_targets = np.array([
                idx for idx in pre_holdout_targets if int(pd.Timestamp(dates[idx]).year) in train_year_set
            ], dtype=np.int64)
            val_targets = np.array([
                idx for idx in pre_holdout_targets if int(pd.Timestamp(dates[idx]).year) in val_year_set
            ], dtype=np.int64)
            if len(train_targets) == 0 or len(val_targets) == 0:
                valid_basin = False
                break
            fold_targets.append({'fold_id': fold['fold_id'], 'train_targets': train_targets, 'val_targets': val_targets})

        if not valid_basin:
            continue

        basin_protocol[basin_id] = {
            'cv_folds': fold_targets,
            'holdout_targets': holdout_targets,
            'pre_holdout_targets': pre_holdout_targets,
        }
        usable_basins.append(basin_id)

    return {
        'holdout_start': str(holdout_start),
        'holdout_end': str(holdout_end),
        'cv_folds': cv_folds,
        'basin_protocol': basin_protocol,
        'usable_basins': usable_basins,
    }


class WindowedTargetDataset(Dataset):
    def __init__(
        self,
        basin_store: Dict[str, Dict[str, np.ndarray]],
        target_map: Dict[str, np.ndarray],
        lookback_days: int,
        horizon_days: int,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
    ):
        self.basin_store = basin_store
        self.lookback_days = lookback_days
        self.horizon_days = horizon_days
        self.feature_mean = feature_mean.astype(np.float32)
        self.feature_std = feature_std.astype(np.float32)
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

    def __getitem__(self, idx: int):
        basin_pos = bisect_right(self.cum_counts, idx)
        basin_id, target_indices = self.entries[basin_pos]
        prev_count = 0 if basin_pos == 0 else self.cum_counts[basin_pos - 1]
        target_idx = int(target_indices[idx - prev_count])
        series = self.basin_store[basin_id]
        sequence_start = target_idx - self.horizon_days - self.lookback_days + 1
        sequence_end = sequence_start + self.lookback_days
        x = series['features'][sequence_start:sequence_end]
        x = (x - self.feature_mean) / self.feature_std
        y = float(series['target'][target_idx])
        date_value = series['dates'][target_idx]
        return {
            'x': torch.from_numpy(x).float(),
            'y': torch.tensor(y, dtype=torch.float32),
            'basin_id': basin_id,
            'target_date': str(np.datetime_as_string(date_value, unit='D')),
        }


def subset_dataset(dataset: Dataset, max_size: Optional[int], seed: int) -> Dataset:
    if max_size is None or len(dataset) <= max_size:
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_size, replace=False))
    return Subset(dataset, indices.tolist())


def make_target_map_for_fold(protocol: Dict, fold_id: int, split_name: str) -> Dict[str, np.ndarray]:
    target_map = {}
    for basin_id, basin_info in protocol['basin_protocol'].items():
        fold_info = next(item for item in basin_info['cv_folds'] if item['fold_id'] == fold_id)
        if split_name == 'train':
            target_map[basin_id] = fold_info['train_targets']
        elif split_name == 'val':
            target_map[basin_id] = fold_info['val_targets']
        elif split_name == 'pre_holdout_all':
            target_map[basin_id] = basin_info['pre_holdout_targets']
        elif split_name == 'holdout':
            target_map[basin_id] = basin_info['holdout_targets']
        else:
            raise ValueError('Unknown split name: ' + split_name)
    return target_map


def feature_stats_from_target_map(
    basin_store: Dict[str, Dict[str, np.ndarray]],
    target_map: Dict[str, np.ndarray],
    horizon_days: int,
) -> Tuple[np.ndarray, np.ndarray]:
    blocks = []
    for basin_id, targets in target_map.items():
        if len(targets) == 0:
            continue
        series = basin_store[basin_id]['features']
        max_target = int(np.max(targets))
        feature_end = max(max_target - horizon_days + 2, 1)
        blocks.append(series[:feature_end])
    matrix = np.concatenate(blocks, axis=0)
    mean = matrix.mean(axis=0).astype(np.float32)
    std = matrix.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VanillaTransformerRegressor(nn.Module):
    def __init__(self, num_features: int, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(num_features, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
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
        x = x[:, -1, :]
        return self.head(x).squeeze(-1)


def make_dataloader(dataset: Dataset, batch_size: int, shuffle: bool, config: Dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config['NUM_WORKERS'],
        pin_memory=config['PIN_MEMORY'],
        drop_last=False,
    )


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: Optional[torch.optim.Optimizer], device: str) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    preds, targets = [], []

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(x)
            loss = nse_loss(pred, y)
            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
        preds.append(pred.detach().cpu().numpy())
        targets.append(y.detach().cpu().numpy())

    pred_np = np.concatenate(preds)
    target_np = np.concatenate(targets)
    return regression_metrics(pred_np, target_np)


def fit_model_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    device: str,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    history = []
    best_state = None
    best_val_nse = -np.inf
    best_epoch = 0
    patience_left = patience
    stopped_early = False

    for epoch in range(1, max_epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        row = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_nse': train_metrics['nse'],
            'val_loss': val_metrics['loss'],
            'val_nse': val_metrics['nse'],
            'val_rmse': val_metrics['rmse'],
            'val_mae': val_metrics['mae'],
        }
        history.append(row)
        print(row, flush=True)
        if val_metrics['nse'] > best_val_nse:
            best_val_nse = val_metrics['nse']
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_left = patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print('Early stopping triggered.', flush=True)
                break
        if STOP_REQUESTED:
            print('Stopping after current trial epoch due to requested stop.', flush=True)
            stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch, stopped_early


def fit_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> Dict[str, float]:
    return run_epoch(model, loader, optimizer, device)


def evaluate_trial_on_cv(
    trial_config: Dict,
    protocol: Dict,
    basin_store: Dict[str, Dict[str, np.ndarray]],
    feature_columns: List[str],
    config: Dict,
    seed: int,
):
    device = config['DEVICE']
    fold_results = []
    stopped_early = False

    for fold in protocol['cv_folds']:
        fold_id = fold['fold_id']
        train_map = make_target_map_for_fold(protocol, fold_id, 'train')
        val_map = make_target_map_for_fold(protocol, fold_id, 'val')
        feature_mean, feature_std = feature_stats_from_target_map(
            basin_store,
            train_map,
            config['PREDICTION_HORIZON_DAYS'],
        )
        train_dataset = WindowedTargetDataset(
            basin_store=basin_store,
            target_map=train_map,
            lookback_days=config['LOOKBACK_DAYS'],
            horizon_days=config['PREDICTION_HORIZON_DAYS'],
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        val_dataset = WindowedTargetDataset(
            basin_store=basin_store,
            target_map=val_map,
            lookback_days=config['LOOKBACK_DAYS'],
            horizon_days=config['PREDICTION_HORIZON_DAYS'],
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        train_dataset = subset_dataset(train_dataset, config['TUNING_MAX_TRAIN_WINDOWS'], seed + fold_id)
        val_dataset = subset_dataset(val_dataset, config['TUNING_MAX_VAL_WINDOWS'], seed + 100 + fold_id)
        train_loader = make_dataloader(train_dataset, batch_size=trial_config['batch_size'], shuffle=True, config=config)
        val_loader = make_dataloader(val_dataset, batch_size=trial_config['batch_size'], shuffle=False, config=config)

        model = VanillaTransformerRegressor(
            num_features=len(feature_columns),
            d_model=trial_config['d_model'],
            nhead=trial_config['nhead'],
            num_layers=trial_config['num_layers'],
            dim_feedforward=trial_config['dim_feedforward'],
            dropout=trial_config['dropout'],
        ).to(device)

        model, history_df, best_epoch, fold_stopped_early = fit_model_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=trial_config['learning_rate'],
            weight_decay=trial_config['weight_decay'],
            max_epochs=config['SEARCH_EPOCHS'],
            patience=config['SEARCH_PATIENCE'],
            device=device,
        )
        best_row = history_df.loc[history_df['val_nse'].idxmax()].to_dict()
        fold_results.append({
            'fold_id': fold_id,
            'best_epoch': int(best_epoch),
            'best_val_nse': float(best_row['val_nse']),
            'best_val_rmse': float(best_row['val_rmse']),
            'best_val_mae': float(best_row['val_mae']),
        })
        if fold_stopped_early:
            stopped_early = True
            break

    fold_df = pd.DataFrame(fold_results)
    summary = None
    if not fold_df.empty and len(fold_df) == len(protocol['cv_folds']):
        summary = {
            'cv_mean_val_nse': float(fold_df['best_val_nse'].mean()),
            'cv_std_val_nse': float(fold_df['best_val_nse'].std(ddof=0)),
            'cv_mean_val_rmse': float(fold_df['best_val_rmse'].mean()),
            'cv_mean_val_mae': float(fold_df['best_val_mae'].mean()),
            'cv_mean_best_epoch': float(fold_df['best_epoch'].mean()),
        }
    return summary, fold_df, stopped_early


def save_plots(search_results_df: pd.DataFrame, final_history_df: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if not search_results_df.empty:
        search_results_df.plot(x='trial', y='cv_mean_val_nse', marker='o', ax=axes[0])
    axes[0].set_title('Random Search CV Mean NSE')
    axes[0].set_ylabel('CV mean NSE')
    axes[0].grid(True)

    if not final_history_df.empty:
        final_history_df.plot(x='epoch', y='train_nse', marker='o', ax=axes[1])
    axes[1].set_title('Final Training NSE History')
    axes[1].set_ylabel('Train NSE')
    axes[1].grid(True)
    plt.tight_layout()
    fig.savefig(output_dir / 'vanilla_transformer_training_plots.png', dpi=150)
    plt.close(fig)


def save_final_resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epochs: int,
    best_hparams: Dict,
    feature_columns: List[str],
    target_column: str,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    config: Dict,
) -> None:
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'completed_epochs': completed_epochs,
            'best_hparams': best_hparams,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'feature_mean': feature_mean,
            'feature_std': feature_std,
            'config': config,
        },
        path,
    )


def main() -> None:
    install_signal_handlers()
    args = parse_args()
    config = load_config(args)
    paths = run_paths(config)
    if paths['done_flag'].exists():
        print('DONE flag already exists. Nothing to do.', flush=True)
        return

    set_seed(config['SEED'])
    state = load_or_init_state(config, paths)

    metadata = load_metadata(config)
    feature_columns = infer_feature_columns(metadata, config)
    target_column = metadata['target_column_in_joined_file']
    joined_df = maybe_filter_basins(read_joined_table(config, metadata), config['BASIN_IDS'])

    print('Using joined file:', choose_joined_path(config), flush=True)
    print('Target column:', target_column, flush=True)
    print('Feature columns:', feature_columns, flush=True)
    print('Rows:', len(joined_df), flush=True)
    print('Basins:', joined_df['basin_id'].nunique(), flush=True)
    print('Date range:', joined_df['date'].min().date(), 'to', joined_df['date'].max().date(), flush=True)
    print('Run directory:', paths['run_dir'], flush=True)

    basin_store = build_basin_store(joined_df, feature_columns, target_column)
    protocol = build_time_protocol(
        basin_store=basin_store,
        lookback_days=config['LOOKBACK_DAYS'],
        horizon_days=config['PREDICTION_HORIZON_DAYS'],
        independent_val_start=config['INDEPENDENT_VAL_START'],
        independent_val_end=config['INDEPENDENT_VAL_END'],
        cv_num_folds=config['CV_NUM_FOLDS'],
    )

    print('Usable basins:', len(protocol['usable_basins']), flush=True)
    print('Independent holdout:', protocol['holdout_start'], 'to', protocol['holdout_end'], flush=True)
    print('Cross-validation folds:', flush=True)
    for fold in protocol['cv_folds']:
        print(
            ' Fold', fold['fold_id'], 'train years', fold['train_years'][0], 'to', fold['train_years'][-1],
            '| val years', fold['val_years'][0], 'to', fold['val_years'][-1],
            flush=True,
        )

    search_results = list(state.get('search_results', []))
    cv_fold_results = list(state.get('cv_fold_results', []))
    completed_trial_signatures = {tuple((k, v) for k, v in sig) for sig in state.get('completed_trial_signatures', [])}
    seen_configs = set(completed_trial_signatures)
    search_rng = random.Random(config['SEED'])
    print('Training device:', config['DEVICE'], flush=True)

    completed_trials = len(search_results)
    while completed_trials < config['SEARCH_TRIALS'] and not STOP_REQUESTED:
        trial_idx = completed_trials + 1
        trial_config = sample_config(SEARCH_SPACE, seen_configs, search_rng)
        print(f'\n=== Random search trial {trial_idx}/{config["SEARCH_TRIALS"]} ===', flush=True)
        print(trial_config, flush=True)
        start_time = time.time()
        summary, fold_df, stopped_early = evaluate_trial_on_cv(
            trial_config,
            protocol,
            basin_store,
            feature_columns,
            config,
            config['SEED'] + trial_idx,
        )
        if summary is None:
            print('Trial interrupted before completing all CV folds. Progress will resume next run.', flush=True)
            state['stopped_early'] = True
            persist_state(state, paths)
            return

        elapsed = time.time() - start_time
        result = dict(trial_config)
        result.update({
            'trial': trial_idx,
            'elapsed_sec': elapsed,
            'cv_mean_best_epoch': summary['cv_mean_best_epoch'],
            'cv_mean_val_nse': summary['cv_mean_val_nse'],
            'cv_std_val_nse': summary['cv_std_val_nse'],
            'cv_mean_val_rmse': summary['cv_mean_val_rmse'],
            'cv_mean_val_mae': summary['cv_mean_val_mae'],
        })
        search_results.append(result)
        fold_df = fold_df.copy()
        fold_df['trial'] = trial_idx
        cv_fold_results.extend(fold_df.to_dict(orient='records'))
        state['search_results'] = search_results
        state['cv_fold_results'] = cv_fold_results
        state['completed_trial_signatures'] = [trial_signature(row) for row in search_results]
        state['stopped_early'] = stopped_early or STOP_REQUESTED
        persist_state(state, paths)
        completed_trials += 1
        if stopped_early or STOP_REQUESTED:
            print('Stopping after completing current trial because stop was requested.', flush=True)
            return

    if completed_trials < config['SEARCH_TRIALS']:
        print('Search phase incomplete; expecting resubmission.', flush=True)
        return

    search_results_df = pd.DataFrame(search_results).sort_values('cv_mean_val_nse', ascending=False).reset_index(drop=True)
    cv_fold_results_df = pd.DataFrame(cv_fold_results)
    best_hparams = state.get('best_hparams') or search_results_df.iloc[0].to_dict()
    final_epochs = state.get('final_epochs') or max(1, min(config['FINAL_MAX_EPOCHS'], int(round(best_hparams['cv_mean_best_epoch']))))
    state['best_hparams'] = best_hparams
    state['final_epochs'] = final_epochs
    state['status'] = 'final_training'
    persist_state(state, paths)

    print('Best random-search configuration:', flush=True)
    print(best_hparams, flush=True)
    print('Final training epochs chosen from CV:', final_epochs, flush=True)

    final_train_map = make_target_map_for_fold(protocol, fold_id=0, split_name='pre_holdout_all')
    final_holdout_map = make_target_map_for_fold(protocol, fold_id=0, split_name='holdout')
    final_feature_mean, final_feature_std = feature_stats_from_target_map(
        basin_store,
        final_train_map,
        config['PREDICTION_HORIZON_DAYS'],
    )
    final_train_dataset = WindowedTargetDataset(
        basin_store=basin_store,
        target_map=final_train_map,
        lookback_days=config['LOOKBACK_DAYS'],
        horizon_days=config['PREDICTION_HORIZON_DAYS'],
        feature_mean=final_feature_mean,
        feature_std=final_feature_std,
    )
    final_holdout_dataset = WindowedTargetDataset(
        basin_store=basin_store,
        target_map=final_holdout_map,
        lookback_days=config['LOOKBACK_DAYS'],
        horizon_days=config['PREDICTION_HORIZON_DAYS'],
        feature_mean=final_feature_mean,
        feature_std=final_feature_std,
    )

    final_batch_size = int(best_hparams['batch_size'])
    final_train_loader = make_dataloader(final_train_dataset, batch_size=final_batch_size, shuffle=True, config=config)
    final_holdout_loader = make_dataloader(final_holdout_dataset, batch_size=final_batch_size, shuffle=False, config=config)

    final_model = VanillaTransformerRegressor(
        num_features=len(feature_columns),
        d_model=int(best_hparams['d_model']),
        nhead=int(best_hparams['nhead']),
        num_layers=int(best_hparams['num_layers']),
        dim_feedforward=int(best_hparams['dim_feedforward']),
        dropout=float(best_hparams['dropout']),
    ).to(config['DEVICE'])
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=float(best_hparams['learning_rate']), weight_decay=float(best_hparams['weight_decay']))

    completed_epochs = int(state.get('final_training_completed_epochs', 0))
    final_history = list(state.get('final_training_history', []))
    if paths['final_resume_ckpt'].exists():
        resume_ckpt = torch.load(paths['final_resume_ckpt'], map_location=config['DEVICE'])
        final_model.load_state_dict(resume_ckpt['model_state_dict'])
        optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
        completed_epochs = int(resume_ckpt['completed_epochs'])
        print(f'Resumed final training from epoch {completed_epochs}.', flush=True)

    for epoch in range(completed_epochs + 1, final_epochs + 1):
        train_metrics = fit_one_epoch(final_model, final_train_loader, optimizer, config['DEVICE'])
        row = {'epoch': epoch, 'train_loss': train_metrics['loss'], 'train_nse': train_metrics['nse']}
        final_history.append(row)
        print(row, flush=True)
        state['final_training_history'] = final_history
        state['final_training_completed_epochs'] = epoch
        persist_state(state, paths)
        if config['SAVE_CHECKPOINTS']:
            save_final_resume_checkpoint(
                paths['final_resume_ckpt'],
                final_model,
                optimizer,
                epoch,
                best_hparams,
                feature_columns,
                target_column,
                final_feature_mean,
                final_feature_std,
                config,
            )
        if STOP_REQUESTED:
            print('Stopping after current final-training epoch because stop was requested.', flush=True)
            state['stopped_early'] = True
            persist_state(state, paths)
            return

    holdout_metrics = run_epoch(final_model, final_holdout_loader, None, config['DEVICE'])
    print('Independent holdout metrics:', holdout_metrics, flush=True)
    state['holdout_metrics'] = holdout_metrics
    state['status'] = 'done'
    state['stopped_early'] = False
    persist_state(state, paths)

    serializable_config = dict(config)
    serializable_config['FEATURE_COLUMNS'] = feature_columns
    serializable_config['TARGET_COLUMN'] = target_column
    serializable_config['FINAL_EPOCHS_FROM_CV'] = final_epochs
    serializable_config['PROTOCOL'] = {
        'holdout_start': protocol['holdout_start'],
        'holdout_end': protocol['holdout_end'],
        'cv_folds': protocol['cv_folds'],
    }
    atomic_write_json(paths['config_json'], serializable_config)
    if config['SAVE_CHECKPOINTS']:
        torch.save(
            {
                'model_state_dict': final_model.state_dict(),
                'feature_columns': feature_columns,
                'target_column': target_column,
                'feature_mean': final_feature_mean,
                'feature_std': final_feature_std,
                'best_hparams': best_hparams,
                'config': serializable_config,
                'holdout_metrics': holdout_metrics,
            },
            paths['best_checkpoint'],
        )
    save_plots(search_results_df, pd.DataFrame(final_history), paths['run_dir'])
    paths['done_flag'].write_text('DONE\n', encoding='utf-8')
    print('Saved config:      ', paths['config_json'], flush=True)
    print('Saved search:      ', paths['search_results_csv'], flush=True)
    print('Saved fold results:', paths['cv_folds_csv'], flush=True)
    print('Saved history:     ', paths['final_history_csv'], flush=True)
    print('Saved holdout:     ', paths['holdout_metrics_json'], flush=True)
    if config['SAVE_CHECKPOINTS']:
        print('Saved checkpoint:  ', paths['best_checkpoint'], flush=True)
    print('Saved plot:        ', paths['plot_png'], flush=True)


if __name__ == '__main__':
    main()
