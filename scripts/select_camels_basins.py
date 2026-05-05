import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

from data.prepare_camels_data import DataPrepConfig, _is_readable_zip, choose_forcing_source, index_camels_zip


def resolve_project_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_DIR / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}, got {type(payload).__name__}")
    return payload


def build_data_prep_config(config: dict) -> DataPrepConfig:
    data_dir = resolve_project_path(config.get("data_dir", "../15529996"))
    processed_dir = resolve_project_path(config.get("processed_dir", "processed"))
    dataset_path = resolve_project_path(
        config.get("data_path", "processed/camels_transformer_windows.npz")
    )
    return DataPrepConfig(
        data_dir=data_dir,
        output_dir=processed_dir,
        dataset_path=dataset_path,
        forcing_product=config.get("forcing_product", "nldas"),
        allow_daymet_model_output_fallback=config.get("allow_daymet_model_output_fallback", True),
    )


def discover_available_basin_ids(cfg: DataPrepConfig) -> List[str]:
    forcing_zip, actual_forcing_product, notes = choose_forcing_source(cfg)

    if not _is_readable_zip(cfg.timeseries_zip):
        raise FileNotFoundError(
            f"Main CAMELS time-series zip is missing or unreadable: {cfg.timeseries_zip}"
        )
    if not _is_readable_zip(forcing_zip):
        raise FileNotFoundError(f"Forcing zip is missing or unreadable: {forcing_zip}")

    forcing_paths = index_camels_zip(forcing_zip, actual_forcing_product)
    target_paths = index_camels_zip(cfg.timeseries_zip, actual_forcing_product)
    available = sorted(set(forcing_paths.forcing) & set(target_paths.streamflow))

    print(f"Forcing archive: {forcing_zip}")
    print(f"Forcing product: {actual_forcing_product}")
    for note in notes:
        print(f"Note: {note}")
    print(f"Target archive:  {cfg.timeseries_zip}")
    print(f"Complete basin pairs: {len(available):,}")
    return available


def choose_basin_ids(available: List[str], count: int, seed: int) -> List[str]:
    if count < 1:
        raise ValueError(f"count must be >= 1, got {count}")
    if count > len(available):
        raise ValueError(
            f"Requested {count} basins, but only {len(available)} complete basin pairs are available"
        )
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(available), size=count, replace=False))
    return [available[idx] for idx in indices]


def write_back_config(config_path: Path, config: dict, basin_ids: List[str]) -> None:
    config["basin_ids"] = basin_ids
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a random subset of available CAMELS basins and write them into train.json."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_DIR / "config" / "train.json",
        help="Path to the training config JSON file.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of basin IDs to select.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for basin sampling.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected basin IDs without writing them back to the config.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_DIR / config_path

    config = load_json(config_path)
    prep_cfg = build_data_prep_config(config)
    available = discover_available_basin_ids(prep_cfg)
    basin_ids = choose_basin_ids(available, args.count, args.seed)

    print(f"Selected {len(basin_ids)} basin IDs:")
    print(json.dumps(basin_ids, indent=2))

    if args.dry_run:
        print("Dry run only; config was not modified.")
        return

    write_back_config(config_path, config, basin_ids)
    print(f"Updated config: {config_path}")


if __name__ == "__main__":
    main()
