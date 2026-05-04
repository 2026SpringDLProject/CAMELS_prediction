import argparse
import json
import re
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CamelsPaths:
    forcing: Dict[str, str]
    streamflow: Dict[str, str]
    gauge_info: Optional[str]


NON_NORMALIZED_FEATURES = frozenset({"basin_code", "doy_sin", "doy_cos"})
TARGET_TRANSFORM = "identity"
PAST_STREAMFLOW_FEATURE = "past_streamflow"
# The static features used for TFT
DEFAULT_STATIC_ATTRIBUTE_GROUPS = ("topo", "clim", "soil", "vege", "geol")
STATIC_ATTRIBUTE_COLUMNS = {
    "topo": (
        "gauge_lat",
        "gauge_lon",
        "elev_mean",
        "slope_mean",
        "area_gages2",
        "area_geospa_fabric",
    ),
    "clim": (
        "p_mean",
        "pet_mean",
        "p_seasonality",
        "frac_snow",
        "aridity",
        "high_prec_freq",
        "high_prec_dur",
        "low_prec_freq",
        "low_prec_dur",
    ),
    "soil": (
        "soil_depth_pelletier",
        "soil_depth_statsgo",
        "soil_porosity",
        "soil_conductivity",
        "max_water_content",
        "sand_frac",
        "silt_frac",
        "clay_frac",
        "water_frac",
        "organic_frac",
        "other_frac",
    ),
    "vege": (
        "frac_forest",
        "lai_max",
        "lai_diff",
        "gvf_max",
        "gvf_diff",
        "dom_land_cover_frac",
        "root_depth_50",
        "root_depth_99",
    ),
    "geol": (
        "glim_1st_class_frac",
        "glim_2nd_class_frac",
        "carbonate_rocks_frac",
        "geol_porostiy",
        "geol_permeability",
    ),
}


@dataclass
class DataPrepConfig:
    data_dir: Path
    output_dir: Path
    dataset_path: Path
    forcing_product: str = "nldas"
    allow_daymet_model_output_fallback: bool = True
    basin_ids: List[str] = field(default_factory=list)
    start_date: Optional[str] = "1980-01-01"
    end_date: Optional[str] = "2014-12-31"
    target_unit: str = "mm/day"
    include_past_streamflow: bool = False
    lookback_days: int = 365
    forecast_horizon_days: int = 1
    prediction_length: int = 1
    stride_days: int = 1
    max_windows: Optional[int] = None
    random_seed: int = 42
    write_flat_files: bool = False

    @property
    def timeseries_zip(self) -> Path:
        return self.data_dir / "basin_timeseries_v1p2_metForcing_obsFlow.zip"

    @property
    def model_output_nldas_zip(self) -> Path:
        return self.data_dir / "basin_timeseries_v1p2_modelOutput_nldas.zip"

    @property
    def model_output_daymet_zip(self) -> Path:
        return self.data_dir / "basin_timeseries_v1p2_modelOutput_daymet.zip"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "camels_transformer_metadata.json"


def config_to_dict(cfg: DataPrepConfig) -> Dict:
    payload = asdict(cfg)
    payload["data_dir"] = str(cfg.data_dir)
    payload["output_dir"] = str(cfg.output_dir)
    payload["dataset_path"] = str(cfg.dataset_path)
    payload["timeseries_zip"] = str(cfg.timeseries_zip)
    payload["model_output_nldas_zip"] = str(cfg.model_output_nldas_zip)
    payload["model_output_daymet_zip"] = str(cfg.model_output_daymet_zip)
    payload["metadata_path"] = str(cfg.metadata_path)
    return payload


def _normalise_basin_id(basin_id) -> str:
    return str(basin_id).strip().zfill(8)


def _is_readable_zip(path: Path) -> bool:
    return path.exists() and zipfile.is_zipfile(path)


def choose_forcing_source(cfg: DataPrepConfig) -> Tuple[Path, str, List[str]]:
    notes: List[str] = []

    if cfg.forcing_product == "nldas":
        if _is_readable_zip(cfg.model_output_nldas_zip):
            return cfg.model_output_nldas_zip, "nldas", notes
        notes.append(f"NLDAS model-output archive not readable/found: {cfg.model_output_nldas_zip}")

        if cfg.allow_daymet_model_output_fallback:
            if _is_readable_zip(cfg.model_output_daymet_zip):
                notes.append(f"Using DAYMET model-output fallback: {cfg.model_output_daymet_zip}")
                return cfg.model_output_daymet_zip, "daymet", notes
            notes.append(
                f"DAYMET model-output fallback not readable/found: {cfg.model_output_daymet_zip}"
            )

    return cfg.timeseries_zip, cfg.forcing_product, notes


def index_camels_zip(zip_path: Path, forcing_product: str = "nldas") -> CamelsPaths:
    forcing_pattern = re.compile(
        rf"basin_mean_forcing/(?:v1p15/)?{re.escape(forcing_product)}/\d{{2}}/(\d{{8}})_.*_forcing_leap\.txt$"
    )
    streamflow_pattern = re.compile(r"usgs_streamflow/\d{2}/(\d{8})_streamflow_qc\.txt$")

    forcing: Dict[str, str] = {}
    streamflow: Dict[str, str] = {}
    gauge_info = None

    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            match = forcing_pattern.search(name)
            if match:
                basin_id = match.group(1)
                if basin_id not in forcing or "/v1p15/" not in name:
                    forcing[basin_id] = name
                continue

            match = streamflow_pattern.search(name)
            if match:
                streamflow[match.group(1)] = name
                continue

            if name.endswith("basin_metadata/gauge_information.txt"):
                gauge_info = name

    return CamelsPaths(forcing=forcing, streamflow=streamflow, gauge_info=gauge_info)


def read_gauge_info(zip_path: Path, gauge_info_path: Optional[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read(gauge_info_path).decode("utf-8", errors="replace")

    rows = []
    for line in text.splitlines()[1:]:
        if not line.strip():
            continue
        parts = line.split()
        huc_02 = parts[0]
        basin_id = parts[1].zfill(8)
        lat = float(parts[-3])
        lon = float(parts[-2])
        area_km2 = float(parts[-1])
        name = " ".join(parts[2:-3])
        rows.append((huc_02, basin_id, name, lat, lon, area_km2))

    return pd.DataFrame(
        rows,
        columns=["huc_02", "basin_id", "gage_name", "lat", "lon", "area_km2"],
    )


def read_forcing(zf: zipfile.ZipFile, member: str, basin_id: str) -> pd.DataFrame:
    with zf.open(member) as fp:
        df = pd.read_csv(fp, sep=r"\s+", skiprows=3, engine="python")

    rename = {
        "Year": "year",
        "Mnth": "month",
        "Day": "day",
        "Hr": "hour",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    df["basin_id"] = basin_id
    return df.drop(columns=["year", "month", "day", "hour"], errors="ignore")


def read_streamflow(zf: zipfile.ZipFile, member: str, basin_id: str) -> pd.DataFrame:
    cols = ["basin_id_file", "year", "month", "day", "QObs(cfs)", "quality_flag"]
    with zf.open(member) as fp:
        df = pd.read_csv(fp, sep=r"\s+", names=cols, engine="python")

    df["basin_id"] = basin_id
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=df["day"]))
    return df.drop(columns=["basin_id_file", "year", "month", "day"])


def cfs_to_mm_per_day(q_cfs: pd.Series, area_km2: float) -> pd.Series:
    return q_cfs * 2446.5755456 / (area_km2 * 1_000_000.0) * 1000.0


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    doy = out["date"].dt.dayofyear.astype(float)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 366.0)
    out["doy_cos"] = np.cos(2 * np.pi * doy / 366.0)
    return out


def basin_code_map(basin_ids: Iterable[str]) -> Dict[str, int]:
    return {basin_id: idx for idx, basin_id in enumerate(sorted(set(basin_ids)))}


def load_static_attributes(
    data_dir: Path,
    basin_ids: Sequence[str],
    groups: Sequence[str],
) -> Tuple[np.ndarray, List[str], Dict]:
    normalized_groups = [str(group).strip().lower() for group in groups]

    static_df = pd.DataFrame({"gauge_id": [_normalise_basin_id(x) for x in basin_ids]})
    feature_cols: List[str] = []
    source_files: Dict[str, str] = {}

    for group in normalized_groups:
        path = data_dir / f"camels_{group}.txt"
        cols = list(STATIC_ATTRIBUTE_COLUMNS[group])
        group_df = pd.read_csv(
            path,
            sep=";",
            dtype={"gauge_id": str},
            na_values=["NA", ""],
            keep_default_na=True,
        )
        group_df["gauge_id"] = group_df["gauge_id"].map(_normalise_basin_id)

        for col in cols:
            group_df[col] = pd.to_numeric(group_df[col], errors="coerce")

        static_df = static_df.merge(group_df[["gauge_id"] + cols], on="gauge_id", how="left")
        feature_cols.extend(cols)
        source_files[group] = str(path)

    if not feature_cols:
        return np.empty((len(basin_ids), 0), dtype=np.float32), [], {
            "groups": normalized_groups,
            "source_files": source_files,
            "missing_counts_before_imputation": {},
            "imputation": "none",
            "normalization": "none",
        }

    raw_values = static_df[feature_cols].to_numpy(dtype=np.float32)
    missing_counts = {
        col: int(count)
        for col, count in zip(feature_cols, np.isnan(raw_values).sum(axis=0))
        if int(count) > 0
    }
    medians = np.nanmedian(raw_values, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32)
    missing_rows, missing_cols = np.where(np.isnan(raw_values))
    if len(missing_rows) > 0:
        raw_values[missing_rows, missing_cols] = medians[missing_cols]

    mean = raw_values.mean(axis=0).astype(np.float32)
    std = raw_values.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    static_features = ((raw_values - mean) / std).astype(np.float32)

    metadata = {
        "groups": normalized_groups,
        "source_files": source_files,
        "columns": feature_cols,
        "missing_counts_before_imputation": missing_counts,
        "imputation": "column_median_over_selected_basins",
        "normalization": "global_z_score_over_selected_basins_after_imputation",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "median": medians.tolist(),
    }
    return static_features, feature_cols, metadata


def write_table(df: pd.DataFrame, path_base: Path) -> Dict[str, Optional[str]]:
    parquet_path = path_base.with_suffix(".parquet")
    csv_path = path_base.with_suffix(".csv")

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return {"csv": str(csv_path), "parquet": str(parquet_path)}


def process_basin_timeseries(
    df: pd.DataFrame,
    basin_id: str,
    basin_code: int,
    cfg: DataPrepConfig,
    forcing_feature_cols: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, np.ndarray]], List[str], List[str], str, int, int, int]:
    rows_before = len(df)
    df = df.replace([-999, -999.0, -99, -99.0], np.nan)

    target_col = "QObs(mm/day)" if cfg.target_unit == "mm/day" else "QObs(cfs)"

    if forcing_feature_cols is None:
        base_non_feature_cols = {"basin_id", "date", "quality_flag", "QObs(cfs)", "QObs(mm/day)"}
        forcing_feature_cols = [c for c in df.columns if c not in base_non_feature_cols]
        forcing_feature_cols = [c for c in forcing_feature_cols if c != "area_km2"]

    model_df = df.dropna(subset=forcing_feature_cols + [target_col]).copy()
    negative_target_rows = int((model_df[target_col] < 0).sum())
    if negative_target_rows > 0:
        model_df = model_df[model_df[target_col] >= 0].copy()
    rows_after = len(model_df)
    if rows_after == 0:
        return None, forcing_feature_cols, [], target_col, rows_before, rows_after, negative_target_rows

    model_df = add_time_features(model_df)
    if cfg.include_past_streamflow:
        model_df[PAST_STREAMFLOW_FEATURE] = model_df[target_col].astype(np.float32)
    model_df["basin_code"] = np.int32(basin_code)

    feature_cols = list(forcing_feature_cols)
    if cfg.include_past_streamflow:
        feature_cols.append(PAST_STREAMFLOW_FEATURE)
    feature_cols += ["doy_sin", "doy_cos", "basin_code"]

    basin_payload = {
        "basin_id": basin_id,
        "features": model_df[feature_cols].to_numpy(dtype=np.float32),
        "targets": model_df[target_col].to_numpy(dtype=np.float32),
        "dates": model_df["date"].to_numpy(dtype="datetime64[D]"),
    }
    return basin_payload, forcing_feature_cols, feature_cols, target_col, rows_before, rows_after, negative_target_rows


def build_indexed_dataset(
    cfg: DataPrepConfig,
) -> Tuple[Dict[str, np.ndarray], Dict]:
    forcing_zip, actual_forcing_product, forcing_source_notes = choose_forcing_source(cfg)

    forcing_paths = index_camels_zip(forcing_zip, actual_forcing_product)
    target_paths = index_camels_zip(cfg.timeseries_zip, actual_forcing_product)

    available_basin_ids = sorted(set(forcing_paths.forcing) & set(target_paths.streamflow))
    requested_basin_ids = [_normalise_basin_id(x) for x in cfg.basin_ids]
    selected_basin_ids = requested_basin_ids or available_basin_ids
    missing = sorted(set(selected_basin_ids) - set(available_basin_ids))
    selected_basin_ids = [b for b in selected_basin_ids if b in available_basin_ids]

    print(f"Forcing archive: {forcing_zip}", flush=True)
    print(f"Forcing product: {actual_forcing_product}", flush=True)
    for note in forcing_source_notes:
        print(f"Note: {note}", flush=True)
    print(f"Target archive:  {cfg.timeseries_zip}", flush=True)
    print(f"Complete basin pairs:   {len(available_basin_ids):,}", flush=True)
    print(f"Selected basins:        {len(selected_basin_ids):,}", flush=True)
    if missing:
        print(
            f"Warning: skipped {len(missing)} requested basins without a complete pair: {missing[:10]}",
            flush=True,
        )

    gauge_df = read_gauge_info(cfg.timeseries_zip, target_paths.gauge_info)
    area_by_basin = gauge_df.set_index("basin_id")["area_km2"]
    basin_code_by_basin = basin_code_map(selected_basin_ids)

    feature_parts: List[np.ndarray] = []
    target_parts: List[np.ndarray] = []
    row_date_parts: List[np.ndarray] = []
    row_basin_index_parts: List[np.ndarray] = []
    window_start_parts: List[np.ndarray] = []
    target_date_parts: List[np.ndarray] = []
    window_basin_index_parts: List[np.ndarray] = []
    basins_in_model: List[str] = []

    forcing_feature_cols: Optional[List[str]] = None
    feature_cols: Optional[List[str]] = None
    target_col: Optional[str] = None
    normalize_feature_idx: Optional[np.ndarray] = None
    total_rows = 0
    joined_rows_before_na = 0
    model_rows_after_na = 0
    negative_target_rows_dropped = 0
    date_min: Optional[np.datetime64] = None
    date_max: Optional[np.datetime64] = None

    start = pd.Timestamp(cfg.start_date) if cfg.start_date else None
    end = pd.Timestamp(cfg.end_date) if cfg.end_date else None

    with zipfile.ZipFile(forcing_zip) as forcing_zf, zipfile.ZipFile(cfg.timeseries_zip) as target_zf:
        for i, basin_id in enumerate(selected_basin_ids, start=1):
            forcing = read_forcing(forcing_zf, forcing_paths.forcing[basin_id], basin_id)
            streamflow = read_streamflow(target_zf, target_paths.streamflow[basin_id], basin_id)

            df = forcing.merge(streamflow, on=["basin_id", "date"], how="inner")
            if start is not None:
                df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]

            area = area_by_basin.get(basin_id, np.nan)
            df["area_km2"] = area
            df["QObs(mm/day)"] = cfs_to_mm_per_day(df["QObs(cfs)"], area)

            (
                basin_payload,
                forcing_feature_cols,
                current_feature_cols,
                target_col,
                rows_before,
                rows_after,
                negative_rows,
            ) = process_basin_timeseries(
                df=df,
                basin_id=basin_id,
                basin_code=basin_code_by_basin.get(basin_id, 0),
                cfg=cfg,
                forcing_feature_cols=forcing_feature_cols,
            )
            joined_rows_before_na += rows_before
            model_rows_after_na += rows_after
            negative_target_rows_dropped += negative_rows

            if basin_payload is None:
                if i % 50 == 0 or i == len(selected_basin_ids):
                    print(f"Loaded {i:>4}/{len(selected_basin_ids)} basins", flush=True)
                continue

            basins_in_model.append(basin_id)
            feature_cols = current_feature_cols

            if normalize_feature_idx is None:
                normalize_feature_idx = np.array(
                    [i for i, c in enumerate(feature_cols) if c not in NON_NORMALIZED_FEATURES],
                    dtype=np.int64,
                )

            features = basin_payload["features"].copy()
            targets = basin_payload["targets"].copy()
            dates = basin_payload["dates"]

            feature_parts.append(features)
            target_parts.append(targets)
            row_date_parts.append(dates.astype("datetime64[D]"))
            basin_slot = np.int32(len(basins_in_model) - 1)
            row_basin_index_parts.append(np.full(len(features), basin_slot, dtype=np.int32))

            if date_min is None or dates[0] < date_min:
                date_min = dates[0]
            if date_max is None or dates[-1] > date_max:
                date_max = dates[-1]

            max_start = len(features) - cfg.lookback_days - cfg.forecast_horizon_days - cfg.prediction_length + 2
            if max_start > 0:
                local_starts = np.arange(0, max_start, cfg.stride_days, dtype=np.int32)
                target_local_idx = local_starts + cfg.lookback_days + cfg.forecast_horizon_days - 1
                window_start_parts.append(local_starts + np.int32(total_rows))
                target_date_parts.append(dates[target_local_idx].astype("datetime64[D]"))
                window_basin_index_parts.append(
                    np.full(len(local_starts), basin_slot, dtype=np.int32)
                )

            total_rows += len(features)

            if i % 50 == 0 or i == len(selected_basin_ids):
                print(f"Loaded {i:>4}/{len(selected_basin_ids)} basins", flush=True)

    all_features = np.concatenate(feature_parts, axis=0).astype(np.float32, copy=False)
    all_targets = np.concatenate(target_parts, axis=0).astype(np.float32, copy=False)
    row_dates = (
        np.concatenate(row_date_parts, axis=0).astype("datetime64[D]")
        if row_date_parts
        else np.empty(0, dtype="datetime64[D]")
    )
    row_basin_index = (
        np.concatenate(row_basin_index_parts, axis=0)
        if row_basin_index_parts
        else np.empty(0, dtype=np.int32)
    )
    window_start = np.concatenate(window_start_parts, axis=0) if window_start_parts else np.empty(0, dtype=np.int32)
    target_dates = (
        np.concatenate(target_date_parts, axis=0).astype("datetime64[D]")
        if target_date_parts
        else np.empty(0, dtype="datetime64[D]")
    )
    window_basin_index = (
        np.concatenate(window_basin_index_parts, axis=0)
        if window_basin_index_parts
        else np.empty(0, dtype=np.int32)
    )

    print(f"Joined rows before NA filtering: {joined_rows_before_na:,}", flush=True)
    print(f"Model rows after filtering:      {model_rows_after_na:,}", flush=True)
    print(f"Basins in model data:            {len(basins_in_model):,}", flush=True)
    print(f"Total windows before max_windows cap: {len(window_start):,}", flush=True)
    if date_min is not None and date_max is not None:
        print(
            f"Date range: {str(date_min)} to {str(date_max)}",
            flush=True,
        )
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}", flush=True)

    if cfg.max_windows is not None and len(window_start) > cfg.max_windows:
        n_basins_in_model = len(basins_in_model)
        if cfg.max_windows < n_basins_in_model:
            print(
                f"Warning: max_windows={cfg.max_windows} is smaller than the number of "
                f"basins ({n_basins_in_model}); keeping one window per basin (total "
                f"{n_basins_in_model}) so every basin is represented in the dataset.",
                flush=True,
            )
            target_per_basin = 1
        else:
            target_per_basin = cfg.max_windows // n_basins_in_model

        keep_indices_list: List[np.ndarray] = []
        for basin_slot in range(n_basins_in_model):
            basin_window_idx = np.flatnonzero(window_basin_index == basin_slot)
            if len(basin_window_idx) <= target_per_basin:
                keep_indices_list.append(basin_window_idx)
            else:
                # Uniform chronological sampling preserves train/val/test coverage per basin.
                sampled = np.linspace(
                    0, len(basin_window_idx) - 1, target_per_basin
                ).round().astype(np.int64)
                sampled = np.unique(sampled)
                keep_indices_list.append(basin_window_idx[sampled])
        keep = np.sort(np.concatenate(keep_indices_list)) if keep_indices_list else np.empty(0, dtype=np.int64)

        before_count = len(window_start)
        window_start = window_start[keep]
        target_dates = target_dates[keep]
        window_basin_index = window_basin_index[keep]
        print(
            f"Stratified per-basin subsample: kept {len(keep):,} of {before_count:,} "
            f"windows (target ≈{target_per_basin}/basin, max_windows={cfg.max_windows}).",
            flush=True,
        )

    if negative_target_rows_dropped > 0:
        print(
            f"Dropped negative runoff rows:    {negative_target_rows_dropped:,}",
            flush=True,
        )

    normalize_feature_indices = (
        normalize_feature_idx.astype(np.int64)
        if normalize_feature_idx is not None
        else np.empty(0, dtype=np.int64)
    )
    static_features, static_feature_cols, static_metadata = load_static_attributes(
        cfg.data_dir,
        basins_in_model,
        DEFAULT_STATIC_ATTRIBUTE_GROUPS,
    )
    print(
        f"Static attribute columns ({len(static_feature_cols)}): {static_feature_cols}",
        flush=True,
    )

    dataset_payload = {
        "dataset_format": np.array(["flat_indexed_v3"]),
        "features": all_features,
        "targets": all_targets,
        "row_date": row_dates.astype("datetime64[D]").astype(str),
        "row_basin_index": row_basin_index,
        "window_start": window_start,
        "window_basin_index": window_basin_index,
        "target_date": target_dates.astype("datetime64[D]").astype(str),
        "feature_columns": np.array(feature_cols),
        "target_name": np.array(["target_runoff"]),
        "lookback_days": np.array([cfg.lookback_days], dtype=np.int32),
        "forecast_horizon_days": np.array([cfg.forecast_horizon_days], dtype=np.int32),
        "prediction_length": np.array([cfg.prediction_length], dtype=np.int32),
        "basin_ids_in_model": np.array(basins_in_model),
        "static_features": static_features,
        "static_feature_columns": np.array(static_feature_cols),
        "normalize_feature_indices": normalize_feature_indices,
        "target_transform": np.array([TARGET_TRANSFORM]),
    }

    metadata = {
        "forcing_archive": str(forcing_zip),
        "target_archive": str(cfg.timeseries_zip),
        "requested_forcing_product": cfg.forcing_product,
        "actual_forcing_product": actual_forcing_product,
        "target_unit": cfg.target_unit,
        "target_column_in_joined_file": target_col,
        "target_column_in_target_file": "target_runoff",
        "feature_columns": feature_cols,
        "basin_code_map": {basin_id: basin_code_by_basin.get(basin_id, 0) for basin_id in basins_in_model},
        "selected_basin_ids": selected_basin_ids,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "forcing_source_notes": forcing_source_notes,
        "static_attributes": static_metadata,
        "available_basin_ids": available_basin_ids,
        "missing_basin_ids": missing,
        "joined_rows_before_na": joined_rows_before_na,
        "model_rows_after_na": model_rows_after_na,
        "negative_target_rows_dropped": negative_target_rows_dropped,
        "basins_in_model": basins_in_model,
    }
    return dataset_payload, metadata


def load_camels_tables(
    cfg: DataPrepConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], str, Dict, Dict]:
    forcing_zip, actual_forcing_product, forcing_source_notes = choose_forcing_source(cfg)

    forcing_paths = index_camels_zip(forcing_zip, actual_forcing_product)
    target_paths = index_camels_zip(cfg.timeseries_zip, actual_forcing_product)

    available_basin_ids = sorted(set(forcing_paths.forcing) & set(target_paths.streamflow))
    requested_basin_ids = [_normalise_basin_id(x) for x in cfg.basin_ids]
    selected_basin_ids = requested_basin_ids or available_basin_ids
    missing = sorted(set(selected_basin_ids) - set(available_basin_ids))
    selected_basin_ids = [b for b in selected_basin_ids if b in available_basin_ids]

    print(f"Forcing archive: {forcing_zip}", flush=True)
    print(f"Forcing product: {actual_forcing_product}", flush=True)
    for note in forcing_source_notes:
        print(f"Note: {note}", flush=True)
    print(f"Target archive:  {cfg.timeseries_zip}", flush=True)
    print(f"Complete basin pairs:   {len(available_basin_ids):,}", flush=True)
    print(f"Selected basins:        {len(selected_basin_ids):,}", flush=True)
    if missing:
        print(
            f"Warning: skipped {len(missing)} requested basins without a complete pair: {missing[:10]}",
            flush=True,
        )

    gauge_df = read_gauge_info(cfg.timeseries_zip, target_paths.gauge_info)
    area_by_basin = gauge_df.set_index("basin_id")["area_km2"]

    records = []
    start = pd.Timestamp(cfg.start_date) if cfg.start_date else None
    end = pd.Timestamp(cfg.end_date) if cfg.end_date else None

    with zipfile.ZipFile(forcing_zip) as forcing_zf, zipfile.ZipFile(cfg.timeseries_zip) as target_zf:
        for i, basin_id in enumerate(selected_basin_ids, start=1):
            forcing = read_forcing(forcing_zf, forcing_paths.forcing[basin_id], basin_id)
            streamflow = read_streamflow(target_zf, target_paths.streamflow[basin_id], basin_id)

            df = forcing.merge(streamflow, on=["basin_id", "date"], how="inner")
            if start is not None:
                df = df[df["date"] >= start]
            if end is not None:
                df = df[df["date"] <= end]

            area = area_by_basin.get(basin_id, np.nan)
            df["area_km2"] = area
            df["QObs(mm/day)"] = cfs_to_mm_per_day(df["QObs(cfs)"], area)
            records.append(df)

            if i % 50 == 0 or i == len(selected_basin_ids):
                print(f"Loaded {i:>4}/{len(selected_basin_ids)} basins", flush=True)

    joined_df = pd.concat(records, ignore_index=True)
    joined_df = joined_df.sort_values(["basin_id", "date"]).reset_index(drop=True)
    joined_df = joined_df.replace([-999, -999.0, -99, -99.0], np.nan)

    target_col = "QObs(mm/day)" if cfg.target_unit == "mm/day" else "QObs(cfs)"

    base_non_feature_cols = {"basin_id", "date", "quality_flag", "QObs(cfs)", "QObs(mm/day)"}
    forcing_feature_cols = [c for c in joined_df.columns if c not in base_non_feature_cols]
    forcing_feature_cols = [c for c in forcing_feature_cols if c != "area_km2"]

    model_df = joined_df.dropna(subset=forcing_feature_cols + [target_col]).copy()
    negative_target_rows_dropped = int((model_df[target_col] < 0).sum())
    if negative_target_rows_dropped > 0:
        model_df = model_df[model_df[target_col] >= 0].copy()

    model_df = add_time_features(model_df)
    if cfg.include_past_streamflow:
        model_df[PAST_STREAMFLOW_FEATURE] = model_df[target_col].astype(np.float32)
    code_by_basin = basin_code_map(model_df["basin_id"])
    model_df["basin_code"] = model_df["basin_id"].map(code_by_basin).astype("int32")

    feature_cols = list(forcing_feature_cols)
    if cfg.include_past_streamflow:
        feature_cols.append(PAST_STREAMFLOW_FEATURE)
    feature_cols += ["doy_sin", "doy_cos", "basin_code"]

    input_df = model_df[["basin_id", "date"] + feature_cols].copy()
    target_df = model_df[["basin_id", "date", target_col]].rename(
        columns={target_col: "target_runoff"}
    )

    print(f"Joined rows before NA filtering: {len(joined_df):,}", flush=True)
    print(f"Model rows after filtering:      {len(model_df):,}", flush=True)
    if negative_target_rows_dropped > 0:
        print(
            f"Dropped negative runoff rows:    {negative_target_rows_dropped:,}",
            flush=True,
        )
    print(f"Basins in model data:            {model_df['basin_id'].nunique():,}", flush=True)
    print(
        f"Date range: {model_df['date'].min().date()} to {model_df['date'].max().date()}",
        flush=True,
    )
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}", flush=True)

    metadata = {
        "forcing_archive": str(forcing_zip),
        "target_archive": str(cfg.timeseries_zip),
        "requested_forcing_product": cfg.forcing_product,
        "actual_forcing_product": actual_forcing_product,
        "target_unit": cfg.target_unit,
        "target_column_in_joined_file": target_col,
        "target_column_in_target_file": "target_runoff",
        "feature_columns": feature_cols,
        "basin_code_map": code_by_basin,
        "selected_basin_ids": selected_basin_ids,
        "start_date": cfg.start_date,
        "end_date": cfg.end_date,
        "forcing_source_notes": forcing_source_notes,
        "negative_target_rows_dropped": negative_target_rows_dropped,
    }
    debug_info = {
        "available_basin_ids": available_basin_ids,
        "missing_basin_ids": missing,
    }
    return input_df, target_df, model_df, feature_cols, target_col, metadata, debug_info


def build_windows_for_basin(
    basin_df: pd.DataFrame,
    feature_columns: Sequence[str],
    target_column: str,
    lookback_days: int,
    horizon_days: int,
    prediction_length: int,
    stride_days: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], List[np.datetime64]]:
    basin_df = basin_df.sort_values("date").reset_index(drop=True)
    values = basin_df[list(feature_columns)].to_numpy(dtype=np.float32)
    targets = basin_df[target_column].to_numpy(dtype=np.float32)
    dates = basin_df["date"].to_numpy()
    basin_ids = basin_df["basin_id"].to_numpy()

    X_parts: List[np.ndarray] = []
    y_parts: List[np.ndarray] = []
    basin_parts: List[str] = []
    date_parts: List[np.datetime64] = []

    max_start = len(basin_df) - lookback_days - horizon_days - prediction_length + 2
    for start_idx in range(0, max(0, max_start), stride_days):
        end_idx = start_idx + lookback_days
        target_start_idx = end_idx + horizon_days - 1
        target_end_idx = target_start_idx + prediction_length
        X_parts.append(values[start_idx:end_idx])
        y_parts.append(targets[target_start_idx:target_end_idx])
        basin_parts.append(basin_ids[target_start_idx])
        date_parts.append(dates[target_start_idx])

    return X_parts, y_parts, basin_parts, date_parts


def build_window_dataset(
    model_df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    cfg: DataPrepConfig,
) -> Dict[str, np.ndarray]:
    X_all: List[np.ndarray] = []
    y_all: List[np.ndarray] = []
    basin_all: List[str] = []
    date_all: List[np.datetime64] = []

    for _, basin_df in model_df.groupby("basin_id", sort=True):
        X, y, basins, dates = build_windows_for_basin(
            basin_df=basin_df,
            feature_columns=feature_cols,
            target_column=target_col,
            lookback_days=cfg.lookback_days,
            horizon_days=cfg.forecast_horizon_days,
            prediction_length=cfg.prediction_length,
            stride_days=cfg.stride_days,
        )
        X_all.extend(X)
        y_all.extend(y)
        basin_all.extend(basins)
        date_all.extend(dates)

    if cfg.max_windows is not None and len(X_all) > cfg.max_windows:
        rng = np.random.default_rng(cfg.random_seed)
        keep = np.sort(rng.choice(len(X_all), size=cfg.max_windows, replace=False))
    else:
        keep = np.arange(len(X_all))

    X_arr = np.stack([X_all[i] for i in keep]).astype(np.float32)
    y_arr = np.stack([y_all[i] for i in keep]).astype(np.float32)
    basin_arr = np.array([basin_all[i] for i in keep])
    date_arr = np.array([np.datetime_as_string(date_all[i], unit="D") for i in keep])

    return {
        "X": X_arr,
        "y": y_arr,
        "basin_id": basin_arr,
        "target_date": date_arr,
        "feature_columns": np.array(feature_cols),
        "target_name": np.array(["target_runoff"]),
    }


def prepare_camels_training_data(cfg: DataPrepConfig) -> Path:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    if cfg.write_flat_files:
        print(
            "write_flat_files=True is not supported in the low-memory indexed pipeline; skipping flat table export.",
            flush=True,
        )

    dataset_payload, metadata = build_indexed_dataset(cfg)
    np.savez_compressed(cfg.dataset_path, **dataset_payload)

    metadata["dataset_path"] = str(cfg.dataset_path)
    metadata["lookback_days"] = cfg.lookback_days
    metadata["forecast_horizon_days"] = cfg.forecast_horizon_days
    metadata["prediction_length"] = cfg.prediction_length
    metadata["stride_days"] = cfg.stride_days
    metadata["max_windows"] = cfg.max_windows
    metadata["dataset_format"] = "flat_indexed_v3"
    metadata["target_transform"] = TARGET_TRANSFORM
    metadata["non_normalized_features"] = sorted(NON_NORMALIZED_FEATURES)
    metadata["data_prep_config"] = config_to_dict(cfg)
    cfg.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote windowed transformer data: {cfg.dataset_path}", flush=True)
    print(f"features shape: {dataset_payload['features'].shape}", flush=True)
    print(f"static features shape: {dataset_payload['static_features'].shape}", flush=True)
    print(f"targets shape: {dataset_payload['targets'].shape}", flush=True)
    print(f"window count: {len(dataset_payload['window_start'])}", flush=True)
    print(f"Wrote metadata: {cfg.metadata_path}", flush=True)
    return cfg.dataset_path


def parse_args() -> DataPrepConfig:
    parser = argparse.ArgumentParser(description="Prepare CAMELS windowed runoff training data.")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--forcing-product", type=str, default="nldas")
    parser.add_argument(
        "--allow-daymet-model-output-fallback",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--basin-id", action="append", default=[])
    parser.add_argument("--start-date", type=str, default="1980-01-01")
    parser.add_argument("--end-date", type=str, default="2014-12-31")
    parser.add_argument("--target-unit", type=str, default="mm/day")
    parser.add_argument(
        "--include-past-streamflow",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lookback-days", type=int, default=365)
    parser.add_argument("--forecast-horizon-days", type=int, default=1)
    parser.add_argument("--prediction-length", type=int, default=1)
    parser.add_argument("--stride-days", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--write-flat-files",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    args = parser.parse_args()
    return DataPrepConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        forcing_product=args.forcing_product,
        allow_daymet_model_output_fallback=args.allow_daymet_model_output_fallback,
        basin_ids=args.basin_id,
        start_date=args.start_date,
        end_date=args.end_date,
        target_unit=args.target_unit,
        include_past_streamflow=args.include_past_streamflow,
        lookback_days=args.lookback_days,
        forecast_horizon_days=args.forecast_horizon_days,
        prediction_length=args.prediction_length,
        stride_days=args.stride_days,
        max_windows=args.max_windows,
        random_seed=args.random_seed,
        write_flat_files=args.write_flat_files,
    )


if __name__ == "__main__":
    prepare_camels_training_data(parse_args())
