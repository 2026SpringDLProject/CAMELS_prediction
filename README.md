# CAMELS Data Loading

Utilities for preparing CAMELS basin time-series data for multi-basin transformer runoff modeling.

## Goal

CAMELS time-series data is organized by basin ID. For each basin, the meteorological forcing time series and the observed streamflow time series are stored as basin-specific files. For example, inside the main CAMELS time-series archive:

```text
basin_mean_forcing/nldas/<region>/<basin_id>_lump_nldas_forcing_leap.txt
usgs_streamflow/<region>/<basin_id>_streamflow_qc.txt
```

The forcing file contains daily meteorological variables such as precipitation, radiation, snow water equivalent, temperature, and vapor pressure. The streamflow file contains observed daily discharge from USGS.

This repository prepares data for multi-basin transformer training. The starting point is to train one transformer model across many basins, rather than training one separate model per basin. A one-model-per-basin setup can still be used later as a comparison experiment.

The notebook reads basin-specific CAMELS files, aligns forcing and streamflow by date, concatenates data from selected basins, and writes model-ready files that can be used for transformer training.

## Data Location

Download the CAMELS data from Zenodo:

<https://zenodo.org/records/15529996>

Place or extract the downloaded files so the project layout is:

```text
courseProject/
├── 15529996/
│   ├── basin_timeseries_v1p2_metForcing_obsFlow.zip
│   ├── basin_timeseries_v1p2_modelOutput_daymet.zip      # optional fallback, if valid
│   ├── basin_timeseries_v1p2_modelOutput_nldas.zip       # optional, if available
│   ├── camels_attributes_v2.0.xlsx
│   └── ...
└── CAMELS_data_load/
    ├── README.md
    └── prepare_camels_transformer_data.ipynb
```

The notebook currently expects the data folder at:

```text
/Users/xshan/Research/GT/cs7643_DL/courseProject/15529996
```

The main required file is:

```text
basin_timeseries_v1p2_metForcing_obsFlow.zip
```

This archive contains both observed streamflow targets and basin-mean meteorological forcing variables such as NLDAS, DAYMET, and Maurer forcing.

Generated model-ready files are written to:

```text
CAMELS_data_load/processed/
```

The `processed/` folder is ignored by Git because it can become large.

## Python Environment

The notebook was tested with the local conda/mamba environment named `cs7643`:

```bash
mamba activate cs7643
```

Essential packages:

```bash
pip install jupyter notebook numpy pandas pyarrow torch matplotlib tqdm
```

Minimum packages needed for data preparation are:

```bash
pip install numpy pandas pyarrow
```

`pyarrow` is used to write Parquet files. The notebook also writes CSV files.

## Data Organization

The notebook first creates flat basin-day tables. Each row corresponds to one basin on one date.

Conceptually, the joined table looks like:

```text
basin_id   date         PRCP   SRAD   SWE   Tmax   Tmin   Vp   doy_sin   doy_cos   basin_code   target_runoff
03366500   1980-01-01   ...    ...    ...   ...    ...    ...  ...       ...       0            ...
03366500   1980-01-02   ...    ...    ...   ...    ...    ...  ...       ...       0            ...
03366500   1980-01-03   ...    ...    ...   ...    ...    ...  ...       ...       0            ...
01013500   1980-01-01   ...    ...    ...   ...    ...    ...  ...       ...       1            ...
01013500   1980-01-02   ...    ...    ...   ...    ...    ...  ...       ...       1            ...
```

The flat files are useful for inspection, filtering, debugging, and reproducibility. The notebook writes:

```text
processed/camels_transformer_inputs.csv
processed/camels_transformer_inputs.parquet
processed/camels_transformer_targets.csv
processed/camels_transformer_targets.parquet
processed/camels_transformer_joined.csv
processed/camels_transformer_joined.parquet
processed/camels_transformer_metadata.json
```

The default target is `QObs(mm/day)`, which is observed streamflow converted from raw USGS discharge `QObs(cfs)` into basin-area-normalized runoff depth. This is usually better for multi-basin training because raw discharge depends strongly on basin size.

## Transformer Samples

A transformer usually does not train on one basin-day row at a time. Instead, each training sample is a sequence window from one basin.

For example, with:

```python
LOOKBACK_DAYS = 365
FORECAST_HORIZON_DAYS = 1
```

one sample means:

```text
Input:  forcing variables from the previous 365 days for one basin
Target: runoff on the next day
```

The transformer input tensor shape is:

```python
X.shape = (num_windows, lookback_days, num_features)
```

During training, each mini-batch usually has shape:

```python
batch_X.shape = (batch_size, lookback_days, num_features)
batch_y.shape = (batch_size,)
```

or:

```python
batch_y.shape = (batch_size, 1)
```

depending on the model and loss-function implementation.

With the notebook defaults, the feature columns are:

```python
[
    'Dayl(s)',
    'PRCP(mm/day)',
    'SRAD(W/m2)',
    'SWE(mm)',
    'Tmax(C)',
    'Tmin(C)',
    'Vp(Pa)',
    'doy_sin',
    'doy_cos',
    'basin_code',
]
```

So with a 365-day lookback, the transformer input shape for one batch might be:

```python
batch_X.shape = (batch_size, 365, 10)
```

Each sequence window should contain data from only one basin. The notebook groups by `basin_id` before creating windows, so a single sample never mixes days from different basins. The final training dataset pools windows from all selected basins, allowing one model to learn across the full basin set.

For smaller experiments, the notebook can export fixed-length sequence windows directly:

```text
processed/camels_transformer_windows.npz
```

This file contains arrays like:

```python
X.shape = (num_windows, lookback_days, num_features)
y.shape = (num_windows,)
```

For the full CAMELS dataset, materializing all possible windows can be very large. In that case, a later training script may prefer to read the flat Parquet/CSV files and create windows on the fly with a PyTorch `Dataset`.

## Notebook

Open and run:

```text
prepare_camels_transformer_data.ipynb
```

The notebook can:

- select one basin, several basins, or all complete basins;
- load meteorological forcing inputs;
- load observed streamflow targets;
- convert streamflow from `QObs(cfs)` to runoff depth `QObs(mm/day)`;
- write both CSV and Parquet input/target files;
- optionally export fixed-length transformer windows as `.npz`.

To choose basins, edit `BASIN_IDS` in the notebook configuration cell. Leave it empty to use all basins with complete forcing and streamflow data:

```python
BASIN_IDS = []
```
