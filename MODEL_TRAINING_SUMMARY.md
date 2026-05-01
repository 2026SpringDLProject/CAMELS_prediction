# Current Model And Training Summary

## 1. Overall Positioning

The current model is a simplified Temporal Fusion Transformer (TFT) style runoff forecasting model implemented in PyTorch.

It is **not** a full reproduction of the original TFT paper. The current implementation keeps the following ideas:

- variable selection over real-valued input features (VSN);
- a learned basin embedding plus 39 numeric CAMELS static catchment attributes for multi-basin training;
- an encoder with LSTM (initialized from basin context) plus self-attention;
- a decoder with horizon embeddings, optional future-known features, and cross-attention to the encoder for multi-step prediction.

It omits or simplifies several full-TFT components:

- static attributes are encoded into the basin context used by the encoder/LSTM/decoder, but the VSN weight network still does not condition directly on static context;
- the GRN uses a self-gate `x * sigmoid(W·x + b)` rather than the full GLU `(W₁·x + b₁) * sigmoid(W₂·x + b₂)`;
- the VSN weight network reads the **raw** feature vector instead of the concatenated, per-feature transformed representations together with a static context;
- no quantile output, no interpretable multi-head attention block in the original form;
- the decoder LSTM is zero-initialized — encoder→decoder handoff happens only through cross-attention.

## 2. Input And Target Definition

Training data is generated from CAMELS forcing and observed streamflow records.

Default input feature set:

- `Dayl(s)`
- `PRCP(mm/day)`
- `SRAD(W/m2)`
- `SWE(mm)`
- `Tmax(C)`
- `Tmin(C)`
- `Vp(Pa)`
- `past_streamflow` (optional; enabled with `include_past_streamflow = true`)
- `doy_sin`
- `doy_cos`
- `basin_code`

When enabled, `past_streamflow` is copied from the observed runoff/flow column for rows inside the historical lookback window only. The prediction target date remains outside the input window, so this provides autoregressive hydrologic state information without leaking the forecast-day target.

Default static context:

- 39 numeric CAMELS attributes from `camels_topo.txt`, `camels_clim.txt`, `camels_soil.txt`, `camels_vege.txt`, and `camels_geol.txt`
- `camels_hydro.txt` is excluded by default because those attributes are derived from observed streamflow and can leak target information into runoff prediction

Target:

- `QObs(mm/day)` by default

The data pipeline builds fixed windows:

- dynamic input window: shape `[N, lookback_days, num_features]`
- static input context: shape `[N, num_static_features]`, gathered from per-basin `static_features`
- `y`: shape `[N, prediction_length]` for multi-step prediction (squeezed to `[N]` when `prediction_length == 1`)

Default settings in `config/train.json`:

- `lookback_days = 90`
- `forecast_horizon_days = 1`
- `prediction_length = 1`
- `stride_days = 30`

This means the model uses the previous 90 days to predict runoff beginning 1 day after the end of the lookback window.

### 2.1 Storage And Normalization Pipeline

The prepared dataset (format `flat_indexed_v3`, `target_transform = "identity"`) stores **raw** features and **raw** targets in mm/day. Normalization is **not** baked into the `.npz` file — it happens at training time.
Static attributes are the exception: the data-prep step stores a per-basin `static_features` matrix that has already been median-imputed and z-scored across the selected basins.

At training time, `compute_train_normalization_stats` computes per-basin statistics using **only** training-window rows (rows whose date is ≤ the latest training-window target date for that basin). This avoids temporal leakage from validation/test periods.

For each basin slot the function produces:

- `feature_mean`, `feature_std` — per-basin z-score stats for the configured feature columns; `basin_code`, `doy_sin`, and `doy_cos` are excluded from normalization;
- `target_mean`, `target_std` — stats of `log1p(target)` per basin; used to z-score the regression target;
- `target_std_raw` — std of raw mm/day targets per basin; used as the fixed denominator inside the Kratzert NSE* loss.

Each `run_epoch` call applies feature z-scoring and target log1p+z-scoring on-the-fly using these per-basin tensors, indexed by `basin_slot`.

## 3. Model Architecture

### 3.1 Input Parsing

`TFTInput` splits the basin code column from the rest of the features:

- real-valued dynamic inputs: all columns except `basin_code`;
- categorical basin identifier: `basin_code`, taken from the first time step (assumed constant across the window).

If `include_past_streamflow` is enabled in the config, historical streamflow is treated as one additional real-valued dynamic input and participates in the same normalization and variable-selection path as the meteorological forcings.

### 3.2 Variable Selection Network (VSN)

The real-valued input features are passed through a Variable Selection Network with three components:

- `FeaturewiseGatedResidualNetwork` — holds **one per-feature GRN** (separate weight tensors per feature, batched in einsum) and lifts each scalar feature into a `d_model` vector. Output shape: `[B, T, F, d_model]`.
- `GatedResidualNetwork` (the weight network) — consumes the **raw** feature vector at each time step and produces `[B, T, F]` logits; input dim = `num_features`, hidden dim = `d_model`, output dim = `num_features`.
- A softmax over the feature axis yields selection weights `[B, T, F]`.
- The weighted sum `(weights * transformed).sum(dim=2)` produces the sequence representation `[B, T, d_model]`.

Note: the weight network here ignores the per-feature transformed representation and any static context. This is a simplification relative to the paper, where weights come from `concat(GRN_ξ_1(ξ_1), …, GRN_ξ_F(ξ_F))` together with a static context vector.

### 3.3 Basin Embedding

The basin code is embedded with `nn.Embedding(num_basins, d_model)`.

When static attributes are enabled, the 39-dimensional static vector is passed through `static_attribute_encoder`, a `GatedResidualNetwork(num_static_features, d_model, d_model)`. The encoded static vector is added to the learned basin embedding to form `basin_context`.

The combined `basin_context` is broadcast across time and added to the VSN output. This gives the model both a learned basin identity signal and explicit catchment attributes while keeping one shared model across many basins.

### 3.4 Encoder

The encoder runs the following stages:

1. The basin context is projected into the LSTM initial state: `h0 = static_to_h0(basin_context)`, `c0 = static_to_c0(basin_context)`. This injects basin identity and explicit static catchment attributes into the recurrence rather than relying only on input-side residual addition.
2. Encoder LSTM `d_model -> lstm_hidden`, single layer, batch-first, initialized with `(h0, c0)`.
3. Multi-head self-attention `lstm_hidden -> lstm_hidden`, no causal mask (the encoder window is fully observed).
4. Post-attention `GatedResidualNetwork` with residual: `encoder_memory = post_attn_grn(x_lstm + attn_out)`. This preserves the LSTM state alongside the attention output rather than replacing it.

Encoder output (`encoder_memory`) shape: `[B, T, lstm_hidden]`.

### 3.5 Decoder

The decoder builds a query sequence of length `prediction_length` purely from static and future-known signals:

- a static context `static_context_proj(basin_context)` projected from `d_model` to `lstm_hidden`, broadcast across all horizon steps;
- a learned horizon embedding `decoder_horizon_embedding(0..prediction_length-1)` that gives each forecast step its own additive query;
- if `num_future_known_features > 0`, future-known features (currently `Dayl(s)`, `doy_sin`, `doy_cos` for each future horizon step) are passed through a single `GatedResidualNetwork` and added to the decoder input.

The decoder input therefore has shape `[B, prediction_length, lstm_hidden]`. It does **not** include the encoder's final time step directly — the encoder is reached only through cross-attention.

The decoder then applies:

1. Decoder LSTM (zero-initialized hidden/cell state — no encoder→decoder state handoff);
2. Cross-attention from decoder states (queries) to `encoder_memory` (keys, values);
3. Post-decoder GRN with residual: `decoder_out = post_decoder_grn(decoder_out + cross_attn_out)`.

This produces either a single-step scalar forecast (when `prediction_length == 1`) or a multi-step sequence forecast.

### 3.6 Output Head

The prediction head is:

- `Linear(lstm_hidden, lstm_hidden)`
- `ReLU`
- `Dropout`
- `Linear(lstm_hidden, 1)`

Output shape:

- single-step (`prediction_length == 1`): `[B]` (final dim is squeezed)
- multi-step: `[B, prediction_length]`

The model's raw output lives in **z-scored log1p space** of the target. It is denormalized to physical mm/day inside the NSE* loss (for training) and inside `denormalize_targets` (for metric reporting).

### 3.7 LSTM Baseline

The training script now supports `model_name = "lstm"` as an LSTM-only baseline. It uses the same dataset, normalization, loss functions, split strategy, and reporting metrics as the TFT model, so differences in validation/test performance are attributable to architecture rather than the training pipeline.

The LSTM baseline keeps:

- the same `TFTInput` parsing of dynamic real-valued features and `basin_code`;
- basin embedding plus optional static catchment attributes as a basin context vector;
- an encoder LSTM over the historical lookback window;
- a decoder LSTM initialized from the encoder final hidden/cell state;
- optional future-known decoder inputs for multi-step forecasts;
- the same MLP output head shape as the TFT model.

It removes:

- Variable Selection Network;
- encoder self-attention;
- decoder cross-attention;
- gated residual post-processing blocks.

Run it with either:

`python train.py --config config/lstm_baseline.json`

or override an existing config:

`python train.py --model-name lstm --output-dir outputs/lstm_baseline`

### 3.8 Extra Returned Diagnostics

The TFT model also returns auxiliary outputs:

- `variable_weights` — VSN selection weights `[B, T, F_real]`;
- `attention_weights` — encoder self-attention `[B, T, T]`;
- `decoder_attention_weights` — decoder cross-attention `[B, prediction_length, T]`.

These are not used in the loss but can be inspected for analysis. The LSTM baseline returns the same dictionary keys with `None` values so the training and evaluation code can share one forward interface.

## 4. Loss Function And Metrics

### 4.1 Training Loss

The training loss is configurable via `loss_name`:

- `"mse"` → `nn.MSELoss()` evaluated in **z-scored log1p space** (the same space the model outputs);
- `"nse"` → `NSELoss`, the Kratzert et al. (2019) basin-averaged NSE* loss.

The Kratzert NSE* loss is computed per basin within each batch:

`L = (1/B) sum_b mean_{n in b} (y_n − ŷ_n)² / (s_b + ε)²`

where:

- `y`, `ŷ` are observed and predicted streamflow in **physical mm/day**. Predictions are denormalized inside the loss with a `log_clip = 10.0` to bound `expm1`, then clamped to non-negative runoff before scoring;
- `s_b` is the **per-basin training-set standard deviation in mm/day**, precomputed by `compute_train_normalization_stats` as `target_std_raw[basin_slot]` and held fixed for the rest of training. This is what makes the loss stable across mini-batches — a within-batch SST denominator collapses when a basin appears only once per shuffled batch;
- `ε = 0.1` (paper default) prevents division by zero in low-flow basins.

This is **not** `1 − NSE`. Smaller is still better, but the absolute magnitude should be interpreted against ranges expected from the paper, not as a deficit-from-perfect-NSE.

If `basin_slot` or any of the per-basin tensors is missing (e.g. dataset without per-basin metadata), the loss falls back to a plain MSE on whatever space `pred` is in.

### 4.2 Reported Metrics

Each epoch the training loop reports, in addition to loss:

- `MAE` — mean absolute error in **physical mm/day**, pooled over all (batch × horizon) elements;
- `RMSE` — root mean squared error in **physical mm/day**, pooled the same way;
- `NSE` — **basin-averaged** Nash–Sutcliffe Efficiency in mm/day.

The NSE metric is computed by accumulating per-basin SSE, sum, and sum-of-squares (in `float64`) across the entire epoch via `scatter_add_`, then computing `1 − SSE_b / SST_b` per basin and averaging across basins with `count > 1` and `SST_b > 1e-8`. This avoids both naive batch-averaging artifacts and contamination from basins with degenerate variance.

When the target is `QObs(mm/day)`, MAE and RMSE remain in `mm/day`.

### 4.3 Per-Basin Test NSE Output

After loading the best checkpoint and running test evaluation, the script writes `basin_test_nse.json` containing:

- `summary` — `n_basins`, `n_valid`, `median_nse`, `mean_nse`, `p10_nse`, `p25_nse`, `p75_nse`, plus counts of basins below 0, below 0.5, and above 0.7;
- `basins` — every basin's `(basin_id, test_nse, rank)` sorted ascending by NSE so the worst-performing basins appear first; basins with insufficient test data (NaN NSE) are placed last.

This is meant for spotting whether a small number of pathological basins are dragging down the basin-averaged NSE.

## 5. Training Procedure

### 5.1 Dataset Preparation

`train.py` first checks whether the prepared `.npz` dataset matches the current configuration. The expected dataset format is `flat_indexed_v3` and `target_transform = "identity"`.

If the prepared dataset does not match the config (different basin list, lookback, horizon, prediction length, date range, forcing product, target unit, basin/seasonal feature flags) or is missing required keys (`row_date`, `row_basin_index`, `window_basin_index`, `normalize_feature_indices`), the script regenerates data from raw CAMELS archives via `data/prepare_camels_data.py`.

### 5.2 Train / Validation / Test Split

The window dataset is split into train / validation / test with default ratios:

- `train_ratio = 0.7`
- `val_ratio = 0.15`
- `test_ratio = 0.15`

Windows are sorted by `target_date` before splitting, making the split temporal. This also defines the per-basin training cutoff used by `compute_train_normalization_stats` to avoid leakage.

### 5.3 Optimizer

Default optimizer: `AdamW` with `lr = 1e-3` and `weight_decay = 1e-4`.

### 5.4 Learning Rate Scheduling

`ReduceLROnPlateau` monitors `val_loss` with `factor = 0.5`, `patience = 3`. The scheduler stays on the loss (rather than NSE) so the signal it sees is monotonic in the optimization objective.

### 5.5 Gradient Handling

Gradient clipping is enabled with `grad_clip = 1.0`. This stabilizes training for the LSTM + attention stack.

### 5.6 Checkpointing

The best checkpoint is selected by **validation NSE (higher is better)**, with a fallback to validation loss when NSE is NaN (e.g. degenerate single-basin runs). This tracks the hydrology metric more directly than tracking only the training loss.

The checkpoint payload includes:

- `model_state_dict`, `optimizer_state_dict`;
- `config`, `feature_columns`, `static_feature_columns`;
- `best_val_median_nse`, `best_val_nse`, `best_val_loss`.

After training, the best checkpoint is reloaded and final test metrics are computed using that checkpoint.

### 5.7 Early Stopping

Early stopping uses the same criterion as the checkpoint: validation NSE (higher is better), falling back to validation loss when NSE is NaN. Default settings:

- `early_stopping_patience = 8`
- `early_stopping_min_delta = 0.0`

If neither metric improves for the configured patience window, training stops and the best checkpoint so far is kept.

### 5.8 Logging And Outputs

For each epoch the script prints:

- `train_loss`, `train_mae`, `train_rmse`, `train_nse`
- `val_loss`, `val_mae`, `val_rmse`, `val_nse`

After training, the script writes:

- `history.json` — per-epoch train/val metrics;
- `metrics.json` — `best_val_median_nse`, `best_val_nse`, `best_val_loss`, full best validation metrics, test metrics, config;
- `training_curve.png` — train vs val loss / MAE / RMSE / NSE over epochs;
- `basin_test_nse.json` — per-basin test NSE rankings (see 4.3);
- `best_model.pt` — best checkpoint.

## 6. Current Default Hyperparameters

From the current config:

- `d_model = 128`
- `lstm_hidden = 64`
- `n_heads = 4`
- `dropout = 0.16963`
- `batch_size = 256`
- `epochs = 10`
- `early_stopping_patience = 8`
- `loss_name = "nse"`
- `lookback_days = 90`
- `stride_days = 30`
- `prediction_length = 1`
- `include_static_attributes = true`
- `static_attribute_groups = ["topo", "clim", "soil", "vege", "geol"]`

This is a moderate-size model for early experimentation. It is small enough to run on a Colab GPU but still expressive enough to overfit on very few basins.

## 7. Practical Interpretation Of The Current Setup

The current training setup is:

- a supervised multi-basin sequence-to-one or sequence-to-many regression model;
- optimized with MSE (in z-scored log1p space) **or** Kratzert NSE* (in physical mm/day);
- monitored with MAE, RMSE, and basin-averaged NSE in physical mm/day;
- using temporal train/val/test splitting with per-basin train-only normalization stats;
- saving the best checkpoint by validation NSE (with val_loss fallback).

This is a strong experimental baseline for CAMELS runoff forecasting and supports comparing:

- number of basins;
- lookback window length;
- single-step vs multi-step prediction;
- model capacity and regularization;
- MSE vs Kratzert NSE* training objectives.
- forcing-only vs autoregressive input (`include_past_streamflow = false/true`).

## 8. Known Limitations

- it is a simplified TFT, not a full TFT implementation (see Section 1 for the list of simplifications);
- static attributes are encoded into the shared basin context, but they do not yet condition the VSN gates directly as in a fuller TFT-style static covariate path;
- categorical static attributes such as dominant land cover, geology class, and precipitation timing are not yet embedded or one-hot encoded;
- the current objective is point forecasting only — no quantile output or probabilistic forecast;
- the basin embedding path is required (`include_basin_id_feature=True` is enforced);
- single-basin runs are supported but the basin-averaged NSE will be reported over a single basin and may degenerate if the test-period `SST_b` is too small;
- the decoder LSTM has no encoder-state handoff — it relies entirely on cross-attention to access encoder information.

## 9. Recommended Next Improvements

If the project moves beyond the current baseline, the most useful next upgrades are:

- route static context into the VSN weight network so feature selection can vary by catchment attributes;
- add categorical static attributes through embeddings or carefully bounded one-hot encodings;
- add quantile loss for probabilistic forecasting;
- add ensemble training across multiple seeds and report median / IQR test NSE;
- compare against simpler baselines (LSTM-only, persistence, climatology) under the same data setup;
- expand to larger basin subsets (toward the full 671-basin CAMELS-US benchmark) to improve generalization.

## 10. Recommended Comparison Protocol

If this model will be compared against other transformer architectures implemented by different teammates, the most important principle is:

- keep the comparison focused on architectural differences rather than accidental differences in data setup, parameter budget, or optimization recipe.

Below is a recommended alignment checklist.

### 10.1 Data And Split Alignment

All models should use exactly the same:

- basin subset
- date range
- target definition
- lookback window
- forecast horizon
- prediction length
- stride
- train/validation/test split

Recommended concrete rule:

- one shared data config file should define the basin IDs, dates, lookback, horizon, prediction length, and split ratios;
- all models must train on windows generated from that exact same config.

This matters more than almost anything else. If one model sees different basins or an easier validation period, the architectural comparison becomes weak.

### 10.2 Input Feature Alignment

All models should receive the same raw input information unless the experiment is explicitly about feature engineering.

Recommended shared feature protocol:

- same meteorological variables
- same day-of-year features
- same handling of basin identity
- same target scaling or normalization scheme

If one model uses `basin_code` and another does not, that should be treated as an architecture choice and explicitly documented.

### 10.3 Parameter Budget Alignment

Comparisons are most convincing when models have roughly similar capacity.

Instead of only matching names like `d_model`, it is better to align by approximate parameter count.

Recommended rule:

- report total trainable parameter count for every model;
- keep models within a similar parameter budget range, for example within about 10-20 percent if possible.

This is important because a larger model can appear better simply due to capacity rather than a better architecture.

### 10.4 Training Recipe Alignment

All models should use the same optimization setup unless there is a strong reason not to.

Recommended shared training settings:

- same optimizer, preferably `AdamW`
- same base learning rate
- same weight decay
- same batch size
- same maximum epoch count
- same scheduler type
- same gradient clipping rule
- same random seed policy

If one model is unusually sensitive and needs a custom recipe, that should be disclosed as part of the result interpretation.

### 10.5 Loss Function Alignment

For the main comparison, all models should optimize the same primary loss.

Recommended rule:

- use one shared training objective for all models, such as `MSELoss`
- evaluate all models with the same reporting metrics

This helps ensure that differences come from the model rather than the optimization target.

If you also want to compare alternative objectives, treat that as a separate ablation:

- architecture comparison: same loss, different models
- loss comparison: same model, different losses

That separation makes conclusions much cleaner.

### 10.6 Reporting Metrics Alignment

All models should report the same metrics on the same split.

Recommended minimum metrics:

- validation MSE or RMSE
- validation MAE
- test MSE or RMSE
- test MAE

For hydrology-focused reporting, it is also worth aligning on:

- NSE
- R2

Recommended rule:

- choose one primary ranking metric before experiments start
- use the other metrics as supporting evidence

For example:

- primary comparison metric: validation MAE or test MAE
- supporting metrics: RMSE and NSE

### 10.7 Single-Step And Multi-Step Experiments Should Be Separate

Single-step and multi-step forecasting should not be mixed into one comparison table unless clearly separated.

Recommended rule:

- compare single-step models against single-step models
- compare multi-step models against multi-step models

If one model predicts `prediction_length = 1` and another predicts `prediction_length = 10`, those are different tasks, not just different architectures.

### 10.8 Early Stopping And Checkpoint Policy

All models should use the same checkpoint selection rule.

Recommended rule:

- select the best checkpoint using validation loss (or validation NSE — whichever is agreed on);
- evaluate test performance only once using that best checkpoint.

Do not compare one model at last epoch and another at best epoch.

### 10.9 Runtime And Stability Reporting

For a stronger comparison, also record:

- parameter count
- training time per epoch
- total training time
- GPU memory usage if available
- performance variance across multiple seeds

This is especially useful if two models have similar accuracy but one is much slower or less stable.

### 10.10 Recommended Shared Baseline Table

Before full experiments, it is helpful to agree on one common baseline configuration shared by all teams.

Recommended shared baseline items:

- same 20-basin subset or same full-basin subset
- same `lookback_days`
- same `prediction_length`
- same target unit
- same dynamic feature set and static attribute groups
- same optimizer and scheduler
- same epoch budget
- same seed set, for example 3 seeds

Then each model variant can be compared under this same protocol.

### 10.11 Practical Recommendation For This Project

For your current collaboration setting, the cleanest agreement would be:

- same basin subset and date range
- same dynamic input features and static attribute groups
- same `lookback_days`, `forecast_horizon_days`, and `prediction_length`
- same train/val/test split
- same `AdamW` optimizer setup
- same training objective (`MSELoss` or Kratzert NSE*)
- same checkpoint selection rule (current code prefers validation median basin NSE, with fallbacks)
- same reported metrics: MAE plus RMSE, and ideally NSE
- same approximate parameter budget across transformer variants

In short:

- align data
- align optimization
- align evaluation
- keep architecture as the main changing factor

That will make the comparison much more defensible in a report or presentation.
