# EOG-Based Eye Gaze Estimation Pipeline

> Full EOG signal-processing + ML pipeline for gaze classification and regression
> using four EyeCon EOG datasets from the University of Malta Centre for Biomedical Cybernetics.

---

## What This Does

1. **Classifies** 300 ms EOG windows by the trial interval they fall in, taken from each dataset's
   `ControlSignal` (1 = first 1 s interval, 2 = second 1 s interval, 3 = final 2 s blink interval).
   The class names `saccade_onset`, `saccade_return_or_second` and `blink` refer to these whole
   intervals, not to detected saccade/blink events, and `rest` never occurs in the EyeCon data.
2. **Regresses** continuous gaze angle (horizontal + vertical, in degrees) from raw EOG

Models compared:
- Trivial baselines (majority class / train-fold mean angle): the floor every model must beat
- Classical ML (SVC / Random Forest / SVR / XGBoost)
- Deep multi-task Conv1D network (shared backbone, two heads)

---

## Setup

```bash
# 1. Clone / open this folder as your workspace
cd eog-gaze-pipeline

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt
```

---

## ⚠ Dataset Download (Do This First)

### Datasets 1–3 — EyeCon page
Go to: https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/

Download and extract each ZIP into the corresponding directory:

| Dataset | ZIP filename | Extract to |
|---------|-------------|------------|
| 1 — Zero-Centred Bipolar | `DATASET.zip` | `data/raw/Dataset_ZeroCentred/` |
| 2 — Monopolar Stationary | `Dataset_Stationary.zip` | `data/raw/Dataset_Stationary/` |
| 3 — Monopolar Non-Stationary | `Dataset_NonStationary.zip` | `data/raw/Dataset_NonStationary/` |

### Dataset 4 — Google Drive (manual)
Download from the Google Drive link on the EyeCon page.
Extract into: `data/raw/Dataset_Isotropic/`

---

## Phase 0 — Verify Downloads

After extraction, run the inspection script:

```bash
python scripts/inspect_raw.py
```

This will:
- Parse one raw trial per dataset
- Print `fs`, channel names, duration, event timestamps
- Save one plot per dataset to `reports/figures/`

**Fill the table below from the script output — do not guess:**

### Data Description Table
*(Populate after running `scripts/inspect_raw.py`)*

| Field | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
|-------|-----------|-----------|-----------|-----------|
| Sampling rate (Hz) | 256 | 256 | 256 | 256 |
| Channel names | `ControlSignal`, `EOG_0`, `EOG_1` | `ControlSignal`, `EOG_0`..`EOG_3` | `ControlSignal`, `EOG_0`..`EOG_3` | `ControlSignal`, `EOG_0`..`EOG_15` |
| Channel count | 3 | 5 | 5 | 17 |
| Electrode positions | Zero-centred Bipolar | Monopolar | Monopolar | Monopolar Isotropic |
| Recording duration, S1 (s) | 1220.07 | 812.56 | 803.93 | 1171.18 |
| Number of subjects | 6 | 10 | 8 | 14 |
| Trials per subject | 300 × 4 s | 200 × 4 s | 200 × 4 s | 288 × 4 s |
| Label file format | `.mat` (Discrete targets) | `.mat` (Continuous) | `.mat` (Continuous) | `.mat` (Continuous) |
| Target angle units | Degrees (°) | Degrees (°) | Degrees (°) | Degrees (°) |

Sampling rates, subject and trial counts are from each dataset's Data Description PDF
(g.tec g.USBamp, Fs = 256 Hz). The loader cannot text-parse the PDFs, so the rates live in
`CFG.data.fs_hz_by_dataset`; an earlier version silently fell back to 250 Hz.

---

## Run the Tests

```bash
pytest tests/ -v
```

Tests that require data will skip cleanly if data is not yet downloaded.
**`test_no_subject_leakage.py` never skips — it uses synthetic data.**

---

## Run the Full Pipeline

Run the entire pipeline end-to-end with the unified CLI runner:

```bash
# Run all phases (inspection, preprocessing, classical models, deep training, ablations, comparison)
python main.py --phase all

# Or run specific phases individually:
python main.py --phase inspect          # Phase 0: Inspect raw datasets & plot signals
python main.py --phase preprocess       # Phases 1–5: Unify, filter, normalize, segment & save CV folds
python main.py --phase train_classical  # Phase 6: Run SVC, RF, SVR, XGBoost + trivial baselines
python main.py --phase train_deep       # Phase 7: Train Deep Multi-Task Conv1D+BiLSTM Model
python main.py --phase ablations        # Phases 9–10: Run head-pose and direction-invariance ablations
python main.py --phase compare          # Phase 8: Generate master comparison table & CSV
python main.py --phase known_start      # Separate: the paper's known-start task (see below)

# Customize architecture and training settings via CLI flags:
python main.py --phase train_deep --model-type conv_bilstm --loss-type uncertainty
python main.py --phase train_deep --model-type conv1d --epochs 150 --lr 5e-4

# Or apply any config override from a YAML file (updates the global config):
python main.py --phase train_deep --config my_config.yaml
```

Notes on artifacts:
- Deep-model checkpoints are per-architecture (`best_<model_type>_fold<i>.pt` in
  `data/processed/checkpoints/`) — retraining one architecture never overwrites or
  reuses another's weights. An existing checkpoint is reused (training skipped) only
  if the config fingerprint stored in it matches the current model/preprocessing/data
  config and input shape; otherwise that fold is retrained.
- The deep model's validation set is one held-out *training* subject per fold, so
  early stopping tracks generalization to unseen subjects.
- Deep-model results are saved per architecture as `reports/deep_model_<type>.json`
  (plus `deep_model_results.json` for the latest run) and all appear in the master
  comparison table.
- `preprocess` also caches the unified+filtered trials (`data/processed/preprocessed_trials.pkl`)
  so the ablations don't re-parse the raw data.

---

## Single-Dataset Mode: Dataset 2 (Monopolar Stationary, 6 Channels)

To eliminate cross-montage variance and utilize the full spatial coverage of the monopolar montage, the pipeline can be focused exclusively on **Dataset 2 (Monopolar Stationary)** using all 6 EOG channels ($H, V + \text{EOG\_0..\text{EOG\_3}}$):

```bash
# Run the complete Dataset 2 6-channel pipeline:
python main.py --config configs/dataset2_all_eog.yaml --phase all
```

### Master Results on Dataset 2 (Monopolar Stationary)

Results after the 2026-09-10 fixes (256 Hz sampling rate, subject-held-out validation,
least-squares calibration). Every number traces to a JSON file in `reports/` and to
`reports/master_results_table.csv`.

Evaluated under strict 5-fold cross-subject GroupKFold (zero test-subject leakage). The target horizontal gaze range is $\pm 27.3^\circ$ (std $= 14.07^\circ$) and vertical range is $\pm 16.0^\circ$ (std $= 8.00^\circ$).

| Method | Dataset | RMSE H (deg) | RMSE V (deg) | MAE H (deg) | MAE V (deg) | F1 (weighted) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| *Trivial: majority class* | Dataset 2 only | — | — | — | — | 0.334 | Always predicts the train fold's most frequent class |
| *Trivial: train-fold mean angle* | Dataset 2 only | 14.07° | 8.00° | 12.10° | 6.85° | — | Always predicts the train fold's mean angle |
| **Classical ML (SVC)** | Dataset 2 only | — | — | — | — | 0.703 | 104 hand-crafted features from 6 channels |
| **Classical ML (RF)** | Dataset 2 only | — | — | — | — | **0.715** | 104 hand-crafted features from 6 channels |
| **Classical ML (SVR)** | Dataset 2 only | 11.75° | 7.25° | 9.53° | 6.00° | — | 104 hand-crafted features from 6 channels |
| **Classical ML (XGB)** | Dataset 2 only | 11.53° | **7.15°** | 9.38° | **5.92°** | — | 104 hand-crafted features; $R^2_H = 0.33$, $R^2_V = 0.20$ |
| **Deep Conv1D+BiLSTM** | Dataset 2 only | **11.41°** | 7.20° | **9.16°** | 5.95° | 0.701 | Multi-task joint loss; $R^2_H = 0.34$, $R^2_V = 0.19$ |
| *Published: Barbara 2023* | Dataset 2 | — | — | — | — | — | *Reports within-subject fixation MAE, not RMSE: see [Comparison with Barbara et al. 2023](#comparison-with-barbara-et-al-2023-bspc-86)* |

Few-shot calibration (per-subject least-squares gain/offset fitted on each test subject's first
30 s) moves the deep model from 11.42° / 7.20° to 11.67° / 7.41° RMSE on the remaining windows
(`reports/calibration_results.json`).

> **Key takeaway**: on unseen subjects the deep model and XGBoost are roughly tied, explaining
> only ~34% of horizontal and ~19% of vertical gaze-angle variance: 19% / 10% lower RMSE than
> always predicting the mean angle. Per-subject calibration slightly *increases* error, so the gap is not a
> per-subject gain/offset problem. The main limitation is the input: a 0.2 Hz high-pass on 300 ms
> windows removes most of the absolute (DC) gaze-position information.

### Experiment: moving-median drift removal

Identical pipeline, except drift is removed by subtracting a 30 s moving-median baseline instead
of the 0.2 Hz high-pass, which keeps each fixation's DC level. The 30 s window was not tuned.
Configs: `configs/dataset2_median_baseline.yaml` (centred window, offline) and
`configs/dataset2_median_baseline_causal.yaml` (past-only window, real-time compatible).
Results are in `reports/experiments/median30*/`.

| RMSE H / V (deg) unless noted | High-pass 0.2 Hz (above) | Median 30 s, centred | Median 30 s, causal |
| :--- | :---: | :---: | :---: |
| Train-fold mean angle | 14.07 / 8.00 | 14.07 / 8.00 | 14.07 / 8.00 |
| XGBoost | 11.53 / 7.15 | **6.61 / 4.75** | 7.54 / 6.54 |
| SVR | 11.75 / 7.25 | 6.73 / 4.90 | 7.75 / 6.64 |
| Deep Conv1D+BiLSTM | 11.41 / 7.20 | 6.67 / 4.88 | 7.97 / 6.92 |
| $R^2$ H / V, XGBoost | 0.33 / 0.20 | 0.78 / 0.65 | 0.71 / 0.33 |
| Classification F1 (weighted), RF | 0.715 | 0.534 | 0.507 |
| Classification F1 (weighted), deep | 0.701 | 0.451 | 0.374 |

- Keeping the DC level cuts regression error by ~40% (H) and ~35% (V) for every model.
- The causal (real-time) variant keeps most of the horizontal gain; vertical is weaker, mostly
  because of one fold (fold-2 V RMSE 9.4–10.4°).
- Trial-interval classification gets worse. Under the high-pass, each saccade becomes a decaying
  transient whose level acts as a "time since saccade" clock that lines up with the fixed
  1 s / 1 s / 2 s trial timing; with the DC level kept, the level encodes (random) gaze position
  instead. A pipeline that needs both tasks should feed the classifier high-passed channels and
  the regressor median-baselined channels.
- Calibration on each subject's first 30 s still increases error (centred: 6.67 / 4.89 →
  7.53 / 5.21), plausibly because the session start is where the baseline estimate is least reliable.
- An earlier version of this README compared these RMSEs with "2.23° / 2.39°" for Barbara 2023.
  Those numbers are not in the paper, which reports fixation MAE; see the comparison below.

### Experiment: robust baselines and context features

Three further steps on the moving-median experiment, all on the same 5 cross-subject folds:

1. **Robust baselines** (`preprocessing.drift_removal_method`). `robust_mean` averages the window
   after dropping samples more than 3 robust SDs from its median (blinks); the mean wanders less
   than the median for uniformly spread gaze targets. `robust_line` fits a line to those samples
   instead, and in past-only mode evaluates it at the current time, removing the half-window lag.
   Configs: `configs/dataset2_robust_mean.yaml` (centred 60 s, offline) and
   `configs/dataset2_robust_line_causal.yaml` (past-only 60 s, real-time).
2. **Context features** for the classical regressors (`src/features/context.py`,
   `preprocessing.context_baselines` / `context_lags_sec`). With only the current 300 ms window,
   predictions are pulled toward the screen centre and depend on the previous saccade, because a
   past-only baseline has partly absorbed the latest fixations. The regressors additionally get
   how three other baselines (robust mean 30 s / 120 s, robust line 120 s) differ from the main
   one, and the signal level 1, 2, 4 and 8 s before the window, all from the past only.
   Config: `configs/dataset2_context_causal.yaml`.
3. **Rolling-range features** (`preprocessing.context_range_windows_sec` /
   `context_range_quantiles`). Cue positions are spread roughly uniformly over a bounded screen
   (±27° H, ±16° V). For such data the mid-range, halfway between a low and a high quantile,
   locates the centre far more precisely than a mean (error shrinking like 1/N instead of
   1/√N), and the range itself is a label-free estimate of the subject's EOG gain. Over the past
   60 / 120 / 240 s and the 10–90% and 5–95% quantiles of each channel, the regressors get the
   window level minus the mid-range, the range, and their ratio.
   Config: `configs/dataset2_range_causal.yaml`.

| RMSE H / V (deg) unless noted | Median 30 s, causal | Robust line 60 s, causal | + context | + context + range | Median 30 s, centred | Robust mean 60 s, centred |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Real-time | yes | yes | yes | yes | no | no |
| XGBoost | 7.54 / 6.54 | 7.67 / 6.06 | 6.87 / 5.60 | **6.34 / 5.36** | 6.61 / 4.75 | **5.61 / 4.76** |
| SVR | 7.75 / 6.64 | 7.73 / 6.14 | 7.31 / 5.82 | 6.97 / 5.68 | 6.73 / 4.90 | 5.74 / 4.92 |
| Deep Conv1D+BiLSTM | 7.97 / 6.92 | 7.37 / 6.08 | — | — | 6.67 / 4.88 | 5.58 / 4.85 |
| $R^2$ H / V, XGBoost | 0.71 / 0.33 | 0.70 / 0.43 | 0.76 / 0.51 | 0.80 / 0.55 | 0.78 / 0.65 | 0.84 / 0.65 |
| Classification F1 (weighted), RF | 0.507 | 0.512 | 0.512 | 0.512 | 0.534 | 0.525 |

- Real-time: the past-only robust line beats the past-only median for every regressor on vertical
  (V RMSE −0.5 to −0.8°); context features and then range features cut XGBoost to 6.34 / 5.36
  (−16% / −18% vs the past-only median). The deep model sees the same windows as "robust line
  60 s" and gets no context or range features.
- Offline: the 60 s robust mean cuts horizontal RMSE by ~1° for every model (XGBoost 6.61 → 5.61).
- Results are in `reports/experiments/{robustline60_causal,context_causal,range_causal,robustmean60}/`.

### Comparison with Barbara et al. 2023 (BSPC 86)

Barbara et al. [2] introduced this dataset, and they score gaze error differently from the RMSE
tables above:

- **Metric:** mean absolute error over *fixation* samples only (after the saccade has ended and
  before the next cue), per subject, then mean ± SD across subjects. Saccades and blinks are not scored.
- **Protocol:** within-subject. Each subject's recording is split into three contiguous subsets
  used to fit, tune and test that subject's own parameters; outlier segments are excluded
  (6.85% short, 5.83% long).
- **Segments:** *short* = single 1 s saccade or 2 s blink windows without subject mistakes;
  *long* = 32 s segments of 8 trials, in which estimation errors build up.

Every regression result JSON now also carries `fixation_mae_{h,v}_deg` (± `_sd_deg`) computed the
same way. A window is a fixation window when it lies in a ControlSignal 1/2 interval, starts at
least `segmentation.fixation_settle_ms` = 400 ms after the cue moved (on Dataset 2, saccades peak
~200 ms and settle by ~400 ms after the cue) and contains no cue change: 15.7% of all windows.
`cv.regression_train_windows` chooses the classical regressors' training windows: `all`,
`fixation` (fixation windows only) or `weighted` (all windows, the others weighted by
`cv.regression_nonfixation_weight` = 0.2). Test windows are never filtered.

Models below are trained **cross-subject**: the test subject is never seen, which is harder than
the paper's protocol. Configs: `dataset2_range_causal{,_weighted,_fixation}.yaml`,
`dataset2_range_centred_{fixation,weighted}.yaml`, and the `dataset2_context_*` configs without
range features.

| Method | Real-time | Training windows | Fixation MAE H (deg) | Fixation MAE V (deg) | All-window RMSE H / V |
| :--- | :---: | :--- | :---: | :---: | :---: |
| *Train-fold mean angle* | — | all | 12.56 ± 0.93 | 7.05 ± 0.60 | 14.07 / 8.00 |
| Robust line 60 s, XGBoost | yes | all | 5.66 ± 1.60 | 4.56 ± 1.26 | 7.67 / 6.06 |
| Robust line 60 s, Deep Conv1D+BiLSTM | yes | all | 5.48 ± 1.43 | 4.58 ± 1.11 | 7.37 / 6.08 |
| + context, XGBoost | yes | all | 4.87 ± 1.55 | 4.18 ± 1.16 | 6.87 / 5.60 |
| + context, XGBoost | yes | fixation | 4.79 ± 1.54 | 4.06 ± 1.25 | 7.88 / 5.96 |
| + context + range, XGBoost | yes | all | 4.47 ± 1.22 | 3.96 ± 1.08 | **6.34 / 5.36** |
| + context + range, XGBoost | yes | weighted | 4.43 ± 1.14 | 3.91 ± 1.10 | 6.35 / 5.39 |
| + context + range, XGBoost | yes | fixation | **4.36 ± 1.12** | **3.78 ± 1.04** | 7.34 / 5.81 |
| Robust mean 60 s, XGBoost | no | all | 3.76 ± 0.84 | 3.48 ± 0.67 | 5.61 / 4.76 |
| + context, XGBoost | no | fixation | 3.45 ± 0.85 | 3.25 ± 0.64 | 7.12 / 5.14 |
| + context + range, XGBoost | no | weighted | 3.23 ± 0.60 | 3.26 ± 0.69 | **5.25 / 4.53** |
| + context + range, XGBoost | no | fixation | **3.13 ± 0.56** | **3.23 ± 0.67** | 6.75 / 5.14 |
| *Published: dual Kalman filter, long 32 s* | yes | same subject | 5.23 ± 2.00 | 6.59 ± 3.10 | — |
| *Published: signal differencing [BSPC 47, 2019], long 32 s* | yes | same subject | 5.82 ± 2.70 | 8.04 ± 2.96 | — |
| *Published: dual Kalman filter, short 1 s* | yes | same subject | 1.64 ± 0.82 | 1.97 ± 0.34 | — |
| *Published: signal differencing, short 1 s* | yes | same subject | 1.51 ± 0.55 | 1.95 ± 0.29 | — |

- **Real-time, without ever seeing the test subject**, XGBoost with context and range features is
  0.9° better than the paper's dual Kalman filter horizontally (4.36 vs 5.23°) and 2.8° better
  vertically (3.78 vs 6.59°) on its long-segment results, and better than the earlier
  signal-differencing method on both axes; so are the all-window and weighted variants. The models
  estimate absolute gaze from each window, so nothing accumulates over a segment; the features do
  use up to 4 min of past signal.
- **Weighted training is the best all-round model:** within 0.15° of the fixation-only model on
  fixation MAE while keeping a low all-window RMSE (real-time 6.35 / 5.39 vs 7.34 / 5.81 for
  fixation-only). Offline it gives the lowest all-window RMSE of any configuration, 5.25 / 4.53
  ($R^2$ 0.86 / 0.68).
- **Range features also narrow the spread across subjects** (fixation MAE SD 1.54 → 1.12° H
  real-time, 0.85 → 0.56° H offline), consistent with them estimating each subject's gain.
- **Not a like-for-like win:** the paper excludes outlier segments and fits every parameter on the
  test subject; this pipeline does neither, but scores 400 ms-settled windows rather than
  EOG-detected fixation samples, and uses whole ~13 min recordings rather than 32 s segments.
- **The short-segment results (~1.5–2°) are out of reach for this absolute task.** They score a
  single saccade at a time from a known starting gaze; the separate
  [known-start evaluation](#separate-evaluation-the-papers-known-start-task) replicates that
  protocol and gets comparable, not clearly better, errors.
- **Caveats:** baseline windows, context and range settings and the training-window choice were
  selected by comparing variants on these same 5 folds (no untouched test set), so the numbers are
  slightly optimistic. The per-subject z-score scale comes from the first 10% of each recording
  (~80 s, usable in real time after that) and the 30 Hz low-pass is zero-phase (a few ms of
  look-ahead).
- Exploratory scripts outside the pipeline also tried, without beating XGBoost with context and
  range features: the paper's within-subject protocol; a causal GRU over long context (the paper's
  suggested future work); larger XGBoost models and absolute-error / pseudo-Huber losses (±0.1°);
  a label-free running per-subject offset correction (−0.1°); stacking a second model on rolling
  statistics of the first model's predictions (worse); and blending with an MLP or the deep model.

### Separate evaluation: the paper's known-start task

The published numbers come from a different and easier task than every table above: each segment
starts at the **known** gaze, and the EOG only has to supply displacements (the paper's Kalman
filter also starts from a known baseline). This evaluation replicates that protocol as closely as
the paper describes it (Sec. 4.5.3 and Appendix E) and is never mixed into the main results:

```bash
python main.py --config configs/dataset2_known_start.yaml --phase known_start
```

It writes `reports/experiments/known_start/known_start_protocol.json` and `known_start_table.csv`
(`src/evaluation/known_start.py`; settings in the `known_start` config section).

**Protocol, as in the paper:**

- **Ground-truth labels come from the EOG** (App. E.1). A sample is a fixation when the
  sample-to-sample difference of neither bipolar channel exceeds 5 robust SDs; a pair of
  opposite-sign spikes in the V difference within 500 ms labels a blink; everything else is a
  saccade. On Dataset 2: 88.9% fixation, 7.9% saccade and 3.2% blink samples.
- **Short analysis:** 1 s saccade windows and 2 s blink windows without subject mistakes. A
  saccade window is a mistake when it contains a blink, or its saccade is missing, anticipatory
  or premature; a blink window, when it does not contain exactly one blink or has stray saccades.
  93% of saccade windows and 84% of blink windows are mistake-free, and each third of a recording
  contributes its first 66 saccade and 33 blink windows, as in the paper.
- **Long analysis:** each third contributes 8 segments of 8 consecutive trials (32 s), mistakes kept.
- **Error** (App. E.2): per segment, the mean |estimate − target| over the fixation-labelled
  samples after the response saccade (saccade windows) or over all fixation-labelled samples
  (blink windows); per subject, the mean over segments; then mean ± SD across subjects.
- **Same subject:** fit on one third, test on another, all 6 orderings. **Unseen subject** (not
  in the paper): fit on the other 9 subjects, with each subject's displacements divided by a
  label-free gain.
- **Outliers:** the paper drops segments with "substantially high" error but gives no threshold.
  Here a segment is dropped when its H or V error lies beyond Tukey's far-out fence,
  Q3 + 3 × IQR, of that subject's segments of the same kind (saccade windows, blink windows or
  long segments; blink windows are fenced separately because they mostly score near 0). With
  detected saccades this drops 1.6% of short and 4.8% of long segments (paper: 6.85% and 5.83%).

**Estimators** (EOG and starting gaze only; a 2-channel H/V linear map without intercept, like
the paper's comparison method [BSPC 47, 2019]):

- *Detected saccades* (signal differencing): eye movements are detected from EOG velocity and
  each adds its displacement; blinks are rejected. The detector finds 93% of cue-driven saccades,
  and 95% of its detections follow a cue.
- *Level change*: the EOG change since the segment start (short segments only).

| Known-start task, MAE H / V (deg) | Fit | All segments | Outlier segments dropped |
| :--- | :--- | :---: | :---: |
| Short, detected saccades | same subject | 1.17 ± 0.49 / 1.39 ± 0.27 | 0.92 ± 0.49 / 1.33 ± 0.24 |
| … saccade windows only | same subject | 1.75 / 2.09 | — |
| Short, level change | same subject | 1.47 ± 0.61 / 2.37 ± 0.60 | 1.28 ± 0.62 / 2.25 ± 0.60 |
| Short, detected saccades | unseen subject | 1.34 ± 0.46 / 1.82 ± 0.54 | 1.11 ± 0.46 / 1.77 ± 0.54 |
| … saccade windows only | unseen subject | 2.02 / 2.73 | — |
| *Published: dual Kalman filter, short* | same subject | — | 1.64 ± 0.82 / 1.97 ± 0.34 |
| *Published: signal differencing, short* | same subject | — | 1.51 ± 0.55 / 1.95 ± 0.29 |
| Long 32 s, detected saccades | same subject | 4.71 ± 1.59 / 8.39 ± 3.13 | 4.21 ± 1.81 / 8.36 ± 3.29 |
| Long 32 s, detected saccades | unseen subject | 4.91 ± 1.41 / 8.73 ± 3.63 | 4.45 ± 1.52 / 8.77 ± 3.70 |
| *Published: dual Kalman filter, long* | same subject | — | 5.23 ± 2.00 / 6.59 ± 3.10 |
| *Published: signal differencing, long* | same subject | — | 5.82 ± 2.70 / 8.04 ± 2.96 |

**How close it gets:**

- **Short segments: comparable, not clearly better.** The combined score (0.92 / 1.33° after
  dropping outliers, 1.17 / 1.39° with none dropped) is below both published methods, but a third
  of the short windows are blink windows. There the target does not move, and once the blink is
  rejected this estimator scores exactly 0. On saccade windows alone it scores 1.75 / 2.09° with
  no segments dropped, about the paper's combined numbers. The paper does not report that split,
  so how much its own blink windows lower its score is unknown.
- **Long segments:** horizontal error is lower than both published methods (4.21 vs 5.23 /
  5.82° after exclusion), vertical error is higher than both (8.36 vs 6.59 / 8.04°). Summing
  detected displacements is essentially the paper's signal-differencing method; the Kalman
  filter's blink and eyelid modelling handles the vertical channel better.
- **Unseen subjects** (not in the paper): 1.34 / 1.82° on short segments with no labels from the
  test subject.
- **Remaining differences from the paper:** ground-truth thresholds come from each whole
  recording rather than from training data only; subject mistakes are judged by fixed rules rather
  than by the authors; the paper gives no outlier threshold, so the far-out fence is a stand-in; the detector and
  blink rules were set by inspecting Dataset 2 events from all subjects; and blink decisions look
  up to 150 ms past the end of a movement.
### Datasets 3 and 4

The same pipeline and 5-fold cross-subject protocol, with the settings chosen on Dataset 2 applied
unchanged, so these datasets also act as a check of those choices on data they were not tuned on.
Two differences, for speed: the CPU-only kernel SVC/SVR are skipped, and XGBoost runs on the GPU
(`cv.xgb_device: cuda`), whose numbers differ slightly from the CPU XGBoost used on Dataset 2.
Configs and result folders are listed in the [Experiment Index](#experiment-index). Dataset 4's
deep-model JSON (59 MB, mostly per-window metadata) is kept out of git; its numbers are in
`reports/experiments/dataset4/robustline60_causal/master_results_table.csv`.

- **Dataset 4 (Monopolar Isotropic):** 14 subjects, 16 electrodes around the eyes plus H and V
  (18 channels), chin rest. Every trial goes from the centre to a 12° target in one of 36
  directions, back to the centre, then a blink, so targets are small and half are at the centre.
- **Dataset 3 (Monopolar Non-Stationary):** 8 subjects with Dataset 2's electrodes and trial
  timing, but free head movement. The target angles are given in a face frame that turns with the
  head, so head pose is already part of the targets. What nothing here models is the slow
  vestibulo-ocular (VOR) eye rotation that counters head movement: it moves the gaze without a
  saccade, so the detected-saccade estimator misses it and a drift baseline can absorb it. The
  Dataset 3 paper (Barbara et al., BSPC 90, 2024) adds a VOR model to its dual Kalman filter.
- **Published numbers:** the Dataset 3 paper reports the known-start task only (within-subject,
  outliers dropped), listed below. The Dataset 4 paper (BSPC 112, 2026) reports per-saccade
  displacement errors in figures only, a different measure, so it has no row here.

| Real-time; 5-fold cross-subject unless noted | Dataset 4 fixation MAE H / V | Dataset 4 RMSE H / V | Dataset 3 fixation MAE H / V | Dataset 3 RMSE H / V |
| :--- | :---: | :---: | :---: | :---: |
| Train-fold mean angle | 3.82 / 3.83 | 4.04 / 4.04 | 8.90 / 6.28 | 10.73 / 7.28 |
| Robust line 60 s, XGBoost | 1.90 / 2.26 | 2.48 / 2.75 | 4.65 / 4.67 | 6.35 / 6.11 |
| Robust line 60 s, Deep Conv1D+BiLSTM | 1.57 / 1.80 | 2.44 / 2.63 | 4.76 / 4.67 | 6.51 / 6.09 |
| + context + range, XGBoost, weighted training | 1.33 / 1.56 | 2.26 / 2.48 | 4.42 / 4.46 | 6.18 / 5.84 |
| + context + range, XGBoost, fixation windows only | 1.11 / 1.51 | 3.12 / 3.37 | 4.28 / 4.36 | 7.04 / 6.27 |
| Known start, same subject, short (detected saccades) | 0.54 / 1.21 | — | 2.72 / 1.97 | — |
| … outlier segments dropped | 0.47 / 1.03 | — | 2.67 / 1.73 | — |
| *Published: dual Kalman filter + VOR model, short* | — | — | 1.85 ± 0.51 / 2.19 ± 0.62 | — |
| *Published: signal differencing, short* | — | — | 3.59 ± 0.74 / 2.52 ± 0.62 | — |
| Known start, same subject, long 32 s (detected saccades) | 2.26 / 9.69 | — | 8.04 / 13.02 | — |
| … outlier segments dropped | 2.02 / 9.69 | — | 7.67 / 12.67 | — |
| *Published: dual Kalman filter + VOR model, long* | — | — | 4.64 ± 1.37 / 6.10 ± 2.58 | — |
| *Published: signal differencing, long* | — | — | 8.13 ± 1.15 / 11.25 ± 5.03 | — |

- **Context and range features with weighted training help on both datasets:** XGBoost's fixation
  MAE drops from 1.90 / 2.26 to 1.33 / 1.56° on Dataset 4 and from 4.65 / 4.67 to 4.42 / 4.46° on
  Dataset 3. The features and the weighting were not run separately here, so the gain belongs to
  the combination. It holds on Dataset 4, whose targets are not spread evenly (centre plus a 12° ring).
- **Compared with always predicting the mean angle,** weighted XGBoost removes
  65% / 45% (H / V) of the fixation error on Dataset 2,
  65% / 59% on Dataset 4 and 50% / 29% on
  Dataset 3. Dataset 4's errors are small in degrees mostly because its targets span only ±12°.
- **Deep model:** on Dataset 4 it beats XGBoost without context features (1.57 / 1.80 vs
  1.90 / 2.26°); on Dataset 3 it scores 4.76 / 4.67° against XGBoost's
  4.65 / 4.67°. It gets no context or range features.
- **Dataset 3 keeps a smaller share of the gain:** weighted XGBoost removes 50% / 29% of the
  mean-angle error versus 65% / 45% on Dataset 2, and long known-start segments reach
  8.04 / 13.02°. Free head movement is the obvious suspect, but Dataset 3 also has fewer subjects (8)
  and nothing here models the VOR eye movements that come with head movement, so the cause is not
  isolated.
- **Known start against the Dataset 3 paper** (outliers dropped; the paper drops 7.77% of short and
  3.91% of long segments, this evaluation 2.8% and 3.4%): on short segments, horizontal error
  (2.67°) lies between the paper's Kalman filter with VOR model (1.85°) and signal differencing
  (3.59°), and vertical error (1.73°) is below both (2.19 / 2.52°). On long segments, horizontal
  error (7.67°) is just below signal differencing (8.13°) but well above the Kalman filter (4.64°),
  and vertical error (12.67 ± 14.92°) is above both (6.10 / 11.25°). The VOR model is the main
  thing this estimator lacks.
- **Known start, long segments, vertical:** blinks the detector misses leave a lasting vertical
  offset that keeps adding up (1.9% of labelled blinks counted as eye
  movements on Dataset 4, 5.1% on Dataset 3).

### Robustness checks

#### Range features when test gaze covers only part of the screen (Dataset 2)

The rolling-range features estimate the screen centre and each subject's EOG gain from the spread
of recent gaze, which Dataset 2's evenly spread cues favour. `scripts/range_stress_test.py`
rebuilds each test subject's recording from a subset of its 200 trials (about 800 s) and scores
real-time XGBoost trained on the other subjects' full recordings. The setup is a 60 s past-only
robust line with weighted training, run on the GPU, with and without the range features. Each
skewed subset has two controls with the same number of trials: random trials joined the same way,
and one contiguous block, which is just as short but has no joins.

| Test recording (mean trials kept) | Context features, fixation MAE H / V | + range features | Gain from range H / V |
| :--- | :---: | :---: | :---: |
| Full recording (200) | 4.82 / 4.12 | 4.39 / 3.88 | +0.43 / +0.24 |
| Right only: both cues H > 0 (51) | 19.29 / 6.49 | 15.53 / 5.68 | +3.75 / +0.81 |
| … random trials, joined (51) | 9.91 / 6.63 | 8.06 / 6.01 | +1.85 / +0.62 |
| … contiguous block (51) | 9.28 / 5.27 | 6.03 / 4.54 | +3.25 / +0.73 |
| Top only: both cues V > 0 (50) | 9.47 / 10.84 | 7.36 / 9.75 | +2.11 / +1.09 |
| … random trials, joined (50) | 9.22 / 6.33 | 7.55 / 5.86 | +1.67 / +0.46 |
| … contiguous block (50) | 10.32 / 4.74 | 6.78 / 4.42 | +3.53 / +0.32 |
| Centre: both cues in the central two thirds (33) | 10.97 / 5.44 | 9.31 / 5.16 | +1.65 / +0.28 |
| … random trials, joined (33) | 10.12 / 6.68 | 8.56 / 6.12 | +1.56 / +0.56 |
| … contiguous block (33) | 7.66 / 5.02 | 6.02 / 4.20 | +1.64 / +0.81 |

- **The range features lower the error in every scenario.** The gain is no smaller when gaze is
  one-sided than in the random-trial controls (right only: +3.75° horizontal vs +1.85°).
- **One-sided gaze defeats the drift baseline itself, with or without range features.** With every
  cue on the right, horizontal error reaches 19.3° (15.5° with range features), against 9.9° (8.1°)
  for random trials. Top-only cues do the same to vertical error (10.8° vs 6.3°). A rolling
  baseline treats the average EOG level over its window as the centre, so a gaze offset that lasts
  that long is removed as drift. Cues kept near the centre cost little (11.0° vs 10.1°). The
  known-start task above does not rely on this assumption.
- **Short recordings are costly on their own.** A contiguous 200 s block roughly doubles horizontal
  error compared with the full recording (7.7–10.3° vs 4.8° without range features). The 120 s
  context baselines and the 240 s range windows have little or no full history in a recording
  that short.
- **Joining trials costs extra with range features:** random trials score 0.8–2.5° worse than a
  contiguous block of the same length (8.06 vs 6.03°, 7.55 vs 6.78°, 8.56 vs 6.02° horizontal).
  Without range features the difference is inconsistent.
- These recordings are shortened, so the absolute numbers are not comparable to the main tables.
  Results: `reports/experiments/range_stress_test/dataset2/range_stress_test.json`.

---

## Experiment Index

Every experiment is one YAML config in `configs/`, and no two experiments share an output folder.
Dataset 2 experiments write to `reports/experiments/<name>/`; Datasets 3 and 4 write to
`reports/experiments/dataset3/<name>/` and `reports/experiments/dataset4/<name>/`, with processed
data under `data/processed_datasetN/<name>/` (git-ignored). Each results folder holds one JSON per
model, a `master_results_table.csv` (or `known_start_table.csv`), and loss curves for deep runs.

| Config | Results | What it tests |
| :--- | :--- | :--- |
| `dataset2_all_eog.yaml` | `reports/` | High-pass baseline, all 6 channels |
| `dataset2_median_baseline{,_causal}.yaml` | `reports/experiments/median30{,_causal}/` | 30 s moving-median drift removal |
| `dataset2_robust_mean.yaml` | `reports/experiments/robustmean60/` | Centred 60 s robust mean (offline) |
| `dataset2_robust_line_causal.yaml` | `reports/experiments/robustline60_causal/` | Past-only 60 s robust line (real-time) |
| `dataset2_context_causal{,_fixation}.yaml`, `dataset2_context_centred_fixation.yaml` | `reports/experiments/context_*/` | Context features |
| `dataset2_range_causal{,_weighted,_fixation}.yaml`, `dataset2_range_centred_{fixation,weighted}.yaml` | `reports/experiments/range_*/` | Context + rolling-range features; training-window choice |
| `dataset2_known_start.yaml` | `reports/experiments/known_start/` | The paper's known-start protocol (separate task) |
| `dataset3_robust_line_causal.yaml` | `reports/experiments/dataset3/robustline60_causal/` | Dataset 3 real-time baseline (no VOR model) |
| `dataset3_range_causal_{weighted,fixation}.yaml` | `reports/experiments/dataset3/range_causal_*/` | Dataset 3 with the best Dataset 2 setup |
| `dataset3_known_start.yaml` | `reports/experiments/dataset3/known_start/` | Dataset 3 known-start protocol |
| `dataset4_robust_line_causal.yaml` | `reports/experiments/dataset4/robustline60_causal/` | Dataset 4 real-time baseline (18 channels) |
| `dataset4_range_causal_{weighted,fixation}.yaml` | `reports/experiments/dataset4/range_causal_*/` | Dataset 4 with the best Dataset 2 setup |
| `dataset4_known_start.yaml` | `reports/experiments/dataset4/known_start/` | Dataset 4 known-start protocol |
| `scripts/nested_cv.py` with `dataset2_range_{causal,centred}_weighted.yaml` | `reports/experiments/nested_cv/dataset2/{realtime,offline}/` | Nested cross-subject selection of baseline window, features and training windows |
| `scripts/range_stress_test.py` with `dataset2_range_causal_weighted.yaml` | `reports/experiments/range_stress_test/dataset2/` | Range features when test gaze covers only part of the screen |
| `scripts/subject_statistics.py` | `reports/experiments/statistics/` | Per-subject Wilcoxon tests and bootstrap confidence intervals |

---

## Project Structure

```
eog-gaze-pipeline/
├── configs/                 # one YAML per experiment (see Experiment Index)
├── data/
│   ├── raw/                 # downloaded ZIPs extracted here
│   └── processed*/          # cached arrays + CV folds per experiment (git-ignored)
├── src/
│   ├── config.py            # ALL tunable parameters — edit here only
│   ├── data/                # schema, loaders, unify, datasets
│   ├── preprocessing/       # filtering, blink detection, normalization
│   ├── features/            # hand-crafted and context/range features for classical ML; feature matrices for the scripts
│   ├── models/              # integration baseline, classical ML, deep model
│   ├── training/            # CV splits, training loop
│   ├── evaluation/          # metrics, paper comparison table, known-start protocol
│   └── ablations/           # head-pose (Phase 9), direction (Phase 10)
├── scripts/
│   ├── inspect_raw.py       # Phase 0 verification
│   ├── nested_cv.py         # nested cross-subject model selection
│   ├── range_stress_test.py # range features under skewed gaze
│   └── subject_statistics.py  # per-subject significance tests and confidence intervals
├── tests/                   # pytest test suite
├── notebooks/               # EDA and results notebooks
├── reports/
│   ├── experiments/         # <name>/ (Dataset 2), dataset3/<name>/, dataset4/<name>/
│   └── figures/             # all saved plots
└── requirements.txt
```

---

## Configuration

All tunable parameters live in [`src/config.py`](src/config.py). Never hard-code magic numbers elsewhere.

Every section (`preprocessing`, `segmentation`, `cv`, `model`, `augmentation`, `data`,
`paths`, `device`) can be overridden from a YAML file passed to `python main.py --config file.yaml`;
unknown sections/keys are rejected with a clear error.

Key parameters (starting points — verify against real `fs` from Phase 0):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `highpass_cutoff_hz` | 0.2 | Drift removal; try 0.1–0.3 |
| `lowpass_cutoff_hz` | 30.0 | Noise filter |
| `window_ms` | 300 | Sliding window length |
| `stride_ms` | 150 | 50% overlap |
| `lambda_regression_loss` | 1.0 | Multi-task loss weight |
| `lr` | 1e-3 | Adam learning rate |
| `batch_size` | 64 | Reduce if GPU OOM |
| `early_stopping_patience` | 10 | Epochs without improvement |
| `cv.svm_max_train_samples` | 50000 | Kernel SVC/SVR train subsample cap (0 = off); test set is never subsampled |
| `data.fs_hz_by_dataset` | 256.0 for all four | Sampling rates from the Data Description PDFs |
| `data.fs_fallback_hz` | 256.0 | Only used for datasets missing from `fs_hz_by_dataset` |
| `preprocessing.drift_removal_method` | `highpass` | Also `polynomial_detrend`, `moving_median`, `robust_mean`, `robust_line` |
| `preprocessing.baseline_window_sec` / `baseline_causal` | 30.0 / false | Window and past-only mode for `robust_mean` / `robust_line` |
| `preprocessing.context_baselines` / `context_lags_sec` | [] / [] | Context features for classical regressors (empty = off) |
| `preprocessing.context_range_windows_sec` / `context_range_quantiles` | [] / [[0.1, 0.9], [0.05, 0.95]] | Rolling-range features (empty windows = off) |
| `segmentation.fixation_settle_ms` | 400 | Settle time after a cue before windows count as fixation windows |
| `cv.regression_train_windows` | `all` | `fixation`: fixation windows only; `weighted`: all windows, others weighted by `cv.regression_nonfixation_weight` (0.2) |
| `known_start.*` | see `KnownStartConfig` | Ground-truth label thresholds, subject-mistake rules, window counts and eye-movement detector of the separate known-start evaluation |

---

## Citations

All four papers must be cited in any work using this pipeline:

1. N. Barbara, T. A. Camilleri, K. P. Camilleri, "A comparison of EOG baseline drift mitigation techniques," *Biomedical Signal Processing and Control*, vol. 57, Mar. 2020.

2. N. Barbara, T. A. Camilleri, K. P. Camilleri, "Real-Time Continuous EOG-based Gaze Angle Estimation with Baseline Drift Compensation Under Stationary Head Conditions," *Biomedical Signal Processing and Control*, vol. 86, Sep. 2023.

3. N. Barbara, T. A. Camilleri, K. P. Camilleri, "Real-Time Continuous EOG-based Gaze Angle Estimation with Baseline Drift Compensation Under Non-Stationary Head Conditions," *Biomedical Signal Processing and Control*, vol. 90, Apr. 2024.

4. N. Barbara, T. A. Camilleri, K. P. Camilleri, "A Systematic Quantitative Analysis on Bipolar Channel Selection for EOG-Based Gaze Displacement Estimation," *Biomedical Signal Processing and Control*.

---

## Reproducing the Headline Numbers

1. Download all four datasets (see above)
2. `python scripts/inspect_raw.py` — verify parsers work
3. Fill the Data Description table above
4. `python scripts/verify_montage.py` — verify bipolar channel pairs + sign
   conventions against the recorded target angles; update `MONOPOLAR_MAPS` in
   [`src/data/unify.py`](src/data/unify.py) if the winners disagree
   (the shipped maps were verified this way on 2026-08-21 — the original
   placeholder mapping had H/V swapped for Datasets 2–4)
5. Run the full pipeline commands above in order (Phases 1–8)
6. Results CSV: `reports/master_results_table.csv`
7. All figures: `reports/figures/`

---

## Appendix: Master Pitfall Checklist

- [ ] Never split by trial — always by `subject_id` (automated in `test_no_subject_leakage.py`)
- [ ] Drift removal happens BEFORE normalization, not after
- [ ] Bipolar sign convention verified per-dataset (Phase 2.4) — `verify_sign_convention()` in `unify.py`
- [ ] Class imbalance (blink rarity) accounted for via `CLASS_WEIGHTS` in `config.py`
- [ ] Head-pose correction on Dataset 3 validated with before/after ablation (Phase 9)
- [ ] Every reported metric traces to a saved JSON file in `reports/`
- [ ] Deep model results reported honestly even if it loses to the classical baseline
