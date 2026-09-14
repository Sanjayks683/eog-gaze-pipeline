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
3. **Replicates the published known-start protocol** (gaze displacement from a known starting
   gaze) as a separate evaluation, so results can be set against Barbara et al.'s papers
4. **Checks the results**: nested cross-validation, a stress test of the range features, and
   per-subject significance tests

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

### Data Description Table

Filled from the `scripts/inspect_raw.py` output and each dataset's Data Description PDF.

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
`CFG.data.fs_hz_by_dataset`; an earlier version silently fell back to 250 Hz. Dataset 1's
`TargetGA.mat` holds one row per cue (each trial's target, then the centre); the loader spreads
the rows over the samples of each cue's interval (`expand_cue_targets` in `src/data/loaders.py`).

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
  signal-differencing method on both axes. So are the all-window and weighted variants, and so is
  the [nested cross-validation](#nested-cross-validation-dataset-2) result, which picks the
  settings without looking at the test subjects (4.49 / 3.81°). The models
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
  Trained the paper's within-subject way, the model does no better
  ([within-subject protocol](#within-subject-protocol-dataset-2)).
- **The short-segment results (~1.5–2°) are out of reach for this absolute task.** They score a
  single saccade at a time from a known starting gaze; the separate
  [known-start evaluation](#separate-evaluation-the-papers-known-start-task) replicates that
  protocol and gets comparable, not clearly better, errors.
- **Caveats:** baseline windows, context and range settings and the training-window choice were
  selected by comparing variants on these same 5 folds (no untouched test set). Nested
  cross-validation re-selects the baseline window, feature set and training windows without the
  test subjects and scores 0.0–0.16° worse than the best numbers here; the context and range
  settings themselves were not re-selected. The per-subject z-score scale comes from the first 10% of each recording
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
- *Detected saccades + head rotation* (recordings with head pose, i.e. Dataset 3): detected
  saccades, plus minus the head's yaw / pitch change between detected movements
  (`known_start.vor_gain` = 1); see [Datasets 3 and 4](#datasets-3-and-4).
- *Fused with cross-subject XGBoost* (`scripts/known_start_fusion.py`): detected saccades plus a
  causal low-pass of (XGBoost estimate − detected-saccade estimate). The time constant is chosen per
  axis on the fit data, from 1 s to ∞. XGBoost is the real-time model with context + range features
  and weighted training, trained 5-fold cross-subject so it never sees the test subject. The saccade
  sum supplies the fast changes, XGBoost the slow level.

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
| **Long 32 s, fused with cross-subject XGBoost** | same subject | 3.62 ± 0.87 / 3.40 ± 0.67 | **3.36 ± 0.85 / 3.38 ± 0.65** |
| Long 32 s, fused with cross-subject XGBoost | unseen subject | 3.89 ± 0.96 / 3.59 ± 0.74 | 3.80 ± 1.01 / 3.56 ± 0.72 |
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
- **Fusion fixes the long-segment vertical error.** Fused with cross-subject XGBoost, long
  segments reach 3.36 / 3.38° (outliers dropped).
  - **Against the paper:** that is below the Kalman filter (5.23 / 6.59°) and signal differencing
    (5.82 / 8.04°) on both axes.
  - **Unseen subjects:** 3.80 / 3.56°.
  - **Time constants chosen:** mostly 2–5 s for vertical and a wide range for horizontal. On short
    segments the fit chooses no fusion, so those numbers are unchanged.
  - **Not like-for-like:** the XGBoost model learns from the other subjects' labelled recordings,
    which the paper's filter does not use.
- **Unseen subjects** (not in the paper): 1.34 / 1.82° on short segments with no labels from the
  test subject.
- **Remaining differences from the paper:** ground-truth thresholds come from each whole
  recording rather than from training data only; subject mistakes are judged by fixed rules rather
  than by the authors; the paper gives no outlier threshold, so the far-out fence is a stand-in; the detector and
  blink rules were set by inspecting Dataset 2 events from all subjects; and blink decisions look
  up to 150 ms past the end of a movement.
### Dataset 1

The same 5-fold cross-subject protocol and Dataset 2 settings, on Dataset 1 (Zero-Centred Bipolar).
- **Recording:** 6 subjects on a chin rest, 300 trials each in three sessions of 100, with only the
  bipolar H and V recorded.
- **Trials:** every trial goes from the centre to a random target (about ±23° H, ±13° V) and back,
  then a blink, so half of the cues are at the centre.
- **Targets:** the loader spreads the per-cue targets over the samples (see the
  [data table](#data-description-table) note). Before that fix, Dataset 1's regression labels were
  meaningless.
- **Differences from Dataset 2:** SVC/SVR are skipped and XGBoost runs on the GPU, as on Datasets 3
  and 4.
- **Published numbers:** the Dataset 1 paper (Barbara et al., BSPC 57, 2020) compares drift-removal
  methods with errors normalised by the screen distance rather than in degrees, so it has no row here.

| Dataset 1, real-time, 5-fold cross-subject | Fixation MAE H / V (deg) | RMSE H / V (deg) |
| :--- | :---: | :---: |
| Train-fold mean angle | 5.91 ± 0.46 / 3.28 ± 0.17 | 6.43 / 3.59 |
| Robust line 60 s, XGBoost | 3.22 ± 0.66 / 3.44 ± 0.79 | 3.83 / 3.50 |
| Robust line 60 s, Deep Conv1D+BiLSTM | 2.89 ± 0.55 / 3.11 ± 0.63 | 3.66 / 3.24 |
| + context + range, XGBoost, weighted training | 1.70 ± 0.47 / **2.18 ± 0.48** | **2.60 / 2.51** |
| + context + range, XGBoost, fixation windows only | **1.52 ± 0.33** / 2.25 ± 0.53 | 4.15 / 3.15 |

| Dataset 1 known start, MAE H / V (deg) | Fit | All segments | Outlier segments dropped |
| :--- | :--- | :---: | :---: |
| Short, detected saccades | same subject | 0.90 ± 0.21 / 1.26 ± 0.13 | 0.72 ± 0.18 / 1.17 ± 0.12 |
| Short, detected saccades | unseen subject | 0.94 ± 0.25 / 1.43 ± 0.22 | 0.80 ± 0.28 / 1.37 ± 0.21 |
| Long 32 s, detected saccades | same subject | 4.14 ± 1.69 / 8.54 ± 2.33 | 3.71 ± 1.73 / 8.23 ± 2.58 |
| Long 32 s, fused with cross-subject XGBoost | same subject | 1.17 ± 0.22 / 1.55 ± 0.27 | **1.15 ± 0.23 / 1.55 ± 0.27** |
| Long 32 s, fused with cross-subject XGBoost | unseen subject | 1.16 ± 0.24 / 1.55 ± 0.28 | 1.14 ± 0.26 / 1.54 ± 0.28 |

- **Context and range features matter most here.** Weighted XGBoost removes 71% / 34% of the
  mean-angle fixation error. It roughly halves the error of XGBoost without the features
  (3.22 / 3.44° → 1.70 / 2.18°). Without them, XGBoost's vertical error is no better than always
  predicting the mean angle (3.44° vs 3.28°): vertical targets span only about ±13°, and every
  trial returns to the centre.
- **Smaller errors than Dataset 2** (1.70° vs 4.43° horizontal with the same setup), although the
  horizontal targets span as far. The montage, sessions and subjects all differ, and none of them
  was tested as the cause.
- **Deep model:** it beats XGBoost without context features (2.89 / 3.11° vs 3.22 / 3.44°), and gets
  no context features itself.
- **Known start:**
  - **Short segments:** 0.72 / 1.17°.
  - **Long segments:** they drift vertically (8.23°), as on the other datasets.
  - **Fused with cross-subject XGBoost:** long segments reach 1.15 / 1.55°. The fit picks a 1 s time
    constant almost everywhere, so here the fused estimate mostly follows XGBoost after the first
    seconds of a segment.
  - Results: `reports/experiments/dataset1/known_start/` and `reports/experiments/known_start_fusion/dataset1/`.

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
  timing, but free head movement, with head pose from a trakSTAR sensor. The target angles are
  given in a face frame that turns with the head. While the eyes hold a screen target, a head
  rotation moves the target by about minus that rotation: within-cue target changes regress on
  head yaw and pitch with slopes of −1.0 to −1.1. The eyes follow with slow vestibulo-ocular (VOR)
  movements, which the detected-saccade estimator misses and a drift baseline can absorb. The
  Dataset 3 paper (Barbara et al., BSPC 90, 2024) adds a VOR model to its dual Kalman filter. Here
  the known-start estimator gets a head-rotation term, and the regression models optional
  head-pose features (`configs/dataset3_range_head_causal_*.yaml`).
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
| + context + range + head pose, XGBoost, weighted training | — | — | 4.20 / 3.95 | 5.99 / 5.25 |
| + context + range + head pose, XGBoost, fixation windows only | — | — | **4.17 / 3.93** | 7.10 / 5.85 |
| Known start, same subject, short (detected saccades) | 0.54 / 1.21 | — | 2.72 / 1.97 | — |
| … outlier segments dropped | 0.47 / 1.03 | — | 2.67 / 1.73 | — |
| … + head rotation, outlier segments dropped | — | — | **1.50 ± 0.31 / 1.50 ± 0.52** | — |
| *Published: dual Kalman filter + VOR model, short* | — | — | 1.85 ± 0.51 / 2.19 ± 0.62 | — |
| *Published: signal differencing, short* | — | — | 3.59 ± 0.74 / 2.52 ± 0.62 | — |
| Known start, same subject, long 32 s (detected saccades) | 2.26 / 9.69 | — | 8.04 / 13.02 | — |
| … outlier segments dropped | 2.02 / 9.69 | — | 7.67 / 12.67 | — |
| … + head rotation, outlier segments dropped | — | — | 5.33 ± 2.00 / 12.42 ± 14.96 | — |
| … fused with cross-subject XGBoost, outlier segments dropped | — | — | **4.01 ± 0.63 / 4.01 ± 1.20** | — |
| *Published: dual Kalman filter + VOR model, long* | — | — | 4.64 ± 1.37 / 6.10 ± 2.58 | — |
| *Published: signal differencing, long* | — | — | 8.13 ± 1.15 / 11.25 ± 5.03 | — |

- **Context and range features with weighted training:** XGBoost's fixation MAE drops from
  1.90 / 2.26 to 1.33 / 1.56° on Dataset 4 and from 4.65 / 4.67 to 4.42 / 4.46° on Dataset 3.
  On Dataset 4 all 14 subjects improve on both axes (Wilcoxon signed-rank p < 0.001; bootstrap
  95% CI of the improvement 0.49–0.66 / 0.59–0.81°). On Dataset 3, 6 of 8 subjects improve
  horizontally and 5 of 8 vertically (p = 0.20 / 0.38; CI −0.04–0.50 / −0.48–0.81°), so with 8
  subjects that gain is not significant. The features and the weighting were not run separately
  here, so the gain belongs to the combination. It holds on Dataset 4, whose targets are not spread
  evenly (centre plus a 12° ring). Tests: `scripts/subject_statistics.py`.
- **Compared with always predicting the mean angle,** weighted XGBoost removes
  65% / 45% (H / V) of the fixation error on Dataset 2,
  65% / 59% on Dataset 4 and 50% / 29% on
  Dataset 3. Dataset 4's errors are small in degrees mostly because its targets span only ±12°.
- **Deep model:** on Dataset 4 it beats XGBoost without context features (1.57 / 1.80 vs
  1.90 / 2.26°); on Dataset 3 it scores 4.76 / 4.67° against XGBoost's
  4.65 / 4.67°. It gets no context or range features.
- **Dataset 3 keeps a smaller share of the gain:** weighted XGBoost removes 50% / 29% of the
  mean-angle error versus 65% / 45% on Dataset 2, and long known-start segments reach
  8.04 / 13.02°. Free head movement is the obvious suspect, but Dataset 3 also has fewer subjects (8),
  so the cause is not isolated; head-pose features recover part of the gap (next point).
- **Head-pose features (Dataset 3 regression):** `dataset3_range_head_causal_*` add head yaw,
  pitch and roll, and their deviation from a past-only 60 s mean.
  - **Weighted training:** fixation MAE falls from 4.42 / 4.46° to 4.20 / 3.95°, and all-window
    RMSE from 6.18 / 5.84° to 5.99 / 5.25°.
  - **Fixation windows only:** fixation MAE falls from 4.28 / 4.36° to 4.17 / 3.93°.
  - Most of the gain is vertical.
- **Known start against the Dataset 3 paper** (outliers dropped; the paper drops 7.77% of short and
  3.91% of long segments, this evaluation 2.9–3.4%):
  - **Short segments:** the head-rotation term brings the error to 1.50 / 1.50°, below the paper's
    Kalman filter with VOR model (1.85 / 2.19°) and signal differencing (3.59 / 2.52°). Without the
    term it is 2.67 / 1.73°. On saccade windows alone (none dropped) the term cuts horizontal
    error from 3.81° to 2.23°.
  - **Long segments:** the term cuts horizontal error from 7.67° to 5.33°, against 4.64° for the
    Kalman filter. Vertical error stays at 12.42 ± 14.96°, far above the paper's 6.10°: missed
    blinks (next point) cause it, and the head term does not touch them.
  - **Unseen subjects:** with the term, 1.63 / 1.92° on short and 5.67 / 9.37° on long segments.
  - **Fused with cross-subject XGBoost** (detected saccades + head rotation,
    `scripts/known_start_fusion.py`): long segments reach 4.01 ± 0.63 / 4.01 ± 1.20°, below the
    Kalman filter with VOR model (4.64 / 6.10°) on both axes, and 4.42 / 4.89° for unseen subjects.
    Short segments stay at 1.51 / 1.52°.
  - **How the term is fitted:** it uses the same fitted H/V map as detected saccades. That map is
    fitted on the cue steps, which are measured before the head moves, so refitting it with the
    head term included would bias it.
- **Known start, long segments, vertical:** blinks the detector misses leave a lasting vertical
  offset that keeps adding up (1.9% of labelled blinks counted as eye
  movements on Dataset 4, 5.1% on Dataset 3).
- **Direction invariance (Phase 10, Dataset 4 deep model):**
  `python main.py --config configs/dataset4_robust_line_causal.yaml --phase ablations` sorts the
  test windows by target direction. Targets within 3° of the centre (71% of windows) have no
  direction and get their own bucket; an earlier version had counted them as 0°.
  - **By direction:** the 2-D gaze RMSE is 1.96° at the centre and 5.56–6.36° across the eight 45°
    direction buckets, so the error hardly depends on direction.
  - **Within a bucket:** the error is larger along the saccade's axis. H RMSE is 4.8–4.9° for
    mostly horizontal targets, and V RMSE 5.4–5.8° for mostly vertical ones.
  - Results: `reports/experiments/dataset4/robustline60_causal/direction_invariance_results.json`.

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
- **Occasional recalibration (anchors).** Every 30, 60 or 120 s the target of the next fixation
  window is given. Until the next anchor, the estimate is that target plus the change in the
  model's prediction since the anchor. Both columns score the same windows (anchors and the time
  before the first one excluded); the model uses range features:

  | Test recording, fixation MAE H / V (deg) | No anchors | Every 30 s | Every 60 s | Every 120 s |
  | :--- | :---: | :---: | :---: | :---: |
  | Full recording | 4.39 / 3.87 | 4.81 / 4.76 | 5.50 / 5.20 | 5.84 / 5.29 |
  | Right only | 15.56 / 5.68 | 8.04 / 7.46 | 9.24 / 7.90 | 12.20 / 8.03 |
  | Top only | 7.35 / 9.74 | 8.99 / 5.45 | 9.14 / 5.65 | 9.99 / 6.02 |
  | Centre only | 9.30 / 5.15 | 9.96 / 6.48 | 11.47 / 7.06 | 11.41 / 6.78 |
  | Random trials, right count | 8.06 / 6.01 | 8.83 / 6.87 | 9.99 / 7.10 | 10.72 / 6.99 |

  - **One-sided gaze:** anchors roughly halve the error on the skewed axis. Right only, horizontal
    falls from 15.6° to 8.0° with an anchor every 30 s; top only, vertical falls from 9.7° to 5.5°.
  - **Everything else gets worse:** the error rises on the other axis, and on every recording
    whose gaze is balanced (full recording 4.39 / 3.87° → 4.81 / 4.76°). Each anchor carries one
    window's prediction error into every estimate until the next anchor, while the drift baseline
    already removes offsets when gaze is balanced.
  - **When to use it:** recalibrating pays off only when gaze is known to stay on one side, and
    probably with several anchors averaged (not tested here).
- These recordings are shortened, so the absolute numbers are not comparable to the main tables.
  Results: `reports/experiments/range_stress_test/dataset2/range_stress_test.json`.

#### Nested cross-validation (Dataset 2)

The experiments above chose the baseline window, feature set and training windows by comparing
results on the same 5 test folds they report. `scripts/nested_cv.py` makes that choice without the
test subjects. Inside each outer fold, a 4-fold cross-subject split of the 8 training subjects
scores all 27 combinations:
- baseline window: 30, 60 or 120 s
- features: engineered, + context, or + context + range
- training windows: all, weighted or fixation

The combination with the best inner fixation MAE (or inner RMSE) is refit on the 8 subjects and
scored once on the 2 held-out subjects. XGBoost runs on the GPU, so numbers differ slightly from
the CPU tables above. The context and range settings themselves (baselines, lags, range windows,
quantiles) keep their configured values and were not part of the selection.

| Dataset 2, XGBoost | Real-time fixation MAE H / V | Real-time RMSE H / V | Offline fixation MAE H / V | Offline RMSE H / V |
| :--- | :---: | :---: | :---: | :---: |
| Nested, selected by fixation MAE | 4.49 ± 1.12 / 3.81 ± 1.03 | 7.49 / 5.75 | 3.13 ± 0.60 / 3.39 ± 0.75 | 6.72 / 5.37 |
| Nested, selected by RMSE | 4.48 ± 1.11 / 3.83 ± 0.89 | 6.42 / 5.19 | 3.34 ± 0.77 / 3.37 ± 0.78 | 5.31 / 4.72 |
| Best combination picked on the test folds (fixation MAE) | 4.43 / 3.72 | 7.46 / 5.66 | 3.11 / 3.25 | 6.80 / 5.19 |
| 60 s baseline, engineered features, all windows | 5.65 / 4.57 | 7.67 / 6.06 | 3.76 / 3.50 | 5.60 / 4.77 |

- **The earlier choice holds up.** Selecting by fixation MAE, every outer fold on both tracks picks
  context + range features with fixation-only training. The real-time track picks a 30 s baseline
  in 4 folds and 60 s in 1; the offline track picks 60 s in 3 and 120 s in 2. Selecting by RMSE
  picks range features in 9 of the 10 folds, with weighted training in 8.
- **Choosing on the test folds was only slightly optimistic.** Nested fixation MAE is 0.06 / 0.09°
  (real-time) and 0.02 / 0.14° (offline) worse than the best combination picked on the test folds,
  and 0.0–0.16° worse than the best CPU results in the comparison table. Real-time, it stays below
  the paper's long-segment Kalman filter results (4.49 / 3.81° vs 5.23 / 6.59°).
- **Selecting by RMSE** gives nearly the same fixation MAE with a much lower all-window RMSE
  (real-time 6.42 / 5.19° vs 7.49 / 5.75°), the same trade-off as weighted training.
- Results: `reports/experiments/nested_cv/dataset2/{realtime,offline}/` (`nested_cv_results.json`
  with each fold's inner scores and choice, and `fixed_candidates.csv` with every combination scored
  on the test folds).

#### Per-subject significance

`scripts/subject_statistics.py` compares two models on each subject's fixation MAE. It runs a
two-sided Wilcoxon signed-rank test and computes a bootstrap 95% confidence interval of the mean
improvement (10,000 resamples of subjects). With 8–14 subjects these tests have little power, and
the p-values are not corrected for multiple comparisons. The reference is always a 60 s real-time
(Dataset 2 offline: centred) baseline with engineered features, trained on all windows.

| Fixation MAE improvement over the reference | Subjects | Mean H / V (deg) | 95% CI H | 95% CI V | Subjects improved H / V | Wilcoxon p H / V |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Dataset 2 real-time, nested selection | 10 | 1.17 / 0.76 | 0.51 – 2.00 | 0.52 – 1.01 | 9 / 10 | 0.010 / 0.002 |
| Dataset 2 offline, nested selection | 10 | 0.64 / 0.11 | 0.37 – 0.89 | −0.13 – 0.31 | 9 / 7 | 0.004 / 0.19 |
| Dataset 3, + context + range, weighted | 8 | 0.23 / 0.22 | −0.04 – 0.50 | −0.48 – 0.81 | 6 / 5 | 0.20 / 0.38 |
| Dataset 3, + context + range, fixation windows | 8 | 0.37 / 0.32 | 0.07 – 0.67 | −0.31 – 0.85 | 6 / 5 | 0.078 / 0.38 |
| Dataset 4, + context + range, weighted | 14 | 0.58 / 0.70 | 0.49 – 0.66 | 0.59 – 0.81 | 14 / 14 | < 0.001 / < 0.001 |
| Dataset 4, + context + range, fixation windows | 14 | 0.79 / 0.76 | 0.70 – 0.89 | 0.64 – 0.87 | 14 / 14 | < 0.001 / < 0.001 |

- **Consistent across subjects:** on Dataset 2 real-time and on Dataset 4, the improvement holds on
  both axes. All 14 Dataset 4 subjects and 9–10 of the 10 Dataset 2 subjects improve.
- **Offline on Dataset 2, the gain is horizontal only.** Vertical error barely changes (7 of 10
  subjects improve; the confidence interval includes 0).
- **Dataset 3 is not significant.** 5–6 of 8 subjects improve, and only the fixation-window model's
  horizontal confidence interval excludes 0.
- Results: `reports/experiments/statistics/statistics.json`.

#### Within-subject protocol (Dataset 2)

The paper fits every parameter on the test subject's own data; the tables above never see the
test subject. `scripts/within_subject.py` scores the real-time XGBoost setup (context + range
features, weighted training) the paper's way. Each subject's recording is split in time into
three parts, and for each of the 6 ordered (fit, test) pairs a model is trained and then scored
on the test part. Test parts are never trained on.

| Dataset 2, real-time XGBoost | Training data | Fixation MAE H / V (deg) |
| :--- | :--- | :---: |
| Within subject | the fit part of the test subject only (~4.5 min) | 5.44 ± 1.21 / 4.30 ± 0.82 |
| Adapted | all other subjects + the fit part of the test subject | **4.27 ± 0.97 / 3.78 ± 0.95** |
| Cross subject | all other subjects | 4.48 ± 1.14 / 3.95 ± 1.17 |

- **One subject's data alone is too little.** Trained on a third of the test subject's own recording,
  XGBoost does worse than the cross-subject model for 9 of 10 subjects on each axis (Wilcoxon
  p = 0.02 / 0.06).
- **Adding the subject's own data to the other subjects helps a little.** Horizontal gains are not
  significant (7 of 10 subjects improve, p = 0.32); vertical gains are (9 of 10, p = 0.02).
- **The protocol does not explain the gap to the paper.** Under its within-subject protocol this
  model scores about the same as cross-subject or worse, so the cross-subject numbers in the
  [comparison](#comparison-with-barbara-et-al-2023-bspc-86) are not an artefact of the easier
  protocol.
- Results: `reports/experiments/within_subject/dataset2/within_subject.json`; paired tests in
  `reports/experiments/statistics/statistics.json`.

---

## Experiment Index

Every experiment is one YAML config in `configs/`, and no two experiments share an output folder.
Dataset 2 experiments write to `reports/experiments/<name>/`; Datasets 1, 3 and 4 write to
`reports/experiments/datasetN/<name>/`, with processed data under `data/processed_datasetN/<name>/`
(git-ignored). Each results folder holds one JSON per
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
| `dataset1_robust_line_causal.yaml` | `reports/experiments/dataset1/robustline60_causal/` | Dataset 1 real-time baseline (bipolar H and V) |
| `dataset1_range_causal_{weighted,fixation}.yaml` | `reports/experiments/dataset1/range_causal_*/` | Dataset 1 with the best Dataset 2 setup |
| `dataset1_known_start.yaml` | `reports/experiments/dataset1/known_start/` | Dataset 1 known-start protocol |
| `dataset3_robust_line_causal.yaml` | `reports/experiments/dataset3/robustline60_causal/` | Dataset 3 real-time baseline (no head pose) |
| `dataset3_range_causal_{weighted,fixation}.yaml` | `reports/experiments/dataset3/range_causal_*/` | Dataset 3 with the best Dataset 2 setup |
| `dataset3_range_head_causal_{weighted,fixation}.yaml` | `reports/experiments/dataset3/range_head_causal_*/` | Dataset 3 with head-pose features added |
| `dataset3_known_start.yaml` | `reports/experiments/dataset3/known_start/` | Dataset 3 known-start protocol |
| `dataset4_robust_line_causal.yaml` | `reports/experiments/dataset4/robustline60_causal/` | Dataset 4 real-time baseline (18 channels) |
| `dataset4_range_causal_{weighted,fixation}.yaml` | `reports/experiments/dataset4/range_causal_*/` | Dataset 4 with the best Dataset 2 setup |
| `dataset4_known_start.yaml` | `reports/experiments/dataset4/known_start/` | Dataset 4 known-start protocol |
| `scripts/nested_cv.py` with `dataset2_range_{causal,centred}_weighted.yaml` | `reports/experiments/nested_cv/dataset2/{realtime,offline}/` | Nested cross-subject selection of baseline window, features and training windows |
| `scripts/range_stress_test.py` with `dataset2_range_causal_weighted.yaml` | `reports/experiments/range_stress_test/dataset2/` | Range features when test gaze covers only part of the screen |
| `scripts/subject_statistics.py` | `reports/experiments/statistics/` | Per-subject Wilcoxon tests and bootstrap confidence intervals |
| `scripts/known_start_fusion.py --dataset datasetN` | `reports/experiments/known_start_fusion/datasetN/` | Known start fused with cross-subject XGBoost |
| `scripts/within_subject.py` with `dataset2_range_causal_weighted.yaml` | `reports/experiments/within_subject/dataset2/` | The paper's within-subject protocol for real-time XGBoost |

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
│   ├── subject_statistics.py  # per-subject significance tests and confidence intervals
│   ├── known_start_fusion.py  # known start fused with cross-subject XGBoost
│   ├── within_subject.py    # the paper's within-subject protocol
│   └── make_figures.py      # result figures from the saved JSON files
├── tests/                   # pytest test suite
├── reports/
│   ├── experiments/         # <name>/ (Dataset 2), dataset1/<name>/, dataset3/<name>/, dataset4/<name>/
│   └── figures/             # raw-signal plots; results/ holds the result figures
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

## Reproducing the Results

1. Download the four datasets (see above) and run `python scripts/inspect_raw.py` to check that
   the parsers read them.
2. `python scripts/verify_montage.py` checks the bipolar channel pairs and signs against the
   recorded target angles. The shipped maps were verified this way on 2026-08-21 (the original
   placeholder maps had H and V swapped for Datasets 2–4); update `MONOPOLAR_MAPS` in
   [`src/data/unify.py`](src/data/unify.py) if the check disagrees.
3. Run each experiment in the [Experiment Index](#experiment-index). Every config's header lists
   its phases, usually `preprocess`, `train_classical`, `train_deep` and `compare` (`known_start`
   for the known-start configs). Each writes to its own results folder, including a
   `master_results_table.csv`.
4. Robustness checks: `scripts/nested_cv.py` for both tracks and `scripts/range_stress_test.py`,
   then `scripts/subject_statistics.py`, which reads the nested results.
5. `python scripts/make_figures.py` writes the result figures to `reports/figures/results/`.
6. `pytest tests/` takes about a minute; tests that need the datasets skip without them.

XGBoost on the GPU (`cv.xgb_device: cuda`: Datasets 1, 3 and 4 and the robustness scripts) gives
slightly different numbers from the CPU XGBoost used for the Dataset 2 configs.

---

## Appendix: Master Pitfall Checklist

- [x] Never split by trial — always by `subject_id` (GroupKFold over subjects; automated in `test_no_subject_leakage.py`)
- [x] Drift removal happens BEFORE normalization, not after (`preprocess_trials`: drift removal, then per-subject z-score)
- [x] Bipolar sign convention verified per dataset — Datasets 2–4 with `scripts/verify_montage.py`, Dataset 1 from the EOG steps at cue onsets (see `src/data/unify.py`)
- [x] Class imbalance (blink rarity) accounted for — `class_weight="balanced"` for RF / SVC, `model.auto_class_weights` for the deep model
- [ ] Head-pose correction on Dataset 3 validated with before/after ablation (Phase 9)
- [x] Every reported metric traces to a saved JSON file in `reports/` (Dataset 4's 59 MB deep-model JSON stays out of git; its numbers are in the master table next to it)
- [x] Deep model results reported honestly even if it loses to the classical baseline
