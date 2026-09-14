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
- **Outliers:** the paper drops segments with "substantially high" error. Here segments are
  dropped without looking at their error: those where the estimator's blink decisions contradict
  the labels (a labelled blink counted as an eye movement, or a response saccade in a blink-free
  window dropped as a blink), the cause the paper names for its outliers. This drops 0.3% of short
  and 7.1% of long segments (paper: 6.85% and 5.83%).

**Estimators** (EOG and starting gaze only; a 2-channel H/V linear map without intercept, like
the paper's comparison method [BSPC 47, 2019]):

- *Detected saccades* (signal differencing): eye movements are detected from EOG velocity and
  each adds its displacement; blinks are rejected. The detector finds 93% of cue-driven saccades,
  and 95% of its detections follow a cue.
- *Level change*: the EOG change since the segment start (short segments only).

| Known-start task, MAE H / V (deg) | Fit | All segments | Label-disagreement segments dropped |
| :--- | :--- | :---: | :---: |
| Short, detected saccades | same subject | 1.17 ± 0.49 / 1.39 ± 0.27 | 1.11 ± 0.47 / 1.38 ± 0.26 |
| … saccade windows only | same subject | 1.75 / 2.09 | — |
| Short, level change | same subject | 1.47 ± 0.61 / 2.37 ± 0.60 | 1.47 ± 0.61 / 2.36 ± 0.59 |
| Short, detected saccades | unseen subject | 1.34 ± 0.46 / 1.82 ± 0.54 | 1.28 ± 0.47 / 1.81 ± 0.54 |
| … saccade windows only | unseen subject | 2.02 / 2.73 | — |
| *Published: dual Kalman filter, short* | same subject | — | 1.64 ± 0.82 / 1.97 ± 0.34 |
| *Published: signal differencing, short* | same subject | — | 1.51 ± 0.55 / 1.95 ± 0.29 |
| Long 32 s, detected saccades | same subject | 4.71 ± 1.59 / 8.39 ± 3.13 | 4.34 ± 1.35 / 8.35 ± 3.20 |
| Long 32 s, detected saccades | unseen subject | 4.91 ± 1.41 / 8.73 ± 3.63 | 4.55 ± 1.19 / 8.69 ± 3.66 |
| *Published: dual Kalman filter, long* | same subject | — | 5.23 ± 2.00 / 6.59 ± 3.10 |
| *Published: signal differencing, long* | same subject | — | 5.82 ± 2.70 / 8.04 ± 2.96 |

**How close it gets:**

- **Short segments: comparable, not clearly better.** The combined score (1.17 / 1.39°) is below
  both published methods, but a third of the short windows are blink windows. There the target
  does not move, and once the blink is rejected this estimator scores exactly 0. On saccade
  windows alone it scores 1.75 / 2.09°, about the paper's combined numbers. The paper does not
  report that split, so how much its own blink windows lower its score is unknown.
- **Long segments:** horizontal error is lower than both published methods (4.34 vs 5.23 /
  5.82° after exclusion), vertical error is higher than both (8.35 vs 6.59 / 8.04°). Summing
  detected displacements is essentially the paper's signal-differencing method; the Kalman
  filter's blink and eyelid modelling handles the vertical channel better.
- **Unseen subjects** (not in the paper): 1.34 / 1.82° on short segments with no labels from the
  test subject.
- **Remaining differences from the paper:** ground-truth thresholds come from each whole
  recording rather than from training data only; subject mistakes are judged by fixed rules rather
  than by the authors; the outlier rule is label-based rather than error-based; the detector and
  blink rules were set by inspecting Dataset 2 events from all subjects; and blink decisions look
  up to 150 ms past the end of a movement.
---

## Project Structure

```
eog-gaze-pipeline/
├── data/
│   ├── raw/                 # downloaded ZIPs extracted here
│   └── processed/           # cached numpy arrays + CV folds
├── src/
│   ├── config.py            # ALL tunable parameters — edit here only
│   ├── data/                # schema, loaders, unify, datasets
│   ├── preprocessing/       # filtering, blink detection, normalization
│   ├── features/            # hand-crafted features for classical ML
│   ├── models/              # integration baseline, classical ML, deep model
│   ├── training/            # CV splits, training loop
│   ├── evaluation/          # metrics, paper comparison table
│   └── ablations/           # head-pose (Phase 9), direction (Phase 10)
├── scripts/
│   └── inspect_raw.py       # Phase 0 verification
├── tests/                   # pytest test suite
├── notebooks/               # EDA and results notebooks
├── reports/figures/         # all saved plots
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
