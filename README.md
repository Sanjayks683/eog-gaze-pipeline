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

> **Note:** these numbers come from the run *before* the 2026-09-10 fixes (250 Hz → 256 Hz
> sampling rate, subject-held-out validation, least-squares calibration, trivial baselines).
> The pipeline is being re-run and this table will be updated from `reports/master_results_table.csv`.

Evaluated under strict 5-fold cross-subject GroupKFold (zero test-subject leakage). The target horizontal gaze range is $\pm 27.3^\circ$ (std $= 14.07^\circ$) and vertical range is $\pm 16.0^\circ$ (std $= 8.00^\circ$).

| Method | Dataset | RMSE H (deg) | RMSE V (deg) | MAE H (deg) | MAE V (deg) | F1 (weighted) | Notes |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| *Trivial: train-fold mean angle* | Dataset 2 only | 14.08° | 8.00° | — | — | — | Always predicts the training folds' mean angle |
| **Classical ML (SVC)** | Dataset 2 only | — | — | — | — | 0.703 | 104 hand-crafted features from 6 channels |
| **Classical ML (RF)** | Dataset 2 only | — | — | — | — | **0.716** | 104 hand-crafted features from 6 channels |
| **Classical ML (SVR)** | Dataset 2 only | 11.82° | 7.28° | 9.59° | 6.02° | — | 104 hand-crafted features from 6 channels |
| **Classical ML (XGB)** | Dataset 2 only | 11.58° | **7.17°** | 9.41° | 5.94° | — | 104 hand-crafted features from 6 channels |
| **Deep Conv1D+BiLSTM** | Dataset 2 only | **11.17°** | 7.23° | **8.87°** | **5.93°** | 0.714 | Multi-task joint loss; $R^2_H = 0.37$, $R^2_V = 0.18$ |
| *Published: Barbara 2023* | Dataset 2 | 2.23° | 2.39° | — | — | — | *BSPC vol. 86 (within-subject calibrated battery model)* |

> **Key takeaway**: on unseen subjects the best model (Conv1D+BiLSTM) explains ~37% of horizontal
> and ~18% of vertical gaze-angle variance, i.e. only 21% / 10% lower RMSE than always predicting
> the mean angle, and several times the error of the within-subject calibrated method in
> Barbara 2023. The main limitation is the input: a 0.2 Hz high-pass on 300 ms windows removes
> most of the absolute (DC) gaze-position information. Per-subject affine calibration does not
> close the gap.

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
