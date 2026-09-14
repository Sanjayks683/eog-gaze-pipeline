# Subject-independent EOG gaze estimation on the four EyeCon datasets: drift-robust context features, head-rotation compensation and complementary fusion

**Draft, 2026-09-15.** Every number is taken from the committed results of this repository; the
README section named in brackets gives its source file. Items marked **TODO** need the authors.

- **Authors:** TODO (names, affiliations, corresponding author)
- **Target journal:** TODO (e.g. Biomedical Signal Processing and Control, where the EyeCon papers appeared)
- **Code:** https://github.com/Sanjayks683/eog-gaze-pipeline (MIT licence)

---

## Abstract

Electrooculography (EOG) is a cheap, camera-free way to estimate gaze, but its baseline drift
makes absolute gaze angles hard to recover, and published EyeCon results fit every parameter on
the test subject and start each segment from a known gaze. We evaluate subject-independent gaze
estimation on all four public EyeCon datasets (6, 10, 8 and 14 subjects) with a 5-fold
cross-subject protocol and fixation mean absolute error (MAE) per subject.

**Real-time, subject-independent estimation.** XGBoost on 300 ms windows reaches 1.70 / 2.18°
(horizontal / vertical) on Dataset 1, 4.49 / 3.81° on Dataset 2, 4.17 / 3.93° on Dataset 3 (free
head) and 0.92 / 1.34° on Dataset 4. The model uses a past-only robust drift baseline plus context
features: baseline disagreement, lagged levels and rolling-range statistics. Its settings were
chosen by nested cross-validation.
- **Significance:** against engineered window features alone, the gain is significant on Datasets
  1, 2 and 4 (Wilcoxon signed-rank p ≤ 0.031), but not on Dataset 3.
- **Head pose:** on Dataset 3, head-pose features lower the error further.

**Published known-start task.** We replicate it and add two estimators:
- **Head rotation:** a term that adds the head's rotation between detected eye movements lowers
  Dataset 3 short-segment error to 1.50 / 1.50°. The published Kalman filter with a VOR model
  reaches 1.85 / 2.19°.
- **Fusion:** a complementary filter fuses summed saccades with the cross-subject model. It removes
  the vertical error that builds up over 32 s segments: 3.36 / 3.38° on Dataset 2 (published
  5.23 / 6.59°) and 4.01 / 4.01° on Dataset 3 (published 4.64 / 6.10°).

**Robustness checks.** Stress tests show that a rolling drift baseline fails when gaze stays on one
side of the screen for longer than its window. A within-subject protocol does not improve the
subject-independent model.

**TODO:** shorten to the journal's word limit.

---

## 1. Introduction

**TODO:** expand with a broader literature review (camera-based vs EOG gaze tracking,
applications in assistive interfaces and sleep research).

Electrooculography measures the corneo-retinal potential with electrodes around the eyes. It is
cheap, works with the eyes closed and in darkness, and needs no camera, but its baseline drifts
slowly, so the signal level is only loosely tied to absolute gaze.

Barbara et al. published four EyeCon datasets with this problem in mind. They compare drift
mitigation techniques (Dataset 1, BSPC 57, 2020), estimate gaze in real time with a dual Kalman
filter for a stationary head (Dataset 2, BSPC 86, 2023) and a non-stationary head with a
vestibulo-ocular reflex (VOR) model (Dataset 3, BSPC 90, 2024), and analyse bipolar channel
selection (Dataset 4, BSPC 112, 2026). Their gaze-angle results share one protocol:
- **Within subject:** parameters are fitted on part of the test subject's own recording.
- **Known start:** each evaluated segment starts from the known gaze.
- **Scoring:** fixation samples, with outlier segments excluded.

Other groups have used the data for other tasks: direction classification on Dataset 3 with a
random window split (Mahmood et al., 2024), and saccade detection for sleep studies (Sensors,
2026). Eye-movement labelling with hidden Markov models (Mifsud et al., 2026) used separate
recordings.

A practical EOG interface would ideally estimate absolute gaze for a new user without collecting
labelled data from that user, and without being told the gaze at the start of every segment. How
well that works on these datasets has not been reported. Absolute estimation also depends on
choices, such as the drift baseline, its window and the training windows, that are easily tuned on
the same folds the results are reported on.

**Contributions**

1. **A subject-independent benchmark on all four EyeCon datasets.** It uses 5-fold cross-subject
   evaluation, per-subject fixation MAE, per-subject significance tests, and nested
   cross-validation for every tuned setting.
2. **Context features for 300 ms windows.** Their rolling-range statistics estimate the screen
   centre and the subject's EOG gain without labels.
3. **A replication of the published known-start protocol,** with a head-rotation term for free-head
   recordings.
4. **A complementary fusion** of summed saccades with a cross-subject absolute model, which removes
   error build-up over long segments.
5. **Robustness analyses:** skewed-gaze stress tests with controls, recalibration anchors, a
   within-subject protocol, and a search over the feature settings. Their negative results define
   where the approach fails.

---

## 2. Data

All four datasets are sampled at 256 Hz (g.tec g.USBamp). Every trial has the same timing: a cue
for 1 s, a second cue for 1 s, then a 2 s blink interval, all marked by a control signal. The
datasets differ as follows [README: Data Description Table]:

| | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
| :--- | :---: | :---: | :---: | :---: |
| Subjects | 6 | 10 | 8 | 14 |
| Channels used | bipolar H, V | H, V + 4 monopolar | H, V + 4 monopolar | H, V + 16 monopolar |
| Head | chin rest | chin rest | free (trakSTAR head pose) | chin rest |
| Trials per subject | 300 (3 sessions) | 200 | 200 | 288 |
| Cues | centre → random target → centre | random → random | random → random | centre → 12° ring (36 directions) → centre |

**Preparation notes.**
- **Dataset 1 targets:** its target file holds one row per cue. We spread each row over the samples
  of its cue interval; stored as-is, the rows would not match the samples.
- **Dataset 3 targets:** its gaze angles are given in a face frame. Within a cue, target changes
  regress on head yaw and pitch with slopes of −1.0 to −1.1, so a head rotation moves the target by
  about minus that rotation.
- **Signs:** the bipolar sign conventions were verified against the targets for every dataset.

---

## 3. Methods

### 3.1 Preprocessing and windows

1. **Bipolar channels:** H and V are formed from the verified electrode pairs.
2. **Drift removal** (Section 3.2).
3. **Low-pass filter:** the signals are filtered at 30 Hz (zero-phase). A 50 Hz notch is added only
   where the spectrum shows a mains peak.
4. **Scaling:** each recording is z-scored per subject, with the scale estimated from its first 10%.
5. **Windows:** the models see 300 ms windows with a 150 ms stride.
6. **Targets:** a window's target is the mean gaze angle over the window.

A window is a *fixation window* when it lies inside a cue interval, starts at least 400 ms after
the cue changed, and contains no further cue change. It approximates the fixation samples scored in
the published work.

### 3.2 Drift baselines

A robust baseline is fitted to a sliding window of the raw signal, after dropping samples more than
3 robust standard deviations from the window median (blinks). Two variants:
- **Real-time:** a past-only robust line, evaluated at the current time.
- **Offline:** a centred robust mean.

The baseline window (30, 60 or 120 s) is selected by nested cross-validation (Section 3.7).

### 3.3 Features

- **Engineered window features:** 17 per channel for each 300 ms window, plus two cross-channel
  features: the H–V correlation and the combined RMS. That makes 104 features for Dataset 2's six
  channels.
  - **Amplitude:** mean, SD, maximum absolute value, maximum, minimum, peak-to-peak and RMS.
  - **Trend and velocity:** linear slope, and the maximum and mean absolute velocity.
  - **Shape:** time above half the peak, and zero crossings of the demeaned signal.
  - **Spectrum** (Welch): band powers at 0–5, 5–15 and 15–30 Hz, and the dominant frequency.
- **Context features:** for each channel,
  - the disagreement between the main baseline and three other baselines (robust mean over 30 s and
    120 s, robust line over 120 s);
  - the signal level 1, 2, 4 and 8 s before the window.
- **Rolling-range features:** over the past 60, 120 and 240 s, and for the 10–90% and 5–95%
  quantile pairs:
  - the window level minus the rolling mid-range, halfway between the two quantiles;
  - the range between them;
  - their ratio.

  Cue positions are spread roughly uniformly over a bounded screen, and for such data the mid-range
  locates the centre with error shrinking like 1/N rather than 1/√N. The range is a label-free
  estimate of the subject's EOG gain.
- **Head-pose features (Dataset 3):** head yaw, pitch and roll over the window, and their deviation
  from a past-only 60 s mean.

In real-time mode no feature uses samples after the window's end, apart from the zero-phase
low-pass filter and the per-subject z-score scale.

### 3.4 Models

- **XGBoost regressors:** one per axis (GPU implementation for Datasets 1, 3 and 4 and for all
  analyses in Section 3.7).
- **Deep multi-task network:** a Conv1D + BiLSTM network with temporal attention. It has a
  classification head for the trial interval and a regression head for gaze. Optionally, the
  context features are standardised per fold and joined to the regression head.
- **Trivial baseline:** always predicting the training folds' mean angle.

**Training windows.** The regressors are trained on one of three window sets:
- all windows;
- fixation windows only;
- all windows, with the non-fixation windows weighted 0.2.

### 3.5 Evaluation

- **Folds:** 5-fold GroupKFold over subjects, so a test subject is never seen in training.
- **Fixation MAE:** computed per subject over fixation windows, then reported as mean ± SD across
  subjects.
- **RMSE:** computed over all windows.
- **Paired comparisons:** a two-sided Wilcoxon signed-rank test on per-subject fixation MAE, plus a
  bootstrap 95% confidence interval of the mean improvement (10,000 resamples of subjects). The
  p-values are not corrected for multiple comparisons.

### 3.6 Known-start protocol

This replicates Barbara et al. (BSPC 86, Section 4.5.3 and Appendix E).

**Labels.** Fixation, saccade and blink labels come from the EOG:
- **Fixation:** a sample where no bipolar channel's sample-to-sample difference exceeds 5 robust SDs.
- **Blink:** a pair of opposite vertical-difference spikes within 500 ms.
- **Saccade:** every other sample.

**Segments.**
- **Short:** 1 s saccade windows and 2 s blink windows without subject mistakes. Each third of a
  recording contributes 66 and 33 of them.
- **Long:** 32 s segments of 8 consecutive trials, 8 per third, with mistakes kept.

**Error.** Per segment, the MAE over its scored fixation samples; per subject, the mean over
segments; then mean ± SD across subjects. Outlier segments are defined per subject and segment kind
by Tukey's far-out fence (Q3 + 3 × IQR). The papers give no threshold for "substantially high"
error, so this rule is our stand-in.

**Fitting.**
- **Same subject:** fitted on one third, tested on another, over all 6 orderings.
- **Unseen subject:** fitted on the other subjects, with a label-free gain per subject.

**Estimators.** Each starts from the known gaze.
- **Detected saccades (signal differencing):** each eye movement detected from EOG velocity adds its
  displacement through a 2 × 2 linear map without intercept; blinks are rejected.
- **Level change:** the EOG change since the segment start, for short segments.
- **Detected saccades + head rotation** (Dataset 3): adds minus the head's yaw (H) and pitch (V)
  change between detected movements, with a VOR gain of 1. It uses the detected-saccade map
  unchanged, because the fitting pairs are cue steps, measured before the head moves.
- **Fused with cross-subject XGBoost:**
  - **Absolute model:** the real-time model of Section 3.4 with context + range features and
    weighted training, trained 5-fold cross-subject. Each window's prediction is held from the
    window's end.
  - **Filter:** the estimate adds, to the detected-saccade estimate, a causal first-order low-pass of
    (absolute estimate − saccade estimate), starting at zero at the known start.
  - **Time constant:** chosen per axis on the fit data from {1, 2, 5, 10, 20, 50, 100, 200 s, ∞}.

### 3.7 Robustness analyses

- **Nested cross-validation.**
  - **Search space:** 27 combinations of baseline window (30 / 60 / 120 s), features (engineered,
    + context, + context + range) and training windows (all, weighted, fixation).
  - **Scheme:** inner 4-fold GroupKFold within each outer training set; the selected combination is
    refit and scored once on the outer test subjects.
  - **Comparison:** every combination is also scored directly on the outer folds.
  - **Range settings (Dataset 2):** a second search covers the rolling-range windows and quantile
    pairs.
- **Skewed-gaze stress test (Dataset 2).**
  - **Skewed recordings:** each test recording is rebuilt from the trials whose cues are both right
    of centre, both above centre, or both near the centre.
  - **Controls:** random trials of the same count, joined the same way, and a contiguous block of the
    same length.
  - **Anchors:** we also score occasional recalibration, where a known gaze is given every 30, 60 or
    120 s.
- **Within-subject protocol (Dataset 2).** Three training sets, each scored on another third of the
  test subject's recording:
  - one third of the test subject's recording;
  - that third plus all other subjects;
  - all other subjects only.
- **Direction invariance (Dataset 4, deep model).** Errors are grouped by target direction, with
  centre targets separate.

---

## 4. Results

### 4.1 Subject-independent real-time estimation

Fixation MAE, horizontal / vertical (degrees), mean ± SD across subjects
[README: Results at a Glance; Dataset 1; Comparison with Barbara et al. 2023; Datasets 3 and 4]:

| Real-time, 5-fold cross-subject | Dataset 1 | Dataset 2 | Dataset 3 | Dataset 4 |
| :--- | :---: | :---: | :---: | :---: |
| Train-fold mean angle | 5.91 / 3.28 | 12.56 / 7.05 | 8.90 / 6.28 | 3.82 / 3.83 |
| Robust line 60 s, XGBoost, engineered features | 3.22 / 3.44 | 5.66 / 4.56 | 4.65 / 4.67 | 1.90 / 2.26 |
| Robust line 60 s, deep Conv1D + BiLSTM | 2.89 / 3.11 | 5.48 / 4.58 | 4.76 / 4.67 | 1.57 / 1.80 |
| + context + range, XGBoost, weighted training | 1.70 / 2.18 | 4.43 / 3.91 | 4.42 / 4.46 | 1.33 / 1.56 |
| + context + range, XGBoost, fixation windows | 1.52 / 2.25 | 4.36 / 3.78 | 4.28 / 4.36 | 1.11 / 1.51 |
| Nested cross-validation (selected by fixation MAE) | — | 4.49 / 3.81 | 4.25 / 4.20 | 0.92 / 1.34 |
| + head-pose features, fixation windows | — | — | 4.17 / 3.93 | — |

Dataset 2's XGBoost rows use the CPU implementation, except its nested cross-validation row; all
other XGBoost results use the GPU.

**Offline results** (centred robust mean baseline):
- **Dataset 2:** 3.13 / 3.39° under nested cross-validation.
- **Dataset 3:** 3.67 / 3.75°.
- **Dataset 4:** 0.66 / 1.02°.

**Per-subject significance** [README: Per-subject significance]. Context + range features against
engineered features only:
- **Dataset 1:** all 6 subjects improve on both axes, p = 0.031 (the smallest attainable with 6
  subjects).
- **Dataset 4:** all 14 subjects improve, p < 0.001.
- **Dataset 2** (nested selection, real-time): the improvement is 1.17 / 0.76°, with bootstrap 95%
  CI 0.51–2.00 / 0.52–1.01°. 9 of 10 subjects improve horizontally and 10 of 10 vertically,
  p = 0.010 / 0.002.
- **Dataset 3:** not significant (6 and 5 of 8 subjects improve, p = 0.20 / 0.38).

**Selection bias** [README: Nested cross-validation]:
- **How optimistic the tuned results were:** choosing settings on the test folds made them
  0.02–0.16° optimistic on Dataset 2, up to 0.28° on Dataset 3 and at most 0.07° on Dataset 4.
- **Choices that held up:** on Dataset 2 every outer fold selected context + range features with
  fixation-only training.
- **Choices that did not:** Datasets 3 and 4 preferred a 30 s baseline in almost every fold, instead
  of the 60 s chosen on Dataset 2.
- **Range settings:** the rolling-range windows and quantiles changed Dataset 2 fixation MAE by at
  most 0.21 / 0.31°.

**Deep model.** Joining the context features to its regression head improved it from 5.48 / 4.58° to
4.95 / 4.23° on Dataset 2. It still trailed XGBoost with the same features.

**Few-shot calibration.** An affine calibration on each subject's first 30 s increased error.

### 4.2 Known-start task against the published methods

Same-subject fitting, outlier segments dropped [README: Separate evaluation: the paper's known-start
task; Datasets 3 and 4]:

| Dataset, segments | Detected saccades | + head rotation | Fused with cross-subject XGBoost | Published: dual Kalman filter | Published: signal differencing |
| :--- | :---: | :---: | :---: | :---: | :---: |
| 2, short | 0.92 ± 0.49 / 1.33 ± 0.24 | — | unchanged | 1.64 ± 0.82 / 1.97 ± 0.34 | 1.51 ± 0.55 / 1.95 ± 0.29 |
| 2, long 32 s | 4.21 ± 1.81 / 8.36 ± 3.29 | — | 3.36 ± 0.85 / 3.38 ± 0.65 | 5.23 ± 2.00 / 6.59 ± 3.10 | 5.82 ± 2.70 / 8.04 ± 2.96 |
| 3, short | 2.67 / 1.73 | 1.50 ± 0.31 / 1.50 ± 0.52 | 1.51 / 1.52 | 1.85 ± 0.51 / 2.19 ± 0.62 (+ VOR) | 3.59 ± 0.74 / 2.52 ± 0.62 |
| 3, long 32 s | 7.67 / 12.67 | 5.33 ± 2.00 / 12.42 ± 14.96 | 4.01 ± 0.63 / 4.01 ± 1.20 | 4.64 ± 1.37 / 6.10 ± 2.58 (+ VOR) | 8.13 ± 1.15 / 11.25 ± 5.03 |
| 1, long 32 s | 3.71 / 8.23 | — | 1.15 / 1.55 | — | — |
| 4, long 32 s | 2.02 / 9.69 | — | 0.85 / 1.35 | — | — |

Our rule drops 0.7–4.9% of segments; the papers drop 3.9–7.8%.

**Short segments.**
- **Dataset 2:** on saccade windows alone, detected saccades score 1.75 / 2.09°. Blink windows,
  where a rejected blink scores zero, lower the combined score.
- **Dataset 3:** the head-rotation term cuts saccade-window error from 3.81° to 2.23° horizontally.

**Long segments.**
- **Where the error comes from:** without fusion, vertical error builds up to 8–13°, because the
  detector misses some blinks (5.1% of labelled blinks on Dataset 3).
- **Choice of time constant:** it is mostly 2–5 s vertically on Dataset 2, and mostly 1 s on
  Datasets 1 and 4. On those two datasets the fused estimate largely follows the absolute model.
- **Unseen subjects:** fusion reaches 3.80 / 3.56° on Dataset 2 and 4.42 / 4.89° on Dataset 3.

### 4.3 Robustness

**Skewed gaze** [README: Range features when test gaze covers only part of the screen].
- **Right-only cues:** horizontal fixation MAE rose to 15.5° with range features, against 8.1° for
  random trials of the same count and 6.0° for a contiguous block of the same length.
- **Top-only cues:** vertical error rose similarly (9.7° against 5.9°).
- **Range features:** they lowered the error in every scenario.
- **Anchors every 30 s:** they halved the error on the skewed axis (right only: 15.6° → 8.0°), but
  raised it on the other axis and on balanced recordings (4.39 / 3.87° → 4.81 / 4.76°).

**Within-subject training** [README: Within-subject protocol].

| Dataset 2, real-time XGBoost | Fixation MAE H / V (deg) |
| :--- | :---: |
| Trained on a third of the test subject's recording | 5.44 / 4.30 |
| Trained on other subjects plus that third | 4.27 / 3.78 |
| Trained on other subjects only | 4.48 / 3.95 |

The subject's own third alone was worse for 9 of 10 subjects on each axis (p = 0.02 / 0.06).

**Direction invariance** (Dataset 4, deep model). The 2-D RMSE was 1.96° at the centre and
5.56–6.36° across the eight direction buckets.

**Figures:** `reports/figures/results/` (TODO: select and caption for the paper).

---

## 5. Discussion

**Context turns a window model into a drift-robust estimator.** A 300 ms window carries gaze plus
the error of the baseline estimate. That error depends on recent gaze, and a regressor without
context hedges towards the screen centre. Baseline disagreement and lagged levels expose the error;
rolling-range statistics estimate where the screen centre and the subject's gain lie.
- **Where it works best:** the gain is largest where cues are spread evenly and the head is still
  (Datasets 1, 2 and 4).
- **Where it helps less:** Dataset 3, where head rotation moves the screen within the face frame.
  There, head-pose features recover part of the gap.

**The known-start results separate two error sources.**
- **Head movement:** with the head free, most of the short-segment error of summed saccades is head
  rotation between movements. Modelling it explicitly beats the published VOR model on short
  segments.
- **Build-up over time:** over long segments the error comes from missed blinks and gain errors, and
  a complementary filter towards an absolute estimate removes it. The published Kalman filters use
  no labelled data from other subjects, so the fused results show what such data adds, not a better
  filter.

**Protocol matters less than expected.** Training on the test subject's own data did not help the
absolute model, and settings chosen on the test folds were only slightly optimistic, except with 8
subjects.

**Failure mode.** A rolling drift baseline cannot tell a lasting gaze offset from drift, so
real-time absolute estimation fails when gaze stays on one side for longer than the baseline window.
Occasional recalibration fixes that axis, but costs accuracy elsewhere. A deployment would need
application knowledge of where the user looks, or several averaged anchors (not tested).

---

## 6. Limitations

- **Few subjects** (6–14 per dataset). Dataset 3's gains are not significant, and the p-values are
  not corrected for multiple comparisons.
- **Settings developed on Dataset 2.** Features and settings were developed on Dataset 2 before
  being applied to the other datasets. Nested cross-validation controls the selection within each
  dataset, but not the choice of candidate families.
- **30 s-baseline results** on Datasets 3 and 4 come from the nested-CV runs. Head-pose features were
  not combined with a 30 s baseline.
- **Known-start replication differs from the papers:**
  - label thresholds come from each whole recording;
  - subject mistakes are judged by fixed rules;
  - the outlier fence is our stand-in;
  - the detector was designed by inspecting Dataset 2.
- **Not strictly causal:** the zero-phase low-pass filter looks a few milliseconds ahead, and the
  z-score scale uses the first 10% of each recording.
- **XGBoost versions and devices:** CPU and GPU implementations give slightly different numbers.
- **Stress-test recordings** are shortened, and their absolute errors are not comparable to the main
  results.

---

## 7. Conclusion

- **Subject-independent real-time gaze estimation** from EOG reaches about 1–2° fixation MAE on the
  EyeCon datasets with small cue ranges, and about 4° on the wider-range ones. Context and
  rolling-range features make most of the difference.
- **On the published known-start task:**
  - modelling head rotation beats the published VOR-based filter on short segments;
  - fusion with a cross-subject absolute model removes long-segment error build-up.
- **The main open problem** is lasting off-centre gaze, which a rolling drift baseline cannot tell
  apart from drift.

---

## Reproducibility

All code, configuration files and result files are in the repository; the Experiment Index in the
README maps each result to its config or script. Pinned package versions are in `requirements.txt`,
and the test suite runs on GitHub Actions.

**TODO:** archive a tagged release (e.g. on Zenodo) and cite its DOI here.

---

## References

**TODO:** complete in the journal's style; check every title, volume and article number.

1. N. Barbara, T. A. Camilleri, K. P. Camilleri, "A comparison of EOG baseline drift mitigation
   techniques," *Biomedical Signal Processing and Control*, vol. 57, 101738, 2020. (Dataset 1)
2. N. Barbara, T. A. Camilleri, K. P. Camilleri, "Real-time continuous EOG-based gaze angle
   estimation with baseline drift compensation under stationary head conditions," *Biomedical
   Signal Processing and Control*, vol. 86, 2023. (Dataset 2)
3. N. Barbara, T. A. Camilleri, K. P. Camilleri, "Real-time continuous EOG-based gaze angle
   estimation with baseline drift compensation under non-stationary head conditions," *Biomedical
   Signal Processing and Control*, vol. 90, 105868, 2024. (Dataset 3)
4. N. Barbara, T. A. Camilleri, K. P. Camilleri, "A systematic quantitative analysis on bipolar
   channel selection for EOG-based gaze displacement estimation," *Biomedical Signal Processing and
   Control*, vol. 112, 108585, 2026. (Dataset 4)
5. N. Barbara et al., signal-differencing gaze estimation method, *Biomedical Signal Processing and
   Control*, vol. 47, 2019. TODO: full title and pages.
6. Mifsud et al., hidden-Markov-model eye-movement labelling from EOG, *Biomedical Signal Processing
   and Control*, 110326, 2026. TODO: authors and title.
7. Mahmood et al., EOG gaze-direction classification on EyeCon Dataset 3, *Engineering, Technology &
   Applied Science Research*, vol. 14, no. 6, 2024. TODO: authors and title.
8. REM-sleep saccade detection tuned on EyeCon data, *Sensors*, vol. 26, no. 4, 1389, 2026. TODO:
   authors and title.
9. T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," *Proc. KDD*, 2016.
