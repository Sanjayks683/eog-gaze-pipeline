"""
tests/test_fixation_eval.py
===========================
Paper-style evaluation: fixation-window flags, fixation MAE, fixation-only
training, causal context features and the published comparison rows.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.config import CFG, _CONFIG_SECTIONS, window_samples, stride_samples
from src.data.datasets import _fixation_flags, make_regression_targets
from src.data.schema import Trial
from src.evaluation.metrics import compute_fixation_metrics
from src.features.context import context_feature_names, make_context_features
from src.models.classical_ml import run_regression_cv
from src.preprocessing.filtering import remove_baseline_drift

FS = 256.0


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _cued_trial(n_trials: int = 30, seed: int = 0, subject: str = "S1") -> Trial:
    """Dataset-2-like recording: cue at P1 for 1 s, P2 for 1 s, blink interval 2 s."""
    rng = np.random.RandomState(seed)
    sec = int(FS)
    cs, target = [], []
    for _ in range(n_trials):
        p1, p2 = rng.uniform(-20, 20, 2), rng.uniform(-20, 20, 2)
        cs += [1] * sec + [2] * sec + [3] * 2 * sec
        target += [p1] * sec + [p2] * 3 * sec
    target = np.array(target)
    n = len(cs)
    drift = np.linspace(0, 30, n)
    channels = {
        "H": target[:, 0] + drift + rng.randn(n) * 0.2,
        "V": target[:, 1] - drift + rng.randn(n) * 0.2,
        "ControlSignal": np.array(cs),
    }
    return Trial(subject_id=subject, trial_id=f"{subject}_trial", dataset_source="dataset2",
                 montage_type="monopolar", fs=FS, channels=channels, target_angle=target)


def test_fixation_flags_need_settled_gaze_inside_saccade_intervals(restore_cfg):
    CFG.segmentation.fixation_settle_ms = 400.0
    trial = _cued_trial(n_trials=2)
    win = window_samples(FS)
    starts = np.arange(0, 8 * int(FS) - win + 1, 8)
    flags = _fixation_flags(trial, starts, win)

    settle, sec = int(round(0.4 * FS)), int(FS)
    cue_intervals = [(0, sec), (sec, 2 * sec), (4 * sec, 5 * sec), (5 * sec, 6 * sec)]
    expected = [any(lo + settle <= s and s + win <= hi for lo, hi in cue_intervals) for s in starts]
    assert flags.tolist() == expected
    assert flags.any() and not flags.all()


def test_fixation_flags_empty_without_targets():
    trial = _cued_trial(n_trials=1)
    trial.target_angle = None
    assert not _fixation_flags(trial, np.arange(0, 500, 38), 77).any()


def test_fixation_mae_averages_subjects_not_windows():
    y_true = np.zeros((6, 2))
    y_pred = np.array([[1, 1], [1, 1], [1, 1], [1, 1], [3, 5], [9, 9]], dtype=float)
    meta = ([{"subject_id": "A", "is_fixation": True}] * 4
            + [{"subject_id": "B", "is_fixation": True}, {"subject_id": "B", "is_fixation": False}])

    m = compute_fixation_metrics(y_true, y_pred, meta)

    assert m["fixation_mae_h_deg"] == pytest.approx(2.0)  # (1 + 3) / 2, not (4*1 + 3) / 5
    assert m["fixation_mae_v_deg"] == pytest.approx(3.0)
    assert m["fixation_mae_h_sd_deg"] == pytest.approx(1.0)
    assert m["n_fixation_windows"] == 5
    assert m["fixation_mae_per_subject"] == {"A": [1.0, 1.0], "B": [3.0, 5.0]}
    assert compute_fixation_metrics(y_true, y_pred, [{"subject_id": "A"}] * 6) == {}


def test_regression_cv_can_train_on_fixation_windows_only(restore_cfg):
    y = np.array([[10.0, 0.0]] * 50 + [[-30.0, 0.0]] * 50)
    X = np.zeros((100, 3))
    meta = [{"subject_id": f"S{i % 2}", "is_fixation": i < 50} for i in range(100)]
    folds = [(np.arange(0, 100, 2), np.arange(1, 100, 2))]

    CFG.cv.regression_train_windows = "all"
    all_windows = run_regression_cv(X, y, folds, model_name="mean", metadata=meta)
    CFG.cv.regression_train_windows = "fixation"
    fixation_only = run_regression_cv(X, y, folds, model_name="mean", metadata=meta)

    CFG.cv.regression_train_windows, CFG.cv.regression_nonfixation_weight = "weighted", 0.25
    weighted = run_regression_cv(X, y, folds, model_name="mean", metadata=meta)

    assert all_windows["fixation_mae_h_deg"] == pytest.approx(20.0)
    assert fixation_only["fixation_mae_h_deg"] == pytest.approx(0.0)
    assert fixation_only["train_windows"] == "fixation"
    # weighted mean = (1 * 10 + 0.25 * -30) / 1.25 = 2
    assert weighted["fixation_mae_h_deg"] == pytest.approx(8.0)
    assert weighted["nonfixation_weight"] == 0.25
    with pytest.raises(ValueError, match="metadata"):
        run_regression_cv(X, y, folds, model_name="mean")


def _preprocess_like_pipeline(trial: Trial) -> dict:
    raw = {k: v.copy() for k, v in trial.channels.items()}
    for ch in ("H", "V"):
        trial.channels[ch] = remove_baseline_drift(raw[ch], FS)
    trial.metadata["zscore_std"] = {"H": 1.0, "V": 1.0}
    return raw


def test_causal_context_features_never_use_samples_after_the_window(restore_cfg):
    pp = CFG.preprocessing
    pp.drift_removal_method, pp.baseline_window_sec, pp.baseline_causal = "robust_line", 20.0, True
    pp.context_baselines = [["robust_mean", 10.0], ["robust_mean", 40.0]]
    pp.context_lags_sec = [1.0, 4.0]
    pp.context_range_windows_sec = [20.0]
    pp.context_range_quantiles = [[0.1, 0.9]]

    def features(trial):
        raw = _preprocess_like_pipeline(trial)
        _, _, meta = make_regression_targets([trial])
        return make_context_features([trial], [raw], meta), meta

    a, meta = features(_cued_trial())
    changed = _cued_trial()
    k = len(changed.channels["H"]) // 2
    for ch in ("H", "V"):
        changed.channels[ch][k:] += 50.0
    b, _ = features(changed)

    assert a.shape == (len(meta), len(context_feature_names(["H", "V"]))) == (len(meta), 14)
    assert np.isfinite(a).all()
    ends = np.array([m["window_start"] for m in meta]) + window_samples(FS)
    before = ends <= k
    assert before.sum() > 100
    assert np.allclose(a[before], b[before])
    assert not np.allclose(a[~before], b[~before])


def test_range_features_find_the_centre_and_spread_of_uniform_fixations(restore_cfg):
    from src.features.context import _range_features

    rng = np.random.RandomState(0)
    sig = 3.0 + 2.0 * np.repeat(rng.uniform(-1, 1, 200), int(FS))  # centre 3, half-range 2
    starts = np.arange(100 * int(FS), 199 * int(FS), 77)
    minus_mid, spread, ratio = _range_features(sig, starts, 77, FS, 60.0, 0.05, 0.95, causal=True)

    level = sig[starts]
    assert np.median(np.abs(minus_mid - (level - 3.0))) < 0.15
    assert np.median(np.abs(spread - 3.6)) < 0.2  # 5-95% of U(-2, 2) spans 3.6
    assert np.allclose(ratio, minus_mid / (spread + 1e-6))


def test_paper_rows_report_fixation_mae_not_invented_rmse(tmp_path, restore_cfg):
    from src.evaluation.compare_to_paper import build_master_table

    CFG.data.datasets_to_load = ["dataset2"]
    rows = {r["method"]: r for r in build_master_table(results_dir=str(tmp_path))}

    dkf_long = rows["Published: Barbara_2023_DKF_long"]
    assert (dkf_long["fixation_mae_h_deg"], dkf_long["fixation_mae_v_deg"]) == (5.23, 6.59)
    assert rows["Published: Barbara_2023_DKF_short"]["fixation_mae_h_deg"] == 1.64
    assert all(r["rmse_h_deg"] is None for m, r in rows.items() if m.startswith("Published"))
