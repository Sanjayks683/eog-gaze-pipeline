"""
tests/test_robustness_scripts.py
================================
Pieces of the nested cross-validation, range stress test and statistics scripts
that can be checked without the datasets.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts.nested_cv import evaluate, run_nested
from scripts.range_stress_test import anchor_predictions, keep_mask, splice, trial_cues
from scripts.subject_statistics import compare
from src.data.schema import Trial

FS = 256.0


def _cued_trial(n_trials: int, seed: int = 0, lead_in: int = 100) -> Trial:
    """Dataset-2-like recording: a lead-in, then cue P1 for 1 s, P2 for 1 s, blink interval 2 s."""
    rng = np.random.RandomState(seed)
    sec = int(FS)
    cs, target = [0] * lead_in, [np.zeros(2)] * lead_in
    for _ in range(n_trials):
        p1, p2 = rng.uniform(-20, 20, 2), rng.uniform(-20, 20, 2)
        cs += [1] * sec + [2] * sec + [3] * 2 * sec
        target += [p1] * sec + [p2] * 3 * sec
    target = np.array(target)
    channels = {"H": target[:, 0].copy(), "V": target[:, 1].copy(), "ControlSignal": np.array(cs)}
    return Trial(subject_id="S1", trial_id="S1_trial", dataset_source="dataset2", montage_type="monopolar",
                 fs=FS, channels=channels, target_angle=target)


def test_stress_test_splices_the_lead_in_and_the_selected_trials_only():
    trial = _cued_trial(n_trials=60)
    cues = trial_cues(trial)
    assert len(cues) == 60 and cues[0][0] == 100

    counts = {}
    right = keep_mask(cues, "right", np.random.RandomState(0), counts)
    counts["right"] = int(right.sum())
    random = keep_mask(cues, "random_right", np.random.RandomState(0), counts)
    assert 0 < counts["right"] < 60 and random.sum() == counts["right"]
    block = keep_mask(cues, "block_right", np.random.RandomState(0), counts)
    assert block.sum() == counts["right"] and np.all(np.diff(np.flatnonzero(block)) == 1)

    spliced = splice(trial, cues, right)
    n = 100 + 4 * int(FS) * counts["right"]
    assert len(spliced.channels["H"]) == len(spliced.target_angle) == n
    assert (spliced.target_angle[100:, 0] > 0).all()
    assert len(trial.channels["H"]) == 100 + 60 * 4 * int(FS)  # the original is untouched


def test_nested_evaluate_scores_only_windows_with_a_prediction():
    y = np.zeros((4, 2))
    pred = np.array([[1.0, 1.0], [np.nan, np.nan], [3.0, 3.0], [1.0, 1.0]])
    out = evaluate(y, pred, np.array(["A", "A", "B", "B"]), np.array([True, True, True, False]))

    assert out["fixation_mae_h_deg"] == pytest.approx(2.0)
    assert out["rmse_h_deg"] == pytest.approx(np.sqrt(11 / 3))
    assert out["score_fixation"] == pytest.approx(2.0)


def test_statistics_compare_pairs_subjects():
    reference = {f"S{i}": [2.0 + i, 3.0 + i] for i in range(10)}
    model = {s: [h - (0.5 + 0.1 * i), v + 0.2] for i, (s, (h, v)) in enumerate(reference.items())}
    out = compare(model, reference, "model", "reference")

    assert out["improvement_deg"] == pytest.approx([0.95, -0.2])
    lo, hi = out["improvement_ci95_deg"]["h"]
    assert 0.5 <= lo < 0.95 < hi <= 1.4
    assert out["subjects_improved"] == {"h": 10, "v": 0}
    assert out["wilcoxon_p"]["h"] < 0.01


def test_anchored_estimates_remove_a_constant_offset_after_each_anchor():
    n = 40
    window_start = np.arange(n) * 10
    y = np.column_stack([np.arange(n, dtype=float), np.zeros(n)])
    fixation = np.ones(n, dtype=bool)
    fixation[0] = False
    out = anchor_predictions(y, y + 5.0, np.array(["A"] * n), fixation, window_start, interval=100)

    anchors = [1, 10, 20, 30]  # first fixation window at or after samples 0, 100, 200, 300
    assert np.isnan(out[anchors]).all() and np.isnan(out[0]).all()
    scored = np.isfinite(out).all(axis=1)
    assert scored.sum() == n - 5 and np.allclose(out[scored], y[scored])


def test_run_nested_picks_the_candidate_that_wins_on_inner_folds():
    rng = np.random.RandomState(0)
    subjects = np.repeat([f"S{i}" for i in range(10)], 20)
    y = rng.randn(200, 2)
    fixation = np.ones(200, dtype=bool)

    def fit_predict(error, train_idx, test_idx, out):
        out[test_idx] = y[test_idx] + error

    selected, fixed, folds, _ = run_nested([0.5, 0.1, 1.0], fit_predict, y, subjects, fixation, name=str)
    assert all(fold["selected"] == {"fixation": "0.1", "rmse": "0.1"} for fold in folds)
    assert np.allclose(selected["fixation"], y + 0.1)
    assert np.allclose(fixed[1.0], y + 1.0)
