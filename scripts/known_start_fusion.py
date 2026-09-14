"""
scripts/known_start_fusion.py
=============================
Known-start estimation fused with a cross-subject absolute gaze model.

Summing detected saccades from a known start is accurate over a second or two, but
its error builds up over a 32 s segment (a missed blink leaves a lasting vertical
offset). The cross-subject XGBoost model has no build-up but a larger error at any
moment. The "fused" estimator in src/evaluation/known_start.py combines the two with
a per-axis complementary filter whose time constant is chosen on the fit data.

The absolute model is the real-time XGBoost of <dataset>_range_causal_weighted.yaml
(past-only 60 s robust line, context + range features, weighted training), trained
5-fold cross-subject on the GPU, so a subject's estimates never use that subject's
data. Its window predictions become a per-sample series, each held from the end of
its window (causal).

    python scripts/known_start_fusion.py --dataset dataset2

Results: reports/experiments/known_start_fusion/<dataset>/ (known_start_protocol.json,
known_start_table.csv, absolute_model.json). Features are cached in
data/processed_nested/<dataset>/realtime_60s.npz (shared with scripts/nested_cv.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nested_cv import evaluate  # noqa: E402
from src.config import CFG, load_config_from_yaml, window_samples  # noqa: E402
from src.evaluation.known_start import run_known_start  # noqa: E402
from src.features.regression_matrix import build_feature_matrix, load_unified_trials, stack_features  # noqa: E402
from src.models.classical_ml import fit_predict_regression_fold  # noqa: E402
from src.training.cv_splits import get_folds  # noqa: E402


def feature_matrix(dataset: str, trials) -> dict:
    pp = CFG.preprocessing
    assert (pp.drift_removal_method, pp.baseline_window_sec, pp.baseline_causal) == ("robust_line", 60.0, True)
    path = os.path.join(CFG.paths.project_root, "data", "processed_nested", dataset, "realtime_60s.npz")
    if os.path.isfile(path):
        with np.load(path) as f:
            print(f"loaded features {path}")
            return {k: f[k] for k in f.files}
    t0 = time.time()
    matrix = build_feature_matrix(trials)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **matrix)
    print(f"built features {path} ({time.time() - t0:.0f}s)")
    return matrix


def absolute_series(trials, matrix: dict, pred: np.ndarray) -> dict:
    """(subject_id, trial_id) -> (n_samples, 2): the latest window prediction whose window has ended."""
    if len({t.subject_id for t in trials}) != len(trials):
        raise ValueError("expected one recording per subject")
    out = {}
    for t in trials:
        rows = np.flatnonzero(matrix["subject"] == t.subject_id)
        ends = matrix["window_start"][rows] + window_samples(t.fs)
        order = np.argsort(ends, kind="stable")
        ends, p = ends[order], pred[rows][order]
        j = np.searchsorted(ends, np.arange(len(t.channels["H"])), side="right") - 1
        series = np.full((len(j), 2), np.nan)
        series[j >= 0] = p[j[j >= 0]]
        out[(t.subject_id, t.trial_id)] = series
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, choices=["dataset1", "dataset2", "dataset3", "dataset4"])
    parser.add_argument("--xgb-device", default="cuda")
    args = parser.parse_args()

    root = CFG.paths.project_root
    load_config_from_yaml(os.path.join(root, "configs", f"{args.dataset}_range_causal_weighted.yaml"))
    CFG.cv.xgb_device = args.xgb_device
    trials = load_unified_trials()
    matrix = feature_matrix(args.dataset, trials)
    X = stack_features(matrix, ["engineered", "context", "range"])
    y, subjects, fixation = matrix["y"].astype(np.float64), matrix["subject"], matrix["fixation"]

    t0 = time.time()
    pred = np.full_like(y, np.nan)
    for train_idx, test_idx in get_folds(subjects):
        p, _, kept = fit_predict_regression_fold(X, y, train_idx, test_idx, "xgb", False, fixation,
                                                 CFG.cv.regression_train_windows)
        pred[kept] = p
    absolute = evaluate(y, pred, subjects, fixation)
    print(f"absolute model: fixation MAE {absolute['fixation_mae_h_deg']:.2f} / {absolute['fixation_mae_v_deg']:.2f}, "
          f"RMSE {absolute['rmse_h_deg']:.2f} / {absolute['rmse_v_deg']:.2f} ({time.time() - t0:.0f}s)")

    out_dir = os.path.join(root, "reports", "experiments", "known_start_fusion", args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "absolute_model.json"), "w") as f:
        json.dump({"description": "5-fold cross-subject real-time XGBoost (context + range features, weighted "
                                  "training) whose window predictions feed the fused known-start estimator",
                   "config": f"configs/{args.dataset}_range_causal_weighted.yaml", "xgb_device": CFG.cv.xgb_device,
                   **absolute}, f, indent=2)
    run_known_start(trials, out_dir, absolute=absolute_series(trials, matrix, pred))


if __name__ == "__main__":
    main()
