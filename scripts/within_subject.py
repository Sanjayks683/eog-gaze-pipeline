"""
scripts/within_subject.py
=========================
The published Dataset 2 results fit every parameter on the test subject's own data;
the main tables here never see the test subject. This script scores the same
XGBoost setup (60 s past-only robust line, context + range features, weighted
training) under a within-subject protocol, so the two can be compared directly.

Each subject's windows are split in time into three contiguous parts. For every
ordered pair of parts (fit, test), 6 per subject, three models are scored on the
test part:

  within subject   trained on the fit part of the same subject only
  adapted          trained on all other subjects plus the fit part of the subject
  cross subject    trained on all other subjects only (no data from the subject)

Fixation MAE is averaged over a subject's 6 orderings, then mean ± SD across
subjects, as in the main tables. Test parts are never used for training, but the
context features of their first windows look back into the neighbouring part
(up to 4 min), as they would in a live recording.

    python scripts/within_subject.py --config configs/dataset2_range_causal_weighted.yaml

Results: reports/experiments/within_subject/<dataset>/within_subject.json. Features are
cached in data/processed_nested/<dataset>/ (shared with scripts/nested_cv.py).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG, load_config_from_yaml  # noqa: E402
from src.evaluation.metrics import compute_fixation_metrics  # noqa: E402
from src.features.regression_matrix import build_feature_matrix, load_unified_trials, stack_features  # noqa: E402
from src.models.classical_ml import fit_predict_regression_fold  # noqa: E402

N_PARTS = 3
VARIANTS = ("within_subject", "adapted", "cross_subject")


def load_features(dataset: str) -> dict:
    pp = CFG.preprocessing
    track = "realtime" if pp.baseline_causal else "offline"
    path = os.path.join(CFG.paths.project_root, "data", "processed_nested", dataset,
                        f"{track}_{pp.baseline_window_sec:g}s.npz")
    if pp.drift_removal_method == ("robust_line" if pp.baseline_causal else "robust_mean") and os.path.isfile(path):
        with np.load(path) as f:
            print(f"loaded features {path}")
            return {k: f[k] for k in f.files}
    t0 = time.time()
    matrix = build_feature_matrix(load_unified_trials())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **matrix)
    print(f"built features {path} ({time.time() - t0:.0f}s)")
    return matrix


def parts_by_subject(subjects: np.ndarray, window_start: np.ndarray) -> dict:
    """subject -> list of N_PARTS index arrays, contiguous in time."""
    out = {}
    for s in np.unique(subjects):
        idx = np.flatnonzero(subjects == s)
        out[s] = np.array_split(idx[np.argsort(window_start[idx], kind="stable")], N_PARTS)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--xgb-device", default="cuda")
    args = parser.parse_args()

    load_config_from_yaml(args.config)
    CFG.cv.xgb_device = args.xgb_device
    dataset = "_".join(CFG.data.datasets_to_load)
    matrix = load_features(dataset)
    blocks = ["engineered", "context", "range"] + (["head"] if matrix.get("head", np.zeros((1, 0))).shape[1] else [])
    X = stack_features(matrix, blocks)
    y, subjects, fixation = matrix["y"].astype(np.float64), matrix["subject"], matrix["fixation"]
    parts = parts_by_subject(subjects, matrix["window_start"])
    train_windows = CFG.cv.regression_train_windows

    per_subject = {v: {} for v in VARIANTS}
    t0 = time.time()
    for s, s_parts in parts.items():
        others = np.flatnonzero(subjects != s)
        cross_pred = np.full_like(y, np.nan)
        test_all = np.concatenate(s_parts)
        p, _, kept = fit_predict_regression_fold(X, y, others, test_all, "xgb", False, fixation, train_windows)
        cross_pred[kept] = p
        scores = {v: [] for v in VARIANTS}
        for fit_i, test_i in itertools.permutations(range(N_PARTS), 2):
            fit, test = s_parts[fit_i], s_parts[test_i]
            preds = {"cross_subject": cross_pred}
            for variant, train in (("within_subject", fit), ("adapted", np.concatenate([others, fit]))):
                pred = np.full_like(y, np.nan)
                p, _, kept = fit_predict_regression_fold(X, y, train, test, "xgb", False, fixation, train_windows)
                pred[kept] = p
                preds[variant] = pred
            for variant, pred in preds.items():
                scored = test[np.isfinite(pred[test]).all(axis=1)]
                m = compute_fixation_metrics(y[scored], pred[scored],
                                             [{"subject_id": s, "is_fixation": bool(f)} for f in fixation[scored]])
                scores[variant].append([m["fixation_mae_h_deg"], m["fixation_mae_v_deg"]])
        for variant in VARIANTS:
            per_subject[variant][str(s)] = np.mean(scores[variant], axis=0).round(4).tolist()
        print(f"{s}: " + " | ".join(f"{v} {per_subject[v][str(s)][0]:.2f} / {per_subject[v][str(s)][1]:.2f}"
                                    for v in VARIANTS) + f" ({time.time() - t0:.0f}s)", flush=True)

    summary = {}
    for variant in VARIANTS:
        values = np.array(list(per_subject[variant].values()))
        summary[variant] = {"fixation_mae_h_deg": float(values[:, 0].mean()), "fixation_mae_v_deg": float(values[:, 1].mean()),
                            "fixation_mae_h_sd_deg": float(values[:, 0].std()), "fixation_mae_v_sd_deg": float(values[:, 1].std()),
                            "fixation_mae_per_subject": per_subject[variant]}
    results = {"dataset": dataset, "config": args.config, "xgb_device": CFG.cv.xgb_device, "feature_blocks": blocks,
               "train_windows": train_windows, "n_parts": N_PARTS, "results": summary,
               "runtime_sec": round(time.time() - t0)}
    out_dir = os.path.join(CFG.paths.project_root, "reports", "experiments", "within_subject", dataset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "within_subject.json"), "w") as f:
        json.dump(results, f, indent=2)
    for variant, r in summary.items():
        print(f"{variant:15s} fixation MAE {r['fixation_mae_h_deg']:.2f} ± {r['fixation_mae_h_sd_deg']:.2f} / "
              f"{r['fixation_mae_v_deg']:.2f} ± {r['fixation_mae_v_sd_deg']:.2f}")
    print(f"Saved {os.path.join(out_dir, 'within_subject.json')}")


if __name__ == "__main__":
    main()
