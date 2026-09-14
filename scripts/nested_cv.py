"""
scripts/nested_cv.py
=====================
Nested cross-subject evaluation of the classical-regression settings (XGBoost).

The earlier experiments picked the drift-baseline window, feature set and training
windows by comparing results on the same 5 test folds they report. Here the outer
loop is the pipeline's 5-fold GroupKFold, and inside each outer fold every candidate
is scored by a 4-fold GroupKFold over the outer-training subjects only. The best
candidate by the inner criterion is refit on all outer-training subjects and scored
once on the held-out subjects. Every candidate is also scored directly on the outer
test folds ("fixed" results, the way the earlier experiments were reported), so the
gap between the two shows how optimistic the earlier selection was.

Candidates: baseline window (30 / 60 / 120 s) x features (engineered / + context /
+ context + range) x training windows (all / weighted / fixation). Criteria: mean of
horizontal and vertical fixation MAE, and mean of horizontal and vertical RMSE.

    python scripts/nested_cv.py --config configs/dataset2_range_causal_weighted.yaml --track realtime
    python scripts/nested_cv.py --config configs/dataset2_range_centred_weighted.yaml --track offline

The config supplies the dataset, channels and context/range settings; the track sets
the baseline method (realtime: past-only robust line, offline: centred robust mean).
Results: reports/experiments/nested_cv/<dataset>/<track>/; feature caches and
predictions: data/processed_nested/<dataset>/ (git-ignored). run_nested() and
save_results() are shared with scripts/nested_range_settings.py.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG, load_config_from_yaml  # noqa: E402
from src.evaluation.metrics import compute_fixation_metrics  # noqa: E402
from src.features.regression_matrix import build_feature_matrix, load_unified_trials, stack_features  # noqa: E402
from src.models.classical_ml import fit_predict_regression_fold  # noqa: E402
from src.training.cv_splits import get_folds  # noqa: E402

TRACKS = {"realtime": ("robust_line", True), "offline": ("robust_mean", False)}
WINDOWS_SEC = (30.0, 60.0, 120.0)
FEATURE_SETS = {"engineered": ["engineered"], "context": ["engineered", "context"],
                "range": ["engineered", "context", "range"]}
TRAIN_WINDOWS = ("all", "weighted", "fixation")
INNER_FOLDS = 4
CRITERIA = ("fixation", "rmse")
INNER_SCORE_KEYS = ("score_fixation", "score_rmse", "fixation_mae_h_deg", "fixation_mae_v_deg",
                    "rmse_h_deg", "rmse_v_deg")


def describe(candidate) -> str:
    window, features, train_windows = candidate
    return f"window={window:g}s features={features} train={train_windows}"


def evaluate(y_true, y_pred, subjects, fixation) -> dict:
    """Fixation MAE and RMSE over the windows that have a prediction (rows with NaN features get none)."""
    scored = np.isfinite(y_pred).all(axis=1)
    y_true, y_pred, subjects, fixation = y_true[scored], y_pred[scored], subjects[scored], fixation[scored]
    meta = [{"subject_id": s, "is_fixation": bool(f)} for s, f in zip(subjects, fixation)]
    fix = compute_fixation_metrics(y_true, y_pred, meta)
    rmse = np.sqrt(np.mean((y_pred - y_true) ** 2, axis=0))
    out = {"rmse_h_deg": float(rmse[0]), "rmse_v_deg": float(rmse[1]), **fix}
    out["score_fixation"] = (out["fixation_mae_h_deg"] + out["fixation_mae_v_deg"]) / 2
    out["score_rmse"] = (out["rmse_h_deg"] + out["rmse_v_deg"]) / 2
    return out


def load_matrices(track: str, trials, cache_dir: str) -> dict:
    method, causal = TRACKS[track]
    matrices = {}
    for window in WINDOWS_SEC:
        path = os.path.join(cache_dir, f"{track}_{window:g}s.npz")
        if os.path.isfile(path):
            with np.load(path) as f:
                matrices[window] = {k: f[k] for k in f.files}
            print(f"loaded features {path}")
            continue
        CFG.preprocessing.drift_removal_method = method
        CFG.preprocessing.baseline_window_sec = window
        CFG.preprocessing.baseline_causal = causal
        t0 = time.time()
        matrices[window] = build_feature_matrix(trials)
        np.savez(path, **matrices[window])
        print(f"built features {path} ({time.time() - t0:.0f}s)")
    ref = matrices[WINDOWS_SEC[0]]
    for window, m in matrices.items():
        for key in ("subject", "window_start", "fixation"):
            if not np.array_equal(m[key], ref[key]):
                raise ValueError(f"window order differs for {window:g}s ({key})")
    return matrices


def run_nested(candidates, fit_predict, y, subjects, fixation, name=describe):
    """
    Outer folds from get_folds, inner GroupKFold(INNER_FOLDS) over each outer fold's
    training subjects. fit_predict(candidate, train_idx, test_idx, out) writes the
    predictions for test_idx into out. Returns (selected predictions per criterion,
    every candidate's outer-fold predictions, per-fold report, runtime in seconds).
    """
    fixed_pred = {c: np.full_like(y, np.nan) for c in candidates}
    selected_pred = {crit: np.full_like(y, np.nan) for crit in CRITERIA}
    folds_report, t_start = [], time.time()
    for k, (tr, te) in enumerate(get_folds(subjects)):
        inner = list(GroupKFold(INNER_FOLDS).split(tr, groups=subjects[tr]))
        inner_scores = {}
        for i, candidate in enumerate(candidates):
            pred = np.full_like(y, np.nan)
            for itr, ite in inner:
                fit_predict(candidate, tr[itr], tr[ite], pred)
            inner_scores[candidate] = evaluate(y[tr], pred[tr], subjects[tr], fixation[tr])
            fit_predict(candidate, tr, te, fixed_pred[candidate])
            print(f"fold {k} [{i + 1}/{len(candidates)}] {name(candidate)}: inner fixation "
                  f"{inner_scores[candidate]['score_fixation']:.3f} rmse {inner_scores[candidate]['score_rmse']:.3f} "
                  f"({time.time() - t_start:.0f}s)", flush=True)
        best = {crit: min(candidates, key=lambda c: inner_scores[c][f"score_{crit}"]) for crit in CRITERIA}
        for crit, candidate in best.items():
            selected_pred[crit][te] = fixed_pred[candidate][te]
        folds_report.append({
            "fold": k,
            "test_subjects": sorted(set(subjects[te].tolist())),
            "selected": {crit: name(c) for crit, c in best.items()},
            "inner_scores": {name(c): {key: s[key] for key in INNER_SCORE_KEYS} for c, s in inner_scores.items()},
        })
        print(f"fold {k} selected: {folds_report[-1]['selected']}", flush=True)
    return selected_pred, fixed_pred, folds_report, round(time.time() - t_start)


def save_results(out_dir: str, header: dict, candidates, name, y, subjects, fixation,
                 selected_pred: dict, fixed_pred: dict, folds_report: list, runtime_sec: int) -> dict:
    """Write nested_cv_results.json and fixed_candidates.csv to out_dir and print the summary."""
    fixed = {name(c): evaluate(y, p, subjects, fixation) for c, p in fixed_pred.items()}
    selected = {crit: evaluate(y, p, subjects, fixation) for crit, p in selected_pred.items()}
    best_fixed = {crit: min(fixed, key=lambda n: fixed[n][f"score_{crit}"]) for crit in selected}
    results = {
        **header,
        "candidates": [name(c) for c in candidates],
        "nested_selected": selected,
        "best_fixed_on_outer_folds": {crit: {"candidate": n, **fixed[n]} for crit, n in best_fixed.items()},
        "fixed_candidates": fixed,
        "folds": folds_report,
        "runtime_sec": runtime_sec,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "nested_cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "fixed_candidates.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["candidate", "fixation_mae_h_deg", "fixation_mae_v_deg", "rmse_h_deg", "rmse_v_deg"])
        for n, r in sorted(fixed.items(), key=lambda kv: kv[1]["score_fixation"]):
            writer.writerow([n, r["fixation_mae_h_deg"], r["fixation_mae_v_deg"], r["rmse_h_deg"], r["rmse_v_deg"]])
    for crit in selected:
        s, b = selected[crit], results["best_fixed_on_outer_folds"][crit]
        print(f"\n[{crit}] nested: fixation MAE {s['fixation_mae_h_deg']:.2f} / {s['fixation_mae_v_deg']:.2f}, "
              f"RMSE {s['rmse_h_deg']:.2f} / {s['rmse_v_deg']:.2f} | best fixed ({b['candidate']}): "
              f"{b['fixation_mae_h_deg']:.2f} / {b['fixation_mae_v_deg']:.2f}, RMSE {b['rmse_h_deg']:.2f} / {b['rmse_v_deg']:.2f}")
    print(f"Saved {os.path.join(out_dir, 'nested_cv_results.json')}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--track", choices=sorted(TRACKS), required=True)
    parser.add_argument("--xgb-device", default="cuda")
    args = parser.parse_args()

    load_config_from_yaml(args.config)
    CFG.cv.xgb_device = args.xgb_device
    dataset = "_".join(CFG.data.datasets_to_load)
    root = CFG.paths.project_root
    out_dir = os.path.join(root, "reports", "experiments", "nested_cv", dataset, args.track)
    cache_dir = os.path.join(root, "data", "processed_nested", dataset)
    os.makedirs(cache_dir, exist_ok=True)

    matrices = load_matrices(args.track, load_unified_trials(), cache_dir)
    ref = matrices[WINDOWS_SEC[0]]
    y, subjects, fixation = ref["y"].astype(np.float64), ref["subject"], ref["fixation"]
    X = {(w, name): stack_features(matrices[w], blocks) for w in WINDOWS_SEC for name, blocks in FEATURE_SETS.items()}
    candidates = list(itertools.product(WINDOWS_SEC, FEATURE_SETS, TRAIN_WINDOWS))

    def fit_predict(candidate, train_idx, test_idx, out):
        window, features, train_windows = candidate
        pred, _, kept = fit_predict_regression_fold(X[(window, features)], y, train_idx, test_idx,
                                                    "xgb", False, fixation, train_windows)
        out[kept] = pred

    selected_pred, fixed_pred, folds_report, runtime = run_nested(candidates, fit_predict, y, subjects, fixation)
    header = {
        "description": "Nested cross-subject selection of drift-baseline window, feature set and training "
                       "windows for XGBoost; outer 5-fold GroupKFold, inner 4-fold GroupKFold on training subjects.",
        "dataset": dataset, "track": args.track, "config": args.config, "xgb_device": CFG.cv.xgb_device,
        "baseline_method": TRACKS[args.track][0], "baseline_causal": TRACKS[args.track][1],
    }
    save_results(out_dir, header, candidates, describe, y, subjects, fixation, selected_pred, fixed_pred,
                 folds_report, runtime)
    np.savez(os.path.join(cache_dir, f"{args.track}_predictions.npz"), y=y, subject=subjects, fixation=fixation,
             **{f"selected_{crit}": p for crit, p in selected_pred.items()},
             **{f"fixed_{i}": fixed_pred[c] for i, c in enumerate(candidates)},
             candidate_names=np.array([describe(c) for c in candidates]))


if __name__ == "__main__":
    main()
