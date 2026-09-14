"""
scripts/subject_statistics.py
======================
Per-subject significance tests and confidence intervals for the main regression
results, on fixation MAE: two-sided Wilcoxon signed-rank tests paired by subject, and
95% bootstrap confidence intervals of the mean over subjects (10,000 resamples).

- Dataset 2: the nested-selected models from scripts/nested_cv.py (both tracks,
  selected by fixation MAE) against the baseline-only candidate of the same track
  (60 s baseline, engineered features, all windows).
- Datasets 3 and 4: XGBoost with context + range features, weighted and fixation-only
  training, against the baseline-only model (engineered features, all windows), refit
  on the GPU from the processed data of the dataset{3,4}_range_causal_weighted configs.

    python scripts/subject_statistics.py

Results: reports/experiments/statistics/statistics.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nested_cv import evaluate  # noqa: E402
from src.config import CFG, load_config_from_yaml  # noqa: E402
from src.data.datasets import load_processed  # noqa: E402
from src.features.engineered import extract_features_batch  # noqa: E402
from src.models.classical_ml import fit_predict_regression_fold  # noqa: E402
from src.training.cv_splits import load_folds  # noqa: E402

REFERENCE = "window=60s features=engineered train=all"
N_BOOT = 10_000


def compare(model: dict, reference: dict, model_name: str, reference_name: str, seed: int = 0) -> dict:
    """Paired per-subject comparison of fixation MAE (lower is better)."""
    subjects = sorted(model)
    m = np.array([model[s] for s in subjects])
    r = np.array([reference[s] for s in subjects])
    rng = np.random.RandomState(seed)
    resample = rng.randint(0, len(m), size=(N_BOOT, len(m)))  # same subjects for model and reference
    boot, boot_diff = m[resample].mean(axis=1), (r - m)[resample].mean(axis=1)

    def ci(samples):
        return {"h": np.percentile(samples[:, 0], [2.5, 97.5]).round(3).tolist(),
                "v": np.percentile(samples[:, 1], [2.5, 97.5]).round(3).tolist()}

    out = {"model": model_name, "reference": reference_name, "n_subjects": len(subjects),
           "model_mean_deg": m.mean(axis=0).round(3).tolist(), "reference_mean_deg": r.mean(axis=0).round(3).tolist(),
           "model_ci95_deg": ci(boot),
           "improvement_deg": (r - m).mean(axis=0).round(3).tolist(),
           "improvement_ci95_deg": ci(boot_diff),
           "subjects_improved": {"h": int((m[:, 0] < r[:, 0]).sum()), "v": int((m[:, 1] < r[:, 1]).sum())},
           "wilcoxon_p": {}}
    for axis, (a, b) in {"h": (m[:, 0], r[:, 0]), "v": (m[:, 1], r[:, 1]),
                         "mean_hv": (m.mean(axis=1), r.mean(axis=1))}.items():
        out["wilcoxon_p"][axis] = float(wilcoxon(a, b, alternative="two-sided").pvalue)
    return out


def dataset2(root: str) -> dict:
    out = {}
    for track in ("realtime", "offline"):
        path = os.path.join(root, "reports", "experiments", "nested_cv", "dataset2", track, "nested_cv_results.json")
        if not os.path.isfile(path):
            print(f"missing {path}; skipping Dataset 2 {track}")
            continue
        r = json.load(open(path))
        out[track] = compare(r["nested_selected"]["fixation"]["fixation_mae_per_subject"],
                             r["fixed_candidates"][REFERENCE]["fixation_mae_per_subject"],
                             f"nested-selected ({track})", f"{REFERENCE} ({track})")
    return out


def refit_dataset(root: str, dataset: str) -> dict:
    load_config_from_yaml(os.path.join(root, "configs", f"{dataset}_range_causal_weighted.yaml"))
    CFG.cv.xgb_device = "cuda"
    X, y, meta = load_processed("regression")
    ctx, _, _ = load_processed("context")
    folds, _ = load_folds()
    cache = os.path.join(root, "data", "processed_nested", dataset, "statistics_engineered.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.isfile(cache):
        eng = np.load(cache)
    else:
        eng = extract_features_batch(X, fs=meta[0]["fs"], channel_names=meta[0]["channel_names"]).astype(np.float32)
        np.save(cache, eng)
    n_ch = len(meta[0]["channel_names"])
    n_basic = (len(CFG.preprocessing.context_baselines) + len(CFG.preprocessing.context_lags_sec)) * n_ch
    features = {"engineered": eng, "range": np.hstack([eng, ctx]).astype(np.float32)}
    subjects = np.array([m["subject_id"] for m in meta])
    fixation = np.array([bool(m["is_fixation"]) for m in meta])
    y = y.astype(np.float64)

    def per_subject(feature_set, train_windows):
        pred = np.full_like(y, np.nan)
        for tr, te in folds:
            p, _, kept = fit_predict_regression_fold(features[feature_set], y, tr, te, "xgb", False,
                                                     fixation, train_windows)
            pred[kept] = p
        return evaluate(y, pred, subjects, fixation)["fixation_mae_per_subject"]

    assert ctx.shape[1] > n_basic, "expected range features in the context matrix"
    reference = per_subject("engineered", "all")
    return {
        "range_weighted": compare(per_subject("range", "weighted"), reference,
                                  "context + range, weighted", "engineered, all windows"),
        "range_fixation": compare(per_subject("range", "fixation"), reference,
                                  "context + range, fixation windows only", "engineered, all windows"),
    }


def main() -> None:
    root = CFG.paths.project_root
    results = {"dataset2": dataset2(root)}
    for dataset in ("dataset3", "dataset4"):
        results[dataset] = refit_dataset(root, dataset)
    out_dir = os.path.join(root, "reports", "experiments", "statistics")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "statistics.json"), "w") as f:
        json.dump(results, f, indent=2)
    for dataset, comparisons in results.items():
        for key, c in comparisons.items():
            print(f"{dataset} {key}: {c['model']} {c['model_mean_deg']} (95% CI H {c['model_ci95_deg']['h']}, "
                  f"V {c['model_ci95_deg']['v']}) vs {c['reference']} {c['reference_mean_deg']} | "
                  f"Wilcoxon p H {c['wilcoxon_p']['h']:.4f}, V {c['wilcoxon_p']['v']:.4f}, "
                  f"mean {c['wilcoxon_p']['mean_hv']:.4f} | improved {c['subjects_improved']}/{c['n_subjects']}")
    print(f"Saved {os.path.join(out_dir, 'statistics.json')}")


if __name__ == "__main__":
    main()
