"""
scripts/nested_range_settings.py
================================
Nested cross-subject selection of the rolling-range feature settings, which
scripts/nested_cv.py kept at their configured values.

Candidates: range windows (30/60/120, 60/120/240, 120/240/480 or 60/240 s) x quantile
pairs (10-90 % + 5-95 %, 5-95 % alone, or 2-98 % + 10-90 %) x training windows
(weighted / fixation), always with the engineered and context features and the
config's drift baseline. Outer 5-fold / inner 4-fold GroupKFold as in nested_cv.py,
and every candidate is also scored on the outer folds directly.

    python scripts/nested_range_settings.py --config configs/dataset2_range_causal_weighted.yaml

Results: reports/experiments/nested_cv/<dataset>/<track>_range_settings/. Only the
range block is rebuilt per setting (no engineered features), cached in
data/processed_nested/<dataset>/.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nested_cv import TRACKS, run_nested, save_results  # noqa: E402
from src.config import CFG, load_config_from_yaml  # noqa: E402
from src.features.regression_matrix import build_feature_matrix, load_unified_trials, stack_features  # noqa: E402
from src.models.classical_ml import fit_predict_regression_fold  # noqa: E402

RANGE_WINDOWS = {"30-60-120s": [30.0, 60.0, 120.0], "60-120-240s": [60.0, 120.0, 240.0],
                 "120-240-480s": [120.0, 240.0, 480.0], "60-240s": [60.0, 240.0]}
QUANTILES = {"q10-90+q05-95": [[0.10, 0.90], [0.05, 0.95]], "q05-95": [[0.05, 0.95]],
             "q02-98+q10-90": [[0.02, 0.98], [0.10, 0.90]]}
TRAIN_WINDOWS = ("weighted", "fixation")
CONFIGURED = ("60-120-240s", "q10-90+q05-95")


def describe(candidate) -> str:
    windows, quantiles, train_windows = candidate
    return f"range={windows} quantiles={quantiles} train={train_windows}"


def _load(path: str) -> dict:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def range_block(trials, windows, quantiles) -> dict:
    """Only the range features for these settings (context baselines, lags and engineered features off)."""
    pp = CFG.preprocessing
    saved = (pp.context_baselines, pp.context_lags_sec, pp.context_range_windows_sec, pp.context_range_quantiles)
    pp.context_baselines, pp.context_lags_sec = [], []
    pp.context_range_windows_sec, pp.context_range_quantiles = windows, quantiles
    try:
        m = build_feature_matrix(trials, engineered=False)
    finally:
        pp.context_baselines, pp.context_lags_sec, pp.context_range_windows_sec, pp.context_range_quantiles = saved
    return {"range": m["range"], "subject": m["subject"], "window_start": m["window_start"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--xgb-device", default="cuda")
    args = parser.parse_args()

    load_config_from_yaml(args.config)
    CFG.cv.xgb_device = args.xgb_device
    pp = CFG.preprocessing
    track = next((t for t, (method, causal) in TRACKS.items()
                  if (method, causal) == (pp.drift_removal_method, pp.baseline_causal)), None)
    if track is None:
        raise ValueError("the config's drift baseline is neither the realtime nor the offline track")
    dataset = "_".join(CFG.data.datasets_to_load)
    root = CFG.paths.project_root
    cache_dir = os.path.join(root, "data", "processed_nested", dataset)
    os.makedirs(cache_dir, exist_ok=True)
    prefix = f"{track}_{pp.baseline_window_sec:g}s"

    trials = load_unified_trials()
    base_path = os.path.join(cache_dir, f"{prefix}.npz")
    if os.path.isfile(base_path):
        base = _load(base_path)
        print(f"loaded features {base_path}")
    else:
        base = build_feature_matrix(trials)
        np.savez(base_path, **base)
        print(f"built features {base_path}")
    y, subjects, fixation = base["y"].astype(np.float64), base["subject"], base["fixation"]
    X_base = stack_features(base, ["engineered", "context"])

    X = {}
    for (w_name, windows), (q_name, quantiles) in itertools.product(RANGE_WINDOWS.items(), QUANTILES.items()):
        path = os.path.join(cache_dir, f"{prefix}_range_{w_name}_{q_name}.npz")
        if os.path.isfile(path):
            block = _load(path)
        else:
            t0 = time.time()
            block = range_block(trials, windows, quantiles)
            np.savez(path, **block)
            print(f"built range features {w_name} {q_name} ({time.time() - t0:.0f}s)", flush=True)
        for key in ("subject", "window_start"):
            if not np.array_equal(block[key], base[key]):
                raise ValueError(f"window order differs for {w_name} {q_name} ({key})")
        X[(w_name, q_name)] = np.hstack([X_base, block["range"]]).astype(np.float32)
    if not np.allclose(X[CONFIGURED][:, X_base.shape[1]:], base["range"], equal_nan=True):
        raise ValueError("rebuilt range features for the configured settings differ from the cached ones")

    candidates = list(itertools.product(RANGE_WINDOWS, QUANTILES, TRAIN_WINDOWS))

    def fit_predict(candidate, train_idx, test_idx, out):
        w_name, q_name, train_windows = candidate
        pred, _, kept = fit_predict_regression_fold(X[(w_name, q_name)], y, train_idx, test_idx,
                                                    "xgb", False, fixation, train_windows)
        out[kept] = pred

    selected_pred, fixed_pred, folds_report, runtime = run_nested(candidates, fit_predict, y, subjects, fixation,
                                                                  describe)
    header = {
        "description": "Nested cross-subject selection of rolling-range windows, quantile pairs and training "
                       "windows for XGBoost with engineered + context + range features; outer 5-fold GroupKFold, "
                       "inner 4-fold GroupKFold on training subjects.",
        "dataset": dataset, "track": track, "config": args.config, "xgb_device": CFG.cv.xgb_device,
        "baseline_method": pp.drift_removal_method, "baseline_causal": pp.baseline_causal,
        "baseline_window_sec": pp.baseline_window_sec,
        "configured_setting": f"range={CONFIGURED[0]} quantiles={CONFIGURED[1]}",
    }
    save_results(os.path.join(root, "reports", "experiments", "nested_cv", dataset, f"{track}_range_settings"),
                 header, candidates, describe, y, subjects, fixation, selected_pred, fixed_pred, folds_report, runtime)


if __name__ == "__main__":
    main()
