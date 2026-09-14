"""
scripts/make_figures.py
=======================
Result figures from the saved JSON files (nothing is retrained):

  cross_dataset_fixation_mae.png  real-time fixation MAE per dataset: mean-angle baseline,
                                  robust line, + context + range (weighted / fixation windows)
  known_start.png                 known-start errors per dataset against the published methods
  per_subject_fixation_mae.png    each subject's fixation MAE, reference vs context + range model
  range_stress_test.png           range features on Dataset 2 recordings with skewed gaze
  nested_cv_candidates.png        every nested-CV combination scored on the outer test folds

    python scripts/make_figures.py

Writes reports/figures/results/; a figure whose inputs are missing is skipped.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.evaluation.known_start import PAPER_KNOWN_START  # noqa: E402

EXPERIMENTS = os.path.join(ROOT, "reports", "experiments")
OUT_DIR = os.path.join(ROOT, "reports", "figures", "results")
DATASETS = ("dataset1", "dataset2", "dataset3", "dataset4")
AXES = (("h", "horizontal"), ("v", "vertical"))


def _load(*parts):
    path = os.path.join(EXPERIMENTS, *parts)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _experiment(dataset: str, name: str):
    """Dataset 2 results live directly under reports/experiments/, the others under datasetN/."""
    return (name,) if dataset == "dataset2" else (dataset, name)


def _label(dataset: str) -> str:
    return f"Dataset {dataset[-1]}"


def _p(p: float) -> str:
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def _save(fig, name: str) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {os.path.join(OUT_DIR, name)}")


CROSS_DATASET_MODELS = (
    ("Train-fold mean angle", "robustline60_causal", "baseline_reg_mean.json"),
    ("Robust line 60 s, XGBoost", "robustline60_causal", "classical_reg_xgb.json"),
    ("+ context + range, weighted training", "range_causal_weighted", "classical_reg_xgb.json"),
    ("+ context + range, fixation windows only", "range_causal_fixation", "classical_reg_xgb.json"),
)


def cross_dataset_fixation_mae() -> None:
    results = {d: [_load(*_experiment(d, folder), name) for _, folder, name in CROSS_DATASET_MODELS]
               for d in DATASETS}
    present = [d for d in DATASETS if any(r and "fixation_mae_h_deg" in r for r in results[d])]
    if not present:
        raise FileNotFoundError("no classical regression results")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    x, width = np.arange(len(present)), 0.8 / len(CROSS_DATASET_MODELS)
    for ax, (key, name) in zip(axes, AXES):
        for i, (label, _, _) in enumerate(CROSS_DATASET_MODELS):
            rows = [results[d][i] or {} for d in present]
            ax.bar(x + (i - (len(CROSS_DATASET_MODELS) - 1) / 2) * width,
                   [r.get(f"fixation_mae_{key}_deg", np.nan) for r in rows], width,
                   yerr=[r.get(f"fixation_mae_{key}_sd_deg", 0.0) for r in rows], capsize=2, label=label)
        ax.set_xticks(x, [_label(d) for d in present])
        ax.set_ylabel(f"{name} fixation MAE (deg)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle("Real-time cross-subject gaze estimation (5-fold; mean ± SD across subjects)")
    _save(fig, "cross_dataset_fixation_mae.png")


def known_start() -> None:
    results = {d: _load(*_experiment(d, "known_start"), "known_start_protocol.json") for d in DATASETS}
    if not any(results.values()):
        raise FileNotFoundError("no known-start results")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    x, width = np.arange(len(DATASETS)), 0.27
    for row, (segment_key, segment) in enumerate((("short_saccades", "short 1-2 s"), ("long_saccades", "long 32 s"))):
        for col, (key, name) in enumerate(AXES):
            ax = axes[row, col]
            ours = [(results[d] or {}).get("same_subject", {}).get(segment_key) for d in DATASETS]
            published = {"Kalman": [np.nan] * len(DATASETS), "differencing": [np.nan] * len(DATASETS)}
            for j, d in enumerate(DATASETS):
                for method, segments, h, v, _ in PAPER_KNOWN_START.get(d, []):
                    if segments == segment:
                        published["Kalman" if "Kalman" in method else "differencing"][j] = h if key == "h" else v
            ax.bar(x - width, [o[f"excluded_mae_{key}_deg"] if o else np.nan for o in ours], width,
                   yerr=[o[f"excluded_mae_{key}_sd_deg"] if o else 0.0 for o in ours], capsize=2,
                   label="This pipeline: detected saccades")
            ax.bar(x, published["Kalman"], width, label="Published: dual Kalman filter (+ VOR model on Dataset 3)")
            ax.bar(x + width, published["differencing"], width, label="Published: signal differencing")
            ax.set_xticks(x, [_label(d) for d in DATASETS])
            ax.set_title(f"{segment}, {name}")
            ax.set_ylabel("fixation MAE (deg)")
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Known-start task, same subject, outlier segments dropped (published results exist for "
                 "Datasets 2 and 3 only)")
    fig.tight_layout()
    _save(fig, "known_start.png")


PER_SUBJECT_PANELS = (("dataset2", "realtime"), ("dataset2", "offline"), ("dataset1", "range_weighted"),
                      ("dataset3", "range_weighted"), ("dataset4", "range_weighted"))


def per_subject_fixation_mae() -> None:
    stats = _load("statistics", "statistics.json")
    panels = [(d, k, stats[d][k]) for d, k in PER_SUBJECT_PANELS
              if stats and "per_subject_deg" in stats.get(d, {}).get(k, {})]
    if not panels:
        raise FileNotFoundError("statistics.json without per-subject values")
    fig, axes = plt.subplots(2, len(panels), figsize=(2.7 * len(panels), 6.5), squeeze=False)
    for j, (dataset, key, comparison) in enumerate(panels):
        per_subject = comparison["per_subject_deg"]
        for i, (axis, name) in enumerate(AXES):
            ax = axes[i, j]
            reference = np.array([per_subject[s]["reference"][i] for s in per_subject])
            model = np.array([per_subject[s]["model"][i] for s in per_subject])
            for a, b in zip(reference, model):
                ax.plot([0, 1], [a, b], marker="o", ms=3, alpha=0.7, color="tab:green" if b < a else "tab:red")
            ax.plot([0, 1], [reference.mean(), model.mean()], marker="s", lw=2.5, color="black")
            ax.set_xticks([0, 1], ["reference", "model"])
            ax.set_xlim(-0.3, 1.3)
            ax.text(0.5, 0.98, _p(comparison["wilcoxon_p"][axis]), transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)
            if i == 0:
                track = {"realtime": "real-time, nested", "offline": "offline, nested"}.get(key, "real-time, weighted")
                ax.set_title(f"{_label(dataset)}\n{track}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{name} fixation MAE (deg)")
    fig.suptitle("Per-subject fixation MAE: engineered features only (reference) vs context + range features "
                 "(model); green = improved, black = mean, Wilcoxon signed-rank p")
    fig.tight_layout()
    _save(fig, "per_subject_fixation_mae.png")


STRESS_LABELS = {"full": "full recording", "right": "right only", "top": "top only", "centre": "centre only"}


def range_stress_test() -> None:
    results = _load("range_stress_test", "dataset2", "range_stress_test.json")
    if not results:
        raise FileNotFoundError("no stress-test results")
    names = list(results["scenarios"])

    def label(name):
        kind, _, skew = name.partition("_")
        if kind in ("random", "block"):
            return f"   {'random trials' if kind == 'random' else 'contiguous block'}, {skew} count"
        return STRESS_LABELS.get(name, name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    y = np.arange(len(names))[::-1]
    for ax, (key, name) in zip(axes, AXES):
        for offset, features, text in ((0.2, "context", "context features"), (-0.2, "range", "+ range features")):
            ax.barh(y + offset, [results["scenarios"][n][features][f"fixation_mae_{key}_deg"] for n in names],
                    0.4, label=text)
        ax.set_yticks(y, [label(n) for n in names])
        ax.set_xlabel(f"{name} fixation MAE (deg)")
        ax.grid(axis="x", alpha=0.3)
    axes[0].legend()
    fig.suptitle("Range features when Dataset 2 test gaze covers only part of the screen "
                 "(real-time XGBoost, weighted training)")
    fig.tight_layout()
    _save(fig, "range_stress_test.png")


def nested_cv_candidates() -> None:
    tracks = [(t, _load("nested_cv", "dataset2", t, "nested_cv_results.json")) for t in ("realtime", "offline")]
    tracks = [(t, r) for t, r in tracks if r]
    if not tracks:
        raise FileNotFoundError("no nested-CV results")
    windows = ("30", "60", "120")
    rows = [(f, tw) for f in ("engineered", "context", "range") for tw in ("all", "weighted", "fixation")]
    fig, axes = plt.subplots(1, len(tracks), figsize=(6.5 * len(tracks), 5), squeeze=False)
    for ax, (track, results) in zip(axes[0], tracks):
        name = lambda w, f, tw: f"window={w}s features={f} train={tw}"
        grid = np.array([[results["fixed_candidates"][name(w, f, tw)]["score_fixation"] for w in windows]
                         for f, tw in rows])
        picks = Counter(fold["selected"]["fixation"] for fold in results["folds"])
        image = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        for i, (f, tw) in enumerate(rows):
            for j, w in enumerate(windows):
                n = picks.get(name(w, f, tw), 0)
                ax.text(j, i, f"{grid[i, j]:.2f}" + (f"\npicked {n}/{len(results['folds'])}" if n else ""),
                        ha="center", va="center", fontsize=7, color="white" if grid[i, j] > grid.mean() else "black")
        ax.set_xticks(range(len(windows)), [f"{w} s" for w in windows])
        ax.set_xlabel("drift-baseline window")
        ax.set_yticks(range(len(rows)), [f"{'+ context + range' if f == 'range' else '+ context' if f == 'context' else f}, "
                                         f"{tw} windows" for f, tw in rows])
        ax.set_title(f"Dataset 2 {'real-time' if track == 'realtime' else 'offline'}: "
                     "mean H/V fixation MAE (deg)", fontsize=10)
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.suptitle("Nested cross-validation: every combination scored on the outer test folds, "
                 "and how often the inner loop picked it")
    fig.tight_layout()
    _save(fig, "nested_cv_candidates.png")


def main() -> None:
    for figure in (cross_dataset_fixation_mae, known_start, per_subject_fixation_mae, range_stress_test,
                   nested_cv_candidates):
        try:
            figure()
        except (FileNotFoundError, KeyError) as e:
            print(f"skipped {figure.__name__}: {e}")


if __name__ == "__main__":
    main()
