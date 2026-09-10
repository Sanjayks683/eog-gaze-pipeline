"""
scripts/verify_montage.py
=========================
Phase 2.4 — verify bipolar montage selection and sign conventions.

For every monopolar dataset, correlates the difference of EVERY ordered
electrode pair with the recorded target gaze angles (H and V) across all
subjects, Fisher-z averages the per-subject correlations, and reports the
best pair per axis. Compare the winners against MONOPOLAR_MAPS in
src/data/unify.py and update the maps if they disagree.

Usage:
    python scripts/verify_montage.py
"""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.loaders import load_dataset2, load_dataset3, load_dataset4
from src.data.unify import MONOPOLAR_MAPS


def fisher_mean(corrs):
    """Average correlations via Fisher z-transform (robust to outliers)."""
    z = np.arctanh(np.clip(corrs, -0.999, 0.999))
    return float(np.mean(z))


def verify_dataset(loader, dataset_name: str, top_k: int = 3) -> None:
    trials = loader()
    eog_keys = sorted(
        [k for k in trials[0].channels if k.startswith("EOG_")],
        key=lambda s: int(s.split("_")[1]),
    )
    print(f"\n=== {dataset_name}: {len(trials)} trials, {len(eog_keys)} channels ===")

    rH, rV = defaultdict(list), defaultdict(list)
    for trial in trials:
        if trial.target_angle is None:
            continue
        n = min(len(trial.channels[eog_keys[0]]), trial.target_angle.shape[0])
        tH, tV = trial.target_angle[:n, 0], trial.target_angle[:n, 1]
        for i, j in combinations(range(len(eog_keys)), 2):
            d = trial.channels[eog_keys[i]][:n] - trial.channels[eog_keys[j]][:n]
            rH[(i, j)].append(np.corrcoef(d, tH)[0, 1])
            rV[(i, j)].append(np.corrcoef(d, tV)[0, 1])

    for axis, rr in (("H", rH), ("V", rV)):
        ranked = sorted(
            ((abs(fisher_mean(v)), fisher_mean(v), np.median(v), len(v), k)
             for k, v in rr.items()),
            reverse=True,
        )
        print(f"  Best {axis} pairs:")
        for _, mz, med, n_subj, (i, j) in ranked[:top_k]:
            print(f"    {eog_keys[i]}-{eog_keys[j]}: fisher_mean_z={mz:+.3f} "
                  f"median_r={med:+.3f} ({n_subj} subjects)")

    m = MONOPOLAR_MAPS.get(dataset_name)
    if m:
        h_sign = "-" if m["flip_h"] else "+"
        v_sign = "-" if m["flip_v"] else "+"
        print(f"  Current MONOPOLAR_MAPS: H = {h_sign}({m['right_key']}-{m['left_key']}), "
              f"V = {v_sign}({m['up_key']}-{m['down_key']})")


def main():
    print("=== Phase 2.4 — Montage verification against target angles ===")
    for loader, name in [
        (load_dataset2, "dataset2"),
        (load_dataset3, "dataset3"),
        (load_dataset4, "dataset4"),
    ]:
        try:
            verify_dataset(loader, name)
        except FileNotFoundError as e:
            print(f"\n[SKIP] {name}: {e}")
    print("\nIf the reported winners disagree with MONOPOLAR_MAPS, update "
          "src/data/unify.py and re-run `python main.py --phase preprocess`.")


if __name__ == "__main__":
    main()
