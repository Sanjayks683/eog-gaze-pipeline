"""
scripts/inspect_raw.py
=======================
Phase 0 validation script.
Loads one raw trial per dataset, plots all channels, and saves figures to reports/figures/.
Run this FIRST after downloading the datasets to confirm the parsers work correctly.

Usage:
    python scripts/inspect_raw.py
"""

import os
import sys
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import CFG


DATASET_LOADERS = {
    "dataset1": ("src.data.loaders", "load_dataset1"),
    "dataset2": ("src.data.loaders", "load_dataset2"),
    "dataset3": ("src.data.loaders", "load_dataset3"),
    "dataset4": ("src.data.loaders", "load_dataset4"),
}


def plot_trial(trial, save_path: str) -> None:
    """Plot all channels of a trial and save to file."""
    ch_names = list(trial.channels.keys())
    n_ch = len(ch_names)
    t = np.arange(trial.n_samples()) / trial.fs

    fig, axes = plt.subplots(n_ch, 1, figsize=(14, 2.5 * n_ch), sharex=True)
    if n_ch == 1:
        axes = [axes]

    fig.suptitle(
        f"Dataset: {trial.dataset_source}  |  Subject: {trial.subject_id}  "
        f"|  Trial: {trial.trial_id}\n"
        f"fs={trial.fs} Hz  |  Duration={trial.duration_s():.2f}s  "
        f"|  Montage={trial.montage_type}",
        fontsize=11,
    )

    for ax, ch in zip(axes, ch_names):
        ax.plot(t, trial.channels[ch], lw=0.8)
        ax.set_ylabel(ch, fontsize=9)
        ax.grid(True, alpha=0.3)

        ts = trial.event_timestamps
        colors = {
            "saccade1_onset": ("g", "S1on"),
            "saccade1_end":   ("g", "S1end"),
            "saccade2_onset": ("b", "S2on"),
            "saccade2_end":   ("b", "S2end"),
            "blink_onset":    ("r", "Bon"),
            "blink_end":      ("r", "Bend"),
        }
        for key, (color, label) in colors.items():
            idx = ts.get(key, -1)
            if idx > 0 and idx < trial.n_samples():
                ax.axvline(idx / trial.fs, color=color, lw=1.2, linestyle="--", alpha=0.7)
                ax.text(idx / trial.fs, ax.get_ylim()[1] * 0.9, label,
                        fontsize=7, color=color, ha="center")

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {save_path}")


def print_data_description_table(trial, dataset_name: str) -> None:
    """Print a table row for the README data description table."""
    ch_names = list(trial.channels.keys())
    print(f"\n{'='*70}")
    print(f"  {dataset_name}")
    print(f"{'='*70}")
    print(f"  fs (Hz):         {trial.fs}")
    print(f"  Channels ({len(ch_names)}):  {ch_names}")
    print(f"  N samples:       {trial.n_samples()}")
    print(f"  Duration (s):    {trial.duration_s():.3f}")
    print(f"  Montage:         {trial.montage_type}")
    print(f"  Has target_angle:{trial.target_angle is not None}")
    print(f"  Has head_pose:   {trial.head_pose is not None}")
    ts = trial.event_timestamps
    known_ts = {k: v for k, v in ts.items() if v != -1}
    print(f"  Event timestamps:{known_ts}")
    print(f"  Metadata:        {trial.metadata}")
    print()
    print("  >>> Copy the above into README.md Phase 0 data description table <<<")


def main():
    os.makedirs(CFG.paths.reports_figures, exist_ok=True)
    print("\n=== EOG Pipeline — Raw Inspection (Phase 0) ===\n")

    for ds_name, (module_name, fn_name) in DATASET_LOADERS.items():
        print(f"\n--- {ds_name} ---")
        try:
            import importlib
            mod = importlib.import_module(module_name)
            loader = getattr(mod, fn_name)
            trials = loader(max_subjects=1)

            if not trials:
                print(f"  [WARN] No trials returned for {ds_name}")
                continue

            trial = trials[0]
            print_data_description_table(trial, ds_name)

            save_path = os.path.join(
                CFG.paths.reports_figures,
                f"raw_trial_{ds_name}.png"
            )
            plot_trial(trial, save_path)

        except FileNotFoundError as e:
            print(f"  [SKIP] {ds_name}: data not yet downloaded.\n  {e}")
        except Exception as e:
            warnings.warn(f"  [ERROR] {ds_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n=== Done. Check reports/figures/ for plots. ===\n")
    print("Fill the README.md Phase 0 data description table with values above.")


if __name__ == "__main__":
    main()
