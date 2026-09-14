"""
scripts/range_stress_test.py
=============================
Do the rolling-range features still help when the test subject's gaze is not spread
over the whole screen?

The range features estimate the screen centre and each subject's EOG gain from the
spread of recent gaze, which Dataset 2's evenly spread random cues favour. Here every
test subject's recording is rebuilt from a subset of its trials so that gaze clusters
in one region, and XGBoost trained on the other subjects' full recordings is scored on
it with and without the range features. Each skewed subset is paired with two controls
of the same number of trials: random trials, spliced the same way, and one contiguous
block, which is shorter but has no splice points. Skewed against random isolates the
skew; random against block isolates the splicing.

Scenarios, from each trial's two cue positions P1 and P2:
  full      every trial (the normal recording)
  right     both cues right of centre (H > 0)
  top       both cues above centre (V > 0)
  centre    both cues within the central two thirds of the subject's H and V range
  random_*  random trials, as many as the matching skewed scenario
  block_*   a contiguous run of trials, as many as the matching skewed scenario

    python scripts/range_stress_test.py --config configs/dataset2_range_causal_weighted.yaml

Results: reports/experiments/range_stress_test/<dataset>/.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.nested_cv import evaluate  # noqa: E402
from src.config import CFG, load_config_from_yaml  # noqa: E402
from src.features.regression_matrix import build_feature_matrix, load_unified_trials, stack_features  # noqa: E402
from src.models.classical_ml import fit_predict_regression_fold  # noqa: E402
from src.training.cv_splits import get_folds  # noqa: E402

FEATURE_SETS = {"context": ["engineered", "context"], "range": ["engineered", "context", "range"]}
SKEWS = ("right", "top", "centre")


def trial_cues(trial):
    """(start, end, P1, P2) for every trial, split at ControlSignal 1 onsets."""
    cs = np.asarray(trial.channels["ControlSignal"])
    target = np.asarray(trial.target_angle)
    starts = np.flatnonzero((cs == 1) & (np.r_[0, cs[:-1]] != 1))
    ends = np.r_[starts[1:], len(cs)]
    cues = []
    for s, e in zip(starts, ends):
        second = np.flatnonzero(cs[s:e] == 2)
        p2 = target[s + second[0]] if len(second) else target[s]
        cues.append((s, e, target[s], p2))
    return cues


def keep_mask(cues, scenario, rng, counts):
    p1 = np.array([c[2] for c in cues])
    p2 = np.array([c[3] for c in cues])
    if scenario == "full":
        return np.ones(len(cues), bool)
    if scenario == "right":
        return (p1[:, 0] > 0) & (p2[:, 0] > 0)
    if scenario == "top":
        return (p1[:, 1] > 0) & (p2[:, 1] > 0)
    if scenario == "centre":
        both = np.vstack([p1, p2])
        limit = (2 / 3) * np.max(np.abs(both), axis=0)
        return np.all(np.abs(p1) <= limit, axis=1) & np.all(np.abs(p2) <= limit, axis=1)
    if scenario.startswith("random_"):
        mask = np.zeros(len(cues), bool)
        mask[rng.choice(len(cues), counts[scenario[len("random_"):]], replace=False)] = True
        return mask
    if scenario.startswith("block_"):
        n = counts[scenario[len("block_"):]]
        start = rng.randint(0, len(cues) - n + 1)
        mask = np.zeros(len(cues), bool)
        mask[start:start + n] = True
        return mask
    raise ValueError(scenario)


def splice(trial, cues, mask):
    """The recording rebuilt from its lead-in plus the kept trials, in order."""
    lead_in = np.arange(0, cues[0][0])
    idx = np.concatenate([lead_in] + [np.arange(s, e) for (s, e, _, _), keep in zip(cues, mask) if keep])
    out = copy.copy(trial)
    out.channels = {k: np.asarray(v)[idx] for k, v in trial.channels.items()}
    for name in ("target_angle", "target_angle_corrected", "head_pose"):
        if getattr(trial, name) is not None:
            setattr(out, name, np.asarray(getattr(trial, name))[idx])
    out.metadata = dict(trial.metadata)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--xgb-device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_config_from_yaml(args.config)
    CFG.cv.xgb_device = args.xgb_device
    dataset = "_".join(CFG.data.datasets_to_load)
    root = CFG.paths.project_root
    out_dir = os.path.join(root, "reports", "experiments", "range_stress_test", dataset)
    os.makedirs(out_dir, exist_ok=True)

    trials = load_unified_trials()
    cues = {t.subject_id: trial_cues(t) for t in trials}
    t0 = time.time()
    full = build_feature_matrix(trials)
    print(f"full-recording features built ({time.time() - t0:.0f}s)")
    subjects_full = full["subject"]
    folds = get_folds(subjects_full)

    controls = tuple(f"{kind}_{k}" for kind in ("random", "block") for k in SKEWS)
    scenario_masks = {s: {} for s in ("full",) + SKEWS + controls}
    rng = np.random.RandomState(args.seed)
    block_rng = np.random.RandomState(args.seed + 1)
    for t in trials:
        c = cues[t.subject_id]
        counts = {}
        for scenario in ("full",) + SKEWS:
            scenario_masks[scenario][t.subject_id] = keep_mask(c, scenario, rng, counts)
            counts[scenario] = int(scenario_masks[scenario][t.subject_id].sum())
        for skew in SKEWS:
            scenario_masks[f"random_{skew}"][t.subject_id] = keep_mask(c, f"random_{skew}", rng, counts)
        for skew in SKEWS:
            scenario_masks[f"block_{skew}"][t.subject_id] = keep_mask(c, f"block_{skew}", block_rng, counts)

    results = {"dataset": dataset, "config": args.config, "xgb_device": CFG.cv.xgb_device, "scenarios": {}}
    for scenario, masks in scenario_masks.items():
        spliced = [splice(t, cues[t.subject_id], masks[t.subject_id]) for t in trials]
        matrix = full if scenario == "full" else build_feature_matrix(spliced)
        spread = {}
        for t in trials:
            kept = [cue for cue, keep in zip(cues[t.subject_id], masks[t.subject_id]) if keep]
            both = np.array([cue[2] for cue in kept] + [cue[3] for cue in kept])
            spread[t.subject_id] = {"trials": len(kept),
                                    "cue_h_5_95": np.percentile(both[:, 0], [5, 95]).round(1).tolist(),
                                    "cue_v_5_95": np.percentile(both[:, 1], [5, 95]).round(1).tolist()}
        entry = {"trials_per_subject": spread}
        for name, blocks in FEATURE_SETS.items():
            X_all = np.vstack([stack_features(full, blocks), stack_features(matrix, blocks)])
            y_all = np.vstack([full["y"], matrix["y"]]).astype(np.float64)
            fix_all = np.r_[full["fixation"], matrix["fixation"]]
            pred = np.full_like(matrix["y"], np.nan, dtype=np.float64)
            offset = len(full["y"])
            for tr, te in folds:
                test_subjects = set(subjects_full[te].tolist())
                test_rows = np.flatnonzero(np.isin(matrix["subject"], list(test_subjects)))
                p, _, kept = fit_predict_regression_fold(X_all, y_all, tr, offset + test_rows, "xgb", False,
                                                         fix_all, CFG.cv.regression_train_windows)
                pred[kept - offset] = p
            entry[name] = evaluate(matrix["y"].astype(np.float64), pred, matrix["subject"], matrix["fixation"])
        ctx, rng_ = entry["context"], entry["range"]
        entry["range_gain_fixation_mae_deg"] = [ctx["fixation_mae_h_deg"] - rng_["fixation_mae_h_deg"],
                                                ctx["fixation_mae_v_deg"] - rng_["fixation_mae_v_deg"]]
        results["scenarios"][scenario] = entry
        print(f"{scenario:15s} context {ctx['fixation_mae_h_deg']:.2f} / {ctx['fixation_mae_v_deg']:.2f} | "
              f"range {rng_['fixation_mae_h_deg']:.2f} / {rng_['fixation_mae_v_deg']:.2f} | gain "
              f"{entry['range_gain_fixation_mae_deg'][0]:+.2f} / {entry['range_gain_fixation_mae_deg'][1]:+.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        with open(os.path.join(out_dir, "range_stress_test.json"), "w") as f:
            json.dump(results, f, indent=2)
    print(f"Saved {os.path.join(out_dir, 'range_stress_test.json')}")


if __name__ == "__main__":
    main()
