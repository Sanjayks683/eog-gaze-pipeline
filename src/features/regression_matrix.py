"""
src/features/regression_matrix.py
==================================
Classical-regression feature matrices built straight from unified trials, for
experiments that need many preprocessing variants (nested cross-validation,
robustness tests) without writing main.py's full processed-data set each time.

It runs the pipeline's own preprocess -> window -> feature code under the current
CFG settings, so a matrix holds the same inputs train_classical would use for that
config: engineered window features, then the context features split into the
baseline/lag block and the rolling-range block.
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np

from src.config import CFG
from src.data.datasets import make_regression_targets
from src.data.loaders import load_all_datasets
from src.data.schema import Trial
from src.data.unify import to_common_schema
from src.features.context import make_context_features
from src.features.engineered import extract_features_batch
from src.preprocessing.normalize import preprocess_trials

FEATURE_BLOCKS = ("engineered", "context", "range", "head")


def load_unified_trials() -> List[Trial]:
    """Raw trials of CFG.data.datasets_to_load, with H and V added."""
    return [to_common_schema(t) for t in load_all_datasets(skip_missing=True)]


def build_feature_matrix(trials: List[Trial], engineered: bool = True) -> Dict[str, np.ndarray]:
    """
    Preprocess copies of `trials` under the current CFG and return plain arrays:
    "engineered", "context" (context baselines + lagged levels), "range" (rolling
    range), "head" (head-pose features; no columns unless context_head_pose), "y"
    (window-mean target angles), "subject", "fixation", "window_start".
    """
    trials = [copy.deepcopy(t) for t in trials]
    raw = [{k: v.copy() for k, v in t.channels.items()} for t in trials]
    trials = preprocess_trials(trials)
    X, y, meta = make_regression_targets(trials)
    fs_values = {m["fs"] for m in meta}
    if len(fs_values) != 1:
        raise ValueError(f"Expected one sampling rate, got {sorted(fs_values)}")
    channel_names = meta[0]["channel_names"]
    ctx = make_context_features(trials, raw, meta)
    pp = CFG.preprocessing
    n_basic = (len(pp.context_baselines) + len(pp.context_lags_sec)) * len(channel_names)
    n_range = len(pp.context_range_windows_sec) * len(pp.context_range_quantiles) * len(channel_names) * 3
    return {
        # engineered=False skips the slow window features when only context blocks are needed
        "engineered": (extract_features_batch(X, fs=fs_values.pop(), channel_names=channel_names).astype(np.float32)
                       if engineered else np.zeros((len(meta), 0), dtype=np.float32)),
        "context": ctx[:, :n_basic],
        "range": ctx[:, n_basic:n_basic + n_range],
        "head": ctx[:, n_basic + n_range:],
        "y": y,
        "subject": np.array([m["subject_id"] for m in meta]),
        "fixation": np.array([bool(m["is_fixation"]) for m in meta]),
        "window_start": np.array([m["window_start"] for m in meta]),
    }


def stack_features(matrix: Dict[str, np.ndarray], blocks: List[str]) -> np.ndarray:
    """Concatenate the named feature blocks column-wise."""
    return np.hstack([matrix[b] for b in blocks]).astype(np.float32)
