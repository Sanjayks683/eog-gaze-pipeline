"""
tests/test_drift.py
===================
Moving-median drift removal and project-relative YAML paths.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import yaml

from src.config import CFG, _CONFIG_SECTIONS, load_config_from_yaml
from src.preprocessing.filtering import remove_baseline_drift

FS = 256.0


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _fixations_with_drift(n_sec: int = 240, seed: int = 0):
    """A new random gaze level every second, plus a slow 5-unit electrode drift."""
    rng = np.random.RandomState(seed)
    levels = np.repeat(rng.uniform(-1, 1, n_sec), int(FS))
    drift = np.linspace(0.0, 5.0, len(levels))
    return levels, levels + drift


@pytest.mark.parametrize("causal", [False, True])
def test_moving_median_keeps_fixation_levels_and_removes_drift(causal, restore_cfg):
    CFG.preprocessing.median_baseline_sec = 30.0
    CFG.preprocessing.median_baseline_causal = causal
    levels, sig = _fixations_with_drift()

    out = remove_baseline_drift(sig, FS, method="moving_median")

    skip = int(30 * FS)
    assert np.corrcoef(out[skip:], levels[skip:])[0, 1] > 0.9
    residual_trend = np.polyfit(np.arange(len(out)), out, 1)[0] * len(out)
    assert abs(residual_trend) < 0.5


def test_causal_moving_median_never_uses_future_samples(restore_cfg):
    CFG.preprocessing.median_baseline_causal = True
    _, sig = _fixations_with_drift()
    k = len(sig) // 2
    changed = sig.copy()
    changed[k:] += 100.0

    a = remove_baseline_drift(sig, FS, method="moving_median")
    b = remove_baseline_drift(changed, FS, method="moving_median")
    assert np.array_equal(a[:k], b[:k])


@pytest.mark.parametrize("method,causal", [
    ("robust_mean", False), ("robust_mean", True), ("robust_line", True),
])
def test_robust_baselines_keep_fixation_levels_and_remove_drift(method, causal, restore_cfg):
    CFG.preprocessing.baseline_window_sec = 30.0
    CFG.preprocessing.baseline_causal = causal
    levels, sig = _fixations_with_drift()

    out = remove_baseline_drift(sig, FS, method=method)

    skip = int(30 * FS)
    assert np.corrcoef(out[skip:], levels[skip:])[0, 1] > 0.9
    residual_trend = np.polyfit(np.arange(len(out)), out, 1)[0] * len(out)
    assert abs(residual_trend) < 0.5


@pytest.mark.parametrize("method", ["robust_mean", "robust_line"])
def test_causal_robust_baselines_never_use_future_samples(method, restore_cfg):
    CFG.preprocessing.baseline_causal = True
    _, sig = _fixations_with_drift()
    k = len(sig) // 2 + 7  # deliberately not on a block boundary
    changed = sig.copy()
    changed[k:] += 100.0

    a = remove_baseline_drift(sig, FS, method=method)
    b = remove_baseline_drift(changed, FS, method=method)
    assert np.array_equal(a[:k], b[:k])


@pytest.mark.parametrize("method,causal", [("robust_mean", False), ("robust_line", True)])
def test_robust_baselines_ignore_blink_spikes(method, causal, restore_cfg):
    CFG.preprocessing.baseline_window_sec = 30.0
    CFG.preprocessing.baseline_causal = causal
    levels, sig = _fixations_with_drift()
    spiky = sig.copy()
    rng = np.random.RandomState(1)
    for onset in rng.choice(len(sig) - 80, 60, replace=False):
        spiky[onset:onset + 80] += 20.0  # 60 large blink-like 300 ms spikes

    clean = remove_baseline_drift(sig, FS, method=method)
    blinked = remove_baseline_drift(spiky, FS, method=method)
    baseline_shift = (spiky - blinked) - (sig - clean)
    assert np.median(np.abs(baseline_shift)) < 0.1


def test_causal_robust_line_removes_lag_on_linear_drift(restore_cfg):
    CFG.preprocessing.baseline_window_sec = 60.0
    CFG.preprocessing.baseline_causal = True
    drift = np.linspace(0.0, 10.0, int(600 * FS))
    tail = slice(int(120 * FS), None)

    lag_mean = np.abs(remove_baseline_drift(drift, FS, method="robust_mean")[tail]).mean()
    lag_line = np.abs(remove_baseline_drift(drift, FS, method="robust_line")[tail]).mean()
    assert lag_mean > 0.4  # half-window lag: 10/600 per s * 30 s = 0.5
    assert lag_line < 0.05


def test_yaml_relative_paths_resolve_against_project_root(tmp_path, restore_cfg):
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({"paths": {"results": "reports/experiments/x"}}),
                    encoding="utf-8")

    load_config_from_yaml(str(path))

    expected = os.path.normpath(os.path.join(CFG.paths.project_root, "reports", "experiments", "x"))
    assert CFG.paths.results == expected
