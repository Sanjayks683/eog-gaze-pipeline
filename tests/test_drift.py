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


def test_yaml_relative_paths_resolve_against_project_root(tmp_path, restore_cfg):
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump({"paths": {"results": "reports/experiments/x"}}),
                    encoding="utf-8")

    load_config_from_yaml(str(path))

    expected = os.path.normpath(os.path.join(CFG.paths.project_root, "reports", "experiments", "x"))
    assert CFG.paths.results == expected
