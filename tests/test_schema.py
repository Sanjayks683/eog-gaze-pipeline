"""
tests/test_schema.py
=====================
Tests that every loader returns valid Trial objects.
Datasets are skipped gracefully if not yet downloaded.
"""

import pytest
import numpy as np


def _check_trial(trial):
    from src.data.schema import REQUIRED_EVENT_KEYS
    assert trial.subject_id, "subject_id must be non-empty"
    assert trial.trial_id, "trial_id must be non-empty"
    assert trial.fs > 0, f"fs must be positive, got {trial.fs}"
    assert len(trial.channels) > 0, "channels dict must not be empty"
    for name, arr in trial.channels.items():
        assert arr is not None and len(arr) > 0, f"Channel {name!r} is empty"
    trial.validate()


def _skip_if_missing(loader_fn, dataset_name):
    """Call loader; if data not present, skip test cleanly."""
    try:
        return loader_fn(max_subjects=1)
    except FileNotFoundError as e:
        pytest.skip(f"{dataset_name} not yet downloaded: {e}")
    except Exception as e:
        if "empty" in str(e).lower() or "not found" in str(e).lower():
            pytest.skip(f"{dataset_name} not available: {e}")
        raise


def test_dataset1_loads():
    from src.data.loaders import load_dataset1
    trials = _skip_if_missing(load_dataset1, "Dataset 1")
    assert len(trials) >= 1, "Dataset 1 must return at least 1 trial"
    for t in trials:
        _check_trial(t)


def test_dataset1_montage_type():
    from src.data.loaders import load_dataset1
    trials = _skip_if_missing(load_dataset1, "Dataset 1")
    for t in trials:
        assert t.montage_type == "bipolar", f"Dataset 1 should be bipolar, got {t.montage_type}"
        assert t.dataset_source == "dataset1"


def test_dataset2_loads():
    from src.data.loaders import load_dataset2
    trials = _skip_if_missing(load_dataset2, "Dataset 2")
    assert len(trials) >= 1
    for t in trials:
        _check_trial(t)


def test_dataset2_montage_type():
    from src.data.loaders import load_dataset2
    trials = _skip_if_missing(load_dataset2, "Dataset 2")
    for t in trials:
        assert t.montage_type == "monopolar"
        assert t.dataset_source == "dataset2"


def test_dataset3_loads():
    from src.data.loaders import load_dataset3
    trials = _skip_if_missing(load_dataset3, "Dataset 3")
    assert len(trials) >= 1
    for t in trials:
        _check_trial(t)


def test_dataset3_has_head_pose_field():
    from src.data.loaders import load_dataset3
    trials = _skip_if_missing(load_dataset3, "Dataset 3")
    for t in trials:
        assert hasattr(t, "head_pose"), "Dataset 3 Trial must have head_pose field"


def test_dataset4_loads():
    from src.data.loaders import load_dataset4
    trials = _skip_if_missing(load_dataset4, "Dataset 4 (manual download required)")
    assert len(trials) >= 1
    for t in trials:
        _check_trial(t)


@pytest.mark.parametrize("loader_name", ["load_dataset1", "load_dataset2", "load_dataset3"])
def test_event_timestamps_monotonic(loader_name):
    import importlib
    mod = importlib.import_module("src.data.loaders")
    loader = getattr(mod, loader_name)
    trials = _skip_if_missing(loader, loader_name)
    for t in trials:
        try:
            t.validate()
        except ValueError as e:
            pytest.fail(f"Event timestamp validation failed for {t}: {e}")


def test_trial_repr_does_not_crash():
    from src.data.schema import Trial
    t = Trial(
        subject_id="S01", trial_id="T001", dataset_source="dataset1",
        montage_type="bipolar", fs=250.0,
        channels={"H": np.zeros(500), "V": np.zeros(500)},
    )
    r = repr(t)
    assert "S01" in r


def test_trial_n_samples():
    from src.data.schema import Trial
    t = Trial(
        subject_id="S01", trial_id="T001", dataset_source="dataset1",
        montage_type="bipolar", fs=500.0,
        channels={"H": np.zeros(1000), "V": np.zeros(1000)},
    )
    assert t.n_samples() == 1000
    assert abs(t.duration_s() - 2.0) < 1e-6


def test_trial_has_bipolar():
    from src.data.schema import Trial
    t = Trial(
        subject_id="S01", trial_id="T001", dataset_source="dataset2",
        montage_type="monopolar", fs=250.0,
        channels={"REOG": np.zeros(500), "LEOG": np.zeros(500)},
    )
    assert not t.has_bipolar()
    t.channels["H"] = np.zeros(500)
    t.channels["V"] = np.zeros(500)
    assert t.has_bipolar()
