"""
tests/test_montage_conversion.py
=================================
Tests for bipolar montage conversion (Phase 2).

Key invariants:
1. Output always has exactly 2 channels: 'H' and 'V'
2. Amplitude scale within order-of-magnitude of Dataset 1 native scale
3. Sign convention: rightward saccade → positive H deflection
"""

import pytest
import numpy as np

from src.data.schema import Trial, make_empty_event_timestamps


def _make_monopolar_trial(dataset: str, right=1.0, left=-1.0, up=0.5, down=-0.5) -> Trial:
    """Synthetic monopolar trial with known H/V values."""
    n = 500
    ts = make_empty_event_timestamps()
    ts["saccade1_onset"] = 100
    ts["saccade1_end"] = 200
    return Trial(
        subject_id="S_test",
        trial_id="T_test",
        dataset_source=dataset,
        montage_type="monopolar",
        fs=250.0,
        channels={
            "REOG": np.full(n, right),
            "LEOG": np.full(n, left),
            "UEOG": np.full(n, up),
            "DEOG": np.full(n, down),
        },
        event_timestamps=ts,
    )


def _patch_monopolar_map(dataset: str):
    """Temporarily patch MONOPOLAR_MAPS with known keys for testing."""
    from src.data import unify
    original = unify.MONOPOLAR_MAPS[dataset].copy()
    unify.MONOPOLAR_MAPS[dataset].update({
        "right_key": "REOG",
        "left_key": "LEOG",
        "up_key": "UEOG",
        "down_key": "DEOG",
        "flip_h": False,
        "flip_v": False,
    })
    return original


def _restore_monopolar_map(dataset: str, original: dict):
    from src.data import unify
    unify.MONOPOLAR_MAPS[dataset].update(original)


def test_dataset1_passthrough_preserves_channels():
    """Dataset 1 passthrough: H and V channels must be present after unification."""
    from src.data.unify import to_common_schema
    trial = Trial(
        subject_id="S01", trial_id="T001", dataset_source="dataset1",
        montage_type="bipolar", fs=250.0,
        channels={"H": np.ones(500), "V": np.zeros(500)},
    )
    result = to_common_schema(trial)
    assert "H" in result.channels
    assert "V" in result.channels


def test_dataset1_heog_veog_keys_mapped():
    """Dataset 1: HEOG/VEOG named channels should be mapped to H/V."""
    from src.data.unify import to_common_schema
    trial = Trial(
        subject_id="S01", trial_id="T001", dataset_source="dataset1",
        montage_type="bipolar", fs=250.0,
        channels={"HEOG": np.ones(500) * 3.0, "VEOG": np.ones(500) * 1.5},
    )
    result = to_common_schema(trial)
    assert "H" in result.channels
    assert "V" in result.channels
    np.testing.assert_allclose(result.channels["H"], 3.0)
    np.testing.assert_allclose(result.channels["V"], 1.5)


@pytest.mark.parametrize("dataset", ["dataset2", "dataset3", "dataset4"])
def test_bipolarize_output_has_H_and_V(dataset):
    """Bipolarized trials must have exactly 'H' and 'V' channels."""
    from src.data.unify import bipolarize
    original = _patch_monopolar_map(dataset)
    try:
        trial = _make_monopolar_trial(dataset)
        result = bipolarize(trial)
        assert "H" in result.channels, f"Missing 'H' channel after bipolarize for {dataset}"
        assert "V" in result.channels, f"Missing 'V' channel after bipolarize for {dataset}"
    finally:
        _restore_monopolar_map(dataset, original)


@pytest.mark.parametrize("dataset", ["dataset2", "dataset3", "dataset4"])
def test_bipolarize_channel_length_preserved(dataset):
    """H and V channels must have same length as input channels."""
    from src.data.unify import bipolarize
    original = _patch_monopolar_map(dataset)
    try:
        trial = _make_monopolar_trial(dataset)
        n = len(trial.channels["REOG"])
        result = bipolarize(trial)
        assert len(result.channels["H"]) == n
        assert len(result.channels["V"]) == n
    finally:
        _restore_monopolar_map(dataset, original)


@pytest.mark.parametrize("dataset", ["dataset2", "dataset3", "dataset4"])
def test_bipolarize_H_equals_right_minus_left(dataset):
    """H = right − left."""
    from src.data.unify import bipolarize
    original = _patch_monopolar_map(dataset)
    try:
        right_val, left_val = 2.5, -1.5
        trial = _make_monopolar_trial(dataset, right=right_val, left=left_val)
        result = bipolarize(trial)
        expected_h = right_val - left_val
        np.testing.assert_allclose(result.channels["H"], expected_h, rtol=1e-5)
    finally:
        _restore_monopolar_map(dataset, original)


@pytest.mark.parametrize("dataset", ["dataset2", "dataset3", "dataset4"])
def test_bipolarize_V_equals_up_minus_down(dataset):
    """V = up − down."""
    from src.data.unify import bipolarize
    original = _patch_monopolar_map(dataset)
    try:
        up_val, down_val = 1.2, -0.8
        trial = _make_monopolar_trial(dataset, up=up_val, down=down_val)
        result = bipolarize(trial)
        expected_v = up_val - down_val
        np.testing.assert_allclose(result.channels["V"], expected_v, rtol=1e-5)
    finally:
        _restore_monopolar_map(dataset, original)


def test_amplitude_within_order_of_magnitude():
    """
    Post-conversion H amplitude must be within one order of magnitude
    of Dataset 1's typical amplitude. This catches gross channel-mapping
    errors (e.g., swapping voltage-scale channels with a reference channel).
    Both signals must have noise so std is non-zero and the ratio is finite.
    """
    from src.data.unify import to_common_schema, bipolarize
    rng = np.random.RandomState(42)

    dataset1_trial = Trial(
        subject_id="S01", trial_id="T001", dataset_source="dataset1",
        montage_type="bipolar", fs=250.0,
        channels={"H": rng.randn(500) * 150, "V": rng.randn(500) * 80},
    )
    result_ds1 = to_common_schema(dataset1_trial)
    amp_ds1 = np.std(result_ds1.channels["H"])

    original = _patch_monopolar_map("dataset2")
    try:
        n = 500
        ts = make_empty_event_timestamps()
        ts["saccade1_onset"] = 100
        ts["saccade1_end"] = 200
        dataset2_trial = Trial(
            subject_id="S_test", trial_id="T_test", dataset_source="dataset2",
            montage_type="monopolar", fs=250.0,
            channels={
                "REOG": rng.randn(n) * 100 + 150,
                "LEOG": rng.randn(n) * 100 - 150,
                "UEOG": rng.randn(n) * 80 + 50,
                "DEOG": rng.randn(n) * 80 - 50,
            },
            event_timestamps=ts,
        )
        result_ds2 = bipolarize(dataset2_trial)
        amp_ds2 = np.std(result_ds2.channels["H"])

        assert amp_ds2 > 0, "Dataset 2 H std is 0 — add noise to test channels"
        ratio = amp_ds1 / amp_ds2
        assert 0.1 <= ratio <= 10.0, (
            f"Amplitude ratio Dataset1/Dataset2 = {ratio:.3f} — "
            "outside one order of magnitude. Check channel mapping."
        )
    finally:
        _restore_monopolar_map("dataset2", original)


def test_rightward_saccade_positive_H():
    """
    A rightward saccade should produce a positive mean H during the saccade window.
    (right electrode more positive than left = positive H)
    """
    from src.data.unify import bipolarize, verify_sign_convention
    original = _patch_monopolar_map("dataset2")
    try:
        n = 500
        ts = make_empty_event_timestamps()
        ts["saccade1_onset"] = 100
        ts["saccade1_end"] = 200

        reog = np.zeros(n)
        leog = np.zeros(n)
        reog[100:200] = 150.0
        leog[100:200] = -150.0

        trial = Trial(
            subject_id="S01", trial_id="T001", dataset_source="dataset2",
            montage_type="monopolar", fs=250.0,
            channels={"REOG": reog, "LEOG": leog, "UEOG": np.zeros(n), "DEOG": np.zeros(n)},
            event_timestamps=ts,
        )
        result = bipolarize(trial)
        h_saccade = result.channels["H"][100:200]
        assert np.mean(h_saccade) > 0, (
            f"Rightward saccade produced negative H (mean={np.mean(h_saccade):.2f}). "
            "Check sign convention or set flip_h=True."
        )
        ok = verify_sign_convention(result, saccade_direction="right", expected_h_sign=1.0)
        assert ok
    finally:
        _restore_monopolar_map("dataset2", original)
