"""
tests/test_known_start.py
=========================
Known-start protocol replication (Barbara et al. 2023) on synthetic EOG with known
gains, saccades, blinks and drift.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from src.config import CFG, _CONFIG_SECTIONS
from src.data.schema import Trial
from src.evaluation.known_start import (
    BLINK, FIXATION, SACCADE,
    detect_eye_movements, ground_truth_labels, prepare_recording,
    run_known_start, run_known_start_protocol,
)

FS = 256.0


@pytest.fixture()
def restore_cfg():
    snapshot = {section: dict(vars(getattr(CFG, section))) for section in _CONFIG_SECTIONS}
    yield CFG
    for section, values in snapshot.items():
        sub = getattr(CFG, section)
        for k, v in values.items():
            setattr(sub, k, v)


def _blink(n: int, at: int, amplitude: float = 250.0, width: int = int(0.25 * FS)) -> np.ndarray:
    out = np.zeros(n)
    b = np.arange(width)
    out[at:at + width] = amplitude * np.sin(np.pi * b / width)
    return out


def _recording(seed: int, gain=(15.0, 10.0), n_trials: int = 200) -> Trial:
    """Cued saccade-saccade-blink trials; the eye follows each cue after 200 ms."""
    rng = np.random.RandomState(seed)
    sec = int(FS)
    cs, target = [3] * sec, [np.zeros(2)] * sec   # lead-in so the first trial has a known start
    for _ in range(n_trials):
        p1 = rng.uniform([-20, -12], [20, 12])
        p2 = rng.uniform([-20, -12], [20, 12])
        cs += [1] * sec + [2] * sec + [3] * 2 * sec
        target += [p1] * sec + [p2] * 3 * sec
    target, cs = np.array(target), np.array(cs)
    n = len(cs)
    lag, k = int(0.2 * FS), int(0.04 * FS)
    gaze = np.r_[np.repeat(target[:1], lag, axis=0), target[:-lag]]
    kernel = np.ones(k) / k
    gaze = np.column_stack([np.convolve(np.pad(g, k, mode="edge"), kernel, "same")[k:-k] for g in gaze.T])

    blink = np.zeros(n)
    for onset in np.flatnonzero(np.diff(cs) != 0) + 1:
        if cs[onset] == 3:
            blink += _blink(n, onset + int(0.5 * FS))
    drift = np.linspace(0, 200, n)
    channels = {
        "H": gain[0] * gaze[:, 0] + drift + rng.randn(n) * 2.0,
        "V": gain[1] * gaze[:, 1] - 0.5 * drift + blink + rng.randn(n) * 2.0,
        "ControlSignal": cs,
    }
    return Trial(subject_id=f"S{seed}", trial_id=f"S{seed}_trial", dataset_source="dataset2",
                 montage_type="monopolar", fs=FS, channels=channels, target_angle=target)


def _step_signal(n: int):
    t = np.arange(n)
    ramp = lambda at: np.clip((t - at * FS) / (0.04 * FS), 0, 1)
    h = 200 * ramp(2) - 100 * ramp(6)
    v = -50 * ramp(2) + 80 * ramp(6) + _blink(n, int(9 * FS), 300.0)
    return np.stack([h, v]) + np.random.RandomState(0).randn(2, n)


def test_detector_measures_saccade_displacements_and_flags_the_blink():
    ev = detect_eye_movements(_step_signal(int(12 * FS)), FS)

    real = ~ev["is_blink"]
    assert real.sum() == 2 and ev["is_blink"].sum() == 1
    assert np.allclose(ev["delta"][real], [[200, -50], [-100, 80]], atol=5)
    assert abs(ev["onset"][ev["is_blink"]][0] - 9 * FS) < 0.05 * FS


def test_ground_truth_labels_mark_saccades_and_the_blink_between_its_peaks():
    n = int(12 * FS)
    control = np.ones(n, dtype=int)
    control[int(8 * FS):int(10 * FS)] = 3

    labels = ground_truth_labels(_step_signal(n), FS, control)

    assert np.mean(labels == FIXATION) > 0.9
    assert (labels[int(2 * FS):int(2.04 * FS)] == SACCADE).mean() > 0.8
    assert (labels[int(6 * FS):int(6.04 * FS)] == SACCADE).mean() > 0.8
    blink = np.flatnonzero(labels == BLINK)
    # From the up-stroke's peak (bump start) to the down-stroke's peak (bump end, 9.25 s).
    assert len(blink) and 9 * FS <= blink[0] < 9.05 * FS and 9.2 * FS < blink[-1] <= 9.27 * FS
    assert not (labels[: int(8 * FS)] == BLINK).any()
    assert not (labels[int(9.3 * FS):] == BLINK).any()


def test_windows_flag_a_blink_during_a_saccade_window_as_a_subject_mistake(restore_cfg):
    trial = _recording(0, n_trials=40)
    cs = trial.channels["ControlSignal"]
    saccade_starts = np.flatnonzero((np.diff(cs) != 0) & np.isin(cs[1:], (1, 2))) + 1
    bad = saccade_starts[20]
    trial.channels["V"] = trial.channels["V"] + _blink(len(cs), bad + int(0.6 * FS))

    windows = prepare_recording(trial)["windows"]

    by_start = {w["start"]: w for w in windows}
    assert by_start[bad]["mistake"] and by_start[bad]["has_blink"]
    # The detector rightly drops that blink, so it is not an estimator/label disagreement.
    assert not by_start[bad]["disagrees"]
    others = [w for w in windows if w["start"] != bad]
    assert np.mean([not w["mistake"] for w in others if w["kind"] == "saccade"]) > 0.9
    assert np.mean([not w["mistake"] for w in others if w["kind"] == "blink"]) > 0.9


def test_known_start_protocol_tracks_gaze_from_the_starting_position(tmp_path, restore_cfg):
    gains = [(15.0, 10.0), (8.0, 6.0), (22.0, 14.0)]
    trials = [_recording(seed, gain) for seed, gain in enumerate(gains)]

    results = run_known_start(trials, results_dir=str(tmp_path))

    stats = results["protocol_stats"]
    assert stats["detector_cue_recall"] > 0.95
    assert stats["short_saccade_windows_per_part"] == 66 and stats["short_blink_windows_per_part"] == 33
    assert stats["long_segments_per_part"] == 8
    assert stats["blink_windows_blink_counted_as_movement"] < 0.1
    assert stats["saccade_windows_response_dropped_as_blink"] < 0.05
    # Same subject: the fitted map is exact, so only per-sample noise remains.
    for key in ("short_saccades", "short_level", "long_saccades"):
        r = results["same_subject"][key]
        assert r["mae_h_deg"] < 0.6 and r["mae_v_deg"] < 0.6, (key, r)
    # Unseen subject: the label-free gain is a quantile spread of a few hundred random
    # saccades, so it carries a few percent of sampling error on shifts up to 40 deg.
    for key in ("short_saccades", "short_level", "long_saccades"):
        r = results["unseen_subject"][key]
        assert r["mae_h_deg"] < 1.5 and r["mae_v_deg"] < 1.0, (key, r)
    assert os.path.isfile(tmp_path / "known_start_protocol.json")
    table = (tmp_path / "known_start_table.csv").read_text(encoding="utf-8")
    assert "Known start, detected saccades" in table and "Published: dual Kalman filter" in table


def test_known_start_protocol_rejects_trials_without_targets():
    trial = _recording(0, n_trials=4)
    trial.target_angle = None
    with pytest.raises(ValueError, match="target_angle"):
        run_known_start_protocol([trial])
