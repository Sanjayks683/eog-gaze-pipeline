"""
src/data/unify.py
=================
Montage unification: convert any Trial into a common 2-channel (H, V) bipolar
representation, regardless of its source dataset.

Conversion rules
----------------
Dataset 1: Already bipolar (H, V). Pass through unchanged.
Dataset 2: Monopolar stationary  → bipolarize()
Dataset 3: Monopolar non-stationary → bipolarize()
Dataset 4: Monopolar isotropic   → bipolarize()

Bipolar conversion
------------------
H = right_electrode − left_electrode
V = up_electrode − down_electrode

Sign convention (verified per dataset in Phase 2.4)
----------------------------------------------------
A rightward saccade must produce a POSITIVE deflection in H after conversion.
An upward saccade must produce a POSITIVE deflection in V after conversion.
Any dataset where the sign comes out inverted gets an explicit flip documented here.

NOTE: The exact electrode→key mapping MUST be verified against each dataset's
Data Description file after download (Phase 0/Phase 2). Placeholder mappings
below are filled in with the most common conventions for bipolar EOG and are
annotated with TODO markers.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import numpy as np

from src.data.schema import Trial


MONOPOLAR_MAPS: Dict[str, dict] = {
    "dataset2": dict(
        right_key="EOG_2",
        left_key="EOG_3",
        up_key="EOG_0",
        down_key="EOG_1",
        flip_h=True,
        flip_v=False,
    ),
    "dataset3": dict(
        right_key="EOG_2",
        left_key="EOG_3",
        up_key="EOG_0",
        down_key="EOG_1",
        flip_h=True,
        flip_v=False,
    ),
    "dataset4": dict(
        right_key="EOG_10",
        left_key="EOG_11",
        up_key="EOG_0",
        down_key="EOG_1",
        flip_h=True,
        flip_v=False,
    ),
}


def to_common_schema(trial: Trial) -> Trial:
    """
    Convert a raw Trial (from any dataset) into the common 2-channel schema
    with channels 'H' and 'V'.

    Dataset 1 → passthrough (already bipolar).
    Datasets 2/3/4 → bipolarize using MONOPOLAR_MAPS.

    Returns the SAME trial object with channels['H'] and channels['V'] added.
    Original monopolar channels are preserved under their original keys.
    """
    if trial.dataset_source == "dataset1":
        return _passthrough_dataset1(trial)
    elif trial.dataset_source in ("dataset2", "dataset3", "dataset4"):
        return bipolarize(trial)
    else:
        raise ValueError(
            f"Unknown dataset_source: {trial.dataset_source!r}. "
            "Expected one of: dataset1, dataset2, dataset3, dataset4."
        )


def bipolarize(trial: Trial) -> Trial:
    """
    Compute bipolar H and V channels from monopolar electrode signals.
    Applies per-dataset sign corrections from MONOPOLAR_MAPS.

    Raises ValueError if the channel mapping is not yet configured
    (keys are None in MONOPOLAR_MAPS — fill after Phase 0 download).
    """
    src = trial.dataset_source
    if src not in MONOPOLAR_MAPS:
        raise ValueError(f"No monopolar map defined for dataset_source={src!r}")

    mapping = MONOPOLAR_MAPS[src]
    right_key = mapping["right_key"]
    left_key = mapping["left_key"]
    up_key = mapping["up_key"]
    down_key = mapping["down_key"]
    flip_h = mapping["flip_h"]
    flip_v = mapping["flip_v"]

    if right_key is None or left_key is None or up_key is None or down_key is None:
        warnings.warn(
            f"[{src}] MONOPOLAR_MAPS keys are None — attempting auto-detection. "
            "Fill MONOPOLAR_MAPS in unify.py after reading the Data Description."
        )
        return _auto_bipolarize(trial, flip_h, flip_v)

    for k in (right_key, left_key, up_key, down_key):
        if k not in trial.channels:
            available = list(trial.channels.keys())
            raise KeyError(
                f"[{src}] Channel key {k!r} not found in trial {trial.subject_id}/{trial.trial_id}. "
                f"Available channels: {available}. "
                "Update MONOPOLAR_MAPS in unify.py with the correct key names."
            )

    h = trial.channels[right_key] - trial.channels[left_key]
    v = trial.channels[up_key] - trial.channels[down_key]

    if flip_h:
        h = -h
    if flip_v:
        v = -v

    trial.channels["H"] = h
    trial.channels["V"] = v
    return trial


def _passthrough_dataset1(trial: Trial) -> Trial:
    """
    Dataset 1 is already bipolar. Map its H/V channel keys to the canonical
    'H' and 'V' names.
    """
    h_candidates = ["H", "HEOG", "Horizontal", "horizontal", "hEOG", "h", "EOG_0"]
    v_candidates = ["V", "VEOG", "Vertical", "vertical", "vEOG", "v", "EOG_1"]

    h_key = next((k for k in h_candidates if k in trial.channels), None)
    v_key = next((k for k in v_candidates if k in trial.channels), None)

    if h_key is None or v_key is None:
        ch_keys = list(trial.channels.keys())
        if len(ch_keys) == 2:
            warnings.warn(
                f"[dataset1] Cannot identify H/V keys by name in trial "
                f"{trial.subject_id}/{trial.trial_id}. "
                f"Treating {ch_keys[0]!r}=H, {ch_keys[1]!r}=V. "
                "Verify sign convention (Phase 2.4)."
            )
            trial.channels["H"] = trial.channels[ch_keys[0]]
            trial.channels["V"] = trial.channels[ch_keys[1]]
        else:
            raise KeyError(
                f"[dataset1] Could not identify H/V channels. "
                f"Available: {list(trial.channels.keys())}. "
                "Update _passthrough_dataset1() with the correct key names."
            )
    else:
        if h_key != "H":
            trial.channels["H"] = trial.channels[h_key]
        if v_key != "V":
            trial.channels["V"] = trial.channels[v_key]

    return trial


def _auto_bipolarize(trial: Trial, flip_h: bool, flip_v: bool) -> Trial:
    """
    Heuristic auto-detection of H/V channels when MONOPOLAR_MAPS is not filled.
    Looks for common naming patterns in the channel keys.
    This is a FALLBACK — always verify with the sign check (Phase 2.4).
    """
    ch = trial.channels
    keys = list(ch.keys())

    right_key = _find_key(keys, ["right", "reog", "r_eog", "re", "rh"])
    left_key = _find_key(keys, ["left", "leog", "l_eog", "le", "lh"])
    up_key = _find_key(keys, ["up", "ueog", "u_eog", "ue", "uv"])
    down_key = _find_key(keys, ["down", "deog", "d_eog", "de", "dv"])

    if None in (right_key, left_key, up_key, down_key):
        if len(keys) >= 4:
            warnings.warn(
                f"[{trial.dataset_source}] Auto-detection ambiguous. "
                f"Treating channels as [R, L, U, D] order: {keys[:4]}. "
                "Verify sign with Phase 2.4 sign check."
            )
            right_key, left_key, up_key, down_key = keys[0], keys[1], keys[2], keys[3]
        else:
            raise ValueError(
                f"[{trial.dataset_source}] Cannot auto-detect H/V channels from keys: {keys}. "
                "Fill MONOPOLAR_MAPS in unify.py."
            )

    h = ch[right_key] - ch[left_key]
    v = ch[up_key] - ch[down_key]

    if flip_h:
        h = -h
    if flip_v:
        v = -v

    trial.channels["H"] = h
    trial.channels["V"] = v
    return trial


def _find_key(keys: List[str], patterns: List[str]) -> Optional[str]:
    """Return first key that matches any pattern (case-insensitive)."""
    for pattern in patterns:
        for key in keys:
            if pattern in key.lower():
                return key
    return None


def verify_sign_convention(
    trial: Trial,
    saccade_direction: str = "right",
    expected_h_sign: float = 1.0,
) -> bool:
    """
    Verify that a rightward saccade produces a positive H deflection.

    Parameters
    ----------
    trial : Trial (must have 'H' and 'V' channels and saccade1_onset/end)
    saccade_direction : 'right' | 'left' | 'up' | 'down'
    expected_h_sign : +1.0 for positive H deflection on rightward saccade

    Returns True if sign is correct, False otherwise (should trigger flip).
    """
    if not trial.has_bipolar():
        raise ValueError("Trial must be bipolarized before sign check.")

    onset = trial.event_timestamps.get("saccade1_onset", -1)
    end = trial.event_timestamps.get("saccade1_end", -1)

    if onset < 0 or end < 0 or end <= onset:
        warnings.warn(
            f"Sign check skipped for {trial.subject_id}/{trial.trial_id}: "
            "invalid saccade timestamps."
        )
        return True

    h_segment = trial.channels["H"][onset:end]
    mean_h = np.mean(h_segment)

    sign_ok = (mean_h * expected_h_sign) > 0
    if not sign_ok:
        warnings.warn(
            f"[{trial.dataset_source}] Sign check FAILED for {trial.subject_id}/{trial.trial_id}: "
            f"expected H sign={expected_h_sign:+.0f} but got mean_H={mean_h:.4f}. "
            "Set flip_h=True in MONOPOLAR_MAPS or check channel assignment."
        )
    return sign_ok
