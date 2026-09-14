"""
src/evaluation/known_start.py
==============================
Known-start gaze evaluation replicating the protocol of Barbara et al. (BSPC 86,
2023; Section 4.5.3 and Appendix E). Kept separate from the main results: it is a
different, easier task, because every segment starts from the true gaze.

Protocol, as in the paper:
  * Ground-truth labels (App. E.1) come from the EOG. A sample is a fixation when
    the sample-to-sample difference of neither bipolar channel exceeds its
    threshold; a positive and a negative spike of the V difference above the blink
    threshold within gt_blink_max_ms label the samples between them as a blink;
    all other samples are saccades. Thresholds come from the subject's recording
    (the paper takes them from training data).
  * Short analysis: 1 s saccade windows (ControlSignal 1/2) and 2 s blink windows
    (ControlSignal 3) without subject mistakes. Each of n_subsets contiguous parts
    of the recording contributes its first short_saccade_windows saccade and
    short_blink_windows blink windows. Long analysis: each part contributes
    long_segments segments of long_trials consecutive trials, mistakes kept.
  * Error (App. E.2): per segment, mean |estimate − target| over its fixation
    samples after the response saccade (saccade windows) or over all its fixation
    samples (blink windows); per subject, mean over segments; mean ± SD across subjects.
  * Same subject: parameters fitted on one part, tested on another, all orderings.
    Unseen subject (not in the paper): fitted on the other subjects, with each
    subject's displacements divided by a label-free gain.
  * Outliers: the paper drops segments with "substantially high" error without
    giving a threshold. Here a segment is an outlier when its horizontal or vertical
    error lies beyond Tukey's far-out fence, Q3 + outlier_iqr_factor x IQR, computed
    per subject and per segment kind. Results are reported with and without them.

Estimators (both see only the EOG and the starting gaze; A is a 2-channel H/V
linear map without intercept, as in Barbara et al., BSPC 47, 2019):
  * detected saccades (signal differencing): eye movements detected from EOG velocity
    each add A · (level after − level before); while a movement is still settling the
    running level stands in for "level after". Whether a movement is a blink is
    decided from the whole movement, up to after_gap_ms + level_window_ms after it ends.
  * level change: A · (EOG now − EOG at segment start); short segments only.
  * detected saccades + head rotation (recordings with head pose, i.e. Dataset 3):
    detected saccades, plus minus vor_gain × the head's yaw (H) and pitch (V) change
    between detected movements. Gaze angles are in a face frame, so while the eyes
    hold a screen target a head rotation moves the gaze by minus that rotation, and
    the eyes follow with slow vestibulo-ocular (VOR) movements the detector does not
    see; inside a detected movement the EOG step already includes the head's share.
    It uses the detected-saccade A unchanged: the gaze shifts A is fitted on are the
    cue steps, taken before the head has moved.
  * fused (given per-sample absolute gaze estimates from a cross-subject model, see
    scripts/known_start_fusion.py): the detected-saccade estimate (with the head term
    when available) plus a causal first-order low-pass, time constant tau, of the
    absolute estimate minus it, starting from 0 at the known start. The saccade sum
    supplies fast changes and the absolute model the slow level, so errors that build
    up over a segment are pulled back. tau is chosen per axis from FUSION_TAUS_SEC on
    the fit data (inf = no fusion).
"""

from __future__ import annotations

import csv
import itertools
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

from src.config import CFG
from src.data.schema import Trial

FIXATION, SACCADE, BLINK = 0, 1, 2
FUSION_TAUS_SEC = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, np.inf)


def _ms(ms: float, fs: float) -> int:
    return max(1, int(round(ms * fs / 1000.0)))


def _runs(mask: np.ndarray):
    """Start (inclusive) and end (exclusive) indices of the True runs of a boolean array."""
    edges = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(np.int8)))
    return edges[::2], edges[1::2]


# ---------------------------------------------------------------------------
# Signals, estimator events and ground-truth labels
# ---------------------------------------------------------------------------

def lowpass_hv(trial: Trial) -> np.ndarray:
    """(2, n) H and V, zero-phase low-passed at CFG.known_start.lowpass_hz."""
    b, a = sp_signal.butter(4, CFG.known_start.lowpass_hz / (0.5 * trial.fs))
    return np.stack([sp_signal.filtfilt(b, a, np.asarray(trial.channels[c], dtype=np.float64))
                     for c in ("H", "V")])


def detect_eye_movements(hv: np.ndarray, fs: float) -> Dict[str, np.ndarray]:
    """
    Estimator-side eye-movement detection (velocity threshold, blink rejection).

    Speed is the Euclidean norm of the per-channel velocity divided by that
    channel's robust velocity SD, so thresholds are gain-free. An event is a run
    where speed exceeds speed_offset_sd that contains a sample above
    speed_onset_sd; events closer than merge_ms are merged. An event is a blink
    when its vertical excursion exceeds blink_excursion_ratio x its net vertical
    displacement AND the part that came back exceeds blink_return_scale x the
    median |vertical displacement| of events shorter than saccade_max_ms; two
    close events whose vertical displacements cancel are blinks too.

    Returns onset, offset (sample indices), before (n_events, 2) = median level
    before, delta (n_events, 2) = median level after − before, is_blink, excursion_v.
    """
    cfg = CFG.known_start
    n = hv.shape[1]
    vel = np.gradient(hv, axis=1)
    scale = 1.4826 * np.median(np.abs(vel - np.median(vel, axis=1, keepdims=True)), axis=1) + 1e-12
    speed = np.sqrt(((vel / scale[:, None]) ** 2).sum(axis=0))

    starts, ends = _runs(speed > cfg.speed_offset_sd)
    merged: List[List[int]] = []
    for s, e in zip(starts, ends):
        if not (speed[s:e] > cfg.speed_onset_sd).any():
            continue
        if merged and s - merged[-1][1] < _ms(cfg.merge_ms, fs):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    merged = [(s, e) for s, e in merged if e - s >= _ms(cfg.min_event_ms, fs)]
    if not merged:
        return {"onset": np.zeros(0, np.int64), "offset": np.zeros(0, np.int64),
                "before": np.zeros((0, 2)), "delta": np.zeros((0, 2)),
                "is_blink": np.zeros(0, bool), "excursion_v": np.zeros(0)}

    onset = np.array([s for s, _ in merged])
    offset = np.array([e for _, e in merged])
    level_win, gap = _ms(cfg.level_window_ms, fs), _ms(cfg.after_gap_ms, fs)
    before = np.zeros((len(onset), 2))
    delta = np.zeros((len(onset), 2))
    excursion_v = np.zeros(len(onset))
    for i, (s, e) in enumerate(zip(onset, offset)):
        lo = max(s - level_win, offset[i - 1] if i else 0, 0)
        hi = min(e + gap + level_win, onset[i + 1] if i + 1 < len(onset) else n, n)
        before[i] = np.median(hv[:, lo:s] if s > lo else hv[:, max(s - 1, 0):s + 1], axis=1)
        a0 = min(e + gap, hi - 1)
        delta[i] = np.median(hv[:, a0:hi], axis=1) - before[i]
        excursion_v[i] = np.max(np.abs(hv[1, s:a0 + 1] - before[i, 1]))

    # Blink: the vertical excursion mostly comes back. "Mostly" is scaled by the subject's
    # median vertical displacement of short (saccade-like) events, so the rule is gain-free
    # and never fires on horizontal saccades whose net vertical change is just noise.
    abs_dv = np.abs(delta[:, 1])
    saccade_like = (offset - onset) < _ms(cfg.saccade_max_ms, fs)
    v_scale = np.median(abs_dv[saccade_like]) if saccade_like.any() else np.median(abs_dv)
    is_blink = ((excursion_v - abs_dv > cfg.blink_return_scale * v_scale)
                & (excursion_v > cfg.blink_excursion_ratio * abs_dv))
    pair_gap = _ms(cfg.blink_pair_ms, fs)
    for i in range(len(onset) - 1):
        dv1, dv2 = delta[i, 1], delta[i + 1, 1]
        if (onset[i + 1] - offset[i] < pair_gap and dv1 * dv2 < 0
                and abs(dv1 + dv2) < cfg.blink_pair_residual * max(abs(dv1), abs(dv2))):
            is_blink[i] = is_blink[i + 1] = True
    return {"onset": onset, "offset": offset, "before": before, "delta": delta,
            "is_blink": is_blink, "excursion_v": excursion_v}


def ground_truth_labels(hv: np.ndarray, fs: float, control: np.ndarray) -> np.ndarray:
    """
    Per-sample FIXATION / SACCADE / BLINK labels from the EOG (paper Appendix E.1).

    Fixation: |sample difference| <= gt_fixation_sd robust SDs on both channels
    (fixation gaps shorter than gt_merge_ms inside movements count as movement).
    Blink: a V-difference spike above the blink threshold followed within
    gt_blink_max_ms by an opposite spike above gt_blink_second_ratio x threshold;
    the samples from the first peak to the largest opposite peak in that time. The
    blink threshold is gt_blink_ratio x the
    median largest |V difference| inside the protocol's blink intervals.
    """
    cfg = CFG.known_start
    diff = np.diff(hv, axis=1, prepend=hv[:, :1])
    sd = 1.4826 * np.median(np.abs(diff - np.median(diff, axis=1, keepdims=True)), axis=1) + 1e-12
    moving = np.any(np.abs(diff) > cfg.gt_fixation_sd * sd[:, None], axis=0)
    starts, ends = _runs(~moving)
    for s, e in zip(starts, ends):
        if 0 < s and e < len(moving) and e - s < _ms(cfg.gt_merge_ms, fs):
            moving[s:e] = True
    labels = np.where(moving, SACCADE, FIXATION).astype(np.int8)

    dv = diff[1]
    bs, be = _runs(control == 3)
    peaks = [np.max(np.abs(dv[s:e])) for s, e in zip(bs, be) if e > s]
    if not peaks:
        return labels
    threshold = cfg.gt_blink_ratio * float(np.median(peaks))
    second = cfg.gt_blink_second_ratio * threshold
    spacing = _ms(20, fs)
    first = {+1: sp_signal.find_peaks(dv, height=threshold, distance=spacing)[0],
             -1: sp_signal.find_peaks(-dv, height=threshold, distance=spacing)[0]}
    closing = {+1: sp_signal.find_peaks(dv, height=second, distance=spacing)[0],
               -1: sp_signal.find_peaks(-dv, height=second, distance=spacing)[0]}
    window = _ms(cfg.gt_blink_max_ms, fs)
    blink_end = -1
    for p, sign in sorted([(p, +1) for p in first[+1]] + [(p, -1) for p in first[-1]]):
        if p <= blink_end:
            continue  # a spike inside the blink just labelled, e.g. its closing spike
        opposite = closing[-sign]
        lo = np.searchsorted(opposite, p, side="right")
        hi = np.searchsorted(opposite, p + window, side="right")
        if hi > lo:
            q = opposite[lo:hi][np.argmax(-sign * dv[opposite[lo:hi]])]
            labels[p:q + 1] = BLINK
            blink_end = q
    return labels


# ---------------------------------------------------------------------------
# Protocol windows
# ---------------------------------------------------------------------------

def protocol_windows(labels: np.ndarray, control: np.ndarray, target: np.ndarray, fs: float) -> List[Dict]:
    """
    Saccade windows (ControlSignal 1/2 intervals) and blink windows (ControlSignal 3
    intervals), each with its trial index, starting gaze, scored fixation samples and
    whether the subject made a mistake in it.

    Saccade-window mistakes: a labelled blink; no saccade starting response_min_ms to
    response_max_ms after the cue; a saccade in the first response_min_ms (anticipation)
    or starting in the last premature_ms (premature next saccade); fewer than
    min_scored_ms of fixation samples after the response. Blink-window mistakes: not
    exactly one labelled blink; a saccade not within blink_margin_ms of it; fewer than
    min_scored_ms of fixation samples. Saccade runs shorter than gt_min_saccade_ms are
    ignored in these checks.
    """
    cfg = CFG.known_start
    n = len(labels)
    bounds = np.r_[0, np.flatnonzero(np.diff(control[:n]) != 0) + 1, n]
    s_start, s_end = _runs(labels == SACCADE)
    long_enough = (s_end - s_start) >= _ms(cfg.gt_min_saccade_ms, fs)
    s_start, s_end = s_start[long_enough], s_end[long_enough]
    b_start, b_end = _runs(labels == BLINK)
    r_min, r_max = _ms(cfg.response_min_ms, fs), _ms(cfg.response_max_ms, fs)
    premature, margin, min_scored = _ms(cfg.premature_ms, fs), _ms(cfg.blink_margin_ms, fs), _ms(cfg.min_scored_ms, fs)

    windows, trial = [], -1
    for a, b in zip(bounds[:-1], bounds[1:]):
        kind = int(control[a])
        trial += kind == 1
        if a == 0 or trial < 0 or kind not in (1, 2, 3):
            continue
        fixation = a + np.flatnonzero(labels[a:b] == FIXATION)
        in_window = (s_start >= a) & (s_start < b)
        if kind in (1, 2):
            response = in_window & (s_start >= a + r_min) & (s_start <= a + r_max)
            after = s_end[response].max() if response.any() else a
            scored = fixation[fixation >= after]
            mistake = bool((labels[a:b] == BLINK).any() or not response.any()
                           or (in_window & (s_start < a + r_min)).any()
                           or ((s_start < a) & (s_end > a)).any()
                           or (in_window & (s_start >= b - premature)).any()
                           or len(scored) < min_scored)
            windows.append({"kind": "saccade", "start": a, "end": b, "trial": trial,
                            "gaze0": target[a - 1], "shift": target[a] - target[a - 1],
                            "response": (int(s_start[response].min()), int(after)) if response.any() else None,
                            "has_blink": bool((labels[a:b] == BLINK).any()),
                            "scored": scored, "mistake": mistake})
        else:
            blinks = (b_start < b) & (b_end > a)
            near = np.zeros(len(s_start), bool)
            for bs, be in zip(b_start[blinks], b_end[blinks]):
                near |= (s_start < be + margin) & (s_end > bs - margin)
            scored = fixation
            mistake = bool(blinks.sum() != 1 or (in_window & ~near).any() or len(scored) < min_scored)
            windows.append({"kind": "blink", "start": a, "end": b, "trial": trial,
                            "gaze0": target[a - 1], "shift": target[a] - target[a - 1],
                            "blink": (int(b_start[blinks][0]), int(b_end[blinks][0])) if blinks.any() else None,
                            "scored": scored, "mistake": mistake})
    return windows


def _estimator_disagrees(w: Dict, events: Dict) -> bool:
    """
    A labelled blink overlapped by a kept movement, or, in a window without a labelled
    blink, a labelled response saccade dropped as a blink. (In a window with a real
    blink the labels mark its strokes as saccade samples, and dropping it is right.)
    """
    overlap = lambda lo, hi: (events["onset"] < hi) & (events["offset"] > lo)
    if w["kind"] == "blink" and w["blink"] is not None:
        return bool((overlap(*w["blink"]) & ~events["is_blink"]).any())
    if w["kind"] == "saccade" and w["response"] is not None and not w["has_blink"]:
        return bool((overlap(*w["response"]) & events["is_blink"]).any())
    return False


def prepare_recording(trial: Trial, absolute: Optional[np.ndarray] = None) -> Dict:
    """
    Signals, estimator events, ground-truth labels, windows, short and long subsets.
    `absolute`: optional (n_samples, 2) causal gaze estimates from a model that never
    saw this subject (NaN where none exists yet), for the fused estimator.
    """
    cfg = CFG.known_start
    fs = trial.fs
    hv = lowpass_hv(trial)
    target = np.asarray(trial.target_angle, dtype=np.float64)
    control = np.asarray(trial.channels["ControlSignal"])
    n = min(hv.shape[1], len(target), len(control))
    hv, target, control = hv[:, :n], target[:n], control[:n]

    events = detect_eye_movements(hv, fs)
    labels = ground_truth_labels(hv, fs, control)
    windows = protocol_windows(labels, control, target, fs)
    for w in windows:
        w["disagrees"] = _estimator_disagrees(w, events)

    trials = sorted({w["trial"] for w in windows})
    parts = [set(p.tolist()) for p in np.array_split(np.array(trials), cfg.n_subsets)]
    short, long_segments = [], []
    for part in parts:
        in_part = [w for w in windows if w["trial"] in part]
        short.append([w for w in in_part if w["kind"] == "saccade" and not w["mistake"]][: cfg.short_saccade_windows]
                     + [w for w in in_part if w["kind"] == "blink" and not w["mistake"]][: cfg.short_blink_windows])
        ordered = sorted(part)
        segs = []
        for k in range(cfg.long_segments):
            seg_trials = set(ordered[k * cfg.long_trials:(k + 1) * cfg.long_trials])
            if len(seg_trials) == cfg.long_trials:
                segs.append([w for w in in_part if w["trial"] in seg_trials])
        long_segments.append(segs)

    keep = ~events["is_blink"]
    level_win = _ms(cfg.level_window_ms, fs)
    return {
        "subject_id": trial.subject_id, "fs": fs, "hv": hv, "target": target, "labels": labels,
        "windows": windows, "short": short, "long": long_segments, "events": events,
        "mv_onset": events["onset"][keep], "mv_before": events["before"][keep], "mv_delta": events["delta"][keep],
        "mv_known": events["offset"][keep] + _ms(cfg.after_gap_ms, fs) + level_win,
        "level_win": level_win,
        "vor": _head_rotation_between_movements(trial, events, n, fs),
        "absolute": None if absolute is None else np.vstack(
            [np.asarray(absolute, dtype=np.float64)[:n], np.full((max(0, n - len(absolute)), 2), np.nan)]),
    }


# ---------------------------------------------------------------------------
# Estimators, fitting and errors
# ---------------------------------------------------------------------------

def _saccade_displacement(d: Dict, start: int, idx: np.ndarray) -> np.ndarray:
    """(len(idx), 2) summed EOG displacement of movements starting at or after `start`."""
    first = np.searchsorted(d["mv_onset"], start)
    onset, known = d["mv_onset"][first:], d["mv_known"][first:]
    before, delta = d["mv_before"][first:], d["mv_delta"][first:]
    if len(onset) == 0:
        return np.zeros((len(idx), 2))
    csum = np.r_[np.zeros((1, 2)), np.cumsum(delta, axis=0)]
    last = np.searchsorted(onset, idx, side="right") - 1
    disp = csum[np.searchsorted(known, idx, side="right")]
    settling = (last >= 0) & (known[np.maximum(last, 0)] > idx)
    j = last[settling]
    disp[settling] = csum[j] + d["hv"][:, idx[settling]].T - before[j]
    return disp


def _level_displacement(d: Dict, start: int, idx: np.ndarray) -> np.ndarray:
    z0 = np.median(d["hv"][:, max(start - d["level_win"], 0):start], axis=1)
    return d["hv"][:, idx].T - z0


def _head_rotation_between_movements(trial: Trial, events: Dict, n: int, fs: float) -> Optional[np.ndarray]:
    """
    (2, n) running sum of the head's yaw (row 0, paired with H) and pitch (row 1, V)
    change in degrees, counting only samples outside detected eye movements (each
    from onset to the end of its after-level window); None without head pose.
    """
    if trial.head_pose is None:
        return None
    cfg = CFG.known_start
    pose = np.asarray(trial.head_pose, dtype=np.float64)
    pose = (pose if pose.shape[0] == 3 else pose.T)[:2, :n]
    for axis in pose:
        bad = ~np.isfinite(axis)
        if bad.any():
            axis[bad] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), axis[~bad])
    b, a = sp_signal.butter(4, cfg.lowpass_hz / (0.5 * fs))
    step = np.diff(sp_signal.filtfilt(b, a, pose, axis=1), axis=1, prepend=pose[:, :1])
    step[:, 0] = 0.0
    moving = np.zeros(n, dtype=bool)
    ends = events["offset"] + _ms(cfg.after_gap_ms, fs) + _ms(cfg.level_window_ms, fs)
    for onset, end in zip(events["onset"], ends):
        moving[onset:end] = True
    step[:, moving] = 0.0
    return np.cumsum(step, axis=1)


def _vor_displacement(d: Dict, start: int, idx: np.ndarray) -> np.ndarray:
    """(len(idx), 2) gaze change since `start` from head rotation between movements."""
    return -CFG.known_start.vor_gain * (d["vor"][:, idx] - d["vor"][:, [start]]).T


def _estimators(d: Dict) -> Tuple[str, ...]:
    return (("saccades", "level") + (("saccades_vor",) if d.get("vor") is not None else ())
            + (("fused",) if d.get("absolute") is not None else ()))


def _fused_predictions(d: Dict, windows: List[Dict], A: np.ndarray, gain: np.ndarray, taus) -> tuple:
    """
    (len(taus), n_scored, 2) fused estimates at the segment's scored samples, one per
    time constant, and (n_scored, 2) targets; None when nothing is scored.
    """
    start, gaze0 = windows[0]["start"], windows[0]["gaze0"]
    idx = np.concatenate([w["scored"] for w in windows])
    if len(idx) == 0:
        return None
    span = np.arange(start, idx.max() + 1)
    base = gaze0 + (_saccade_displacement(d, start, span) / gain) @ A.T
    if d.get("vor") is not None:
        base = base + _vor_displacement(d, start, span)
    diff = d["absolute"][span] - base
    diff[~np.isfinite(diff)] = 0.0  # no absolute estimate yet: no correction
    at = idx - start
    out = np.empty((len(taus), len(idx), 2))
    for i, tau in enumerate(taus):
        if np.isinf(tau):
            out[i] = base[at]
            continue
        alpha = 1.0 - np.exp(-1.0 / (tau * d["fs"]))
        out[i] = base[at] + sp_signal.lfilter([alpha], [1.0, alpha - 1.0], diff, axis=0)[at]
    return out, d["target"][idx]


def _choose_taus(fit_segments: List[tuple], A: np.ndarray) -> np.ndarray:
    """Per axis, the FUSION_TAUS_SEC value with the lowest mean error over (recording, windows, gain) segments."""
    errors = []
    for d, windows, gain in fit_segments:
        fused = _fused_predictions(d, windows, A, gain, FUSION_TAUS_SEC)
        if fused is not None:
            errors.append(np.abs(fused[0] - fused[1]).mean(axis=1))
    if not errors:
        return np.array([np.inf, np.inf])
    return np.array(FUSION_TAUS_SEC)[np.mean(errors, axis=0).argmin(axis=0)]


def _fit_no_intercept(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """A (2 x 2) minimising ||X A' − Y||; predictions are X @ A.T."""
    return np.linalg.lstsq(X, Y, rcond=None)[0].T


def training_pairs(d: Dict, windows: List[Dict], gain: np.ndarray) -> Dict[str, tuple]:
    """Per saccade window: (EOG displacement / gain, gaze shift) for both fitted estimators."""
    sacc = [w for w in windows if w["kind"] == "saccade" and len(w["scored"])]
    if not sacc:
        return {"saccades": (np.zeros((0, 2)), np.zeros((0, 2))), "level": (np.zeros((0, 2)), np.zeros((0, 2)))}
    Y = np.array([w["shift"] for w in sacc])
    X_sac = np.array([_saccade_displacement(d, w["start"], np.array([w["end"] - 1]))[0] for w in sacc])
    X_lvl = np.array([_level_displacement(d, w["start"], w["scored"]).mean(axis=0) for w in sacc])
    return {"saccades": (X_sac / gain, Y), "level": (X_lvl / gain, Y)}


def _fit_estimators(pairs: Dict[str, tuple], d: Dict, fit_segments: Optional[List[tuple]] = None,
                    tau_log: Optional[list] = None) -> Dict[str, np.ndarray]:
    """
    A per estimator; the head-rotation estimator shares the detected-saccade A. With
    absolute estimates, "fused" gets that A and per-axis time constants chosen on
    fit_segments, a list of (recording, windows, gain); the choice is appended to tau_log.
    """
    A = {est: _fit_no_intercept(*pairs[est]) for est in ("saccades", "level")}
    if d.get("vor") is not None:
        A["saccades_vor"] = A["saccades"]
    if d.get("absolute") is not None:
        taus = _choose_taus(fit_segments or [], A["saccades"])
        A["fused"] = (A["saccades"], taus)
        if tau_log is not None:
            tau_log.append(taus.tolist())
    return A


def segment_error(d: Dict, windows: List[Dict], A: np.ndarray, gain: np.ndarray, estimator: str) -> np.ndarray:
    """Mean |estimate − target| over the scored samples of consecutive windows from one known start."""
    if estimator == "fused":
        A_saccades, taus = A
        pred, target = _fused_predictions(d, windows, A_saccades, gain, tuple(taus))
        return np.array([np.abs(pred[0, :, 0] - target[:, 0]).mean(), np.abs(pred[1, :, 1] - target[:, 1]).mean()])
    start, gaze0 = windows[0]["start"], windows[0]["gaze0"]
    idx = np.concatenate([w["scored"] for w in windows])
    disp = (_level_displacement if estimator == "level" else _saccade_displacement)(d, start, idx)
    pred = gaze0 + (disp / gain) @ A.T
    if estimator == "saccades_vor":
        pred = pred + _vor_displacement(d, start, idx)
    return np.abs(pred - d["target"][idx]).mean(axis=0)


def _label_free_gain(d: Dict) -> np.ndarray:
    """Per-channel spread (90th − 10th percentile) of the subject's non-blink movement displacements."""
    return np.subtract(*np.percentile(d["mv_delta"], [90, 10], axis=0)) + 1e-12


def _evaluate(d: Dict, A: Dict[str, np.ndarray], gain: np.ndarray, short_windows: List[Dict],
              long_segments: List[List[Dict]], acc: Dict[str, list]) -> None:
    for w in short_windows:
        for est in _estimators(d):
            acc[f"short_{est}"].append((segment_error(d, [w], A[est], gain, est), w["kind"]))
    for seg in long_segments:
        for est in _estimators(d):
            if est != "level":
                acc[f"long_{est}"].append((segment_error(d, seg, A[est], gain, est), "segment"))


def _outliers(errs: np.ndarray) -> np.ndarray:
    """Segments whose H or V error lies beyond Tukey's far-out fence, Q3 + k x IQR."""
    q1, q3 = np.percentile(errs, [25, 75], axis=0)
    return np.any(errs > q3 + CFG.known_start.outlier_iqr_factor * (q3 - q1), axis=1)


def _summary(per_subject: List[list]) -> Dict[str, float]:
    full, kept, excluded, total = [], [], 0, 0
    by_kind = {"saccade": [], "blink": []}
    for segments in per_subject:
        if not segments:
            continue
        errs = np.array([e for e, _ in segments])
        kinds = np.array([k for _, k in segments])
        flags = np.zeros(len(errs), dtype=bool)
        for kind in np.unique(kinds):  # blink windows score near 0, so fence each kind separately
            flags[kinds == kind] = _outliers(errs[kinds == kind])
        full.append(errs.mean(axis=0))
        kept.append(errs[~flags].mean(axis=0) if (~flags).any() else errs.mean(axis=0))
        for kind, values in by_kind.items():
            if (kinds == kind).any():
                values.append(errs[kinds == kind].mean(axis=0))
        excluded += int(flags.sum())
        total += len(flags)
    full, kept = np.array(full), np.array(kept)
    out = {
        "mae_h_deg": float(full[:, 0].mean()), "mae_v_deg": float(full[:, 1].mean()),
        "mae_h_sd_deg": float(full[:, 0].std()), "mae_v_sd_deg": float(full[:, 1].std()),
        "excluded_fraction": excluded / total if total else 0.0,
        "excluded_mae_h_deg": float(kept[:, 0].mean()), "excluded_mae_v_deg": float(kept[:, 1].mean()),
        "excluded_mae_h_sd_deg": float(kept[:, 0].std()), "excluded_mae_v_sd_deg": float(kept[:, 1].std()),
        "n_segments": total,
    }
    for kind, values in by_kind.items():
        if values:
            v = np.array(values)
            out[f"{kind}_windows_mae_h_deg"] = float(v[:, 0].mean())
            out[f"{kind}_windows_mae_v_deg"] = float(v[:, 1].mean())
    return out


def _protocol_stats(data: List[Dict]) -> Dict[str, float]:
    fs_ms = lambda d, ms: _ms(ms, d["fs"])
    rows = []
    for d in data:
        labels, windows, ev = d["labels"], d["windows"], d["events"]
        sacc = [w for w in windows if w["kind"] == "saccade"]
        blink = [w for w in windows if w["kind"] == "blink"]
        lo, hi = fs_ms(d, CFG.known_start.response_min_ms), fs_ms(d, CFG.known_start.response_max_ms)
        cue = np.array([w["start"] for w in sacc])
        onset = d["mv_onset"]
        k = np.searchsorted(cue, onset, side="right") - 1
        lag = onset - cue[np.maximum(k, 0)]
        near = (k >= 0) & (lag >= lo) & (lag <= hi)
        responded = np.zeros(len(cue), bool)
        responded[k[near]] = True
        rows.append({
            "label_fixation_fraction": float(np.mean(labels == FIXATION)),
            "label_saccade_fraction": float(np.mean(labels == SACCADE)),
            "label_blink_fraction": float(np.mean(labels == BLINK)),
            "saccade_windows_mistake_free": float(np.mean([not w["mistake"] for w in sacc])),
            "blink_windows_mistake_free": float(np.mean([not w["mistake"] for w in blink])),
            "short_saccade_windows_per_part": float(np.mean([sum(w["kind"] == "saccade" for w in p) for p in d["short"]])),
            "short_blink_windows_per_part": float(np.mean([sum(w["kind"] == "blink" for w in p) for p in d["short"]])),
            "long_segments_per_part": float(np.mean([len(p) for p in d["long"]])),
            "detector_cue_recall": float(responded.mean()),
            "detector_event_precision": float(near.mean()) if len(onset) else float("nan"),
            "blink_windows_blink_counted_as_movement": float(np.mean([w["disagrees"] for w in blink if w["blink"] is not None])),
            "saccade_windows_response_dropped_as_blink": float(np.mean([w["disagrees"] for w in sacc if w["response"] is not None])),
        })
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}


def run_known_start_protocol(trials: List[Trial], absolute: Optional[Dict[tuple, np.ndarray]] = None) -> Dict:
    """
    Evaluate the paper's known-start protocol on continuous recordings with targets and
    ControlSignal. `absolute` maps (subject_id, trial_id) to per-sample cross-subject gaze
    estimates and adds the fused estimator.
    """
    cfg = CFG.known_start
    absolute = absolute or {}
    data = [prepare_recording(t, absolute.get((t.subject_id, t.trial_id))) for t in trials
            if t.target_angle is not None and "ControlSignal" in t.channels and t.has_bipolar()]
    if not data:
        raise ValueError("Known-start protocol needs trials with target_angle, ControlSignal, H and V.")
    subjects = sorted({d["subject_id"] for d in data})
    by_subject = {s: [d for d in data if d["subject_id"] == s] for s in subjects}
    has_head = all(d["vor"] is not None for d in data)
    if not has_head:  # all recordings or none, so every subject has the same rows
        for d in data:
            d["vor"] = None
    has_absolute = all(d["absolute"] is not None for d in data)
    if not has_absolute:
        for d in data:
            d["absolute"] = None
    keys = ("short_saccades", "short_level", "long_saccades")
    keys += ("short_saccades_vor", "long_saccades_vor") if has_head else ()
    keys += ("short_fused", "long_fused") if has_absolute else ()
    tau_log = {fit: {"short": [], "long": []} for fit in ("same_subject", "unseen_subject")}
    ones = np.ones(2)

    same = {k: [] for k in keys}
    for s in subjects:
        acc = {k: [] for k in keys}
        for d in by_subject[s]:
            for fit_i, test_i in itertools.permutations(range(cfg.n_subsets), 2):
                short_pairs = training_pairs(d, d["short"][fit_i], ones)
                long_pairs = training_pairs(d, [w for seg in d["long"][fit_i] for w in seg], ones)
                A_short = _fit_estimators(short_pairs, d, [(d, [w], ones) for w in d["short"][fit_i]],
                                          tau_log["same_subject"]["short"])
                A_long = _fit_estimators(long_pairs, d, [(d, seg, ones) for seg in d["long"][fit_i]],
                                         tau_log["same_subject"]["long"])
                _evaluate(d, A_short, ones, d["short"][test_i], [], acc)
                _evaluate(d, A_long, ones, [], d["long"][test_i], acc)
        for k in keys:
            same[k].append(acc[k])

    gain = {id(d): _label_free_gain(d) for d in data}
    unseen = {k: [] for k in keys}
    for s in subjects:
        others = [d for o in subjects if o != s for d in by_subject[o]]
        pairs = [training_pairs(d, [w for part in d["short"] for w in part], gain[id(d)]) for d in others]
        pooled = {est: (np.vstack([p[est][0] for p in pairs]), np.vstack([p[est][1] for p in pairs]))
                  for est in ("saccades", "level")}
        fit_short = [(o, [w], gain[id(o)]) for o in others for part in o["short"] for w in part] if has_absolute else None
        fit_long = [(o, seg, gain[id(o)]) for o in others for part in o["long"] for seg in part] if has_absolute else None
        A_short = _fit_estimators(pooled, others[0], fit_short, tau_log["unseen_subject"]["short"])
        A_long = _fit_estimators(pooled, others[0], fit_long, tau_log["unseen_subject"]["long"])
        acc = {k: [] for k in keys}
        for d in by_subject[s]:
            _evaluate(d, A_short, gain[id(d)], [w for part in d["short"] for w in part], [], acc)
            _evaluate(d, A_long, gain[id(d)], [], [seg for part in d["long"] for seg in part], acc)
        for k in keys:
            unseen[k].append(acc[k])

    return {
        "protocol": "known start, replicating Barbara et al. BSPC 86 (2023) Sec. 4.5.3 and App. E: "
                    "EOG-derived fixation/saccade/blink labels, mistake-free short windows, 8-trial long "
                    "segments, per-sample fixation MAE per segment, mean ± SD across subjects",
        "datasets": sorted({t.dataset_source for t in trials}),
        "subjects": subjects,
        "config": dict(vars(cfg)),
        "protocol_stats": _protocol_stats(data),
        "same_subject": {k: _summary(v) for k, v in same.items()},
        "unseen_subject": {k: _summary(v) for k, v in unseen.items()},
        **({"fusion_taus_sec": {fit: {length: _tau_counts(log) for length, log in logs.items()}
                                for fit, logs in tau_log.items()}} if has_absolute else {}),
    }


def _tau_counts(log: list) -> Dict[str, Dict[str, int]]:
    """How often each time constant was chosen, per axis."""
    arr = np.array(log, dtype=np.float64).reshape(-1, 2)
    return {axis: {f"{tau:g}": int(np.sum(arr[:, i] == tau)) for tau in FUSION_TAUS_SEC if np.any(arr[:, i] == tau)}
            for i, axis in enumerate(("h", "v"))}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Published known-start results: (method, segments, fixation MAE H, V, excluded outlier share).
PAPER_KNOWN_START = {
    "dataset2": [  # Barbara et al., BSPC 86 (2023), Tables 2-3
        ("Published: dual Kalman filter", "short 1-2 s", 1.64, 1.97, 0.0685),
        ("Published: signal differencing [BSPC 47, 2019]", "short 1-2 s", 1.51, 1.95, 0.0685),
        ("Published: dual Kalman filter", "long 32 s", 5.23, 6.59, 0.0583),
        ("Published: signal differencing [BSPC 47, 2019]", "long 32 s", 5.82, 8.04, 0.0583),
    ],
    "dataset3": [  # Barbara et al., BSPC 90 (2024), Tables 2-3
        ("Published: dual Kalman filter + VOR model", "short 1-2 s", 1.85, 2.19, 0.0777),
        ("Published: signal differencing [BSPC 47, 2019]", "short 1-2 s", 3.59, 2.52, 0.0777),
        ("Published: dual Kalman filter + VOR model", "long 32 s", 4.64, 6.10, 0.0391),
        ("Published: signal differencing [BSPC 47, 2019]", "long 32 s", 8.13, 11.25, 0.0391),
    ],
}

_ROW_LABELS = {
    "short_saccades": ("detected saccades", "short 1-2 s"),
    "short_saccades_vor": ("detected saccades + head rotation", "short 1-2 s"),
    "short_level": ("level change since start", "short 1-2 s"),
    "long_saccades": ("detected saccades", "long 32 s"),
    "long_saccades_vor": ("detected saccades + head rotation", "long 32 s"),
    "short_fused": ("fused with cross-subject XGBoost", "short 1-2 s"),
    "long_fused": ("fused with cross-subject XGBoost", "long 32 s"),
}


def known_start_rows(results: Dict) -> List[Dict]:
    rows = []
    for fit_key, fit_label in (("same_subject", "same subject"), ("unseen_subject", "unseen subject")):
        for key, (method, segments) in _ROW_LABELS.items():
            if key not in results[fit_key]:
                continue
            r = results[fit_key][key]
            rows.append({"method": f"Known start, {method}", "fit": fit_label, "segments": segments,
                         "mae_h_deg": r["mae_h_deg"], "mae_v_deg": r["mae_v_deg"],
                         "mae_h_sd_deg": r["mae_h_sd_deg"], "mae_v_sd_deg": r["mae_v_sd_deg"],
                         "excluded_mae_h_deg": r["excluded_mae_h_deg"], "excluded_mae_v_deg": r["excluded_mae_v_deg"],
                         "excluded_fraction": r["excluded_fraction"]})
    datasets = results.get("datasets", [])
    published = PAPER_KNOWN_START.get(datasets[0], []) if len(datasets) == 1 else []
    for method, segments, h, v, frac in published:
        rows.append({"method": method, "fit": "same subject", "segments": segments,
                     "mae_h_deg": None, "mae_v_deg": None, "mae_h_sd_deg": None, "mae_v_sd_deg": None,
                     "excluded_mae_h_deg": h, "excluded_mae_v_deg": v, "excluded_fraction": frac})
    return rows


def run_known_start(trials: List[Trial], results_dir: str = None,
                    absolute: Optional[Dict[tuple, np.ndarray]] = None) -> Dict:
    """Run the protocol, print the table, and save known_start_protocol.json + known_start_table.csv."""
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)
    results = run_known_start_protocol(trials, absolute)
    with open(os.path.join(results_dir, "known_start_protocol.json"), "w") as f:
        json.dump(results, f, indent=2)
    rows = known_start_rows(results)
    with open(os.path.join(results_dir, "known_start_table.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fmt = lambda v: f"{v:5.2f}" if v is not None else "   - "
    print("\nProtocol stats:")
    for k, v in results["protocol_stats"].items():
        print(f"  {k:40s} {v:.3f}")
    print(f"\n{'Method':<48} {'Fit':<15} {'Segments':<12} {'MAE H / V':>13} {'excluded H / V':>16} {'excl.':>6}")
    for r in rows:
        print(f"{r['method']:<48} {r['fit']:<15} {r['segments']:<12} {fmt(r['mae_h_deg'])} / {fmt(r['mae_v_deg'])}"
              f"   {fmt(r['excluded_mae_h_deg'])} / {fmt(r['excluded_mae_v_deg'])} {r['excluded_fraction']:6.1%}")
    print(f"Saved: {os.path.join(results_dir, 'known_start_protocol.json')}")
    return results
