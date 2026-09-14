"""
src/data/loaders.py
===================
Per-dataset raw loaders. Each returns a list[Trial] with raw (pre-unification)
channel data read directly from disk.

IMPORTANT: fs, channel names, and subject counts are read from the actual
Data Description files bundled with each dataset — never hard-coded here.
If a dataset directory is empty or missing, a clear FileNotFoundError is raised
with instructions on where to download the data.

Dataset download URLs (Phase 0):
  Dataset 1 (Zero-Centred Bipolar):  https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/
  Dataset 2 (Monopolar Stationary):  same page
  Dataset 3 (Monopolar Non-Stationary): same page
  Dataset 4 (Monopolar Isotropic):   Google Drive link — MANUAL download required
"""

from __future__ import annotations

import os
import re
import glob
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import scipy.io as sio

from src.data.schema import Trial, make_empty_event_timestamps
from src.config import CFG


def _check_dir(path: str, dataset_name: str, download_hint: str = "") -> None:
    """Raise FileNotFoundError if path is empty or missing."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"\n[{dataset_name}] Directory not found: {path}\n"
            f"Please download and extract the dataset ZIP there.\n"
            + (f"Download: {download_hint}" if download_hint else "")
        )
    files = [f for f in os.listdir(path) if not f.startswith(".")]
    if len(files) == 0:
        raise FileNotFoundError(
            f"\n[{dataset_name}] Directory exists but is EMPTY: {path}\n"
            f"Please extract the dataset ZIP into this folder.\n"
            + (f"Download: {download_hint}" if download_hint else "")
        )


def _load_mat_safe(filepath: str) -> dict:
    """Load .mat file, trying scipy first, then h5py for v7.3 files."""
    try:
        return sio.loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        try:
            import h5py
            data = {}
            with h5py.File(filepath, "r") as f:
                def _visit(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        data[name.replace("/", "_")] = np.array(obj)
                f.visititems(_visit)
            return data
        except ImportError:
            raise ImportError(
                "This .mat file uses HDF5 format (MATLAB v7.3). "
                "Install h5py: pip install h5py"
            )


def _infer_fs_from_description(desc_path: str, fallback: Optional[float] = None) -> float:
    """
    Parse the sampling rate from a text-based Data Description file.
    Looks for patterns like 'sampling rate: 250 Hz' or 'fs = 500'.
    Returns fallback if parsing fails and fallback is provided, else raises.
    """
    if not os.path.isfile(desc_path):
        if fallback is not None:
            warnings.warn(f"Data Description not found at {desc_path}. Using fallback fs={fallback}")
            return fallback
        raise FileNotFoundError(f"Data Description file not found: {desc_path}")

    with open(desc_path, "r", errors="ignore") as f:
        text = f.read()

    patterns = [
        r"sampling\s+(?:rate|frequency)[^\d]*(\d+)\s*(?:Hz|hz|HZ)",
        r"fs\s*[=:]\s*(\d+)",
        r"(\d+)\s*(?:Hz|hz|HZ)\s+sampling",
        r"sampled\s+at\s+(\d+)\s*(?:Hz|hz|HZ)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1))

    if fallback is not None:
        warnings.warn(f"Could not parse a sampling rate from {desc_path}. Using fallback fs={fallback}")
    return fallback


def _resolve_fs(dataset_name: str, root: str) -> Optional[float]:
    """
    Sampling rate for one dataset. CFG.data.fs_hz_by_dataset (values taken from
    each dataset's Data Description PDF) wins; otherwise parse a text Data
    Description file under `root`, falling back to CFG.data.fs_fallback_hz.
    Returns None only when there is neither a table entry nor a description file.
    """
    fs = (getattr(CFG.data, "fs_hz_by_dataset", None) or {}).get(dataset_name)
    if fs:
        return float(fs)
    desc_candidates = glob.glob(os.path.join(root, "**", "*escription*"), recursive=True) + \
                      glob.glob(os.path.join(root, "**", "*.txt"), recursive=True)
    if not desc_candidates:
        return None
    return _infer_fs_from_description(desc_candidates[0], fallback=CFG.data.fs_fallback_hz)


def load_dataset1(max_subjects: Optional[int] = None) -> List[Trial]:
    """
    Load Dataset 1: Zero-Centred Bipolar EOG.
    Channels are already differenced (H = right-left, V = up-down).
    Returns list of Trial objects with montage_type='bipolar'.

    Directory expected: data/raw/Dataset_ZeroCentred/
    """
    root = CFG.paths.dataset1_dir
    _check_dir(
        root, "Dataset 1 (Zero-Centred Bipolar)",
        "https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/"
    )

    fs = _resolve_fs("dataset1", root)

    trials: List[Trial] = []

    data_files = (
        glob.glob(os.path.join(root, "**", "*.mat"), recursive=True) +
        glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    )
    data_files = sorted(data_files)

    if not data_files:
        raise FileNotFoundError(
            f"No .mat or .csv data files found under {root}. "
            "Check that the ZIP was extracted correctly."
        )

    subject_dirs = _discover_subject_structure(root)

    if subject_dirs:
        trials = _load_structured(subject_dirs, "dataset1", "bipolar", fs, max_subjects)
    else:
        trials = _load_flat(data_files, "dataset1", "bipolar", fs, max_subjects)

    print(f"[Dataset 1] Loaded {len(trials)} trials from {root}")
    return trials


def load_dataset2(max_subjects: Optional[int] = None) -> List[Trial]:
    """
    Load Dataset 2: Monopolar, Stationary.
    Returns list of Trial objects with montage_type='monopolar'.

    Directory expected: data/raw/Dataset_Stationary/
    """
    root = CFG.paths.dataset2_dir
    _check_dir(
        root, "Dataset 2 (Monopolar Stationary)",
        "https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/"
    )

    fs = _resolve_fs("dataset2", root)

    data_files = sorted(
        glob.glob(os.path.join(root, "**", "*.mat"), recursive=True) +
        glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    )
    if not data_files:
        raise FileNotFoundError(f"No data files found under {root}.")

    subject_dirs = _discover_subject_structure(root)
    if subject_dirs:
        trials = _load_structured(subject_dirs, "dataset2", "monopolar", fs, max_subjects)
    else:
        trials = _load_flat(data_files, "dataset2", "monopolar", fs, max_subjects)

    print(f"[Dataset 2] Loaded {len(trials)} trials from {root}")
    return trials


def load_dataset3(max_subjects: Optional[int] = None) -> List[Trial]:
    """
    Load Dataset 3: Monopolar, Non-Stationary.
    Includes head pose / position channels.
    Returns Trial objects with head_pose populated.

    Directory expected: data/raw/Dataset_NonStationary/
    """
    root = CFG.paths.dataset3_dir
    _check_dir(
        root, "Dataset 3 (Monopolar Non-Stationary)",
        "https://www.um.edu.mt/cbc/ourprojects/eyecon/eogdataset/"
    )

    fs = _resolve_fs("dataset3", root)

    data_files = sorted(
        glob.glob(os.path.join(root, "**", "*.mat"), recursive=True) +
        glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    )
    if not data_files:
        raise FileNotFoundError(f"No data files found under {root}.")

    subject_dirs = _discover_subject_structure(root)
    if subject_dirs:
        trials = _load_structured(subject_dirs, "dataset3", "monopolar", fs, max_subjects,
                                  has_head_pose=True)
    else:
        trials = _load_flat(data_files, "dataset3", "monopolar", fs, max_subjects,
                            has_head_pose=True)

    print(f"[Dataset 3] Loaded {len(trials)} trials from {root}")
    return trials


def load_dataset4(max_subjects: Optional[int] = None) -> List[Trial]:
    """
    Load Dataset 4: Monopolar, Isotropic Directions.
    ⚠ MANUAL DOWNLOAD required from Google Drive.
    Extract into: data/raw/Dataset_Isotropic/

    Returns list of Trial objects with montage_type='monopolar'.
    """
    root = CFG.paths.dataset4_dir
    _check_dir(
        root, "Dataset 4 (Monopolar Isotropic)",
        "Google Drive link — see README.md for URL. Download manually and extract here."
    )

    fs = _resolve_fs("dataset4", root)

    data_files = sorted(
        glob.glob(os.path.join(root, "**", "*.mat"), recursive=True) +
        glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)
    )
    if not data_files:
        raise FileNotFoundError(f"No data files found under {root}.")

    subject_dirs = _discover_subject_structure(root)
    if subject_dirs:
        trials = _load_structured(subject_dirs, "dataset4", "monopolar", fs, max_subjects)
    else:
        trials = _load_flat(data_files, "dataset4", "monopolar", fs, max_subjects)

    print(f"[Dataset 4] Loaded {len(trials)} trials from {root}")
    return trials


def _discover_subject_structure(root: str) -> List[str]:
    """
    Return a sorted list of subject subdirectories if the dataset uses a
    per-subject folder layout (e.g., S01/, S02/, Subject_1/).
    Searches recursively to handle nested extraction folders.
    """
    candidates = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "__MACOSX" in dirpath:
            continue
        base = os.path.basename(dirpath)
        if re.match(r"(?i)^(sub|s\d+|subject)", base):
            candidates.append(dirpath)
    return sorted(list(set(candidates)))


def _parse_mat_trial(
    mat_data: dict,
    subject_id: str,
    trial_id: str,
    dataset_source: str,
    montage_type: str,
    fs: Optional[float],
    has_head_pose: bool = False,
) -> Trial:
    """
    Convert a loaded .mat dict into a Trial.
    Heuristically detects channel arrays vs. metadata scalars.
    Looks for common key patterns used in MATLAB biomedical datasets.
    """
    channels: Dict[str, np.ndarray] = {}
    event_timestamps = make_empty_event_timestamps()
    target_angle: Optional[np.ndarray] = None
    head_pose: Optional[np.ndarray] = None
    head_position: Optional[np.ndarray] = None
    detected_fs: Optional[float] = fs

    EOG_CHANNEL_PATTERNS = re.compile(
        r"(?i)(eog|heog|veog|hori|vert|left|right|up|down|chan|ch\d+|signal|data)",
        re.IGNORECASE
    )
    FS_PATTERNS = re.compile(r"(?i)(fs|srate|samplingrate|sampling_rate|freq)")
    ANGLE_PATTERNS = re.compile(r"(?i)(angle|gaze|target|degree|deg)")
    POSE_PATTERNS = re.compile(r"(?i)(head|pose|position|imu|accel|gyro)")
    EVENT_PATTERNS = re.compile(r"(?i)(event|onset|trigger|marker|timestamp|label)")

    for key, val in mat_data.items():
        if key.startswith("__"):
            continue
        val_arr = np.atleast_1d(np.array(val, dtype=object))

        if FS_PATTERNS.search(key) and np.isscalar(val):
            detected_fs = float(val)
            continue

        if ANGLE_PATTERNS.search(key) and isinstance(val, np.ndarray) and val.ndim >= 1:
            arr = np.array(val, dtype=np.float64)
            if arr.ndim == 2 and arr.shape[0] in (1, 2) and arr.shape[1] > 10:
                arr = arr.T
            if arr.shape[-1] in (1, 2) or (arr.ndim == 1 and len(arr) > 10):
                if arr.ndim == 1:
                    arr = arr[:, np.newaxis]
                if arr.shape[1] == 1:
                    arr = np.hstack([arr, np.zeros_like(arr)])
                target_angle = arr
            continue

        if has_head_pose and POSE_PATTERNS.search(key) and isinstance(val, np.ndarray):
            # Head_Position (x, y, z in m) matches POSE_PATTERNS too; storing it in
            # head_pose used to overwrite Head_Pose (yaw, pitch, roll in deg).
            if re.search(r"(?i)position", key):
                head_position = np.array(val, dtype=np.float64)
            else:
                head_pose = np.array(val, dtype=np.float64)
            continue

        if EVENT_PATTERNS.search(key):
            _parse_events_into(val, event_timestamps)
            continue

        if EOG_CHANNEL_PATTERNS.search(key) and isinstance(val, np.ndarray):
            if val.ndim == 1 and len(val) > 10:
                channels[key] = np.array(val, dtype=np.float64)
            elif val.ndim == 2:
                for i in range(val.shape[0] if val.shape[0] < val.shape[1] else val.shape[1]):
                    if val.shape[0] < val.shape[1]:
                        channels[f"{key}_{i}"] = val[i].astype(np.float64)
                    else:
                        channels[f"{key}_{i}"] = val[:, i].astype(np.float64)
            continue

        if isinstance(val, np.ndarray) and val.ndim == 1 and len(val) > 100:
            channels[f"ch_{key}"] = np.array(val, dtype=np.float64)

    if detected_fs is None:
        raise ValueError(
            f"Could not determine fs for subject={subject_id}, trial={trial_id}, "
            f"dataset={dataset_source}. "
            "Add fs to the Data Description parser or set it manually."
        )

    return Trial(
        subject_id=subject_id,
        trial_id=trial_id,
        dataset_source=dataset_source,
        montage_type=montage_type,
        fs=detected_fs,
        channels=channels,
        event_timestamps=event_timestamps,
        target_angle=target_angle,
        head_pose=head_pose,
        metadata={"source_file": trial_id,
                  **({"head_position": head_position} if head_position is not None else {})},
    )


def _parse_events_into(val, event_timestamps: Dict[str, int]) -> None:
    """Try to parse event/label array into event_timestamps dict."""
    try:
        arr = np.atleast_1d(np.array(val)).flatten()
        int_arr = arr[~np.isnan(arr.astype(float))].astype(int)
        keys_in_order = [
            "saccade1_onset", "saccade1_end",
            "saccade2_onset", "saccade2_end",
            "blink_onset", "blink_end",
        ]
        for i, k in enumerate(keys_in_order):
            if i < len(int_arr):
                event_timestamps[k] = int(int_arr[i])
    except Exception:
        pass


def _load_structured(
    subject_dirs: List[str],
    dataset_source: str,
    montage_type: str,
    fs: Optional[float],
    max_subjects: Optional[int],
    has_head_pose: bool = False,
) -> List[Trial]:
    """Load from per-subject directory layout."""
    trials: List[Trial] = []
    dirs = subject_dirs[:max_subjects] if max_subjects else subject_dirs
    for subj_dir in dirs:
        subject_id = os.path.basename(subj_dir)
        mat_files = sorted(
            glob.glob(os.path.join(subj_dir, "**", "*.mat"), recursive=True) +
            glob.glob(os.path.join(subj_dir, "**", "*.csv"), recursive=True)
        )
        
        combined_data = {}
        for fpath in mat_files:
            try:
                if fpath.endswith(".mat"):
                    mat_data = _load_mat_safe(fpath)
                else:
                    mat_data = _load_csv_as_mat(fpath)
                combined_data.update(mat_data)
            except Exception as e:
                warnings.warn(f"Skipping {fpath}: {e}")
                
        if combined_data:
            try:
                trial = _parse_mat_trial(
                    combined_data, subject_id, f"{subject_id}_trial",
                    dataset_source, montage_type, fs, has_head_pose
                )
                if "ControlSignal" in combined_data:
                    trial.channels["ControlSignal"] = np.array(combined_data["ControlSignal"]).flatten()
                if "TargetGA" in combined_data:
                    trial.metadata["TargetGA"] = np.array(combined_data["TargetGA"])
                trial.validate()
                trials.append(trial)
            except Exception as e:
                warnings.warn(f"Failed to parse trial for {subject_id}: {e}")
                
    return trials


def _load_flat(
    data_files: List[str],
    dataset_source: str,
    montage_type: str,
    fs: Optional[float],
    max_subjects: Optional[int],
    has_head_pose: bool = False,
) -> List[Trial]:
    """Load from flat directory layout — infer subject_id from filename."""
    trials: List[Trial] = []
    seen_subjects: Dict[str, int] = {}

    for fpath in data_files:
        fname = os.path.basename(fpath)
        m = re.search(r"(?i)(s\d+|sub\d+|subject\d+)", fname)
        subject_id = m.group(0).upper() if m else f"S{len(seen_subjects)+1:02d}"
        seen_subjects[subject_id] = seen_subjects.get(subject_id, 0) + 1

        if max_subjects and len(seen_subjects) > max_subjects:
            break

        trial_id = f"T{seen_subjects[subject_id]:03d}_{fname}"
        try:
            if fpath.endswith(".mat"):
                mat_data = _load_mat_safe(fpath)
            else:
                mat_data = _load_csv_as_mat(fpath)
            trial = _parse_mat_trial(
                mat_data, subject_id, trial_id,
                dataset_source, montage_type, fs, has_head_pose
            )
            trial.validate()
            trials.append(trial)
        except Exception as e:
            warnings.warn(f"Skipping {fpath}: {e}")
    return trials


def _load_csv_as_mat(fpath: str) -> dict:
    """Load CSV file and return as a mat-like dict."""
    import pandas as pd
    df = pd.read_csv(fpath)
    return {col: df[col].values for col in df.columns}


def load_all_datasets(skip_missing: bool = True) -> List[Trial]:
    """
    Attempt to load all four datasets. If skip_missing=True, datasets whose
    directories are empty are skipped with a warning instead of raising.

    Honors CFG.data.datasets_to_load: only the listed datasets are loaded
    (e.g. ["dataset2"] for a single-dataset Stationary run).
    """
    all_trials: List[Trial] = []
    loaders = [
        ("dataset1", load_dataset1),
        ("dataset2", load_dataset2),
        ("dataset3", load_dataset3),
        ("dataset4", load_dataset4),
    ]
    selected = set(getattr(CFG.data, "datasets_to_load", None) or
                   ["dataset1", "dataset2", "dataset3", "dataset4"])
    unknown = selected - {name for name, _ in loaders}
    if unknown:
        raise ValueError(
            f"CFG.data.datasets_to_load contains unknown dataset(s): {sorted(unknown)}. "
            "Valid names: dataset1, dataset2, dataset3, dataset4."
        )
    loaders = [(name, fn) for name, fn in loaders if name in selected]
    if not loaders:
        raise ValueError("CFG.data.datasets_to_load is empty — nothing to load.")
    if len(loaders) < 4:
        print(f"Dataset scope (CFG.data.datasets_to_load): {sorted(selected)}")
    for name, loader in loaders:
        try:
            all_trials.extend(loader())
        except FileNotFoundError as e:
            if skip_missing:
                warnings.warn(f"Skipping {name}: {e}")
            else:
                raise
    print(f"\nTotal trials loaded across all datasets: {len(all_trials)}")
    return all_trials
