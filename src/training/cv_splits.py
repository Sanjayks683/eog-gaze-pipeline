"""
src/training/cv_splits.py
==========================
Subject-wise cross-validation splits.

RULE: splits are always by subject_id — NEVER by trial.
Splits are saved to disk and reused by every subsequent phase so
all models (baselines and deep) are evaluated on identical folds.
"""

from __future__ import annotations

import os
import pickle
from typing import List, Tuple, Dict

import numpy as np
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from src.config import CFG


FoldList = List[Tuple[np.ndarray, np.ndarray]]


def get_folds(
    subject_ids: np.ndarray,
    strategy: str = None,
    k: int = None,
) -> FoldList:
    """
    Generate cross-validation folds stratified by subject.

    Parameters
    ----------
    subject_ids : (n_windows,) array of subject IDs (one per window)
    strategy    : "group_kfold" or "leave_one_group_out" (default from CFG)
    k           : number of folds for GroupKFold (default from CFG)

    Returns
    -------
    List of (train_indices, test_indices) arrays.
    """
    if strategy is None:
        strategy = CFG.cv.strategy
    if k is None:
        k = CFG.cv.k

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)
    print(f"CV split: {n_subjects} unique subjects, strategy={strategy}")

    if strategy == "group_kfold" and n_subjects < k:
        print(
            f"  Warning: {n_subjects} subjects < k={k}. "
            "Switching to LeaveOneGroupOut."
        )
        strategy = "leave_one_group_out"

    X_dummy = np.zeros((len(subject_ids), 1))

    if strategy == "group_kfold":
        splitter = GroupKFold(n_splits=k)
    elif strategy == "leave_one_group_out":
        splitter = LeaveOneGroupOut()
    else:
        raise ValueError(f"Unknown CV strategy: {strategy!r}")

    folds: FoldList = []
    for train_idx, test_idx in splitter.split(X_dummy, groups=subject_ids):
        folds.append((train_idx, test_idx))

    print(f"  Generated {len(folds)} folds.")
    for i, (tr, te) in enumerate(folds):
        tr_subj = np.unique(subject_ids[tr])
        te_subj = np.unique(subject_ids[te])
        print(f"  Fold {i}: train={len(tr_subj)} subjects, test={len(te_subj)} subjects")

    return folds


def save_folds(folds: FoldList, metadata: List[dict], path: str = None) -> str:
    """
    Save fold assignments and window metadata to disk.

    Parameters
    ----------
    folds    : output of get_folds()
    metadata : list of dicts (one per window) with subject_id, dataset_source, etc.
    path     : file path; defaults to CFG.paths.folds_file

    Returns
    -------
    The path where folds were saved.
    """
    if path is None:
        path = CFG.paths.folds_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {"folds": folds, "metadata": metadata}
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"Folds saved: {path}")
    return path


def load_folds(path: str = None) -> Tuple[FoldList, List[dict]]:
    """
    Load fold assignments from disk.

    Returns
    -------
    (folds, metadata)
    """
    if path is None:
        path = CFG.paths.folds_file
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Folds file not found: {path}. "
            "Run the dataset pipeline first to generate and save folds."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    folds = payload["folds"]
    metadata = payload["metadata"]
    print(f"Folds loaded: {path}  ({len(folds)} folds)")
    return folds, metadata


def get_subject_ids_from_metadata(metadata: List[dict]) -> np.ndarray:
    """Extract subject_id array from window metadata list."""
    return np.array([m["subject_id"] for m in metadata])


def get_dataset_mask(metadata: List[dict], dataset_source: str) -> np.ndarray:
    """Boolean mask selecting windows from a specific dataset."""
    return np.array([m["dataset_source"] == dataset_source for m in metadata])
