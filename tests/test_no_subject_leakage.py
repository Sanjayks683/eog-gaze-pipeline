"""
tests/test_no_subject_leakage.py
==================================
AUTOMATED leakage test — Phase 5.2.

Verifies that for EVERY fold, train and test subject sets are disjoint.
This test must NEVER be skipped. It is the guard against future
refactors accidentally introducing data leakage.
"""

import numpy as np
import pytest

from src.training.cv_splits import get_folds


def _make_subject_ids(n_subjects: int, windows_per_subject: int) -> np.ndarray:
    """Create a flat array of subject IDs, one per window."""
    return np.repeat(
        [f"S{i:02d}" for i in range(n_subjects)],
        windows_per_subject
    )


@pytest.mark.parametrize("n_subjects,strategy", [
    (10, "group_kfold"),
    (5, "group_kfold"),
    (4, "leave_one_group_out"),
    (3, "leave_one_group_out"),
])
def test_no_subject_leakage(n_subjects, strategy):
    """
    For every fold, train_subjects ∩ test_subjects must be empty.
    This is an automated test, not a manual check.
    """
    subject_ids = _make_subject_ids(n_subjects, windows_per_subject=50)
    folds = get_folds(subject_ids, strategy=strategy, k=min(5, n_subjects))

    assert len(folds) > 0, "No folds generated"

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        train_subjects = set(subject_ids[train_idx])
        test_subjects = set(subject_ids[test_idx])
        overlap = train_subjects & test_subjects

        assert len(overlap) == 0, (
            f"SUBJECT LEAKAGE DETECTED in fold {fold_i}!\n"
            f"  train_subjects = {sorted(train_subjects)}\n"
            f"  test_subjects  = {sorted(test_subjects)}\n"
            f"  overlap        = {sorted(overlap)}\n"
            "Fix the splitting logic in cv_splits.py immediately."
        )


def test_all_subjects_appear_in_test_exactly_once_logo():
    """
    In LeaveOneGroupOut, each subject must appear in test exactly once.
    """
    n_subjects = 6
    subject_ids = _make_subject_ids(n_subjects, windows_per_subject=30)
    folds = get_folds(subject_ids, strategy="leave_one_group_out")

    test_subject_sets = [set(subject_ids[test_idx]) for _, test_idx in folds]

    all_test_subjects = [s for fold_set in test_subject_sets for s in fold_set]
    unique_subjects = np.unique(subject_ids)

    for subj in unique_subjects:
        count = all_test_subjects.count(subj)
        assert count == 1, (
            f"Subject {subj} appears {count} times in test sets "
            "(expected exactly 1 for LeaveOneGroupOut)"
        )


def test_indices_cover_all_windows():
    """
    The union of all train+test indices across folds should cover all windows.
    """
    n_subjects = 8
    n_windows = n_subjects * 40
    subject_ids = _make_subject_ids(n_subjects, windows_per_subject=40)
    folds = get_folds(subject_ids, strategy="group_kfold", k=5)

    all_test_indices = set()
    for _, test_idx in folds:
        all_test_indices.update(test_idx.tolist())

    assert all_test_indices == set(range(n_windows)), (
        "Not all window indices appear in test sets. "
        "GroupKFold should cover every window exactly once across all folds."
    )


def test_indices_are_valid():
    """Train and test indices must be valid non-empty arrays."""
    subject_ids = _make_subject_ids(10, windows_per_subject=20)
    folds = get_folds(subject_ids, strategy="group_kfold", k=5)

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        assert len(train_idx) > 0, f"Fold {fold_i}: train_idx is empty"
        assert len(test_idx) > 0, f"Fold {fold_i}: test_idx is empty"
        assert train_idx.max() < len(subject_ids), \
            f"Fold {fold_i}: train index out of bounds"
        assert test_idx.max() < len(subject_ids), \
            f"Fold {fold_i}: test index out of bounds"


def test_auto_switch_to_logo_when_few_subjects():
    """
    If n_subjects < k, get_folds should silently switch to LOGO.
    Resulting folds must still be leakage-free.
    """
    subject_ids = _make_subject_ids(3, windows_per_subject=20)
    folds = get_folds(subject_ids, strategy="group_kfold", k=5)
    assert len(folds) == 3, "Should produce 3 folds (one per subject) via LOGO"

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        overlap = set(subject_ids[train_idx]) & set(subject_ids[test_idx])
        assert len(overlap) == 0, f"Leakage in fold {fold_i} after auto-switch to LOGO"
