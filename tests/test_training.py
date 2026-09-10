"""
tests/test_training.py
======================
Checkpoint config fingerprints, the subject-held-out validation split,
and the trivial (majority class / mean angle) baselines.
"""

from __future__ import annotations

import numpy as np
import torch

from src.config import CFG
from src.models.classical_ml import run_classification_cv, run_regression_cv
from src.models.deep_multitask import EOGMultiTaskNet
from src.training.train import config_fingerprint, load_checkpoint, validation_indices


def test_fingerprint_changes_with_config_and_shape():
    base = config_fingerprint((100, 2, 75))
    assert base == config_fingerprint((100, 2, 75))
    assert base != config_fingerprint((100, 6, 75))
    orig_lr = CFG.model.lr
    try:
        CFG.model.lr = orig_lr * 2
        assert base != config_fingerprint((100, 2, 75))
    finally:
        CFG.model.lr = orig_lr


def test_checkpoint_from_other_config_is_rejected(tmp_path):
    dev = torch.device("cpu")
    model = EOGMultiTaskNet(in_channels=2)
    path = str(tmp_path / "ckpt.pt")

    torch.save({"model_type": "conv1d", "fingerprint": "aaa",
                "state_dict": model.state_dict()}, path)
    assert load_checkpoint(path, EOGMultiTaskNet(in_channels=2), "conv1d", dev, "aaa")
    assert not load_checkpoint(path, EOGMultiTaskNet(in_channels=2), "conv1d", dev, "bbb")

    torch.save(model.state_dict(), path)
    assert not load_checkpoint(path, EOGMultiTaskNet(in_channels=2), "conv1d", dev, "aaa")


def test_validation_holds_out_whole_training_subjects():
    subjects = np.array([f"S{i}" for i in range(10) for _ in range(50)])
    metadata = [{"subject_id": s} for s in subjects]
    train_idx = np.arange(100, 500)

    val_pos = validation_indices(train_idx, metadata, seed=0)

    train_subjects = subjects[train_idx]
    val_subjects = set(train_subjects[val_pos])
    remaining = set(np.delete(train_subjects, val_pos))
    assert len(val_subjects) == 1
    assert val_subjects.isdisjoint(remaining)


def test_validation_falls_back_to_random_windows_without_metadata():
    val_pos = validation_indices(np.arange(200), None, seed=0)
    assert len(val_pos) == 20 and len(np.unique(val_pos)) == 20


def test_trivial_baselines():
    rng = np.random.RandomState(0)
    X = rng.randn(200, 3)
    y_cls = np.array([1] * 150 + [2] * 50)
    y_reg = rng.randn(200, 2) * 10.0
    evens, odds = np.arange(0, 200, 2), np.arange(1, 200, 2)
    folds = [(evens, odds), (odds, evens)]

    clf = run_classification_cv(X, y_cls, folds, model_name="majority")
    assert np.isclose(clf["pooled_f1_weighted"], 0.75 * (2 * 0.75 / 1.75))

    reg = run_regression_cv(X, y_reg, folds, model_name="mean")
    err = np.vstack([y_reg[te] - y_reg[tr].mean(axis=0) for tr, te in folds])
    assert np.isclose(reg["pooled_rmse_h_deg"], np.sqrt(np.mean(err[:, 0] ** 2)))
    assert np.isclose(reg["pooled_rmse_v_deg"], np.sqrt(np.mean(err[:, 1] ** 2)))
