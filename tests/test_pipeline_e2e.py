"""
tests/test_pipeline_e2e.py
===========================
End-to-end integration test suite using synthetic Trial objects.

Validates that:
  1. Data loading and unification schema work end-to-end.
  2. Preprocessing, windowing, and fold generation run cleanly.
  3. Feature extraction and classical ML models execute without error.
  4. Deep models (Conv1D, LSTM, Conv1D+BiLSTM) train for 1 epoch with
     both static and Kendall uncertainty loss weighting modes.
  5. Metrics computation and paper comparison table generation function correctly.
"""

from __future__ import annotations

import os
import tempfile
import pytest
import numpy as np
import torch

from src.config import CFG
from src.data.schema import Trial, make_empty_event_timestamps
from src.data.unify import to_common_schema
from src.preprocessing.normalize import preprocess_trials
from src.data.datasets import (
    make_classification_windows,
    make_regression_targets,
)
from src.training.cv_splits import get_folds, get_subject_ids_from_metadata
from src.features.engineered import extract_features_batch
from src.models.classical_ml import run_classification_cv, run_regression_cv
from src.models.deep_multitask import build_model, build_loss, MultiTaskLoss
from src.training.train import train_cv
from src.evaluation.compare_to_paper import build_master_table


def _make_synthetic_trial(
    subject_id: str,
    trial_id: str,
    dataset_source: str = "dataset2",
    duration_s: float = 2.0,
    fs: float = 250.0,
) -> Trial:
    """Create a valid synthetic Trial object for pipeline integration testing."""
    n_samples = int(duration_s * fs)
    t = np.linspace(0, duration_s, n_samples)

    ch = {
        "EOG_0": (np.sin(2 * np.pi * 1.5 * t) * 50.0).astype(np.float64),
        "EOG_1": (-np.sin(2 * np.pi * 1.5 * t) * 50.0).astype(np.float64),
        "EOG_2": (np.cos(2 * np.pi * 1.5 * t) * 30.0).astype(np.float64),
        "EOG_3": (-np.cos(2 * np.pi * 1.5 * t) * 30.0).astype(np.float64),
        "ControlSignal": np.zeros(n_samples, dtype=np.float64),
    }

    ch["ControlSignal"][50:100] = 1
    ch["ControlSignal"][150:200] = 2
    ch["ControlSignal"][300:350] = 3

    events = make_empty_event_timestamps()
    events["saccade1_onset"] = 50
    events["saccade1_end"] = 100

    target_angle = np.zeros((n_samples, 2), dtype=np.float32)
    target_angle[50:, 0] = 15.0

    head_pose = np.random.randn(n_samples, 3).astype(np.float32)

    trial = Trial(
        subject_id=subject_id,
        trial_id=trial_id,
        dataset_source=dataset_source,
        montage_type="monopolar",
        fs=fs,
        channels=ch,
        event_timestamps=events,
        target_angle=target_angle,
        head_pose=head_pose,
    )
    trial.validate()
    return trial


def test_full_pipeline_synthetic_e2e():
    """Run full pipeline phases 1 through 8 on synthetic trials in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        orig_proc = CFG.paths.data_processed
        orig_res = CFG.paths.results
        CFG.paths.data_processed = tmp_dir
        CFG.paths.results = tmp_dir

        try:
            subjects = ["S01", "S02", "S03", "S04"]
            trials = []
            for s in subjects:
                t1 = _make_synthetic_trial(s, f"{s}_T1", dataset_source="dataset2")
                t2 = _make_synthetic_trial(s, f"{s}_T2", dataset_source="dataset3")
                trials.extend([t1, t2])

            trials = [to_common_schema(t) for t in trials]
            trials = preprocess_trials(trials)

            for t in trials:
                assert t.has_bipolar()
                assert "H" in t.channels and "V" in t.channels

            X_cls, y_cls, meta_cls = make_classification_windows(trials)
            X_reg, y_reg, meta_reg = make_regression_targets(trials)

            assert X_cls.ndim == 3 and X_cls.shape[1] == 2
            assert len(X_cls) == len(y_cls) == len(meta_cls)
            assert X_reg.shape[0] == len(y_reg)

            subject_ids = get_subject_ids_from_metadata(meta_cls)
            folds = get_folds(subject_ids, strategy="group_kfold", k=2)
            assert len(folds) == 2

            Xf = extract_features_batch(X_cls, fs=250.0)
            assert Xf.shape[0] == len(X_cls)

            clf_res = run_classification_cv(Xf, y_cls, folds, model_name="rf")
            assert "pooled_f1_weighted" in clf_res

            reg_res = run_regression_cv(Xf, y_reg, folds, model_name="svr")
            assert "pooled_rmse_h_deg" in reg_res

            saved_model_cfg = (CFG.model.model_type, CFG.model.loss_type,
                               CFG.model.max_epochs, CFG.model.batch_size)
            try:
                for model_type in ["conv1d", "lstm", "conv_bilstm"]:
                    for loss_type in ["uncertainty", "static"]:
                        CFG.model.model_type = model_type
                        CFG.model.loss_type = loss_type
                        CFG.model.max_epochs = 1
                        CFG.model.batch_size = 16

                        deep_res = train_cv(
                            X_cls, y_cls, y_reg, folds,
                            results_dir=tmp_dir, checkpoint_dir=tmp_dir
                        )
                        assert deep_res["model"] in ("EOGMultiTaskNet", "EOGLSTMNet", "EOGConvLSTMNet")
                        assert "pooled_rmse_h_deg" in deep_res
            finally:
                (CFG.model.model_type, CFG.model.loss_type,
                 CFG.model.max_epochs, CFG.model.batch_size) = saved_model_cfg

            rows = build_master_table(results_dir=tmp_dir)
            assert len(rows) > 0

        finally:
            CFG.paths.data_processed = orig_proc
            CFG.paths.results = orig_res


def test_data_augmentation_tensor():
    """Verify PyTorch tensor batch augmentation shape and output integrity."""
    from src.preprocessing.augmentation import augment_eog_batch_tensor

    batch = torch.randn(8, 2, 75)
    aug = augment_eog_batch_tensor(batch, scale_range=(0.9, 1.1), noise_std=0.01)

    assert aug.shape == batch.shape
    assert not torch.isnan(aug).any()
    assert not torch.equal(batch, aug)


def test_all_eog_channel_mode_synthetic():
    """Verify that window_channel_mode='all_eog' stacks 6 channels and runs through models."""
    from src.features.engineered import get_feature_names

    orig_mode = CFG.data.window_channel_mode
    orig_in_ch = CFG.model.in_channels
    try:
        CFG.data.window_channel_mode = "all_eog"
        trials = [_make_synthetic_trial("S01", "S01_T1", dataset_source="dataset2")]
        trials = [to_common_schema(t) for t in trials]
        trials = preprocess_trials(trials)

        X_cls, y_cls, meta_cls = make_classification_windows(trials)
        assert X_cls.shape[1] == 6, f"Expected 6 channels, got {X_cls.shape[1]}"
        assert meta_cls[0]["channel_names"] == ["H", "V", "EOG_0", "EOG_1", "EOG_2", "EOG_3"]

        names = get_feature_names(n_channels=6, channel_names=meta_cls[0]["channel_names"])
        Xf = extract_features_batch(X_cls, fs=250.0, channel_names=meta_cls[0]["channel_names"])
        assert Xf.shape[1] == len(names)

        CFG.model.in_channels = 6
        model = build_model(model_type="conv_bilstm", device=torch.device("cpu"))
        x_tensor = torch.randn(4, 6, 75)
        out_cls, out_reg = model(x_tensor)
        assert out_cls.shape == (4, CFG.model.num_classes)
        assert out_reg.shape == (4, CFG.model.regression_outputs)
    finally:
        CFG.data.window_channel_mode = orig_mode
        CFG.model.in_channels = orig_in_ch

