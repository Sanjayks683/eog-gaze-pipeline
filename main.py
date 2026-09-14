"""
main.py
=======
Unified Command-Line Interface (CLI) for the EOG Eye Gaze Pipeline.

Usage:
    python main.py --phase inspect
    python main.py --phase preprocess
    python main.py --phase train_classical
    python main.py --phase train_deep --model-type conv_bilstm
    python main.py --phase ablations
    python main.py --phase compare
    python main.py --phase all
"""

from __future__ import annotations

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CFG, load_config_from_yaml


def run_inspect():
    print("\n" + "=" * 60)
    print("  Phase 0: Raw Data Inspection")
    print("=" * 60)
    from scripts.inspect_raw import main as inspect_main
    inspect_main()


def run_preprocess():
    print("\n" + "=" * 60)
    print("  Phases 1-5: Load, Unify, Preprocess, Segment & Fold Splits")
    print("=" * 60)
    from src.data.loaders import load_all_datasets
    from src.data.unify import to_common_schema
    from src.preprocessing.normalize import preprocess_trials
    from src.data.datasets import (
        make_classification_windows,
        make_regression_targets,
        save_processed,
        save_preprocessed_trials,
        print_class_balance,
    )
    from src.training.cv_splits import get_folds, save_folds, get_subject_ids_from_metadata

    print("[1/3] Loading datasets...")
    trials = load_all_datasets(skip_missing=True)
    if not trials:
        print("No datasets found. Please download datasets into data/raw/ as described in README.md.")
        return

    print("[2/3] Unifying montages, filtering, and normalizing...")
    trials = [to_common_schema(t) for t in trials]
    use_context = bool(CFG.preprocessing.context_baselines or CFG.preprocessing.context_lags_sec)
    raw_signals = [{k: v.copy() for k, v in t.channels.items()} for t in trials] if use_context else None
    trials = preprocess_trials(trials)

    print("[3/3] Windowing and saving processed arrays...")
    X_cls, y_cls, meta = make_classification_windows(trials)
    X_reg, y_reg, _ = make_regression_targets(trials)

    print_class_balance(y_cls)
    save_processed(X_cls, y_cls, meta, "classification")
    save_processed(X_reg, y_reg, meta, "regression")
    if use_context:
        from src.features.context import make_context_features
        save_processed(make_context_features(trials, raw_signals, meta), y_reg, meta, "context")
    save_preprocessed_trials(trials)

    subject_ids = get_subject_ids_from_metadata(meta)
    folds = get_folds(subject_ids)
    save_folds(folds, meta)
    print("Preprocessing completed successfully.")


def _sync_in_channels(X: np.ndarray) -> None:
    """Point CFG.model.in_channels at the actual channel count in X so every
    model built afterwards matches the data (bipolar=2, all_eog=2+EOG_*)."""
    import numpy as np
    n_ch = int(X.shape[1])
    if getattr(CFG.model, "in_channels", None) != n_ch:
        print(f"Setting CFG.model.in_channels = {n_ch} (from data)")
        CFG.model.in_channels = n_ch


def _window_channel_names(meta: list):
    """Channel names recorded by preprocessing, if present."""
    for m in meta:
        names = m.get("channel_names")
        if names:
            return names
    return None


def _extract_features_respecting_fs(X, meta):
    """
    Extract hand-crafted features using the per-window sampling rate recorded
    in the metadata. Uses the fast single-fs batch path when all windows share
    one fs (the normal case), otherwise falls back to per-window extraction.
    """
    import numpy as np
    from src.features.engineered import extract_features_batch, extract_features

    fs_values = sorted({m["fs"] for m in meta if m.get("fs") is not None})
    missing_fs = sum(1 for m in meta if m.get("fs") is None)
    if missing_fs or not fs_values:
        raise ValueError(
            f"{missing_fs} window(s) are missing 'fs' in their metadata. "
            "Re-run `python main.py --phase preprocess` to regenerate metadata."
        )
    channel_names = _window_channel_names(meta)
    if len(fs_values) == 1:
        return extract_features_batch(X, fs=fs_values[0], channel_names=channel_names)
    print(f"Mixed sampling rates across windows {fs_values} — "
          "extracting features per window with its own fs.")
    return np.stack([
        extract_features(X[i], fs=meta[i]["fs"], channel_names=channel_names) for i in range(len(X))
    ])


def run_train_classical():
    print("\n" + "=" * 60)
    print("  Phase 6: Classical Baseline Model Training")
    print("=" * 60)
    import numpy as np
    from src.data.datasets import load_processed, load_preprocessed_trials
    from src.training.cv_splits import load_folds
    from src.models.classical_ml import (
        run_classification_cv,
        run_regression_cv,
        save_results,
    )

    X_cls, y_cls, meta = load_processed("classification")
    X_reg, y_reg, _ = load_processed("regression")
    folds, _ = load_folds()

    _sync_in_channels(X_cls)

    print("Extracting hand-crafted features for classical ML...")
    Xf_cls = _extract_features_respecting_fs(X_cls, meta)
    Xf_reg = _extract_features_respecting_fs(X_reg, meta)
    if CFG.preprocessing.context_baselines or CFG.preprocessing.context_lags_sec:
        X_ctx, _, _ = load_processed("context")
        print(f"Adding {X_ctx.shape[1]} context features to the regression features")
        Xf_reg = np.hstack([Xf_reg, X_ctx])

    print("\n" + "="*60)
    print("  Phase 6: Classical Machine Learning Baselines")
    print("="*60)

    print("\nTrivial baselines: majority class / train-fold mean angle")
    save_results(run_classification_cv(Xf_cls, y_cls, folds, model_name="majority"),
                 "baseline_clf_majority")
    save_results(run_regression_cv(Xf_reg, y_reg, folds, model_name="mean", metadata=meta),
                 "baseline_reg_mean")

    for clf in CFG.cv.classical_classifiers:
        print(f"\nTraining Classical Classifier: {clf.upper()}")
        r = run_classification_cv(Xf_cls, y_cls, folds, model_name=clf)
        save_results(r, f"classical_clf_{clf}")

    for reg in CFG.cv.classical_regressors:
        print(f"\nTraining Classical Regressor: {reg.upper()}")
        r = run_regression_cv(Xf_reg, y_reg, folds, model_name=reg, metadata=meta)
        save_results(r, f"classical_reg_{reg}")


def run_train_deep():
    print("\n" + "=" * 60)
    print(f"  Phase 7: Deep Multi-Task Training ({CFG.model.model_type.upper()})")
    print("=" * 60)
    from src.data.datasets import load_processed
    from src.training.cv_splits import load_folds
    from src.training.train import train_cv

    X_cls, y_cls, meta = load_processed("classification")
    X_reg, y_reg, _ = load_processed("regression")
    folds, _ = load_folds()

    _sync_in_channels(X_cls)

    context = None
    if CFG.model.context_features:
        context, _, _ = load_processed("context")
        CFG.model.context_dim = int(context.shape[1])
        print(f"The deep model gets {context.shape[1]} context features")
    train_cv(X_cls, y_cls, y_reg, folds, metadata=meta, context=context)


def run_ablations():
    print("\n" + "=" * 60)
    print("  Phases 9 & 10: Ablation Studies")
    print("=" * 60)
    from scripts.run_ablations import (
        rebuild_angle_preds,
        run_head_pose_ablation,
        run_direction_ablation_full,
    )

    angle_pred, angle_true, test_meta = rebuild_angle_preds()

    print("\n--- Phase 9: Head-Pose Ablation ---")
    run_head_pose_ablation(angle_pred, angle_true, test_meta)

    print("\n--- Phase 10: Direction-Invariance Ablation ---")
    run_direction_ablation_full(angle_pred, angle_true, test_meta)


def run_calibrate():
    print("\n" + "=" * 60)
    print("  Few-Shot Subject Calibration Evaluation")
    print("=" * 60)
    from src.evaluation.calibration import run_calibration_pipeline
    run_calibration_pipeline(model_type=CFG.model.model_type)


def run_known_start():
    print("\n" + "=" * 60)
    print("  Known-start protocol (separate from the main results)")
    print("=" * 60)
    from src.data.loaders import load_all_datasets
    from src.data.unify import to_common_schema
    from src.evaluation.known_start import run_known_start as evaluate

    trials = [to_common_schema(t) for t in load_all_datasets(skip_missing=True)]
    if not trials:
        print("No datasets found. Please download datasets into data/raw/ as described in README.md.")
        return
    evaluate(trials)


def run_compare():
    print("\n" + "=" * 60)
    print("  Phase 8: Master Comparison Table")
    print("=" * 60)
    from src.evaluation.compare_to_paper import run_comparison
    run_comparison()


def main():
    parser = argparse.ArgumentParser(description="EOG Eye Gaze Estimation Pipeline Runner")
    parser.add_argument(
        "--phase",
        choices=["inspect", "preprocess", "train_classical", "train_deep", "ablations", "calibrate",
                 "compare", "known_start", "all"],
        default="all",
        help="Pipeline phase to run",
    )
    parser.add_argument(
        "--model-type",
        choices=["conv1d", "lstm", "conv_bilstm"],
        default=None,
        help="Deep model architecture type (default: CFG.model.model_type)",
    )
    parser.add_argument(
        "--loss-type",
        choices=["static", "uncertainty", "range_preserving"],
        default=None,
        help="Loss weighting mode (default: CFG.model.loss_type)",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--grid-search", action="store_true", help="Enable classical ML grid search")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    args = parser.parse_args()

    if args.config:
        load_config_from_yaml(args.config)

    if args.model_type is not None:
        CFG.model.model_type = args.model_type
    if args.loss_type is not None:
        CFG.model.loss_type = args.loss_type
    if args.epochs is not None:
        CFG.model.max_epochs = args.epochs
    if args.batch_size is not None:
        CFG.model.batch_size = args.batch_size
    if args.lr is not None:
        CFG.model.lr = args.lr
    if args.grid_search:
        CFG.cv.use_grid_search = True

    if args.phase == "inspect":
        run_inspect()
    elif args.phase == "preprocess":
        run_preprocess()
    elif args.phase == "train_classical":
        run_train_classical()
    elif args.phase == "train_deep":
        run_train_deep()
    elif args.phase == "ablations":
        run_ablations()
    elif args.phase == "calibrate":
        run_calibrate()
    elif args.phase == "compare":
        run_compare()
    elif args.phase == "known_start":
        run_known_start()
    elif args.phase == "all":
        run_inspect()
        try:
            run_preprocess()
            run_train_classical()
            run_train_deep()
            run_ablations()
            run_compare()
        except FileNotFoundError as e:
            print(f"\n[INFO] Pipeline stopped: raw data missing. Follow dataset download instructions in README.md.\n  {e}")


if __name__ == "__main__":
    main()
