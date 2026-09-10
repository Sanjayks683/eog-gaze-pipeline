"""
scripts/retrain_lstm.py
========================
Retrain the deep model from scratch using the current architecture
configured in CFG.model.model_type.

Usage:
    python scripts/retrain_lstm.py
    python scripts/retrain_lstm.py --model-type conv_bilstm
"""

import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import CFG
from src.data.datasets import load_processed
from src.training.train import train_cv
from src.training.cv_splits import load_folds


def main():
    parser = argparse.ArgumentParser(description="Retrain deep model")
    parser.add_argument("--model-type", type=str, default=None,
                        help="Model type: conv1d, lstm, conv_bilstm (default: CFG)")
    args = parser.parse_args()

    if args.model_type:
        CFG.model.model_type = args.model_type

    print(f"Loading preprocessed data for {CFG.model.model_type.upper()} training...")
    X_cls, y_cls, meta = load_processed("classification")
    X_reg, y_reg, _ = load_processed("regression")
    
    print("Loading CV folds...")
    folds, fold_meta = load_folds()

    print(f"Starting Deep Learning Training ({CFG.model.model_type.upper()})...")
    train_cv(X_cls, y_cls, y_reg, folds, metadata=meta)

    
if __name__ == "__main__":
    main()
