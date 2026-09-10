"""
src/evaluation/compare_to_paper.py
====================================
Build the master results table comparing:
  1. Trivial baselines: majority class / train-fold mean angle (Phase 6)
  2. Classical ML (Phase 6)
  3. Deep multi-task model (Phase 7)
  4. Published paper numbers (manually entered below)

Saves as CSV to reports/ and prints to stdout.
Units: degrees (°). Any condition mismatch vs. published papers is noted.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np

from src.config import CFG


PAPER_RESULTS = {
    "Barbara_2020_BSPC": {
        "description": "EOG baseline drift mitigation; stationary conditions",
        "dataset": "Dataset 2 (most comparable)",
        "rmse_h_deg": None,
        "rmse_v_deg": None,
        "mae_h_deg": None,
        "mae_v_deg": None,
        "notes": "Fill from Table in: Barbara et al., BSPC vol.57, Mar.2020",
    },
    "Barbara_2023_BSPC": {
        "description": "Real-time continuous EOG gaze estimation, stationary head",
        "dataset": "Dataset 2",
        "rmse_h_deg": 2.23,
        "rmse_v_deg": 2.39,
        "mae_h_deg": None,
        "mae_v_deg": None,
        "notes": "Barbara et al., BSPC vol.86, Sep.2023 (battery model under stationary head)",
    },
    "Barbara_2024_BSPC": {
        "description": "Real-time EOG gaze estimation, non-stationary head",
        "dataset": "Dataset 3",
        "rmse_h_deg": None,
        "rmse_v_deg": None,
        "mae_h_deg": None,
        "mae_v_deg": None,
        "notes": "Fill from: Barbara et al., BSPC vol.90, Apr.2024",
    },
    "Barbara_BSPC_Isotropic": {
        "description": "Bipolar channel selection, isotropic directions",
        "dataset": "Dataset 4",
        "rmse_h_deg": None,
        "rmse_v_deg": None,
        "mae_h_deg": None,
        "mae_v_deg": None,
        "notes": "Fill from: Barbara et al., BSPC (systematic quantitative analysis)",
    },
}


def load_result_safe(name: str, results_dir: str) -> Optional[Dict]:
    path = os.path.join(results_dir, f"{name}.json")
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _detect_dataset_label() -> str:
    """Determine dataset scope label based on config or processed metadata."""
    dsl = getattr(CFG.data, "datasets_to_load", None)
    if dsl == ["dataset2"] or (dsl and len(dsl) == 1 and "dataset2" in dsl):
        return "Dataset 2 only"
    elif dsl and len(dsl) < 4:
        return f"Subset ({', '.join(dsl)})"

    try:
        from src.data.datasets import load_processed
        _, _, meta = load_processed("classification")
        sources = sorted({m.get("dataset_source") for m in meta if m.get("dataset_source")})
        if sources == ["dataset2"]:
            return "Dataset 2 only"
        elif len(sources) == 1:
            return f"{sources[0].capitalize()} only"
        elif len(sources) < 4:
            return f"Subset ({', '.join(sources)})"
    except Exception:
        pass

    return "Pooled"


def build_master_table(results_dir: str = None) -> List[Dict]:
    """
    Build the master comparison table.

    Returns list of row dicts with keys:
      method, dataset, rmse_h_deg, rmse_v_deg, mae_h_deg, mae_v_deg,
      f1_weighted, notes
    """
    if results_dir is None:
        results_dir = CFG.paths.results

    rows = []
    dataset_label = _detect_dataset_label()

    d = load_result_safe("baseline_clf_majority", results_dir)
    if d:
        rows.append({
            "method": "Baseline: majority class",
            "dataset": dataset_label,
            "rmse_h_deg": None, "rmse_v_deg": None,
            "mae_h_deg": None, "mae_v_deg": None,
            "f1_weighted": d.get("pooled_f1_weighted"),
            "notes": "Always predicts the train fold's most frequent class",
        })
    d = load_result_safe("baseline_reg_mean", results_dir)
    if d:
        rows.append({
            "method": "Baseline: train-fold mean angle",
            "dataset": dataset_label,
            "rmse_h_deg": d.get("pooled_rmse_h_deg"),
            "rmse_v_deg": d.get("pooled_rmse_v_deg"),
            "mae_h_deg": d.get("pooled_mae_h_deg"),
            "mae_v_deg": d.get("pooled_mae_v_deg"),
            "f1_weighted": None,
            "notes": "Always predicts the train fold's mean H/V angle",
        })

    for clf_name in ["svc", "rf"]:
        d = load_result_safe(f"classical_clf_{clf_name}", results_dir)
        if d:
            rows.append({
                "method": f"Classical ML ({clf_name.upper()}) — classification",
                "dataset": dataset_label,
                "rmse_h_deg": None, "rmse_v_deg": None,
                "mae_h_deg": None, "mae_v_deg": None,
                "f1_weighted": d.get("pooled_f1_weighted"),
                "notes": "",
            })

    for reg_name in ["svr", "xgb"]:
        d = load_result_safe(f"classical_reg_{reg_name}", results_dir)
        if d:
            rows.append({
                "method": f"Classical ML ({reg_name.upper()}) — regression",
                "dataset": dataset_label,
                "rmse_h_deg": d.get("pooled_rmse_h_deg"),
                "rmse_v_deg": d.get("pooled_rmse_v_deg"),
                "mae_h_deg": d.get("pooled_mae_h_deg"),
                "mae_v_deg": d.get("pooled_mae_v_deg"),
                "f1_weighted": None,
                "notes": "",
            })

    for model_type in ["conv1d", "lstm", "conv_bilstm"]:
        deep = load_result_safe(f"deep_model_{model_type}", results_dir)
        if deep:
            arch = deep.get("model", model_type)
            rows.append({
                "method": f"Deep multi-task ({model_type} / {arch})",
                "dataset": dataset_label,
                "rmse_h_deg": deep.get("pooled_rmse_h_deg"),
                "rmse_v_deg": deep.get("pooled_rmse_v_deg"),
                "mae_h_deg": deep.get("pooled_mae_h_deg"),
                "mae_v_deg": deep.get("pooled_mae_v_deg"),
                "f1_weighted": deep.get("pooled_f1_weighted"),
                "notes": "Multi-task: classification + regression jointly",
            })
    if not any(r["method"].startswith("Deep multi-task") for r in rows):
        rows.append({
            "method": "Deep multi-task",
            "dataset": "—",
            "rmse_h_deg": None, "rmse_v_deg": None,
            "mae_h_deg": None, "mae_v_deg": None,
            "f1_weighted": None,
            "notes": "Run Phase 7 to generate deep_model_<type>.json",
        })

    for paper_key, paper_data in PAPER_RESULTS.items():
        if dataset_label == "Dataset 2 only" and "Dataset 2" not in paper_data["dataset"]:
            continue
        if dataset_label == "Dataset 2 only" and paper_data["rmse_h_deg"] is None:
            continue
        rows.append({
            "method": f"Published: {paper_key}",
            "dataset": paper_data["dataset"],
            "rmse_h_deg": paper_data["rmse_h_deg"],
            "rmse_v_deg": paper_data["rmse_v_deg"],
            "mae_h_deg": paper_data["mae_h_deg"],
            "mae_v_deg": paper_data["mae_v_deg"],
            "f1_weighted": None,
            "notes": paper_data["notes"],
        })

    return rows


def print_master_table(rows: List[Dict]) -> None:
    """Pretty-print the master results table."""
    print("\n" + "=" * 110)
    print(f"{'Method':<40} {'Dataset':<20} {'RMSE H°':>8} {'RMSE V°':>8} {'MAE H°':>8} {'F1 W':>8}")
    print("=" * 110)
    for row in rows:
        def fmt(v): return f"{v:.3f}" if v is not None else "  N/A "
        print(
            f"{row['method']:<40} {row['dataset']:<20} "
            f"{fmt(row['rmse_h_deg']):>8} {fmt(row['rmse_v_deg']):>8} "
            f"{fmt(row['mae_h_deg']):>8} {fmt(row['f1_weighted']):>8}"
        )
        if row.get("notes"):
            print(f"  -> {row['notes']}")
    print("=" * 110)


def save_master_table_csv(rows: List[Dict], results_dir: str = None) -> str:
    """Save master table as CSV."""
    if results_dir is None:
        results_dir = CFG.paths.results
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "master_results_table.csv")
    fieldnames = ["method", "dataset", "rmse_h_deg", "rmse_v_deg",
                  "mae_h_deg", "mae_v_deg", "f1_weighted", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Master table saved: {path}")
    return path


def run_comparison(results_dir: str = None) -> None:
    """Full comparison pipeline: build → print → save CSV."""
    rows = build_master_table(results_dir)
    print_master_table(rows)
    save_master_table_csv(rows, results_dir)
