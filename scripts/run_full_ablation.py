#!/usr/bin/env python3
"""Run full ablation study and dump all results to JSON.

Evaluates every checkpoint on the same test set(s) and saves
reproducible results for paper and SBIR proposal.

Usage:
    python scripts/run_full_ablation.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from services.ml.model import create_model

LABEL_NAMES = {0: "normal", 1: "maneuver", 2: "decay", 3: "breakup"}


def evaluate_checkpoint(ckpt_path: str, X: np.ndarray, y_true: np.ndarray,
                        size: str, n_features: int) -> dict:
    """Evaluate a single checkpoint. Returns metrics dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = create_model(size, n_features=n_features)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    X_tensor = torch.from_numpy(X[:, :, :n_features]).float()
    with torch.no_grad():
        out = model(X_tensor, causal=True)
        preds = out["classifications"].argmax(dim=-1).reshape(-1).numpy()

    y_flat = y_true.reshape(-1)
    results = {"accuracy": float((preds == y_flat).mean())}

    for c in range(4):
        tp = int(((preds == c) & (y_flat == c)).sum())
        fp = int(((preds == c) & (y_flat != c)).sum())
        fn = int(((preds != c) & (y_flat == c)).sum())
        total = int((y_flat == c).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(total, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)

        name = LABEL_NAMES[c]
        results[name] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": total,
            "tp": tp, "fp": fp, "fn": fn,
        }

    results["predicted_distribution"] = {
        LABEL_NAMES.get(k, str(k)): int(v)
        for k, v in zip(*np.unique(preds, return_counts=True))
    }

    return results


def main():
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    # Load test sets
    # IMM-UKF fused v3 (11 features) — primary test set
    X_imm_11 = np.load("data/ml_imm_fused_v3/X_test.npy")
    y_imm = np.load("data/ml_imm_fused_v3/y_test.npy")

    # Also load 6-feat version for older models
    X_imm_6 = X_imm_11[:, :, :6]
    X_imm_7 = X_imm_11[:, :, :7]

    print(f"Test set: {X_imm_11.shape[0]:,} sequences, {X_imm_11.shape[-1]} features")
    print(f"Labels: {dict(zip(*np.unique(y_imm.ravel(), return_counts=True)))}")
    print()

    all_results = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_set": "data/ml_imm_fused_v3/X_test.npy",
            "test_sequences": int(X_imm_11.shape[0]),
            "test_timesteps": int(X_imm_11.shape[0] * X_imm_11.shape[1]),
        },
        "models": {},
    }

    # Define all checkpoints to evaluate
    checkpoints = [
        # Progression: 6-feat models
        ("v0.8_imm_scratch_6feat", "checkpoints/v08_imm_fused/best_model.pt", "small", 6),
        ("v0.8b_twostage_6feat", "checkpoints/v08b_twostage/best_model.pt", "small", 6),
        ("v0.9_medium_twostage_7feat", "checkpoints/v09_medium_finetune/best_model.pt", "medium", 7),
        ("v0.9c_balanced_weights_7feat", "checkpoints/v09c_balanced/best_model.pt", "medium", 7),

        # A/B: 6 vs 7 features (dt_hours effect)
        ("ab_6feat_baseline", "checkpoints/ab_6feat/best_model.pt", "small", 6),
        ("ab_7feat_with_dt_hours", "checkpoints/ab_7feat/best_model.pt", "small", 7),

        # A/B/C: 6 vs 11 features + oversample (v3 normalization)
        ("ab2_6feat_v3norm", "checkpoints/ab2_6feat/best_model.pt", "small", 6),
        ("ab2_11feat_v3norm", "checkpoints/ab2_11feat/best_model.pt", "small", 11),
        ("ab2_11feat_oversample", "checkpoints/ab2_11feat_os/best_model.pt", "small", 11),

        # Final models
        ("v1.1_medium_11feat_twostage", "checkpoints/v11_medium_11feat_finetune/best_model.pt", "medium", 11),
        ("v1.1b_medium_11feat_oversample", "checkpoints/v11b_final/best_model.pt", "medium", 11),
    ]

    for name, path, size, n_feat in checkpoints:
        if not os.path.exists(path):
            print(f"  SKIP {name}: {path} not found")
            continue

        X = {6: X_imm_6, 7: X_imm_7, 11: X_imm_11}[n_feat]
        print(f"  Evaluating {name} ({size}, {n_feat}D)...", end=" ", flush=True)
        try:
            metrics = evaluate_checkpoint(path, X, y_imm, size, n_feat)
            metrics["checkpoint"] = path
            metrics["model_size"] = size
            metrics["n_features"] = n_feat
            all_results["models"][name] = metrics
            man = metrics["maneuver"]
            dec = metrics["decay"]
            print(f"Man R={man['recall']:.3f} F1={man['f1']:.3f} | "
                  f"Dec R={dec['recall']:.3f} F1={dec['f1']:.3f} | "
                  f"Acc={metrics['accuracy']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}")

    # Add data statistics
    all_results["data_stats"] = {
        "raw_tles": 232_380_556,
        "parsed_sequences": 8_613_300,
        "labeled_timesteps": 430_665_000,
        "n_features": 11,
        "imm_ukf_200sat": {
            "rule_v1_anomalies": 812,
            "imm_ukf_anomalies": 34_576,
            "ratio": 42.6,
        },
        "imm_ukf_500sat": {
            "satellites": 499,
            "timesteps": 998_000,
            "maneuver_labels": 148_278,
            "decay_labels": 24_503,
        },
    }

    # Save
    out_path = out_dir / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # Also print a summary table
    print("\n" + "=" * 90)
    print(f"{'Model':<40s} {'Feat':>4s} {'Size':>6s} {'Man R':>6s} {'Man F1':>6s} "
          f"{'Dec R':>6s} {'Dec F1':>6s} {'Acc':>6s}")
    print("-" * 90)
    for name, m in all_results["models"].items():
        man = m["maneuver"]
        dec = m["decay"]
        print(f"{name:<40s} {m['n_features']:>4d} {m['model_size']:>6s} "
              f"{man['recall']:>6.3f} {man['f1']:>6.3f} "
              f"{dec['recall']:>6.3f} {dec['f1']:>6.3f} {m['accuracy']:>6.3f}")


if __name__ == "__main__":
    main()
