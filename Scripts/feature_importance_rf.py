#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np

"""
Command Line command to run:
python Scripts/feature_importance_rf.py --model path/to/model.pkl --output path/to/output.png --top-k 10 --dump-json path/to/importances.json

    
"""
def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot and dump Random Forest feature importances.")
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default="Data/Instance-Algorithm Datasets/random_forest_selector_train.pkl",
        help="Path to pickled (RandomForest model, feature_order) tuple.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default="Data/Instance-Algorithm Datasets/rf_feature_importance.png",
        help="Optional path for a bar-plot PNG. If omitted, no plot is written.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=9,
        help="If set, only show the top-k features.",
    )
    parser.add_argument(
        "--dump-json",
        type=pathlib.Path,
        default="Data/Instance-Algorithm Datasets/rf_feature_importance.json",
        help="Optional path to dump sorted importances as JSON.",
    )
    return parser.parse_args(raw_args)


def load_model(path: pathlib.Path) -> Tuple[object, List[str]]:
    model, feature_order = joblib.load(path)
    if feature_order is None:
        raise ValueError("Model file does not include feature_order.")
    return model, list(feature_order)


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)
    model, feature_order = load_model(args.model)
    if not hasattr(model, "feature_importances_"):
        raise SystemExit("Model does not expose feature_importances_.")

    importances = np.asarray(model.feature_importances_, dtype=float)
    pairs = sorted(zip(feature_order, importances), key=lambda kv: kv[1], reverse=True)
    if args.top_k is not None and args.top_k > 0:
        pairs = pairs[: args.top_k]

    print("Feature importances (sorted):")
    for name, val in pairs:
        print(f"  {name:30s} {val:.4f}")

    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_json.open("w", encoding="utf-8") as fh:
            json.dump([{"feature": n, "importance": float(v)} for n, v in pairs], fh, indent=2)
        print(f"Wrote JSON importances to {args.dump_json}")

    if args.output:
        names = [n for n, _ in pairs]
        vals = [v for _, v in pairs]
        plt.figure(figsize=(8, 2 + 0.2 * len(pairs)))
        bars = plt.barh(range(len(pairs)), vals, color="#4F6BED")
        plt.gca().invert_yaxis()
        plt.yticks(range(len(pairs)), names)
        plt.xlabel("Gini importance")
        plt.title("Random Forest Feature Importances")
        # Give annotations a little breathing room so they do not touch the frame.
        max_val = max(vals) if vals else 0.0
        plt.xlim(0, max_val * 1.1 if max_val else 1.0)
        # Annotate values on bars for readability.
        for bar, val in zip(bars, vals):
            plt.text(val + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2, f"{val:.3f}", va="center")
        plt.tight_layout()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output, dpi=200)
        plt.close()
        print(f"Wrote bar plot to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
