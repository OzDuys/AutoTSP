#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, Iterable, List, Tuple

import joblib

# Ensure we can import AutoTSP from project root.
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INSTANCE_DATASETS_DIR = PROJECT_ROOT / "Instance Datasets"
INSTANCE_ALGO_DIR = PROJECT_ROOT / "Instance-Algorithm Datasets"
DEFAULT_PROBLEMS_PATH = INSTANCE_DATASETS_DIR / "problems.jsonl"
DEFAULT_RESULTS_PATH = INSTANCE_ALGO_DIR / "Full Dataset" / "results.jsonl"
DEFAULT_MODEL_OUT = INSTANCE_ALGO_DIR / "Full Dataset" / "random_forest_selector.pkl"

from AutoTSP.features import FeatureExtractor
from AutoTSP.selectors import RandomForestSelector


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a RandomForest selector from benchmark results.")
    parser.add_argument(
        "--problems",
        type=pathlib.Path,
        default=DEFAULT_PROBLEMS_PATH,
        help="JSONL file of problem instances used during benchmarking.",
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_PATH,
        help="JSONL file of benchmark results (one row per problem/algorithm).",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=5.0,
        help="Maximum allowed elapsed time to consider a run feasible (seconds).",
    )
    parser.add_argument(
        "--model-out",
        type=pathlib.Path,
        default=DEFAULT_MODEL_OUT,
        help="Destination path for the pickled (model, feature_order) tuple.",
    )
    parser.add_argument(
        "--training-out",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL path to dump the derived training rows (features + best_solver).",
    )
    return parser.parse_args(raw_args)


def load_jsonl(path: pathlib.Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def best_solver_per_problem(results: List[dict], time_budget: float) -> Dict[str, Tuple[float, float, str]]:
    """
    Return mapping: problem_id -> (cost, elapsed, solver_name) for best cost within budget.
    """
    best: Dict[str, Tuple[float, float, str]] = {}
    for row in results:
        pid = row.get("problem_id")
        algo = row.get("algorithm")
        cost = row.get("cost")
        elapsed = row.get("elapsed", float("inf"))
        status = row.get("status")
        if pid is None or algo is None or cost is None:
            continue
        if status not in {"complete", "success"}:
            continue
        if elapsed is None or elapsed > time_budget:
            continue
        prev = best.get(pid)
        if prev is None or cost < prev[0] or (cost == prev[0] and elapsed < prev[1]):
            best[pid] = (float(cost), float(elapsed), algo)
    return best


def build_training_rows(problems: List[dict], results: List[dict], time_budget: float) -> List[dict]:
    problems_by_id = {row.get("problem_id"): row for row in problems if row.get("problem_id")}
    best_map = best_solver_per_problem(results, time_budget)
    rows: List[dict] = []
    for pid, (cost, elapsed, algo) in best_map.items():
        problem = problems_by_id.get(pid)
        if not problem:
            continue
        features = FeatureExtractor.extract(problem).values
        features = dict(features)
        features["time_budget"] = float(time_budget)
        features["remaining_budget"] = float(time_budget)  # training data assumed full budget available
        rows.append({"problem_id": pid, "best_solver": algo, "cost": cost, "elapsed": elapsed, "features": features})
    return rows


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)
    problems = load_jsonl(args.problems)
    results = load_jsonl(args.results)
    rows = build_training_rows(problems, results, args.time_budget)

    if not rows:
        raise SystemExit("No training rows derived; check inputs.")

    rf = RandomForestSelector()
    rf.fit(rows)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump((rf.model, rf.feature_order), args.model_out)
    print(f"Saved model to {args.model_out} (features: {len(rf.feature_order or [])})")

    if args.training_out:
        args.training_out.parent.mkdir(parents=True, exist_ok=True)
        with args.training_out.open("w", encoding="utf-8") as out:
            for row in rows:
                out.write(json.dumps(row))
                out.write("\n")
        print(f"Wrote training rows to {args.training_out} ({len(rows)} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
