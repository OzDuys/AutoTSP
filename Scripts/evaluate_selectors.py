#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import joblib
import numpy as np

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
DEFAULT_TRAINING_ROWS = INSTANCE_ALGO_DIR / "Full Dataset" / "rf_training.jsonl"

# Prefer the training-time model location if present; fall back to the copy under AutoTSP.
DEFAULT_MODEL_CANDIDATES = [
    INSTANCE_ALGO_DIR / "Full Dataset" / "random_forest_selector.pkl",
    PROJECT_ROOT / "AutoTSP" / "selectors" / "selector ml models" / "random_forest_selector.pkl",
]

from AutoTSP.features import FeatureExtractor
from AutoTSP.selectors import RandomForestSelector, RuleBasedSelector
from AutoTSP.solvers import SOLVER_REGISTRY


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-based vs RandomForest selectors.")
    parser.add_argument(
        "--problems",
        type=pathlib.Path,
        default=DEFAULT_PROBLEMS_PATH,
        help="JSONL file of problem instances.",
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_PATH,
        help="JSONL file of benchmark results (one row per problem/algorithm).",
    )
    parser.add_argument(
        "--model",
        type=pathlib.Path,
        default=None,
        help="Path to pickled (RandomForest model, feature_order) tuple. "
        "If omitted, tries common default locations.",
    )
    parser.add_argument(
        "--time-budget",
        type=float,
        default=5.0,
        help="Per-instance time budget used when defining the oracle and regrets (seconds).",
    )
    parser.add_argument(
        "--penalty-type",
        type=str,
        choices=["nn_full", "geom"],
        default="nn_full",
        help="Scheme for defining per-instance failure penalties: "
        "'nn_full' scales nn_probe_cost up to a full-tour estimate; "
        "'geom' uses a simple geometric upper bound (n * max_edge_length).",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="If set, evaluate on at most this many problems "
        "(by taking the first N problem_ids from the problems file).",
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=1,
        help="Minimum number of instances a selector must cover to be reported.",
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


def pick_model_path(explicit: Optional[pathlib.Path]) -> pathlib.Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(explicit)
        return explicit
    for cand in DEFAULT_MODEL_CANDIDATES:
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "No RandomForest selector model found. "
        "Tried: " + ", ".join(str(p) for p in DEFAULT_MODEL_CANDIDATES),
    )


def group_results_by_problem(results: Iterable[Mapping[str, Any]]) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for row in results:
        pid = row.get("problem_id")
        if not pid:
            continue
        grouped.setdefault(str(pid), []).append(row)
    return grouped


def is_valid_run(row: Mapping[str, Any], time_budget: float) -> bool:
    status = row.get("status")
    if status not in {"complete", "success"}:
        return False
    elapsed = row.get("elapsed")
    if elapsed is None:
        return False
    try:
        elapsed_f = float(elapsed)
    except Exception:
        return False
    return elapsed_f <= time_budget


@dataclass
class PerInstanceOutcome:
    problem_id: str
    selector_name: str
    chosen_algo: str
    cost: float
    elapsed_solver: float
    feature_time: float
    selector_overhead: float
    oracle_algo: str
    oracle_cost: float
    oracle_elapsed: float

    @property
    def total_time(self) -> float:
        return self.feature_time + self.selector_overhead + self.elapsed_solver

    @property
    def regret(self) -> float:
        return self.cost - self.oracle_cost


def compute_penalty_base(
    problems: Iterable[Mapping[str, Any]],
    relevant_ids: Optional[Iterable[str]] = None,
    penalty_type: str = "nn_full",
) -> Dict[str, float]:
    """
    Compute an instance-specific base cost for failure penalties.

    penalty_type:
      - 'nn_full': approximate full-tour length by scaling nn_probe_cost.
      - 'geom': simple geometric bound n * max_edge_length.
    """
    allowed: Optional[set[str]] = None
    if relevant_ids is not None:
        allowed = {str(pid) for pid in relevant_ids}

    penalty: Dict[str, float] = {}

    if penalty_type == "nn_full":
        # Prefer precomputed features from RF training rows if available.
        if DEFAULT_TRAINING_ROWS.exists():
            with DEFAULT_TRAINING_ROWS.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    pid = row.get("problem_id")
                    if not pid:
                        continue
                    pid_str = str(pid)
                    if allowed is not None and pid_str not in allowed:
                        continue
                    features = row.get("features") or {}
                    n = features.get("n_nodes")
                    base = features.get("nn_probe_cost")
                    if base is None or n is None:
                        continue
                    try:
                        n_f = float(n)
                        base_f = float(base)
                    except Exception:
                        continue
                    if not math.isfinite(base_f) or base_f <= 0.0 or not math.isfinite(n_f) or n_f <= 0.0:
                        continue
                    # Mirror FeatureExtractor's probe_size definition.
                    probe_size = max(1, int(max(1, 0.1 * n_f)))
                    full_est = base_f * (n_f / float(probe_size))
                    if full_est > 0.0 and math.isfinite(full_est):
                        penalty[pid_str] = full_est

        if penalty:
            return penalty

        # Fallback: recompute features directly from problem definitions.
        for problem in problems:
            pid = problem.get("problem_id")
            if not pid:
                continue
            pid_str = str(pid)
            if allowed is not None and pid_str not in allowed:
                continue
            try:
                features = FeatureExtractor.extract(problem)
            except Exception:
                continue
            values = features.values
            n = values.get("n_nodes")
            base = values.get("nn_probe_cost")
            if base is None or n is None:
                continue
            try:
                n_f = float(n)
                base_f = float(base)
            except Exception:
                continue
            if not math.isfinite(base_f) or base_f <= 0.0 or not math.isfinite(n_f) or n_f <= 0.0:
                continue
            probe_size = max(1, int(max(1, 0.1 * n_f)))
            full_est = base_f * (n_f / float(probe_size))
            if full_est > 0.0 and math.isfinite(full_est):
                penalty[pid_str] = full_est
        return penalty

    if penalty_type == "geom":
        # Geometric bound based on coordinates or distance matrix.
        for problem in problems:
            pid = problem.get("problem_id")
            if not pid:
                continue
            pid_str = str(pid)
            if allowed is not None and pid_str not in allowed:
                continue

            n = problem.get("num_cities") or problem.get("n_nodes")
            try:
                n_f = float(n) if n is not None else 0.0
            except Exception:
                n_f = 0.0
            if n_f <= 0.0:
                continue

            coords = problem.get("coordinates")
            dist_matrix = problem.get("distance_matrix")

            base_f: Optional[float] = None

            if coords is not None:
                arr = np.asarray(coords, dtype=float)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    span = arr.max(axis=0) - arr.min(axis=0)
                    diag = float(np.linalg.norm(span))
                    if math.isfinite(diag) and diag > 0.0:
                        base_f = n_f * diag
            elif dist_matrix is not None:
                arr = np.asarray(dist_matrix, dtype=float)
                if arr.ndim == 2 and arr.size > 0:
                    max_edge = float(np.max(arr))
                    if math.isfinite(max_edge) and max_edge > 0.0:
                        base_f = n_f * max_edge

            if base_f is not None and base_f > 0.0 and math.isfinite(base_f):
                penalty[pid_str] = base_f

        return penalty

    raise ValueError(f"Unknown penalty_type: {penalty_type}")


def build_oracle_and_sba(
    grouped: Mapping[str, List[Mapping[str, Any]]],
    penalty_base: Mapping[str, float],
    failure_penalty_multiplier: float,
    time_budget: float,
) -> Tuple[Dict[str, Mapping[str, Any]], Optional[str]]:
    """
    Returns:
      oracle_by_pid: problem_id -> best row within budget
      sba_algo: name of single-best algorithm (or None if cannot be determined)
    """
    oracle_by_pid: Dict[str, Mapping[str, Any]] = {}
    algo_stats: Dict[str, Dict[str, float]] = {}

    for pid, rows in grouped.items():
        base = penalty_base.get(pid)
        if base is None or not math.isfinite(base):
            continue
        penalty_cost = base * float(failure_penalty_multiplier)

        best_cost = math.inf
        best_algo: Optional[str] = None
        best_elapsed = float(time_budget)

        for r in rows:
            algo = str(r.get("algorithm"))
            if is_valid_run(r, time_budget):
                try:
                    cost = float(r.get("cost", math.inf))
                    elapsed = float(r.get("elapsed", time_budget))
                except Exception:
                    cost = penalty_cost
                    elapsed = float(time_budget)
            else:
                cost = penalty_cost
                elapsed = float(time_budget)

            stats = algo_stats.setdefault(algo, {"count": 0.0, "sum_cost": 0.0})
            stats["count"] += 1.0
            stats["sum_cost"] += cost

            if cost < best_cost:
                best_cost = cost
                best_algo = algo
                best_elapsed = elapsed

        if best_algo is not None:
            oracle_by_pid[pid] = {"algorithm": best_algo, "cost": best_cost, "elapsed": best_elapsed}

    if not algo_stats:
        return oracle_by_pid, None

    # Define SBA as algorithm with lowest average cost.
    sba_algo = min(
        algo_stats.items(),
        key=lambda kv: (kv[1]["sum_cost"] / max(kv[1]["count"], 1.0)),
    )[0]

    return oracle_by_pid, sba_algo


def evaluate_selectors(
    problems: List[Mapping[str, Any]],
    grouped_results: Mapping[str, List[Mapping[str, Any]]],
    oracle_by_pid: Mapping[str, Mapping[str, Any]],
    sba_algo: Optional[str],
    rf_model_path: pathlib.Path,
    penalty_base: Mapping[str, float],
    failure_penalty_multiplier: float,
    time_budget: float,
) -> List[PerInstanceOutcome]:
    # Load RF model and feature order.
    model, feature_order = joblib.load(rf_model_path)
    rf_selector = RandomForestSelector(model=model, feature_order=feature_order)
    rb_selector = RuleBasedSelector()

    outcomes: List[PerInstanceOutcome] = []

    for problem in problems:
        pid = problem.get("problem_id")
        if not pid:
            continue
        pid_str = str(pid)
        oracle_row = oracle_by_pid.get(pid_str)
        rows = grouped_results.get(pid_str)
        base = penalty_base.get(pid_str)
        if oracle_row is None or not rows or base is None or not math.isfinite(base):
            continue
        penalty_cost = base * float(failure_penalty_multiplier)

        # Pre-compute features once per problem.
        feat_start = time.perf_counter()
        features = FeatureExtractor.extract(problem)
        feature_time = features.elapsed
        feat_elapsed = time.perf_counter() - feat_start
        # Use the measured feature_time from the extractor, but keep feat_elapsed as a sanity check.
        selector_features = dict(features.values)
        selector_features["time_budget"] = float(time_budget)
        remaining_budget = max(0.0, float(time_budget) - feature_time)
        selector_features["remaining_budget"] = remaining_budget

        # Index available runs by algorithm name for quick lookup.
        runs_by_algo: Dict[str, Mapping[str, Any]] = {}
        for r in rows:
            algo = str(r.get("algorithm"))
            # Prefer valid, within-budget runs.
            if algo not in runs_by_algo or is_valid_run(r, time_budget):
                runs_by_algo[algo] = r

        def add_outcome(selector_name: str, algo_name: str, selector_overhead: float) -> None:
            row = runs_by_algo.get(algo_name)
            if row is None or not is_valid_run(row, time_budget):
                # Treat missing/invalid runs as a penalised outcome.
                cost = penalty_cost
                elapsed_solver = float(time_budget)
            else:
                try:
                    cost = float(row.get("cost", penalty_cost))
                except Exception:
                    cost = penalty_cost
                try:
                    elapsed_solver = float(row.get("elapsed", time_budget))
                except Exception:
                    elapsed_solver = float(time_budget)

            outcomes.append(
                PerInstanceOutcome(
                    problem_id=pid_str,
                    selector_name=selector_name,
                    chosen_algo=algo_name,
                    cost=cost,
                    elapsed_solver=elapsed_solver,
                    feature_time=feature_time,
                    selector_overhead=selector_overhead,
                    oracle_algo=str(oracle_row.get("algorithm")),
                    oracle_cost=float(oracle_row.get("cost", penalty_cost)),
                    oracle_elapsed=float(oracle_row.get("elapsed", time_budget)),
                )
            )

        # Rule-based selector.
        start = time.perf_counter()
        rb_solver_cls = rb_selector.predict(selector_features, remaining_budget)
        rb_overhead = time.perf_counter() - start
        add_outcome("rule_based", rb_solver_cls.name, rb_overhead)

        # RandomForest selector.
        start = time.perf_counter()
        rf_solver_cls = rf_selector.predict(selector_features, remaining_budget)
        rf_overhead = time.perf_counter() - start
        add_outcome("random_forest", rf_solver_cls.name, rf_overhead)

        # Single best algorithm baseline (if available).
        if sba_algo is not None:
            add_outcome("single_best", sba_algo, selector_overhead=0.0)

        # Oracle policy (upper bound).
        add_outcome("oracle", str(oracle_row.get("algorithm")), selector_overhead=0.0)

    return outcomes


def summarise(outcomes: Iterable[PerInstanceOutcome], min_instances: int = 1) -> None:
    by_selector: Dict[str, List[PerInstanceOutcome]] = {}
    for o in outcomes:
        by_selector.setdefault(o.selector_name, []).append(o)

    header = (
        f"{'selector':<16}"
        f"{'n_instances':>12}"
        f"{'top1_acc_vs_oracle':>20}"
        f"{'avg_cost':>14}"
        f"{'avg_regret':>14}"
        f"{'median_regret':>16}"
        f"{'avg_total_time':>16}"
    )
    print(header)

    for name, rows in sorted(by_selector.items()):
        if len(rows) < min_instances:
            continue
        costs = [r.cost for r in rows]
        regrets = [r.regret for r in rows if math.isfinite(r.regret)]
        total_times = [r.total_time for r in rows]
        top1 = sum(1 for r in rows if r.chosen_algo == r.oracle_algo) / max(len(rows), 1)

        avg_cost = statistics.fmean(costs) if costs else float("nan")
        avg_regret = statistics.fmean(regrets) if regrets else float("nan")
        median_regret = statistics.median(regrets) if regrets else float("nan")
        avg_time = statistics.fmean(total_times) if total_times else float("nan")

        line = (
            f"{name:<16}"
            f"{len(rows):>12d}"
            f"{top1:>20.3f}"
            f"{avg_cost:>14.3f}"
            f"{avg_regret:>14.3f}"
            f"{median_regret:>16.3f}"
            f"{avg_time:>16.4f}"
        )
        print(line)


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)

    problems = load_jsonl(args.problems)
    results = load_jsonl(args.results)

    # Optional subsetting: restrict to at most N distinct problem_ids.
    if args.max_problems is not None and args.max_problems > 0:
        seen_ids: list[str] = []
        seen_set: set[str] = set()
        for row in problems:
            pid = row.get("problem_id")
            if not pid:
                continue
            pid_str = str(pid)
            if pid_str in seen_set:
                continue
            seen_ids.append(pid_str)
            seen_set.add(pid_str)
            if len(seen_ids) >= args.max_problems:
                break
        subset_ids = set(seen_ids)
        problems = [p for p in problems if str(p.get("problem_id")) in subset_ids]
        results = [r for r in results if str(r.get("problem_id")) in subset_ids]

    grouped = group_results_by_problem(results)
    penalty_base = compute_penalty_base(problems, relevant_ids=grouped.keys(), penalty_type=args.penalty_type)
    if not penalty_base:
        raise SystemExit("Could not compute nn_probe_cost-based penalties for any instances.")
    # Rebuild oracle/SBA using penalised costs so that timeouts/failed runs are properly penalised.
    oracle_by_pid, sba_algo = build_oracle_and_sba(
        grouped,
        penalty_base=penalty_base,
        failure_penalty_multiplier=2.0,
        time_budget=args.time_budget,
    )

    if not oracle_by_pid:
        raise SystemExit("No valid oracle runs found within the given time budget.")

    model_path = pick_model_path(args.model)
    print(f"Using RandomForest model from: {model_path}")
    if sba_algo is not None:
        print(f"Single-best algorithm baseline: {sba_algo}")
    else:
        print("Single-best algorithm baseline could not be determined.")

    outcomes = evaluate_selectors(
        problems=problems,
        grouped_results=grouped,
        oracle_by_pid=oracle_by_pid,
        sba_algo=sba_algo,
        rf_model_path=model_path,
        penalty_base=penalty_base,
        failure_penalty_multiplier=2.0,
        time_budget=args.time_budget,
    )

    if not outcomes:
        raise SystemExit("No evaluation outcomes produced; check inputs and time budget.")

    print(f"Evaluated selectors on {len({o.problem_id for o in outcomes})} instances.")
    summarise(outcomes, min_instances=args.min_instances)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
