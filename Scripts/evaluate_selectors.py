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
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

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
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "Scripts" / "Selector_evaluation_plots"

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
        "--training-rows",
        type=pathlib.Path,
        default=None,
        help="Optional JSONL of precomputed training rows (features + labels) to speed up penalty computation.",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Ignore any precomputed training rows when computing penalties and recompute features from problems.",
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
    parser.add_argument(
        "--num-single-best",
        type=int,
        default=1,
        help="Number of top fixed algorithms to include as baselines (by average penalised cost).",
    )
    parser.add_argument(
        "--plot-dir",
        type=pathlib.Path,
        default=DEFAULT_PLOTS_DIR,
        help="If set, write summary plots (bar charts and time–cost scatter) into this directory.",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="If set, skip instances with more than this many cities when evaluating selectors.",
    )
    parser.add_argument(
        "--metrics-out",
        type=pathlib.Path,
        default=None,
        help="Optional path to write selector-level summary metrics as JSONL (defaults to plot-dir/evaluation.jsonl if plot-dir is set).",
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
    num_cities: int
    cost: float
    cost_normalized: float
    elapsed_solver: float
    feature_time: float
    selector_overhead: float
    oracle_algo: str
    oracle_cost: float
    oracle_elapsed: float
    top2_hit: bool
    top3_hit: bool
    solved: bool

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
    training_rows_path: Optional[pathlib.Path] = None,
    force_recompute: bool = False,
    max_cities: Optional[int] = None,
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
        training_rows_file = training_rows_path or DEFAULT_TRAINING_ROWS
        if not force_recompute and training_rows_file.exists():
            with training_rows_file.open("r", encoding="utf-8") as fh:
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
            if max_cities is not None:
                num_cities_val = problem.get("num_cities") or problem.get("n_nodes")
                try:
                    n_int = int(num_cities_val) if num_cities_val is not None else 0
                except Exception:
                    n_int = 0
                if n_int > max_cities:
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
) -> Tuple[Dict[str, Mapping[str, Any]], List[str]]:
    """
    Returns:
      oracle_by_pid: problem_id -> best row within budget
      sba_algos: names of fixed algorithms sorted by average cost (may be empty)
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

            if is_valid_run(r, time_budget) and cost < best_cost:
                best_cost = cost
                best_algo = algo
                best_elapsed = elapsed

        if best_algo is not None:
            oracle_by_pid[pid] = {"algorithm": best_algo, "cost": best_cost, "elapsed": best_elapsed}

    if not algo_stats:
        return oracle_by_pid, []

    # Sort algorithms by average cost ascending.
    sba_sorted = sorted(
        algo_stats.items(),
        key=lambda kv: (kv[1]["sum_cost"] / max(kv[1]["count"], 1.0)),
    )
    sba_algos = [name for name, _ in sba_sorted]

    return oracle_by_pid, sba_algos


def evaluate_selectors(
    problems: List[Mapping[str, Any]],
    grouped_results: Mapping[str, List[Mapping[str, Any]]],
    oracle_by_pid: Mapping[str, Mapping[str, Any]],
    sba_algos: List[str],
    rf_model_path: pathlib.Path,
    penalty_base: Mapping[str, float],
    failure_penalty_multiplier: float,
    time_budget: float,
    max_cities: Optional[int] = None,
    num_single_best: int = 1,
) -> List[PerInstanceOutcome]:
    # Load RF model and feature order.
    model, feature_order = joblib.load(rf_model_path)
    rf_selector = RandomForestSelector(model=model, feature_order=feature_order)
    rb_selector = RuleBasedSelector()

    outcomes: List[PerInstanceOutcome] = []

    for problem in tqdm(problems, desc="Evaluating selectors", unit="problem"):
        pid = problem.get("problem_id")
        if not pid:
            continue
        pid_str = str(pid)
        num_cities_val = problem.get("num_cities") or problem.get("n_nodes")
        try:
            num_cities_int = int(num_cities_val) if num_cities_val is not None else 0
        except Exception:
            num_cities_int = 0
        if max_cities is not None and num_cities_int > max_cities:
            continue
        oracle_row = oracle_by_pid.get(pid_str)
        rows = grouped_results.get(pid_str)
        base = penalty_base.get(pid_str)
        if oracle_row is None or not rows or base is None or not math.isfinite(base):
            continue
        penalty_cost = base * float(failure_penalty_multiplier)
        oracle_cost = float(oracle_row.get("cost", penalty_cost))

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

        # Build per-algorithm penalised costs for top-k accuracy.
        algo_costs: List[Tuple[str, float]] = []
        for algo_name, r in runs_by_algo.items():
            if is_valid_run(r, time_budget):
                try:
                    c_val = float(r.get("cost", math.inf))
                except Exception:
                    c_val = penalty_cost
            else:
                c_val = penalty_cost
            algo_costs.append((algo_name, c_val))
        algo_costs.sort(key=lambda x: x[1])
        top2 = {a for a, _ in algo_costs[:2]}
        top3 = {a for a, _ in algo_costs[:3]}

        def add_outcome(selector_name: str, algo_name: str, selector_overhead: float) -> None:
            row = runs_by_algo.get(algo_name)
            if row is None or not is_valid_run(row, time_budget):
                # Treat missing/invalid runs as a penalised outcome.
                cost = penalty_cost
                elapsed_solver = float(time_budget)
                solved = False
            else:
                try:
                    cost = float(row.get("cost", penalty_cost))
                except Exception:
                    cost = penalty_cost
                try:
                    elapsed_solver = float(row.get("elapsed", time_budget))
                except Exception:
                    elapsed_solver = float(time_budget)
                solved = True

            norm_cost = cost / oracle_cost if oracle_cost and oracle_cost > 0 else float("nan")
            outcomes.append(
                PerInstanceOutcome(
                    problem_id=pid_str,
                    selector_name=selector_name,
                    chosen_algo=algo_name,
                    num_cities=num_cities_int,
                    cost=cost,
                    cost_normalized=norm_cost,
                    elapsed_solver=elapsed_solver,
                    feature_time=feature_time,
                    selector_overhead=selector_overhead,
                    oracle_algo=str(oracle_row.get("algorithm")),
                    oracle_cost=float(oracle_row.get("cost", penalty_cost)),
                    oracle_elapsed=float(oracle_row.get("elapsed", time_budget)),
                    top2_hit=algo_name in top2,
                    top3_hit=algo_name in top3,
                    solved=solved,
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

        # Single best algorithm baselines (if available).
        for idx, algo_name in enumerate(sba_algos):
            add_outcome(f"single_best_{idx+1}", algo_name, selector_overhead=0.0)
            if idx + 1 >= num_single_best:
                break

        # Oracle policy (upper bound).
        add_outcome("oracle", str(oracle_row.get("algorithm")), selector_overhead=0.0)

    return outcomes


def summarise(outcomes: Iterable[PerInstanceOutcome], min_instances: int = 1, metrics_out: Optional[pathlib.Path] = None) -> None:
    by_selector: Dict[str, List[PerInstanceOutcome]] = {}
    for o in outcomes:
        by_selector.setdefault(o.selector_name, []).append(o)

    metrics_records: List[dict] = []

    header = (
        f"{'selector':<16}"
        f"{'n_instances':>12}"
        f"{'top1_acc_vs_oracle':>20}"
        f"{'top2_acc':>12}"
        f"{'top3_acc':>12}"
        f"{'avg_cost':>14}"
        f"{'avg_cost_norm':>16}"
        f"{'avg_regret':>14}"
        f"{'median_regret':>16}"
        f"{'avg_total_time':>16}"
        f"{'solved_pct':>12}"
    )
    print(header)

    for name, rows in sorted(by_selector.items()):
        if len(rows) < min_instances:
            continue
        costs = [r.cost for r in rows]
        costs_norm = [r.cost_normalized for r in rows if math.isfinite(r.cost_normalized)]
        regrets = [r.regret for r in rows if math.isfinite(r.regret)]
        total_times = [r.total_time for r in rows]
        top1 = sum(1 for r in rows if r.chosen_algo == r.oracle_algo) / max(len(rows), 1)
        top2 = sum(1 for r in rows if r.top2_hit) / max(len(rows), 1)
        top3 = sum(1 for r in rows if r.top3_hit) / max(len(rows), 1)
        solved_pct = sum(1 for r in rows if r.solved) / max(len(rows), 1)

        avg_cost = statistics.fmean(costs) if costs else float("nan")
        avg_cost_norm = statistics.fmean(costs_norm) if costs_norm else float("nan")
        avg_regret = statistics.fmean(regrets) if regrets else float("nan")
        median_regret = statistics.median(regrets) if regrets else float("nan")
        avg_time = statistics.fmean(total_times) if total_times else float("nan")

        line = (
            f"{name:<16}"
            f"{len(rows):>12d}"
            f"{top1:>20.3f}"
            f"{top2:>12.3f}"
            f"{top3:>12.3f}"
            f"{avg_cost:>14.3f}"
            f"{avg_cost_norm:>16.3f}"
            f"{avg_regret:>14.3f}"
            f"{median_regret:>16.3f}"
            f"{avg_time:>16.4f}"
            f"{solved_pct:>12.3f}"
        )
        print(line)

        metrics_records.append(
            {
                "selector": name,
                "n_instances": len(rows),
                "top1_acc": top1,
                "top2_acc": top2,
                "top3_acc": top3,
                "avg_cost": avg_cost,
                "avg_cost_normalized": avg_cost_norm,
                "avg_regret": avg_regret,
                "median_regret": median_regret,
                "avg_total_time": avg_time,
                "solved_pct": solved_pct,
            }
        )

    if metrics_out:
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with metrics_out.open("w", encoding="utf-8") as fh:
            for rec in metrics_records:
                fh.write(json.dumps(rec))
                fh.write("\n")


def plot_summary(outcomes: Iterable[PerInstanceOutcome], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_selector: Dict[str, List[PerInstanceOutcome]] = {}
    for o in outcomes:
        by_selector.setdefault(o.selector_name, []).append(o)

    def selector_stats(name: str, rows: List[PerInstanceOutcome]) -> Dict[str, float]:
        costs_norm = [r.cost_normalized for r in rows if math.isfinite(r.cost_normalized)]
        regrets = [r.regret for r in rows if math.isfinite(r.regret)]
        total_times = [r.total_time for r in rows]
        return {
            "avg_cost_norm": statistics.fmean(costs_norm) if costs_norm else float("nan"),
            "avg_regret": statistics.fmean(regrets) if regrets else float("nan"),
            "avg_time": statistics.fmean(total_times) if total_times else float("nan"),
        }

    stats = {name: selector_stats(name, rows) for name, rows in by_selector.items()}

    # Bar plot: average regret.
    selectors = list(stats.keys())
    avg_regrets = [stats[s]["avg_regret"] for s in selectors]
    plt.figure(figsize=(8, 4))
    plt.bar(selectors, avg_regrets, color="#4F6BED")
    plt.ylabel("Average regret")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_regret.png", dpi=200)
    plt.close()

    # Bar plot: average normalized cost.
    avg_cost_norms = [stats[s]["avg_cost_norm"] for s in selectors]
    plt.figure(figsize=(8, 4))
    plt.bar(selectors, avg_cost_norms, color="#F28E2B")
    plt.ylabel("Average normalized cost")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_cost_normalized.png", dpi=200)
    plt.close()

    # Scatter: per-instance total time vs raw cost (cluster view).
    cmap = plt.get_cmap("tab10")
    color_map: Dict[str, str] = {name: cmap(i % 10) for i, name in enumerate(selectors)}

    plt.figure(figsize=(6, 6))
    for name in selectors:
        if name == "oracle":
            continue
        rows = by_selector.get(name, [])
        if not rows:
            continue
        xs = [r.total_time for r in rows if math.isfinite(r.cost)]
        ys = [r.cost for r in rows if math.isfinite(r.cost)]
        if not xs or not ys:
            continue
        plt.scatter(xs, ys, s=15, alpha=0.7, color=color_map[name], label=name)
    plt.xlabel("Total time per instance (s)")
    plt.ylabel("Cost per instance")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "time_vs_cost.png", dpi=200)
    plt.close()

    plt.figure(figsize=(6, 6))
    for name in selectors:
        rows = by_selector.get(name, [])
        if not rows:
            continue
        xs = [r.total_time for r in rows if math.isfinite(r.cost)]
        ys = [r.cost for r in rows if math.isfinite(r.cost)]
        if not xs or not ys:
            continue
        plt.scatter(xs, ys, s=15, alpha=0.7, color=color_map[name], label=name)
    plt.xlabel("Total time per instance (s)")
    plt.ylabel("Cost per instance")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "time_vs_cost_with_oracle.png", dpi=200)
    plt.close()

    # Boxplot of per-instance regrets.
    data = []
    labels = []
    for name, rows in by_selector.items():
        vals = [r.regret for r in rows if math.isfinite(r.regret)]
        if vals:
            data.append(vals)
            labels.append(name)
    if data:
        plt.figure(figsize=(10, 5))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.ylabel("Regret")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(out_dir / "regret_boxplot.png", dpi=200)
        plt.close()


def plot_selection_patterns(outcomes: Iterable[PerInstanceOutcome], out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_selector: Dict[str, List[PerInstanceOutcome]] = {}
    for o in outcomes:
        by_selector.setdefault(o.selector_name, []).append(o)

    # 1) Cumulative accuracy vs city count (instances sorted by size).
    plt.figure(figsize=(8, 6))
    # Prefer a specific order if present.
    preferred_order = ["oracle", "random_forest", "rule_based"]
    preferred_order += [s for s in sorted(by_selector.keys()) if s.startswith("single_best")]
    seen = set()
    ordered_selectors: List[str] = []
    for s in preferred_order:
        if s in by_selector and s not in seen:
            ordered_selectors.append(s)
            seen.add(s)
    for s in sorted(by_selector.keys()):
        if s not in seen:
            ordered_selectors.append(s)
            seen.add(s)

    cmap_sel = plt.get_cmap("tab10")
    selector_colors: Dict[str, str] = {name: cmap_sel(i % 10) for i, name in enumerate(ordered_selectors)}

    for name in ordered_selectors:
        rows = by_selector.get(name, [])
        if not rows:
            continue
        rows_sorted = sorted(rows, key=lambda r: r.num_cities)
        sizes = [r.num_cities for r in rows_sorted]
        correct_flags = [r.chosen_algo == r.oracle_algo for r in rows_sorted]
        cum_correct = 0
        cum_acc: List[float] = []
        for i, ok in enumerate(correct_flags, start=1):
            if ok:
                cum_correct += 1
            cum_acc.append(cum_correct / i)
        plt.plot(sizes, cum_acc, marker="o", label=name)

    plt.xlabel("Number of cities (sorted)")
    plt.ylabel("Cumulative accuracy vs oracle")
    plt.xscale("log")
    plt.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "selection_cumulative_accuracy.png", dpi=200)
    plt.close()

    # 2) Algorithm choice vs city size for each selector (color-coded algorithms).
    algo_colors: Dict[str, str] = {
        "simulated_annealing": "#1b9e77",
        "lkh": "#d95f02",
        "concorde_exact": "#7570b3",
        "three_opt": "#e7298a",
        "two_opt": "#a6761d",
        "simple_nearest_neighbor": "#66a61e",
        "multi_start_nearest_neighbor": "#e6ab02",
        "iterated_local_search": "#e6a0c4",
        "genetic_algorithm": "#ffd92f",
        "ant_colony": "#a1d99b",
        "christofides": "#1f78b4",
        "shinka_spatial_heuristic": "#b15928",
        "branch_and_bound": "#fb9a99",
        "held_karp": "#cab2d6",
        "cutting_plane": "#b2df8a",
    }
    default_color = "#999999"

    plt.figure(figsize=(10, 4))
    selector_positions: Dict[str, float] = {}
    row_spacing = 0.5  # Pull selectors closer together so the plot is less tall.
    for idx, name in enumerate(ordered_selectors):
        selector_positions[name] = float(idx) * row_spacing
    for name in ordered_selectors:
        rows = by_selector.get(name, [])
        if not rows:
            continue
        y = selector_positions[name]
        xs = [r.num_cities for r in rows]
        cs = [algo_colors.get(r.chosen_algo, default_color) for r in rows]
        plt.scatter(xs, [y] * len(xs), c=cs, s=20, alpha=0.8, label=name)

    # Build a separate legend for algorithms.
    algo_labels = sorted(set(r.chosen_algo for r in outcomes))
    handles = []
    labels = []
    for algo in algo_labels:
        color = algo_colors.get(algo, default_color)
        handles.append(plt.Line2D([], [], marker="o", linestyle="", color=color))
        labels.append(algo)
    plt.legend(handles, labels, title="Algorithms", bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    plt.yticks(list(selector_positions.values()), list(selector_positions.keys()))
    plt.xlabel("Number of cities")
    plt.xscale("log")
    plt.ylabel("Selector")
    plt.tight_layout()
    plt.savefig(out_dir / "selection_algorithms_vs_size.png", dpi=200, bbox_inches="tight")
    plt.close()

    # 3) Total time vs city size for each selector.
    plt.figure(figsize=(8, 6))
    for name in ordered_selectors:
        rows = by_selector.get(name, [])
        if not rows:
            continue
        xs = [r.num_cities for r in rows]
        ys = [r.total_time for r in rows]
        if not xs or not ys:
            continue
        plt.scatter(xs, ys, s=15, alpha=0.7, color=selector_colors.get(name, "#666666"), label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of cities")
    plt.ylabel("Total time per instance (s)")
    plt.legend(loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_dir / "time_vs_size.png", dpi=200)
    plt.close()

    # 4) Write a JSONL log of selections for further custom plotting.
    log_path = out_dir / "selection_log.jsonl"
    with log_path.open("w", encoding="utf-8") as fh:
        for o in outcomes:
            fh.write(
                json.dumps(
                    {
                        "problem_id": o.problem_id,
                        "selector": o.selector_name,
                        "chosen_algo": o.chosen_algo,
                        "oracle_algo": o.oracle_algo,
                        "num_cities": o.num_cities,
                        "regret": o.regret,
                    }
                )
            )
            fh.write("\n")


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)

    problems = load_jsonl(args.problems)
    results = load_jsonl(args.results)

    # Optional subsetting: restrict by max_cities before anything else.
    if args.max_cities is not None:
        max_c = args.max_cities
        filtered_ids: set[str] = set()
        filtered_problems = []
        for p in problems:
            num_cities_val = p.get("num_cities") or p.get("n_nodes")
            try:
                n_int = int(num_cities_val) if num_cities_val is not None else 0
            except Exception:
                n_int = 0
            if n_int > max_c:
                continue
            pid = p.get("problem_id")
            if pid:
                filtered_ids.add(str(pid))
            filtered_problems.append(p)
        problems = filtered_problems
        if filtered_ids:
            results = [r for r in results if str(r.get("problem_id")) in filtered_ids]

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
    penalty_base = compute_penalty_base(
        problems,
        relevant_ids=grouped.keys(),
        penalty_type=args.penalty_type,
        training_rows_path=args.training_rows,
        force_recompute=args.force_recompute_features,
        max_cities=args.max_cities,
    )
    if not penalty_base:
        raise SystemExit("Could not compute nn_probe_cost-based penalties for any instances.")
    # Rebuild oracle/SBA using penalised costs so that timeouts/failed runs are properly penalised.
    oracle_by_pid, sba_algos = build_oracle_and_sba(
        grouped,
        penalty_base=penalty_base,
        failure_penalty_multiplier=2.0,
        time_budget=args.time_budget,
    )

    if not oracle_by_pid:
        raise SystemExit("No valid oracle runs found within the given time budget.")

    model_path = pick_model_path(args.model)
    print(f"Using RandomForest model from: {model_path}")
    if sba_algos:
        print(f"Single-best algorithm baseline(s): {sba_algos[: max(1, args.num_single_best)]}")
    else:
        print("Single-best algorithm baselines could not be determined.")

    outcomes = evaluate_selectors(
        problems=problems,
        grouped_results=grouped,
        oracle_by_pid=oracle_by_pid,
        sba_algos=sba_algos[: max(1, args.num_single_best)],
        rf_model_path=model_path,
        penalty_base=penalty_base,
        failure_penalty_multiplier=2.0,
        time_budget=args.time_budget,
        max_cities=args.max_cities,
        num_single_best=max(1, args.num_single_best),
    )

    if not outcomes:
        raise SystemExit("No evaluation outcomes produced; check inputs and time budget.")

    print(f"Evaluated selectors on {len({o.problem_id for o in outcomes})} instances.")
    metrics_path = args.metrics_out
    if metrics_path is None and args.plot_dir:
        metrics_path = args.plot_dir / "evaluation.jsonl"
    summarise(outcomes, min_instances=args.min_instances, metrics_out=metrics_path)

    if args.plot_dir:
        args.plot_dir.mkdir(parents=True, exist_ok=True)
        plot_summary(outcomes, args.plot_dir)
        plot_selection_patterns(outcomes, args.plot_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
