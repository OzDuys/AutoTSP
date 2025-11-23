#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pathlib
import signal
import sys
from dataclasses import asdict
from typing import Iterable, Iterator, Tuple, Any

import numpy as np

# Resolve project layout relative to this script.
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

INSTANCE_DATASETS_DIR = PROJECT_ROOT / "Instance Datasets"
INSTANCE_ALGO_DIR = PROJECT_ROOT / "Instance-Algorithm Datasets"
DEFAULT_PROBLEMS_PATH = INSTANCE_DATASETS_DIR / "problems.jsonl"
DEFAULT_RESULTS_PATH = INSTANCE_ALGO_DIR / "Full Dataset" / "results.jsonl"

from AutoTSP import AlgorithmResult, SOLVER_SPECS, get_solver

# Build callables for multiprocessing while keeping the solver metadata handy.
ALGORITHM_SPECS = SOLVER_SPECS
ALGORITHMS = {
    name: (lambda dist_matrix, time_limit=5.0, _spec=spec: _spec.cls().solve(dist_matrix, time_limit=time_limit))
    for name, spec in SOLVER_SPECS.items()
}

def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TSP algorithms on generated problem instances.")
    parser.add_argument(
        "--problems",
        type=pathlib.Path,
        default=DEFAULT_PROBLEMS_PATH,
        help="JSONL file containing problem instances.",
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_PATH,
        help="Destination JSONL file for algorithm outcomes.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        choices=sorted(ALGORITHMS.keys()),
        help="Subset of algorithms to execute (default: all).",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=5.0,
        help="Per-algorithm time budget in seconds.",
    )
    parser.add_argument(
        "--memory-limit",
        type=float,
        default=1024.0,
        help="Per-algorithm memory budget in megabytes.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run algorithms even if results already exist for a problem.",
    )
    return parser.parse_args(raw_args)


def iter_jsonl(path: pathlib.Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_existing_results(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    records: dict[tuple[str, str], dict] = {}
    if not path.exists():
        return records
    for row in iter_jsonl(path):
        pid = row.get("problem_id")
        algo = row.get("algorithm")
        if not pid or not algo:
            continue
        records[(pid, algo)] = row
    return records


def ensure_problem_id(problem: dict) -> str:
    if "problem_id" in problem:
        return problem["problem_id"]
    if problem.get("coordinates") is not None:
        coords = np.asarray(problem.get("coordinates"), dtype=float)
    elif problem.get("distance_matrix") is not None:
        coords = np.asarray(problem.get("distance_matrix"), dtype=float)
    else:
        coords = np.empty((0, 0), dtype=float)
    metric = problem.get("metric", "euclidean") or "euclidean"
    problem_type = problem.get("problem_type", "unknown") or "unknown"
    origin = problem.get("origin", "synthetic") or "synthetic"
    transformation = problem.get("transformation", None)
    source_name = problem.get("source_name", None)
    hasher = hashlib.sha1()
    hasher.update(str(coords.shape[0]).encode())
    hasher.update(problem_type.encode())
    hasher.update(metric.encode())
    hasher.update(origin.encode())
    hasher.update(str(transformation).encode())
    hasher.update(str(source_name).encode())
    hasher.update(coords.tobytes())
    digest = hasher.hexdigest()
    problem["problem_id"] = digest
    return digest


def compute_distance_matrix(problem: dict) -> np.ndarray:
    if "distance_matrix" in problem:
        return np.asarray(problem["distance_matrix"], dtype=float)
    coordinates = np.asarray(problem.get("coordinates"), dtype=float)
    metric = problem.get("metric", "euclidean")
    if metric == "manhattan":
        diff = coordinates[:, None, :] - coordinates[None, :, :]
        return np.abs(diff).sum(axis=-1)
    diff = coordinates[:, None, :] - coordinates[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def format_context(num_cities: object, iteration: int | None, problem_type: str | None, origin: str | None) -> str:
    parts: list[str] = []
    if num_cities is not None:
        parts.append(f"cities={num_cities}")
    if iteration is not None:
        parts.append(f"instance={iteration}")
    if problem_type:
        parts.append(f"type={problem_type}")
    if origin:
        parts.append(f"origin={origin}")
    return f" ({', '.join(parts)})" if parts else ""


def base_record(problem: dict, algo_name: str) -> dict:
    spec = ALGORITHM_SPECS[algo_name]
    category = getattr(spec, "category", None)
    if category is None:
        family = getattr(spec, "family", None)
        category = family.value if family is not None else "unknown"
    return {
        "algorithm": algo_name,
        "algorithm_category": category,
        "problem_id": problem.get("problem_id"),
        "problem_type": problem.get("problem_type", "unknown"),
        "metric": problem.get("metric", "euclidean"),
        "num_cities": problem.get("num_cities"),
        "instance_index": problem.get("instance_index"),
        "directed": problem.get("directed", False),
        "origin": problem.get("origin", "synthetic"),
        "source_name": problem.get("source_name"),
        "transformation": problem.get("transformation"),
    }


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer, np.int64, np.int32)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    return value


def _run_solver_worker(
    algo_name: str,
    problem: dict,
    time_limit: float,
    memory_limit: int | None,
    queue: mp.Queue,
) -> None:
    try:
        import resource
    except ImportError:
        resource = None

    if resource and memory_limit:
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except (ValueError, OSError):
            pass

    try:
        dist = compute_distance_matrix(problem)
        solver = ALGORITHMS[algo_name]
        result = solver(dist, time_limit=time_limit)
        queue.put(("ok", result))
    except Exception as exc:  # noqa: BLE001
        queue.put(("error", {"type": type(exc).__name__, "message": str(exc)}))


def execute_with_limits(
    problem: dict,
    algo_name: str,
    time_limit: float,
    memory_limit: int | None,
) -> Tuple[AlgorithmResult | None, dict | None]:
    queue: mp.Queue = mp.Queue()
    if "coordinates" not in problem and "distance_matrix" not in problem:
        return None, {"status": "infeasible", "reason": "missing_geometry", "error": "No coordinates or distance matrix"}

    process = mp.Process(
        target=_run_solver_worker,
        args=(algo_name, problem, time_limit, memory_limit, queue),
    )
    process.start()
    process.join(timeout=time_limit + 1.0)

    if process.is_alive():
        process.terminate()
        process.join()
        return None, {"status": "infeasible", "reason": "timeout"}

    result_obj: AlgorithmResult | None = None
    failure: dict | None = None

    if not queue.empty():
        status, payload = queue.get()
        if status == "ok":
            result_obj = payload
        else:
            failure = {"status": "infeasible", "reason": payload.get("type"), "error": payload.get("message")}
    else:
        if process.exitcode and process.exitcode < 0:
            signum = -process.exitcode
            if signum == signal.SIGKILL or signum == signal.SIGABRT:
                failure = {"status": "infeasible", "reason": "memory_limit_exceeded"}
        if failure is None:
            failure = {"status": "infeasible", "reason": "unknown_failure", "error": "Worker exited without result"}

    return result_obj, failure


def serialize_result(problem: dict, algorithm: str, result: AlgorithmResult) -> dict:
    record = base_record(problem, algorithm)
    record.update(asdict(result))
    return record


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)
    if not args.problems.exists():
        raise SystemExit(f"Problem file not found: {args.problems}")

    selected_algorithms = args.algorithms or list(ALGORITHMS.keys())
    existing = load_existing_results(args.results)
    infeasible_thresholds: dict[str, int] = {}
    for (pid, algo_name), record in existing.items():
        status = record.get("status")
        reason = record.get("reason")
        spec = ALGORITHM_SPECS.get(algo_name)
        family_attr = getattr(spec, "family", None) if spec else None
        family_name = family_attr.value if hasattr(family_attr, "value") else str(family_attr).lower()
        # Only record thresholds for non-metaheuristic algorithms; heuristics/metaheuristics can still be useful at larger sizes.
        if status == "infeasible" and reason in {"timeout", "memory_limit_exceeded"}:
            if family_name in {"metaheuristic", "heuristic"}:
                continue
            num_cities = record.get("num_cities")
            if isinstance(num_cities, int):
                infeasible_thresholds[algo_name] = min(
                    infeasible_thresholds.get(algo_name, num_cities), num_cities
                )
    appended = 0
    reused = 0
    args.results.parent.mkdir(parents=True, exist_ok=True)
    memory_bytes = int(args.memory_limit * 1024 * 1024) if args.memory_limit else None
    iteration_tracker: dict[tuple[str, int], int] = {}

    with args.results.open("a", encoding="utf-8") as out:
        for problem in iter_jsonl(args.problems):
            problem_id = ensure_problem_id(problem)
            num_cities = problem.get("num_cities")
            problem_type = problem.get("problem_type", "unknown")
            iteration_idx: int | None = None
            if isinstance(num_cities, int):
                key_iter = (problem_type, num_cities)
                iteration_idx = iteration_tracker.get(key_iter, 0) + 1
                iteration_tracker[key_iter] = iteration_idx
            context = format_context(num_cities, iteration_idx, problem_type, problem.get("origin"))
            for algo_name in selected_algorithms:
                spec = ALGORITHM_SPECS[algo_name]
                key = (problem_id, algo_name)
                if not args.overwrite and key in existing:
                    reused += 1
                    print(
                        f"{algo_name} on problem {problem_id}{context} -> cached ({existing[key].get('status')})"
                    )
                    continue
                threshold = infeasible_thresholds.get(algo_name)
                # Only skip if we already timed out on a strictly larger instance size and the algorithm is not a heuristic/metaheuristic.
                family_attr = getattr(spec, "family", None) if spec else None
                family_name = family_attr.value if hasattr(family_attr, "value") else str(family_attr).lower()
                if (
                    threshold is not None
                    and isinstance(num_cities, int)
                    and num_cities > threshold
                    and family_name not in {"metaheuristic", "heuristic"}
                ):
                    record = base_record(problem, algo_name)
                    record.update(
                        {
                            "status": "infeasible",
                            "reason": "prior_infeasible_smaller_instance",
                        }
                    )
                    out.write(json.dumps(to_jsonable(record)))
                    out.write("\n")
                    existing[key] = record
                    appended += 1
                    print(
                        f"{algo_name} on problem {problem_id}{context} -> infeasible (skipped due to prior threshold)"
                    )
                    continue
                if problem.get("directed", False) and not spec.supports_directed:
                    record = base_record(problem, algo_name)
                    record.update(
                        {
                            "status": "unsupported",
                            "reason": "directed_not_supported",
                        }
                    )
                    out.write(json.dumps(to_jsonable(record)))
                    out.write("\n")
                    existing[key] = record
                    appended += 1
                    print(
                        f"{algo_name} on problem {problem_id}{context} -> unsupported (directed not supported)"
                    )
                    continue
                result_obj, failure = execute_with_limits(problem, algo_name, args.time_limit, memory_bytes)
                if result_obj:
                    record = serialize_result(problem, algo_name, result_obj)
                    if result_obj.status != "complete":
                        record["status"] = "infeasible"
                        record["reason"] = result_obj.status
                else:
                    record = base_record(problem, algo_name)
                    record.update(failure or {"status": "infeasible", "reason": "unknown_failure"})
                if record.get("status") != "infeasible":
                    record["status"] = record.get("status", "complete")
                else:
                    nc = record.get("num_cities")
                    if isinstance(nc, int):
                        reason = record.get("reason")
                        if reason in {"timeout", "memory_limit_exceeded"}:
                            prev = infeasible_thresholds.get(algo_name)
                            if prev is None or nc < prev:
                                infeasible_thresholds[algo_name] = nc
                out.write(json.dumps(to_jsonable(record)))
                out.write("\n")
                existing[key] = record
                appended += 1
                print(f"{algo_name} on problem {problem_id}{context} -> {record.get('status')}")

    print(f"Completed {appended} new runs. Reused {reused} cached results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
