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
from typing import Iterable, Iterator, Tuple

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms import ALGORITHMS, AlgorithmResult


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TSP algorithms on generated problem instances.")
    parser.add_argument(
        "--problems",
        type=pathlib.Path,
        default=pathlib.Path("data/problems.jsonl"),
        help="JSONL file containing problem instances.",
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path("data/results.jsonl"),
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
    coords = np.asarray(problem.get("coordinates"), dtype=float)
    digest = hashlib.sha1(coords.tobytes()).hexdigest()
    problem["problem_id"] = digest
    return digest


def distance_matrix(coordinates: list[list[float]] | np.ndarray) -> np.ndarray:
    coords = np.asarray(coordinates, dtype=float)
    diff = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def format_context(num_cities: object, iteration: int | None) -> str:
    parts: list[str] = []
    if num_cities is not None:
        parts.append(f"cities={num_cities}")
    if iteration is not None:
        parts.append(f"instance={iteration}")
    return f" ({', '.join(parts)})" if parts else ""


def _run_solver_worker(
    algo_name: str,
    coordinates: list[list[float]],
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
        dist = distance_matrix(coordinates)
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
    coordinates = problem.get("coordinates")
    if coordinates is None:
        return None, {"status": "infeasible", "reason": "missing_coordinates", "error": "No coordinates in problem"}

    process = mp.Process(
        target=_run_solver_worker,
        args=(algo_name, coordinates, time_limit, memory_limit, queue),
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
    record = asdict(result)
    record.update(
        {
            "algorithm": algorithm,
            "problem_id": problem["problem_id"],
            "num_cities": problem["num_cities"],
        }
    )
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
        if status == "infeasible":
            num_cities = record.get("num_cities")
            if isinstance(num_cities, int):
                infeasible_thresholds[algo_name] = min(
                    infeasible_thresholds.get(algo_name, num_cities), num_cities
                )
    appended = 0
    reused = 0
    args.results.parent.mkdir(parents=True, exist_ok=True)
    memory_bytes = int(args.memory_limit * 1024 * 1024) if args.memory_limit else None
    city_iterations: dict[int, int] = {}

    with args.results.open("a", encoding="utf-8") as out:
        for problem in iter_jsonl(args.problems):
            problem_id = ensure_problem_id(problem)
            num_cities = problem.get("num_cities")
            iteration_idx: int | None = None
            if isinstance(num_cities, int):
                iteration_idx = city_iterations.get(num_cities, 0) + 1
                city_iterations[num_cities] = iteration_idx
            context = format_context(num_cities, iteration_idx)
            for algo_name in selected_algorithms:
                key = (problem_id, algo_name)
                if not args.overwrite and key in existing:
                    reused += 1
                    print(
                        f"{algo_name} on problem {problem_id}{context} -> cached ({existing[key].get('status')})"
                    )
                    continue
                threshold = infeasible_thresholds.get(algo_name)
                if (
                    threshold is not None
                    and isinstance(num_cities, int)
                    and num_cities >= threshold
                ):
                    record = {
                        "algorithm": algo_name,
                        "problem_id": problem_id,
                        "num_cities": num_cities,
                        "status": "infeasible",
                        "reason": "prior_infeasible_smaller_instance",
                    }
                    out.write(json.dumps(record))
                    out.write("\n")
                    existing[key] = record
                    appended += 1
                    print(
                        f"{algo_name} on problem {problem_id}{context} -> infeasible (skipped due to prior threshold)"
                    )
                    continue
                result_obj, failure = execute_with_limits(problem, algo_name, args.time_limit, memory_bytes)
                if result_obj:
                    record = serialize_result(problem, algo_name, result_obj)
                    if result_obj.status != "complete":
                        record["status"] = "infeasible"
                        record["reason"] = result_obj.status
                else:
                    record = {
                        "algorithm": algo_name,
                        "problem_id": problem_id,
                        "num_cities": num_cities,
                        **(failure or {"status": "infeasible", "reason": "unknown_failure"}),
                    }
                if record.get("status") != "infeasible":
                    record["status"] = record.get("status", "complete")
                else:
                    nc = record.get("num_cities")
                    if isinstance(nc, int):
                        prev = infeasible_thresholds.get(algo_name)
                        if prev is None or nc < prev:
                            infeasible_thresholds[algo_name] = nc
                out.write(json.dumps(record))
                out.write("\n")
                existing[key] = record
                appended += 1
                print(f"{algo_name} on problem {problem_id}{context} -> {record.get('status')}")

    print(f"Completed {appended} new runs. Reused {reused} cached results.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
