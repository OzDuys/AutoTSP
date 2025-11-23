#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pathlib
from datetime import datetime, timezone
from typing import Iterable, List

import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTANCE_ALGO_DIR = PROJECT_ROOT / "Instance-Algorithm Datasets"
DEFAULT_RESULTS_PATH = INSTANCE_ALGO_DIR / "Full Dataset" / "results.jsonl"


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate benchmarking results for the interactive visualiser."
    )
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_PATH,
        help="Input JSONL file containing algorithm runs.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("visualizer/visualization_data.json"),
        help="Destination JSON file consumed by the web visualiser.",
    )
    return parser.parse_args(raw_args)


def load_records(path: pathlib.Path) -> List[dict]:
    records: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def build_dataframe(records: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    numeric_cols = ["num_cities", "elapsed", "cost"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def safe_value(value: str | None, fallback: str = "unknown") -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    return str(value)


def bucket_num_cities(n: float | int | None) -> str | None:
    if n is None or math.isnan(n):
        return None
    n_int = int(n)
    bins = [0, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1_000_000_000]
    for i in range(len(bins) - 1):
        low, high = bins[i], bins[i + 1]
        if low <= n_int < high:
            if high >= 1_000_000_000:
                return f"{low}+"
            return f"{low}-{high - 1}"
    return f"{n_int}+"


def ordered_buckets_from_set(buckets: set[str]) -> list[str]:
    def bucket_key(bucket: str) -> int:
        if bucket.endswith("+"):
            return int(bucket.rstrip("+"))
        parts = bucket.split("-")
        try:
            return int(parts[0])
        except Exception:
            return 1_000_000_000

    return sorted(buckets, key=bucket_key)


def aggregate(df: pd.DataFrame) -> dict:
    df["problem_type"] = df["problem_type"].apply(safe_value)
    df["algorithm_category"] = df.get("algorithm_category", pd.Series(index=df.index)).apply(
        lambda value: safe_value(value, "uncategorised")
    )
    df["algorithm"] = df["algorithm"].apply(safe_value)
    df["city_bucket"] = df["num_cities"].apply(bucket_num_cities)
    df = df.dropna(subset=["city_bucket"])

    completed = df[df["status"] == "complete"].copy()
    if completed.empty:
        raise SystemExit("No completed runs available for aggregation.")

    group_columns = ["algorithm_category", "algorithm", "city_bucket"]
    metrics = ["elapsed", "cost"]

    problem_types = sorted(completed["problem_type"].dropna().unique().tolist())
    problem_types = ["all"] + problem_types

    rows = []
    for ptype in problem_types:
        subset_all = df if ptype == "all" else df[df["problem_type"] == ptype]
        subset_completed = completed if ptype == "all" else completed[completed["problem_type"] == ptype]
        if subset_completed.empty:
            continue

        # Attempt counts (includes failures) for coverage
        attempt_counts = (
            subset_all.groupby(group_columns, dropna=False)["status"]
            .count()
            .reset_index()
            .rename(columns={"status": "attempts"})
        )

        for metric in metrics:
            if metric not in subset_completed.columns:
                continue
            grouped = (
                subset_completed.dropna(subset=[metric])
                .groupby(group_columns, dropna=False)[metric]
                .agg(["mean", "median", "std", "count", "min", "max"])
                .reset_index()
            )
            for record in grouped.to_dict(orient="records"):
                count = int(record["count"])
                std_value = float(record["std"]) if not math.isnan(record["std"]) else 0.0
                sem = std_value / math.sqrt(count) if count > 0 else 0.0
                ci95 = 1.96 * sem if count > 1 else 0.0
                median_value = float(record["median"]) if not math.isnan(record["median"]) else float(record["mean"])
                min_value = float(record["min"]) if not math.isnan(record["min"]) else float(record["mean"])
                max_value = float(record["max"]) if not math.isnan(record["max"]) else float(record["mean"])
                attempts_row = attempt_counts[
                    (attempt_counts["algorithm_category"] == record["algorithm_category"])
                    & (attempt_counts["algorithm"] == record["algorithm"])
                    & (attempt_counts["city_bucket"] == record["city_bucket"])
                ]
                attempts = int(attempts_row["attempts"].iloc[0]) if not attempts_row.empty else count
                success_rate = (count / attempts) if attempts > 0 else 0.0
                rows.append(
                    {
                        "problem_type": ptype,
                        "algorithm_category": safe_value(record["algorithm_category"], "uncategorised"),
                        "algorithm": safe_value(record["algorithm"]),
                        "num_cities_bucket": safe_value(record["city_bucket"]),
                        "metric": metric,
                        "mean": float(record["mean"]),
                        "median": median_value,
                        "std": std_value,
                        "count": count,
                        "attempts": attempts,
                        "success_rate": success_rate,
                        "ci95": ci95,
                        "range_min": min_value,
                        "range_max": max_value,
                    }
                )

        # Joint metric for Pareto plotting (mean runtime vs cost)
        if {"elapsed", "cost"}.issubset(subset_completed.columns):
            joint_group = subset_completed.groupby(group_columns, dropna=False)
            for keys, group in joint_group:
                if group.empty:
                    continue
                category, algorithm, city_bucket = keys
                runtime_stats = group["elapsed"].dropna()
                cost_stats = group["cost"].dropna()
                if runtime_stats.empty or cost_stats.empty:
                    continue
                runtime_mean = float(runtime_stats.mean())
                runtime_min = float(runtime_stats.min())
                runtime_max = float(runtime_stats.max())
                cost_mean = float(cost_stats.mean())
                cost_min = float(cost_stats.min())
                cost_max = float(cost_stats.max())
                count = int(min(runtime_stats.shape[0], cost_stats.shape[0]))
                attempts_row = attempt_counts[
                    (attempt_counts["algorithm_category"] == category)
                    & (attempt_counts["algorithm"] == algorithm)
                    & (attempt_counts["city_bucket"] == city_bucket)
                ]
                attempts = int(attempts_row["attempts"].iloc[0]) if not attempts_row.empty else count
                success_rate = (count / attempts) if attempts > 0 else 0.0
                rows.append(
                    {
                        "problem_type": ptype,
                        "algorithm_category": safe_value(category, "uncategorised"),
                        "algorithm": safe_value(algorithm),
                        "num_cities_bucket": safe_value(city_bucket),
                        "metric": "runtime_cost",
                        "runtime_mean": runtime_mean,
                        "runtime_min": runtime_min,
                        "runtime_max": runtime_max,
                        "cost_mean": cost_mean,
                        "cost_min": cost_min,
                        "cost_max": cost_max,
                        "count": count,
                        "attempts": attempts,
                        "success_rate": success_rate,
                    }
                )

    city_bucket_set = set(completed["city_bucket"].dropna().unique().tolist())
    metadata = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "source_results": str(df.attrs.get("source_results", "")),
        "metrics": [metric for metric in metrics if metric in completed.columns],
        "problem_types": problem_types,
        "algorithm_categories": sorted(completed["algorithm_category"].dropna().unique().tolist()),
        "algorithms": sorted(completed["algorithm"].dropna().unique().tolist()),
        "city_buckets": ordered_buckets_from_set(city_bucket_set),
        "supports_runtime_cost": {"elapsed", "cost"}.issubset(completed.columns),
    }

    return {"metadata": metadata, "records": rows}


def main(raw_args: Iterable[str] | None = None) -> None:
    args = parse_args(raw_args)
    if not args.results.exists():
        raise SystemExit(f"No results file found at {args.results}")

    records = load_records(args.results)
    if not records:
        raise SystemExit("Results file is empty.")

    df = build_dataframe(records)
    df.attrs["source_results"] = str(args.results)
    payload = aggregate(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    print(f"Wrote aggregated data to {args.output}")


if __name__ == "__main__":
    main()
