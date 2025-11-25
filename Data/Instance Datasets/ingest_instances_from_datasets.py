#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable, Optional

import numpy as np

# Ensure we can import generate_synthetic_problems helpers
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_synthetic_problems  # type: ignore
from generate_synthetic_problems import compute_problem_id, parse_tsplib, record_to_jsonable  # type: ignore


INSTANCE_DATASETS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_ROOT_DIR = "Data/Instance Datasets/External TSP datasets"
DEFAULT_PROBLEMS_PATH = INSTANCE_DATASETS_DIR / "tsp_problems_external.jsonl"


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest all TSPLIB-style files under a root directory into JSONL.")
    parser.add_argument(
        "--root",
        type=pathlib.Path,
        default=DEFAULT_ROOT_DIR,
        help="Root directory to scan.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_PROBLEMS_PATH,
        help="Destination JSONL file (appended if exists).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to ingest (useful for quick tests).",
    )
    parser.add_argument(
        "--max-cities",
        type=int,
        default=None,
        help="Skip instances with more than this many cities.",
    )
    return parser.parse_args(raw_args)


def find_candidate_files(root: pathlib.Path) -> list[pathlib.Path]:
    candidates: list[pathlib.Path] = []
    for path in root.rglob("*"):
        if path.suffix.lower() in {".tsp", ".atsp", ".txt", ".dat", ".tw"}:
            candidates.append(path)
    return sorted(candidates)


def parse_numeric_matrix(path: pathlib.Path) -> Optional[dict]:
    """Attempt to parse a file containing a full distance matrix (whitespace-separated)."""
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            tokens = []
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                parts = stripped.replace(",", " ").split()
                for p in parts:
                    try:
                        tokens.append(float(p))
                    except ValueError:
                        return None
    except OSError:
        return None
    if not tokens:
        return None
    length = len(tokens)
    n = int(round(length ** 0.5))
    if n * n != length:
        return None
    mat = np.asarray(tokens, dtype=float).reshape(n, n)
    return {
        "coordinates": None,
        "distance_matrix": mat,
        "metric": "explicit",
        "problem_type": "matrix_explicit",
        "origin": "matrix_file",
        "source_name": path.stem,
        "directed": True,
        "num_cities": n,
    }


def parse_header_matrix(path: pathlib.Path) -> Optional[dict]:
    """
    Parse files where the first line is an integer n, followed by n lines of n values.
    Ignores any remaining lines (e.g., time-window metadata).
    """
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            first = None
            for line in fh:
                stripped = line.strip()
                if stripped:
                    first = stripped
                    break
            if first is None:
                return None
            n = int(first.split()[0])
            rows = []
            for _ in range(n):
                line = fh.readline()
                if not line:
                    break
                parts = line.strip().replace(",", " ").split()
                if len(parts) < n:
                    return None
                try:
                    row = [float(p) for p in parts[:n]]
                except ValueError:
                    return None
                rows.append(row)
            if len(rows) != n:
                return None
            mat = np.asarray(rows, dtype=float)
    except Exception:
        return None
    return {
        "coordinates": None,
        "distance_matrix": mat,
        "metric": "explicit",
        "problem_type": "matrix_explicit",
        "origin": "matrix_file",
        "source_name": path.stem,
        "directed": True,
        "num_cities": n,
    }


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)
    files = find_candidate_files(args.root)
    if args.limit:
        files = files[: args.limit]
    seen: set[str] = set()
    existing_out = []
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = obj.get("problem_id")
            if pid:
                seen.add(pid)
        existing_out.append(f"(pre-existing records: {len(seen)})")

    output_records = []
    for fpath in files:
        base: Optional[dict] = None
        if fpath.suffix.lower() in {".tsp", ".atsp"}:
            try:
                base = parse_tsplib(fpath)
            except Exception as exc:  # noqa: BLE001
                print(f"[WARN] Skipping {fpath} ({exc})")
                continue
        else:
            base = parse_numeric_matrix(fpath)
            if base is None:
                base = parse_header_matrix(fpath)
            if base is None:
                continue
        coords = base.get("coordinates")
        mat = base.get("distance_matrix")
        n = coords.shape[0] if coords is not None else (mat.shape[0] if mat is not None else None)
        if args.max_cities and n and n > args.max_cities:
            continue
        record = dict(base)
        if n is not None:
            record["num_cities"] = int(n)
        record["problem_id"] = compute_problem_id(record)
        if record["problem_id"] in seen:
            continue
        seen.add(record["problem_id"])
        output_records.append(record_to_jsonable(record))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as out:
        for rec in output_records:
            out.write(json.dumps(rec))
            out.write("\n")

    print(f"Ingested {len(output_records)} new TSPLIB files into {args.output}. {len(seen)} total unique problems.")
    if existing_out:
        print(" ".join(existing_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
