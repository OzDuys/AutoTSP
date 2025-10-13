#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from datetime import datetime
from typing import Iterable

import numpy as np


def create_instance(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coordinates = rng.random((num_cities, 2)) * scale
    digest = hashlib.sha1(coordinates.tobytes()).hexdigest()
    return {
        "num_cities": num_cities,
        "problem_id": digest,
        "coordinates": coordinates.tolist(),
        "scale": scale,
    }


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random TSP problem instances.")
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000],
        help="City counts to generate.",
    )
    parser.add_argument(
        "--instances-per-count",
        type=int,
        default=10,
        help="How many instances to generate per city count.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Coordinates drawn uniformly in [0, scale).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("data/problems.jsonl"),
        help="Destination JSONL file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args(raw_args)


def main(raw_args: Iterable[str] | None = None) -> None:
    args = parse_args(raw_args)
    rng = np.random.default_rng(args.seed)

    timestamp = datetime.utcnow().isoformat() + "Z"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for count in args.counts:
            for _ in range(args.instances_per_count):
                instance = create_instance(count, rng, args.scale)
                record = {
                    "created_at": timestamp,
                    "seed": args.seed,
                    **instance,
                }
                fh.write(json.dumps(record))
                fh.write("\n")


if __name__ == "__main__":
    main()
