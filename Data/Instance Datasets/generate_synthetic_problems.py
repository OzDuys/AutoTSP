#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import sys
from datetime import datetime, timezone
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np


# Paths relative to this script (Instance Datasets root).
THIS_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_PROBLEMS_PATH = THIS_DIR / "tsp_problems_synth.jsonl"


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


def uniform_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coordinates = rng.random((num_cities, 2)) * scale
    return {
        "coordinates": coordinates,
        "metric": "euclidean",
        "problem_type": "uniform_euclidean",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def clustered_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    clusters = max(2, min(num_cities // 5, 10))
    centers = rng.random((clusters, 2)) * scale
    assignments = rng.integers(0, clusters, size=num_cities)
    coordinates = centers[assignments] + rng.normal(loc=0.0, scale=scale * 0.05, size=(num_cities, 2))
    coordinates = np.clip(coordinates, 0, scale)
    return {
        "coordinates": coordinates,
        "metric": "euclidean",
        "problem_type": "clustered_euclidean",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def grid_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    side = max(2, math.ceil(math.sqrt(num_cities)))
    grid_x, grid_y = np.meshgrid(np.linspace(0, scale, side), np.linspace(0, scale, side))
    grid = np.stack([grid_x.flatten(), grid_y.flatten()], axis=1)[:num_cities]
    jitter = rng.normal(loc=0.0, scale=scale * 0.02, size=grid.shape)
    coordinates = np.clip(grid + jitter, 0, scale)
    return {
        "coordinates": coordinates,
        "metric": "euclidean",
        "problem_type": "grid_euclidean",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def circular_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    radius = scale / 2.5
    angles = np.linspace(0, 2 * math.pi, num_cities, endpoint=False)
    noise = rng.normal(scale=0.1, size=num_cities)
    x = (scale / 2) + (radius + noise) * np.cos(angles + rng.normal(scale=0.05, size=num_cities))
    y = (scale / 2) + (radius + noise) * np.sin(angles + rng.normal(scale=0.05, size=num_cities))
    coordinates = np.stack([x, y], axis=1)
    coordinates = np.clip(coordinates, 0, scale)
    return {
        "coordinates": coordinates,
        "metric": "euclidean",
        "problem_type": "circular_euclidean",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def manhattan_uniform(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coordinates = rng.random((num_cities, 2)) * scale
    return {
        "coordinates": coordinates,
        "metric": "manhattan",
        "problem_type": "uniform_manhattan",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def rotated_grid_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    base = grid_euclidean(num_cities, rng, scale)
    coords = base["coordinates"]
    angle = rng.uniform(0, 2 * math.pi)
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    center = coords.mean(axis=0, keepdims=True)
    coords = (coords - center) @ rotation.T + center
    base["coordinates"] = np.clip(coords, 0, scale)
    base["problem_type"] = "rotated_grid_euclidean"
    return base


def ring_of_clusters(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    clusters = max(3, min(num_cities // 8, 12))
    angles = rng.uniform(0, 2 * math.pi, size=clusters)
    radius = scale / 3
    centers = np.stack(
        [(scale / 2) + radius * np.cos(angles), (scale / 2) + radius * np.sin(angles)],
        axis=1,
    )
    assignments = rng.integers(0, clusters, size=num_cities)
    coordinates = centers[assignments] + rng.normal(loc=0.0, scale=scale * 0.03, size=(num_cities, 2))
    coordinates = np.clip(coordinates, 0, scale)
    return {
        "coordinates": coordinates,
        "metric": "euclidean",
        "problem_type": "ring_clusters",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def noisy_manhattan(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coordinates = rng.random((num_cities, 2)) * scale
    coordinates += rng.normal(scale=scale * 0.01, size=coordinates.shape)
    coordinates = np.clip(coordinates, 0, scale)
    return {
        "coordinates": coordinates,
        "metric": "manhattan",
        "problem_type": "noisy_manhattan",
        "origin": "synthetic",
        "transformation": "noisy_coords",
        "source_name": None,
        "directed": False,
    }


def directed_euclidean_skew(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coords = rng.random((num_cities, 2)) * scale
    diff = coords[:, None, :] - coords[None, :, :]
    base = np.linalg.norm(diff, axis=-1)
    skew = rng.normal(loc=0.0, scale=0.1, size=(num_cities, num_cities))
    asym = base * (1.0 + 0.2 * skew)
    np.fill_diagonal(asym, 0.0)
    asym = np.clip(asym, 0.0, None)
    return {
        "coordinates": coords,
        "distance_matrix": asym,
        "metric": "asymmetric_euclidean",
        "problem_type": "directed_euclidean_skew",
        "origin": "synthetic",
        "transformation": "skewed_distances",
        "source_name": None,
        "directed": True,
    }


def corridor_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    xs = rng.random(num_cities) * scale
    ys = rng.random(num_cities) * (scale * 0.1)
    coordinates = np.stack([xs, ys], axis=1)
    return {
        "coordinates": coordinates,
        "metric": "euclidean",
        "problem_type": "corridor_euclidean",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def clustered_outliers(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    base = clustered_euclidean(num_cities, rng, scale)
    coords = base["coordinates"]
    outliers = max(1, num_cities // 20)
    idx = rng.choice(num_cities, size=outliers, replace=False)
    coords[idx] = rng.random((outliers, 2)) * scale * 1.5
    base["coordinates"] = coords
    base["problem_type"] = "clustered_outliers"
    return base


def anisotropic_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coords = rng.random((num_cities, 2)) * scale
    coords[:, 1] *= scale * 0.1
    return {
        "coordinates": coords,
        "metric": "euclidean",
        "problem_type": "anisotropic_euclidean",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": False,
    }


def noisy_euclidean(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    coords = rng.random((num_cities, 2)) * scale
    coords += rng.normal(scale=scale * 0.02, size=coords.shape)
    coords = np.clip(coords, 0, scale)
    return {
        "coordinates": coords,
        "metric": "euclidean",
        "problem_type": "noisy_euclidean",
        "origin": "synthetic",
        "transformation": "noisy_coords",
        "source_name": None,
        "directed": False,
    }


def asymmetric_random(num_cities: int, rng: np.random.Generator, scale: float) -> dict:
    mat = rng.random((num_cities, num_cities)) * scale
    np.fill_diagonal(mat, 0.0)
    return {
        "distance_matrix": mat,
        "metric": "asymmetric",
        "problem_type": "asymmetric_random",
        "origin": "synthetic",
        "transformation": None,
        "source_name": None,
        "directed": True,
    }


PROBLEM_GENERATORS: Dict[str, Callable[[int, np.random.Generator, float], dict]] = {
    "uniform_euclidean": uniform_euclidean,
    "clustered_euclidean": clustered_euclidean,
    "grid_euclidean": grid_euclidean,
    "circular_euclidean": circular_euclidean,
    "manhattan_uniform": manhattan_uniform,
    "rotated_grid_euclidean": rotated_grid_euclidean,
    "ring_clusters": ring_of_clusters,
    "corridor_euclidean": corridor_euclidean,
    "clustered_outliers": clustered_outliers,
    "anisotropic_euclidean": anisotropic_euclidean,
    "noisy_euclidean": noisy_euclidean,
    "noisy_manhattan": noisy_manhattan,
    "asymmetric_random": asymmetric_random,
    "directed_euclidean_skew": directed_euclidean_skew,
}


# ---------------------------------------------------------------------------
# Procedural transformations (for TSPLIB or other coordinate sets)
# ---------------------------------------------------------------------------


def apply_transform(
    coordinates: np.ndarray,
    transform: str,
    rng: np.random.Generator,
    scale: float,
) -> np.ndarray:
    coords = coordinates.copy()
    if transform == "rotate":
        angle = rng.uniform(0, 2 * math.pi)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        center = coords.mean(axis=0, keepdims=True)
        coords = (coords - center) @ rotation.T + center
    elif transform == "jitter":
        magnitude = max(scale * 0.02, 1.0)
        coords = coords + rng.normal(scale=magnitude, size=coords.shape)
    elif transform == "rescale":
        min_vals = coords.min(axis=0)
        max_vals = coords.max(axis=0)
        span = np.maximum(max_vals - min_vals, 1e-6)
        coords = (coords - min_vals) / span * scale
    elif transform == "mirror":
        coords[:, 0] = coords[:, 0].max() - coords[:, 0]
    elif transform == "cluster_perturb":
        clusters = max(2, min(coords.shape[0] // 10, 8))
        assignments = rng.integers(0, clusters, size=coords.shape[0])
        offsets = rng.normal(scale=scale * 0.03, size=(clusters, 2))
        coords = coords + offsets[assignments]
    else:
        raise ValueError(f"Unknown transformation: {transform}")
    return coords


PROCEDURAL_TRANSFORMS = {"rotate", "jitter", "rescale", "mirror", "cluster_perturb"}


# ---------------------------------------------------------------------------
# TSPLIB parsing helpers
# ---------------------------------------------------------------------------


class TSPLibError(RuntimeError):
    """Raised when a TSPLIB instance cannot be parsed."""


def parse_tsplib(path: pathlib.Path) -> dict:
    metadata: Dict[str, str] = {}
    coords: List[List[float]] = []
    edge_weights: List[float] = []
    section: Optional[str] = None

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("COMMENT"):
                continue
            if line == "EOF":
                break
            if ":" in line and section is None:
                key, value = [token.strip() for token in line.split(":", 1)]
                metadata[key.upper()] = value
                continue
            upper = line.upper()
            if upper == "NODE_COORD_SECTION":
                section = "NODE_COORD_SECTION"
                continue
            if upper == "EDGE_WEIGHT_SECTION":
                section = "EDGE_WEIGHT_SECTION"
                continue
            if upper in {"DISPLAY_DATA_SECTION", "TOUR_SECTION"}:
                # Stop parsing at unsupported sections.
                break
            if section == "NODE_COORD_SECTION":
                parts = line.split()
                if len(parts) < 3:
                    raise TSPLibError(f"Invalid coordinate line in {path}: {line}")
                x, y = float(parts[-2]), float(parts[-1])
                coords.append([x, y])
            elif section == "EDGE_WEIGHT_SECTION":
                try:
                    edge_weights.extend(float(token) for token in line.split())
                except ValueError:
                    # Stop if non-numeric content appears (e.g., start of another section)
                    break
            else:
                # Unsupported section
                continue

    if "DIMENSION" not in metadata:
        raise TSPLibError(f"Missing DIMENSION in {path}")
    dimension = int(metadata["DIMENSION"])
    edge_weight_type = metadata.get("EDGE_WEIGHT_TYPE", "EUC_2D").upper()
    edge_weight_format = metadata.get("EDGE_WEIGHT_FORMAT", "FULL_MATRIX").upper()
    problem_type = metadata.get("TYPE", "TSP").upper()

    if coords:
        if len(coords) != dimension:
            raise TSPLibError(f"Coordinate count mismatch in {path}")
        coordinates = np.asarray(coords, dtype=float)
        metric = "euclidean" if edge_weight_type in {"EUC_2D", "CEIL_2D", "ATT"} else "euclidean"
        return {
            "coordinates": coordinates,
            "distance_matrix": None,
            "metric": metric,
            "problem_type": f"tsplib_{edge_weight_type.lower()}",
            "origin": "tsplib",
            "source_name": path.stem,
            "directed": problem_type == "ATSP",
        }

    if not edge_weights:
        raise TSPLibError(f"No data section parsed for {path}")

    def build_matrix(weights: list[float]) -> np.ndarray:
        fmt = edge_weight_format
        n = dimension
        matrix = np.zeros((n, n), dtype=float)
        if fmt == "FULL_MATRIX":
            if len(weights) != n * n:
                raise TSPLibError(f"Edge weight count mismatch in {path}")
            return np.asarray(weights, dtype=float).reshape(n, n)
        if fmt in {"UPPER_ROW", "UPPER_DIAG_ROW", "LOWER_ROW", "LOWER_DIAG_ROW"}:
            idx = 0
            for i in range(n):
                j_start = i if "DIAG" in fmt else i + 1
                if fmt.startswith("UPPER"):
                    for j in range(j_start, n):
                        matrix[i, j] = weights[idx]
                        matrix[j, i] = weights[idx]
                        idx += 1
                else:  # LOWER
                    for j in range(0, j_start + (1 if "DIAG" in fmt else 0)):
                        if i == j and "DIAG" not in fmt:
                            continue
                        matrix[i, j] = weights[idx]
                        matrix[j, i] = weights[idx]
                        idx += 1
            if idx != len(weights):
                raise TSPLibError(f"Edge weight count mismatch in {path}")
            return matrix
        raise TSPLibError(f"Unsupported EDGE_WEIGHT_FORMAT '{edge_weight_format}' in {path}")

    matrix = build_matrix(edge_weights)
    directed = problem_type == "ATSP"
    return {
        "coordinates": None,
        "distance_matrix": matrix,
        "metric": "explicit",
        "problem_type": f"tsplib_{edge_weight_type.lower()}",
        "origin": "tsplib",
        "source_name": path.stem,
        "directed": directed,
        "num_cities": dimension,
    }


def load_tsplib_instances(
    directory: pathlib.Path,
    rng: np.random.Generator,
    max_cities: Optional[int],
    limit: Optional[int],
    transforms: List[str],
    scale: float,
) -> List[dict]:
    instances: List[dict] = []
    files = sorted(list(directory.glob("*.tsp")) + list(directory.glob("*.atsp")))
    if limit:
        files = files[:limit]
    for tsp_path in files:
        try:
            base = parse_tsplib(tsp_path)
        except TSPLibError as exc:
            print(f"[WARN] Skipping {tsp_path.name}: {exc}", file=sys.stderr)
            continue
        if max_cities and base.get("coordinates") is not None and base["coordinates"].shape[0] > max_cities:
            continue
        if max_cities and base.get("distance_matrix") is not None and base["distance_matrix"].shape[0] > max_cities:
            continue

        instances.append(base)

        coords = base.get("coordinates")
        if coords is None:
            # Transformations currently only supported for coordinate-based instances
            continue
        for transform in transforms:
            transform = transform.lower()
            if transform not in PROCEDURAL_TRANSFORMS:
                print(f"[WARN] Unknown TSPLIB transform '{transform}'", file=sys.stderr)
                continue
            new_coords = apply_transform(coords, transform, rng, scale)
            transformed = {
                "coordinates": new_coords,
                "distance_matrix": None,
                "metric": base["metric"],
                "problem_type": f"{base['problem_type']}_{transform}",
                "origin": "tsplib_transformed",
                "source_name": base["source_name"],
                "transformation": transform,
                "directed": base["directed"],
            }
            instances.append(transformed)
    return instances


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def compute_problem_id(record: dict) -> str:
    hasher = hashlib.sha1()
    hasher.update(str(record.get("num_cities")).encode())
    hasher.update(str(record.get("metric")).encode())
    hasher.update(str(record.get("directed", False)).encode())
    # Only hash the graph content (coordinates or distance matrix); ignore source-specific metadata.
    if record.get("coordinates") is not None:
        coords = np.asarray(record["coordinates"], dtype=float)
        hasher.update(coords.tobytes())
    elif record.get("distance_matrix") is not None:
        dist = np.asarray(record["distance_matrix"], dtype=float)
        hasher.update(dist.tobytes())
    return hasher.hexdigest()


def record_to_jsonable(record: dict) -> dict:
    payload = dict(record)
    if payload.get("coordinates") is not None:
        payload["coordinates"] = np.asarray(payload["coordinates"], dtype=float).tolist()
        payload.pop("distance_matrix", None)
    elif payload.get("distance_matrix") is not None:
        payload["distance_matrix"] = np.asarray(payload["distance_matrix"], dtype=float).tolist()
    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic and TSPLIB-based TSP problem instances.")
    parser.add_argument(
        "--counts",
        nargs="+",
        type=int,
        default=[5,8, 10,12,16, 20,30, 50,70, 100,150, 200, 300, 500, 750, 1000, 2000],
        help="City counts to generate for synthetic instances.",
    )
    parser.add_argument(
        "--instances-per-count",
        type=int,
        default=1,
        help="Number of synthetic instances per city count (per problem type).",
    )
    parser.add_argument(
        "--problem-types",
        nargs="+",
        choices=sorted(PROBLEM_GENERATORS.keys()),
        default=list(PROBLEM_GENERATORS.keys()),
        help="Synthetic problem structures to generate.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=100.0,
        help="Coordinate scale; synthetic coordinates sampled within [0, scale).",
    )
    parser.add_argument(
        "--tsplib-dir",
        type=pathlib.Path,
        default=None,
        help="Directory containing TSPLIB .tsp files to include.",
    )
    parser.add_argument(
        "--tsplib-limit",
        type=int,
        default=None,
        help="Maximum number of TSPLIB files to ingest.",
    )
    parser.add_argument(
        "--tsplib-max-cities",
        type=int,
        default=None,
        help="Skip TSPLIB instances with more than this many cities.",
    )
    parser.add_argument(
        "--tsplib-transform",
        nargs="*",
        default=[],
        help=f"Procedural transforms to apply to TSPLIB coordinates ({', '.join(sorted(PROCEDURAL_TRANSFORMS))}).",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=DEFAULT_PROBLEMS_PATH,
        help="Destination JSONL file.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    return parser.parse_args(raw_args)


def generate_synthetic(args: argparse.Namespace, rng: np.random.Generator) -> List[dict]:
    records: List[dict] = []
    for count in args.counts:
        for problem_type in args.problem_types:
            generator = PROBLEM_GENERATORS[problem_type]
            for instance_idx in range(1, args.instances_per_count + 1):
                data = generator(count, rng, args.scale)
                record = {
                    **data,
                    "num_cities": count,
                    "instance_index": instance_idx,
                    "scale": args.scale,
                    "created_at": None,
                    "seed": args.seed,
                    "source_name": data.get("source_name"),
                    "transformation": data.get("transformation"),
                }
                record["problem_id"] = compute_problem_id(record)
                records.append(record)
    return records


def main(raw_args: Iterable[str] | None = None) -> None:
    args = parse_args(raw_args)
    rng = np.random.default_rng(args.seed)
    emitted: List[dict] = []

    emitted.extend(generate_synthetic(args, rng))

    if args.tsplib_dir and args.tsplib_dir.exists():
        transforms = [t.lower() for t in args.tsplib_transform]
        tsplib_records = load_tsplib_instances(
            args.tsplib_dir,
            rng,
            max_cities=args.tsplib_max_cities,
            limit=args.tsplib_limit,
            transforms=transforms,
            scale=args.scale,
        )
        for record in tsplib_records:
            record = dict(record)
            record["num_cities"] = (
                record["coordinates"].shape[0] if record.get("coordinates") is not None else record["distance_matrix"].shape[0]
            )
            record["instance_index"] = 1
            record["scale"] = args.scale
            record["seed"] = args.seed
            record["created_at"] = None
            record.setdefault("transformation", None)
            record["problem_id"] = compute_problem_id(record)
            emitted.append(record)
    elif args.tsplib_dir:
        print(f"[WARN] TSPLIB directory '{args.tsplib_dir}' does not exist; skipping.", file=sys.stderr)

    timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for record in emitted:
            serializable = record_to_jsonable(record)
            serializable["created_at"] = timestamp
            fh.write(json.dumps(serializable))
            fh.write("\n")


if __name__ == "__main__":
    main()
