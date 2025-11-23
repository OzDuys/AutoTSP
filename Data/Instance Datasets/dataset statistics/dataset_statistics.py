#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from collections import Counter, defaultdict
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


INSTANCE_DATASETS_DIR = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_PROBLEMS_PATH = INSTANCE_DATASETS_DIR / "problems.jsonl"
DEFAULT_OUTPUT_DIR = INSTANCE_DATASETS_DIR / "dataset statistics"


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a problems JSONL file (counts, city distribution, plots).")
    parser.add_argument(
        "--problems",
        type=pathlib.Path,
        default=DEFAULT_PROBLEMS_PATH,
        help="Path to problems JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write plots and summary JSON.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many problem types to show in the bar chart.",
    )
    return parser.parse_args(raw_args)


def iter_jsonl(path: pathlib.Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize(problems: List[dict]) -> dict:
    # Collect only integer num_cities values via explicit loop (type checker friendly)
    num_cities: List[int] = []
    for p in problems:
        v = p.get("num_cities")
        if isinstance(v, int):
            num_cities.append(v)
    metrics = Counter(p.get("metric", "unknown") or "unknown" for p in problems)
    origins = Counter(p.get("origin", "unknown") or "unknown" for p in problems)
    types = Counter(p.get("problem_type", "unknown") or "unknown" for p in problems)
    directed = Counter(bool(p.get("directed", False)) for p in problems)
    summary = {
        "count": len(problems),
        "num_cities": {
            "min": int(np.min(num_cities)) if num_cities else None,
            "max": int(np.max(num_cities)) if num_cities else None,
            "median": float(np.median(num_cities)) if num_cities else None,
            "mean": float(np.mean(num_cities)) if num_cities else None,
            "p10": float(np.percentile(num_cities, 10)) if num_cities else None,
            "p90": float(np.percentile(num_cities, 90)) if num_cities else None,
            "count": len(num_cities),
        },
        "metrics": metrics,
        "origins": origins,
        "problem_types": types,
        "directed": directed,
    }
    return summary


def plot_num_cities(num_cities: List[int], out_path: pathlib.Path) -> None:
    if not num_cities:
        return
    data = np.array([v for v in num_cities if v is not None and v >= 0])
    if data.size == 0:
        return

    extreme_ratio = data.max() / (np.median(data) + 1e-9)
    # If highly skewed, make composite figure
    if extreme_ratio > 50 and data.max() > 500:
        fig, axes = plt.subplots(2, 2, figsize=(12, 6))
        fig.suptitle("Distribution of num_cities")

        # Linear zoom (0–200)
        zoom_cut = 200
        zoom_data = data[data <= zoom_cut]
        axes[0, 0].hist(zoom_data, bins=min(50, len(np.unique(zoom_data))), color="#4F6BED", edgecolor="black", alpha=0.8)
        axes[0, 0].set_title(f"Linear (<= {zoom_cut})")
        axes[0, 0].set_xlabel("Number of cities")
        axes[0, 0].set_ylabel("Count")

        # Full range with log x
        positive = data[data > 0]
        if positive.size > 0:
            # Geometric bins
            min_pos = max(1, positive.min())
            bins = np.geomspace(min_pos, positive.max(), 60)
            axes[0, 1].hist(positive, bins=bins, color="#4F6BED", edgecolor="black", alpha=0.8)
            axes[0, 1].set_xscale("log")
            axes[0, 1].set_title("Full range (log x)")
            axes[0, 1].set_xlabel("Number of cities (log scale)")
            axes[0, 1].set_ylabel("Count")

        # CDF (linear up to zoom_cut)
        sorted_data = np.sort(data)
        cdf = np.arange(1, sorted_data.size + 1) / sorted_data.size
        axes[1, 0].plot(sorted_data, cdf, color="#F28E2B")
        axes[1, 0].set_xlim(0, zoom_cut)
        axes[1, 0].set_title("CDF (0–200)")
        axes[1, 0].set_xlabel("Number of cities")
        axes[1, 0].set_ylabel("Cumulative fraction")

        # CDF (log x full)
        axes[1, 1].plot(sorted_data, cdf, color="#F28E2B")
        axes[1, 1].set_xscale("log")
        axes[1, 1].set_title("CDF (log x)")
        axes[1, 1].set_xlabel("Number of cities (log scale)")
        axes[1, 1].set_ylabel("Cumulative fraction")

        plt.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
        return

    # Default simple histogram
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=50, color="#4F6BED", edgecolor="black", alpha=0.8)
    plt.xlabel("Number of cities")
    plt.ylabel("Count")
    plt.title("Distribution of num_cities")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

def plot_num_cities_by_category(problems: List[dict], key: str, out_path: pathlib.Path, max_categories: int = 12) -> None:
    # Collect num_cities per category
    records = [(p.get("num_cities"), p.get(key, "unknown") or "unknown") for p in problems]
    data_by_cat = defaultdict(list)
    for n, cat in records:
        if isinstance(n, int) and n >= 0:
            data_by_cat[cat].append(n)
    if not data_by_cat:
        return
    # Rank categories
    sorted_cats = sorted(data_by_cat.items(), key=lambda kv: len(kv[1]), reverse=True)
    top = sorted_cats[:max_categories]
    remainder = sorted_cats[max_categories:]
    if remainder:
        other_vals = []
        for _, vals in remainder:
            other_vals.extend(vals)
        top.append(("other", other_vals))
    categories = [c for c, _ in top]
    arrays = [np.array(v) for _, v in top]
    all_data = np.concatenate(arrays)
    if all_data.size == 0:
        return

    extreme_ratio = all_data.max() / (np.median(all_data) + 1e-9)
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(len(categories))]

    def make_bins_linear(data):
        uniq = len(np.unique(data))
        return min(50, uniq) if uniq > 1 else 1

    def make_bins_log(pos_data):
        min_pos = max(1, pos_data.min())
        return np.geomspace(min_pos, pos_data.max(), 60)

    if extreme_ratio > 50 and all_data.max() > 500:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        # Place suptitle high so legend can sit beneath it
        fig.suptitle(f"num_cities distribution stacked by {key}", y=0.985)

        # Linear zoom (<=200)
        zoom_cut = 200
        zoom_arrays = [a[a <= zoom_cut] for a in arrays]
        bins_linear = make_bins_linear(np.concatenate(zoom_arrays)) if np.concatenate(zoom_arrays).size else 1
        axes[0].hist(zoom_arrays, bins=bins_linear, stacked=True, color=colors, edgecolor="black", alpha=0.85, label=categories)
        axes[0].set_title(f"Zoom (<= {zoom_cut})")
        axes[0].set_xlabel("Number of cities")
        axes[0].set_ylabel("Count")

        # Full range (log x)
        pos_arrays = [a[a > 0] for a in arrays]
        pos_concat = np.concatenate(pos_arrays)
        if pos_concat.size:
            bins_log = make_bins_log(pos_concat)
            axes[1].hist(pos_arrays, bins=bins_log, stacked=True, color=colors, edgecolor="black", alpha=0.85, label=categories)
            axes[1].set_xscale("log")
        axes[1].set_title("Full range (log x)")
        axes[1].set_xlabel("Number of cities (log scale)")
        axes[1].set_ylabel("Count")

        # Legend just below suptitle, above axes (keeps top area tidy)
        handles, labels = axes[1].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=min(5, len(categories)), frameon=False)
        # Adjust subplot region to leave room at top for title + legend
        fig.subplots_adjust(top=0.88, bottom=0.12, wspace=0.30)
        # Tight layout within reserved rect (avoid compressing legend/title zone)
        plt.tight_layout(rect=(0, 0, 1, 0.88))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        return

    # Simple stacked histogram (linear)
    bins_linear = make_bins_linear(all_data)
    plt.figure(figsize=(10, 5))
    plt.hist(arrays, bins=bins_linear, stacked=True, color=colors, edgecolor="black", alpha=0.85, label=categories)
    plt.xlabel("Number of cities")
    plt.ylabel("Count")
    plt.title(f"Distribution of num_cities by {key} (stacked)")
    plt.legend(loc="upper right", fontsize="small", frameon=False)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()

def plot_top_counts(counter: Counter, title: str, out_path: pathlib.Path, top_k: int) -> None:
    items = counter.most_common(top_k)
    if not items:
        return
    labels, counts = zip(*items)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(labels)), counts, color="#F28E2B")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main(raw_args: Iterable[str] | None = None) -> int:
    args = parse_args(raw_args)
    if not args.problems.exists():
        raise SystemExit(f"Problems file not found: {args.problems}")
    problems = list(iter_jsonl(args.problems))
    summary = summarize(problems)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_json_path = args.output_dir / "dataset_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump(
            summary,
            fh,
            ensure_ascii=False,
            indent=2,
            default=lambda o: dict(o) if isinstance(o, Counter) else str(o),
        )

    num_cities: List[int] = []
    for p in problems:
        v = p.get("num_cities")
        if isinstance(v, int):
            num_cities.append(v)
    plot_num_cities(num_cities, args.output_dir / "num_cities_hist.png")
    plot_num_cities_by_category(problems, "origin", args.output_dir / "num_cities_by_origin.png")
    plot_num_cities_by_category(problems, "problem_type", args.output_dir / "num_cities_by_problem_type.png")
    plot_top_counts(summary["problem_types"], "Top problem types", args.output_dir / "problem_types_top.png", args.top_k)
    plot_top_counts(summary["origins"], "Origins", args.output_dir / "origins.png", args.top_k)
    plot_top_counts(summary["metrics"], "Metrics", args.output_dir / "metrics.png", args.top_k)

    print(f"Wrote summary to {summary_json_path} and plots to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
