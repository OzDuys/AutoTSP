#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTANCE_ALGO_DIR = PROJECT_ROOT / "Instance-Algorithm Datasets"
DEFAULT_RESULTS_PATH = INSTANCE_ALGO_DIR / "Full Dataset" / "results.jsonl"
DEFAULT_FIGURE_PATH = INSTANCE_ALGO_DIR / "Full Dataset" / "pngs" / "results.png"


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise TSP benchmarking results with standard deviation bands.")
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=DEFAULT_RESULTS_PATH,
        help="Input JSONL file with algorithm runs.",
    )
    parser.add_argument(
        "--figure",
        type=pathlib.Path,
        default=DEFAULT_FIGURE_PATH,
        help="Base path for rendered plots (one file per problem type).",
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


def print_summary(df: pd.DataFrame) -> None:
    completed = df[df["status"] == "complete"]
    if completed.empty:
        print("No completed runs available for summary.")
        return
    summary = (
        completed.groupby("algorithm")
        .agg(avg_elapsed=("elapsed", "mean"), avg_cost=("cost", "mean"), runs=("elapsed", "count"))
        .reset_index()
    )
    for _, row in summary.iterrows():
        print(
            f"{row['algorithm']}: runs={int(row['runs'])} avg_elapsed={row['avg_elapsed']:.4f}s avg_cost={row['avg_cost']:.2f}"
        )


def sanitize(name: str | None) -> str:
    if not name:
        return "unknown"
    return "".join(ch if ch.isalnum() else "_" for ch in name.lower())


def lineplot_with_sd(data: pd.DataFrame, hue: str, title_prefix: str, path: pathlib.Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(
        data=data,
        x="num_cities",
        y="elapsed",
        hue=hue,
        estimator="mean",
        errorbar="sd",
        err_style="band",
        ax=axes[0],
    )
    axes[0].set_title(f"{title_prefix} Runtime (mean ± 1σ)")
    axes[0].set_xlabel("Number of Cities")
    axes[0].set_ylabel("Elapsed Time (s)")

    sns.lineplot(
        data=data,
        x="num_cities",
        y="cost",
        hue=hue,
        estimator="mean",
        errorbar="sd",
        err_style="band",
        ax=axes[1],
    )
    axes[1].set_title(f"{title_prefix} Tour Cost (mean ± 1σ)")
    axes[1].set_xlabel("Number of Cities")
    axes[1].set_ylabel("Tour Cost")

    for ax in axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved figure to {path}")


def render(df: pd.DataFrame, output: pathlib.Path) -> None:
    completed = df[df["status"] == "complete"].copy()
    if completed.empty:
        raise SystemExit("No completed runs to plot.")

    if not output.suffix:
        output = output.with_suffix(".png")

    # Overall plots
    overall_path = output.with_name(f"{output.stem}_overall_algorithms{output.suffix}")
    lineplot_with_sd(completed, hue="algorithm", title_prefix="Overall", path=overall_path)
    if "algorithm_category" in completed.columns:
        overall_cat_path = output.with_name(f"{output.stem}_overall_categories{output.suffix}")
        lineplot_with_sd(
            completed,
            hue="algorithm_category",
            title_prefix="Overall Categories",
            path=overall_cat_path,
        )

    # Per-problem-type plots
    for problem_type, subset in completed.groupby(completed["problem_type"].fillna("unknown")):
        safe_name = sanitize(problem_type)
        pretty_name = problem_type.replace("_", " ").title()
        title_prefix = pretty_name
        algo_path = output.with_name(f"{output.stem}_{safe_name}_algorithms{output.suffix}")
        lineplot_with_sd(subset, hue="algorithm", title_prefix=title_prefix, path=algo_path)

        if "algorithm_category" in subset.columns:
            category_path = output.with_name(f"{output.stem}_{safe_name}_categories{output.suffix}")
            lineplot_with_sd(
                subset,
                hue="algorithm_category",
                title_prefix=f"{pretty_name} Categories",
                path=category_path,
            )

    # Per-origin plots
    if "origin" in completed.columns:
        for origin, subset in completed.groupby(completed["origin"].fillna("unknown")):
            safe_name = sanitize(origin)
            pretty_name = origin.replace("_", " ").title()
            origin_algo_path = output.with_name(f"{output.stem}_origin_{safe_name}_algorithms{output.suffix}")
            lineplot_with_sd(subset, hue="algorithm", title_prefix=f"{pretty_name} Origin", path=origin_algo_path)
            if "algorithm_category" in subset.columns:
                origin_cat_path = output.with_name(f"{output.stem}_origin_{safe_name}_categories{output.suffix}")
                lineplot_with_sd(
                    subset,
                    hue="algorithm_category",
                    title_prefix=f"{pretty_name} Origin Categories",
                    path=origin_cat_path,
                )

    # Focused plots per algorithm family
    if "algorithm_category" in completed.columns:
        for category, subset in completed.groupby(completed["algorithm_category"].fillna("uncategorised")):
            safe_name = sanitize(category)
            pretty_name = category.replace("_", " ").title()
            cat_algo_path = output.with_name(f"{output.stem}_family_{safe_name}{output.suffix}")
            lineplot_with_sd(subset, hue="algorithm", title_prefix=f"{pretty_name} Algorithms", path=cat_algo_path)


def main(raw_args: Iterable[str] | None = None) -> None:
    args = parse_args(raw_args)
    if not args.results.exists():
        raise SystemExit(f"No results file found at {args.results}")
    records = load_records(args.results)
    if not records:
        raise SystemExit("Results file is empty.")
    df = build_dataframe(records)
    print_summary(df)
    render(df, args.figure)


if __name__ == "__main__":
    main()
