#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Sequence

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.axes import Axes


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
INSTANCE_ALGO_DIR = PROJECT_ROOT / "Data/Instance-Algorithm Datasets"   # Data/Instance-Algorithm Datasets
DEFAULT_RESULTS_PATH = INSTANCE_ALGO_DIR / "combined_results.jsonl"
DEFAULT_FIGURE_PATH = PROJECT_ROOT / "Scripts" / "Solver Comparison Plots" / "combined_results.png"
FAMILIES_TO_PLOT = ["approximation", "exact", "heuristic", "metaheuristic"]
SMOOTHING_ALPHA = 0.1


def parse_args(raw_args: Sequence[str] | None = None) -> argparse.Namespace:
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


def aggregate_and_smooth(data: pd.DataFrame, value_col: str, hue: str, alpha: float) -> pd.DataFrame:
    required_cols = {hue, "num_cities", value_col}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for plotting: {', '.join(sorted(missing))}")

    subset = data[[hue, "num_cities", value_col]].dropna()
    if subset.empty:
        return pd.DataFrame(columns=[hue, "num_cities", "smoothed"])

    grouped = (
        subset.groupby([hue, "num_cities"], as_index=False)
        .agg(mean=(value_col, "mean"))
        .sort_values([hue, "num_cities"])
        .reset_index(drop=True)
    )

    grouped["smoothed"] = (
        grouped.groupby(hue, group_keys=False)["mean"].transform(lambda s: s.ewm(alpha=alpha, adjust=False).mean())
    )
    return grouped


def plot_metric_with_band(
    ax: Axes,
    aggregated: pd.DataFrame,
    hue: str,
    palette_map: dict,
) -> None:
    if aggregated.empty:
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", transform=ax.transAxes)
        return

    sns.lineplot(
        data=aggregated,
        x="num_cities",
        y="smoothed",
        hue=hue,
        estimator=None,
        palette=palette_map,
        ax=ax,
    )


def lineplot_with_sd(data: pd.DataFrame, hue: str, title_prefix: str, path: pathlib.Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    elapsed_agg = aggregate_and_smooth(data, "elapsed", hue, SMOOTHING_ALPHA)
    cost_agg = aggregate_and_smooth(data, "cost", hue, SMOOTHING_ALPHA)
    unique_hues = data[hue].dropna().unique().tolist()
    if unique_hues:
        palette = sns.color_palette("tab10", n_colors=len(unique_hues))
        palette_map = {label: palette[idx] for idx, label in enumerate(unique_hues)}
    else:
        palette_map = {}

    plot_metric_with_band(axes[0], elapsed_agg, hue, palette_map)
    axes[0].set_title(f"{title_prefix} Runtime (mean ± 1σ)")
    axes[0].set_xlabel("Number of Cities")
    axes[0].set_ylabel("Elapsed Time (s)")

    plot_metric_with_band(axes[1], cost_agg, hue, palette_map)
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
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=5, fontsize='small')
    fig.tight_layout(rect=(0, 0.1, 1, 1))

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
        completed["algorithm_category"] = completed["algorithm_category"].fillna("uncategorised")
        completed["algorithm_category_normalised"] = completed["algorithm_category"].str.lower()

        overall_cat_path = output.with_name(f"{output.stem}_overall_categories{output.suffix}")
        lineplot_with_sd(
            completed,
            hue="algorithm_category",
            title_prefix="Overall Categories",
            path=overall_cat_path,
        )

        for family in FAMILIES_TO_PLOT:
            subset = completed[completed["algorithm_category_normalised"] == family]
            if subset.empty:
                continue
            pretty_name = family.replace("_", " ").title()
            family_path = output.with_name(f"{output.stem}_family_{sanitize(family)}{output.suffix}")
            lineplot_with_sd(
                subset,
                hue="algorithm",
                title_prefix=f"{pretty_name} Algorithms",
                path=family_path,
            )


def main(raw_args: Sequence[str] | None = None) -> None:
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
