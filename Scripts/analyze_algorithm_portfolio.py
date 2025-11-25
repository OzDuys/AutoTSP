#!/usr/bin/env python3
"""
Empirical analysis for the TSP algorithm portfolio.

This script loads the combined benchmark logs, computes baseline
statistics, and renders static figures used in the report:
 - global runtime/quality trends and success rates,
 - Pareto fronts by instance family and size bucket,
 - relationships between instance features and relative performance.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = PROJECT_ROOT / "Data/Instance-Algorithm Datasets/combined_results.jsonl"
FEATURES_PATH = PROJECT_ROOT / "Data/Instance-Algorithm Datasets/rf_training_train.jsonl"
FIG_DIR = PROJECT_ROOT / "Scripts/Algorithm_portfolio_plots"
SUCCESS_STATUSES = {"complete", "success"}


def load_results(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    df["elapsed"] = pd.to_numeric(df["elapsed"], errors="coerce")
    df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
    return df


def normalise_cost(success_df: pd.DataFrame) -> pd.DataFrame:
    best_cost = success_df.groupby("problem_id")["cost"].transform("min")
    out = success_df.copy()
    out["norm_cost"] = out["cost"] / best_cost
    return out


def baseline_stats(df: pd.DataFrame, success_df: pd.DataFrame) -> pd.DataFrame:
    total = df.groupby("algorithm")["status"].count()
    successes = success_df.groupby("algorithm")["status"].count()
    failures = df[~df["status"].isin(SUCCESS_STATUSES)].groupby("algorithm")["status"].count()
    stats = pd.DataFrame(
        {
            "total_runs": total,
            "successful_runs": successes,
            "failed_runs": failures,
        }
    ).fillna(0)
    stats["success_rate"] = stats["successful_runs"] / stats["total_runs"]
    agg = (
        success_df.groupby("algorithm")
        .agg(
            mean_elapsed=("elapsed", "mean"),
            median_elapsed=("elapsed", "median"),
            mean_cost=("cost", "mean"),
            median_cost=("cost", "median"),
            mean_norm_cost=("norm_cost", "mean"),
            median_norm_cost=("norm_cost", "median"),
        )
        .fillna(np.nan)
    )
    return stats.join(agg)


def loglog_lineplot(data: pd.DataFrame, y: str, title: str, path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metrics = [("elapsed", "Runtime (s)"), ("norm_cost", "Normalised cost")]

    for ax, (metric, ylabel) in zip(axes, metrics):
        sns.lineplot(
            data=data,
            x="num_cities",
            y=metric,
            hue="algorithm",
            estimator="mean",
            errorbar="sd",
            err_style="band",
            ax=ax,
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of cities (log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} vs size")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Algorithm",
        loc="center right",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="x-small",
    )
    for ax in axes:
        ax.get_legend().remove()

    fig.tight_layout(rect=(0, 0, 0.82, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def success_rate_plot(stats: pd.DataFrame, path: Path) -> None:
    ordered = stats.sort_values("success_rate", ascending=False).reset_index()
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=ordered, x="algorithm", y="success_rate", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
    ax.set_title("Successful runs by algorithm")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pareto_mask(points: pd.DataFrame, x_col: str, y_col: str) -> List[bool]:
    arr = points[[x_col, y_col]].to_numpy()
    mask: List[bool] = []
    for i, p in enumerate(arr):
        dominated = False
        for j, q in enumerate(arr):
            if i == j:
                continue
            if (q[0] <= p[0] and q[1] <= p[1]) and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        mask.append(not dominated)
    return mask


def pareto_plots(success_df: pd.DataFrame, families: Iterable[str] | None, path_prefix: Path) -> None:
    bins = [0, 50, 200, 1000, np.inf]
    labels = ["5-50", "50-200", "200-1000", ">1000"]
    success_df = success_df.copy()
    success_df["size_bucket"] = pd.cut(success_df["num_cities"], bins=bins, labels=labels, include_lowest=True)

    all_families = families if families is not None else sorted(success_df["problem_type"].dropna().unique())
    algorithms = sorted(success_df["algorithm"].unique())
    palette = sns.color_palette("tab20", n_colors=len(algorithms))
    color_map = dict(zip(algorithms, palette))
    marker_map = {label: m for label, m in zip(labels, ["o", "s", "D", "^"])}

    for family in all_families:
        fam_df = success_df[success_df["problem_type"] == family]
        if fam_df.empty:
            continue
        agg = (
            fam_df.groupby(["size_bucket", "algorithm"])
            .agg(med_elapsed=("elapsed", "median"), med_cost=("cost", "median"), runs=("elapsed", "count"))
            .reset_index()
        )
        agg = agg[agg["runs"] >= 3]
        if agg.empty:
            continue

        fig, ax = plt.subplots(figsize=(9.5, 5.5))
        for bucket in labels:
            sub = agg[agg["size_bucket"] == bucket]
            if sub.empty:
                continue
            pareto = pareto_mask(sub, "med_elapsed", "med_cost")
            sub = sub.assign(pareto=pareto)
            for _, row in sub.iterrows():
                ax.scatter(
                    row["med_elapsed"],
                    row["med_cost"],
                    s=80,
                    marker=marker_map.get(bucket, "o"),
                    edgecolors="black" if row["pareto"] else "none",
                    linewidths=1.2 if row["pareto"] else 0.0,
                    alpha=0.85,
                    color=color_map.get(row["algorithm"], "gray"),
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Median runtime (s, log)")
        ax.set_ylabel("Median cost (log)")
        ax.set_title(f"Runtime-quality trade-off: {family.replace('_', ' ').title()}")

        # Legends: algorithms by colour (right), size buckets by marker (top).
        algo_handles = [
            plt.Line2D([0], [0], marker="o", color=color_map[a], linestyle="", markersize=7, label=a)
            for a in algorithms
            if a in agg["algorithm"].unique()
        ]
        bucket_handles = [
            plt.Line2D([0], [0], marker=marker_map[b], color="black", linestyle="", markersize=7, label=b)
            for b in labels
            if b in agg["size_bucket"].unique()
        ]
        legend1 = ax.legend(
            handles=algo_handles,
            title="Algorithm",
            bbox_to_anchor=(1.02, 0.5),
            loc="center left",
            fontsize="x-small",
        )
        legend2 = ax.legend(
            handles=bucket_handles,
            title="Size bucket",
            loc="upper left",
            fontsize="small",
        )
        ax.add_artist(legend1)
        fig.tight_layout(rect=(0, 0, 0.76, 0.95))
        out_path = path_prefix.with_name(f"pareto_{family}.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=250)
        plt.close(fig)


def pareto_overall(success_df: pd.DataFrame, path: Path) -> None:
    bins = [0, 50, 200, 1000, np.inf]
    labels = ["5-50", "50-200", "200-1000", ">1000"]
    df = success_df.copy()
    df["size_bucket"] = pd.cut(df["num_cities"], bins=bins, labels=labels, include_lowest=True)
    agg = (
        df.groupby(["size_bucket", "algorithm"])
        .agg(med_elapsed=("elapsed", "median"), med_cost=("cost", "median"), runs=("elapsed", "count"))
        .reset_index()
    )
    agg = agg[agg["runs"] >= 5]
    if agg.empty:
        return

    algorithms = sorted(df["algorithm"].unique())
    palette = sns.color_palette("tab20", n_colors=len(algorithms))
    color_map = dict(zip(algorithms, palette))
    marker_map = {label: m for label, m in zip(labels, ["o", "s", "D", "^"])}

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for bucket in labels:
        sub = agg[agg["size_bucket"] == bucket]
        if sub.empty:
            continue
        pareto = pareto_mask(sub, "med_elapsed", "med_cost")
        sub = sub.assign(pareto=pareto)
        for _, row in sub.iterrows():
            ax.scatter(
                row["med_elapsed"],
                row["med_cost"],
                s=80,
                marker=marker_map.get(bucket, "o"),
                edgecolors="black" if row["pareto"] else "none",
                linewidths=1.2 if row["pareto"] else 0.0,
                alpha=0.85,
                color=color_map.get(row["algorithm"], "gray"),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Median runtime (s, log)")
    ax.set_ylabel("Median cost (log)")
    ax.set_title("Runtime-quality trade-off: all families")

    algo_handles = [
        plt.Line2D([0], [0], marker="o", color=color_map[a], linestyle="", markersize=7, label=a)
        for a in algorithms
        if a in agg["algorithm"].unique()
    ]
    bucket_handles = [
        plt.Line2D([0], [0], marker=marker_map[b], color="black", linestyle="", markersize=7, label=b)
        for b in labels
        if b in agg["size_bucket"].unique()
    ]
    legend_algo = ax.legend(
        handles=algo_handles,
        title="Algorithm",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize="x-small",
    )
    legend_bucket = ax.legend(
        handles=bucket_handles,
        title="Size bucket",
        loc="upper left",
        fontsize="small",
    )
    ax.add_artist(legend_algo)
    fig.tight_layout(rect=(0, 0, 0.76, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=250)
    plt.close(fig)


def load_features(path: Path) -> pd.DataFrame:
    df = pd.read_json(path, lines=True)
    feature_cols = df["features"].apply(pd.Series)
    df = pd.concat([df.drop(columns=["features"]), feature_cols], axis=1)
    return df


def feature_plots(success_df: pd.DataFrame, best_per_problem: pd.DataFrame, path_prefix: Path) -> None:
    feat_df = load_features(FEATURES_PATH)
    merged = feat_df.merge(
        success_df[
            [
                "problem_id",
                "algorithm",
                "algorithm_category",
                "norm_cost",
                "elapsed",
                "num_cities",
                "metric",
                "problem_type",
            ]
        ],
        on="problem_id",
        how="left",
    )
    merged = merged.dropna(subset=["norm_cost"])
    if merged.empty:
        return

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.scatterplot(
        data=merged,
        x="n_nodes",
        y="norm_cost",
        hue="algorithm_category",
        alpha=0.6,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of cities (log)")
    ax.set_ylabel("Normalised cost (log)")
    ax.set_title("Relative cost across sizes and algorithm families")
    fig.tight_layout()
    path_prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_prefix.with_name("feature_cost_vs_size.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.scatterplot(
        data=merged,
        x="nn_probe_cost_norm",
        y="norm_cost",
        hue="algorithm_category",
        style="is_metric",
        alpha=0.6,
        ax=ax,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nearest-neighbour probe cost (normalised, log)")
    ax.set_ylabel("Normalised cost (log)")
    ax.set_title("Quality vs probe difficulty")
    fig.tight_layout()
    fig.savefig(path_prefix.with_name("feature_vs_probe.png"), dpi=200)
    plt.close(fig)

    best = best_per_problem.copy()
    bins = [0, 50, 200, 1000, np.inf]
    labels = ["5-50", "50-200", "200-1000", ">1000"]
    best["size_bucket"] = pd.cut(best["num_cities"], bins=bins, labels=labels, include_lowest=True)
    counts = (
        best.groupby(["size_bucket", "best_algorithm"])
        .size()
        .reset_index(name="count")
        .sort_values(["size_bucket", "count"], ascending=[True, False])
    )
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.barplot(data=counts, x="size_bucket", y="count", hue="best_algorithm", ax=ax)
    ax.set_xlabel("Size bucket")
    ax.set_ylabel("Number of instances where algorithm is best")
    ax.set_title("Which algorithm wins per size bucket")
    fig.tight_layout()
    fig.savefig(path_prefix.with_name("best_algo_by_size.png"), dpi=200)
    plt.close(fig)


def compute_best_per_problem(success_df: pd.DataFrame) -> pd.DataFrame:
    idx = success_df.groupby("problem_id")["norm_cost"].idxmin()
    best = success_df.loc[idx, ["problem_id", "algorithm", "norm_cost", "num_cities", "problem_type", "metric"]].copy()
    best = best.rename(columns={"algorithm": "best_algorithm"})
    return best


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(RESULTS_PATH)
    success = results[results["status"].isin(SUCCESS_STATUSES)].copy()
    success = normalise_cost(success)
    stats = baseline_stats(results, success)
    print("Baseline statistics (per algorithm):")
    print(stats.sort_values("success_rate", ascending=False).round(3))

    loglog_lineplot(success, "elapsed", "Runtime vs problem size", FIG_DIR / "global_elapsed.png")
    loglog_lineplot(success, "norm_cost", "Normalised cost vs problem size", FIG_DIR / "global_cost.png")
    success_rate_plot(stats, FIG_DIR / "success_rates.png")

    pareto_plots(success, None, FIG_DIR / "pareto_placeholder.png")
    pareto_overall(success, FIG_DIR / "pareto_overall.png")

    best_per_problem = compute_best_per_problem(success)
    feature_plots(success, best_per_problem, FIG_DIR / "feature_placeholder.png")


if __name__ == "__main__":
    main()
