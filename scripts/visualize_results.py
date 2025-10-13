#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pathlib
from typing import Iterable, List

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def parse_args(raw_args: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise TSP benchmarking results with standard deviation bands.")
    parser.add_argument(
        "--results",
        type=pathlib.Path,
        default=pathlib.Path("data/results.jsonl"),
        help="Input JSONL file with algorithm runs.",
    )
    parser.add_argument(
        "--figure",
        type=pathlib.Path,
        default=pathlib.Path("data/results.png"),
        help="Destination for rendered plot (PNG).",
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


def render(df: pd.DataFrame, output: pathlib.Path) -> None:
    completed = df[df["status"] == "complete"].copy()
    if completed.empty:
        raise SystemExit("No completed runs to plot.")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.lineplot(
        data=completed,
        x="num_cities",
        y="elapsed",
        hue="algorithm",
        estimator="mean",
        errorbar="sd",
        err_style="band",
        ax=axes[0],
    )
    axes[0].set_title("Runtime by City Count (mean ± 1σ)")
    axes[0].set_xlabel("Number of Cities")
    axes[0].set_ylabel("Elapsed Time (s)")

    sns.lineplot(
        data=completed,
        x="num_cities",
        y="cost",
        hue="algorithm",
        estimator="mean",
        errorbar="sd",
        err_style="band",
        ax=axes[1],
    )
    axes[1].set_title("Tour Cost by City Count (mean ± 1σ)")
    axes[1].set_xlabel("Number of Cities")
    axes[1].set_ylabel("Tour Cost")

    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200)
    print(f"Saved figure to {output}")


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
