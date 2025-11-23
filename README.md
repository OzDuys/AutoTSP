# AutoTSP: Instance → Features → Selector → Solver

This repository benchmarks and automates Travelling Salesman Problem (TSP) solving. The pipeline is:

**Instance dataset → Feature extraction → AutoTSP selector → Chosen solver → Solution and benchmarking.**

The layout of the repo mirrors this pipeline.

## Repository Layout

- `AutoTSP/` – Python package containing:
  - `core.py` – end‑to‑end `AutoTSP` pipeline (features → selector → solver).
  - `features.py` – fast instance feature extraction.
  - `solvers/` – solver implementations and taxonomy utilities.
  - `selectors/` + `selector.py` – rule‑based and RandomForest selectors.
- `Instance Datasets/` – instance generation, ingestion, and statistics:
  - `generate_problems.py` – synthetic and TSPLIB‑based instance generation. By default writes to `Instance Datasets/problems.jsonl`.
  - `ingest_instances_from_datasets.py` – scans `Instance Datasets/datasets/` (TSPLIB and related collections) and appends instances into `Instance Datasets/problems.jsonl`.
  - `dataset statistics/dataset_statistics.py` – dataset summary and plots for a given problems JSONL.
  - `datasets/` – raw TSPLIB and other benchmark collections.
  - `problems.jsonl` – canonical instance set used by downstream stages.
- `Instance-Algorithm Datasets/` – algorithm runs, selector training, and aggregate outputs:
  - `run_algorithms.py` – runs all registered AutoTSP solvers on a problems JSONL. By default reads `Instance Datasets/problems.jsonl` and writes to `Instance-Algorithm Datasets/Full Dataset/results.jsonl`.
  - `solver_comparison_plots.py` – seaborn plots (runtime/cost with std‑dev bands) from a results JSONL. By default uses `Instance-Algorithm Datasets/Full Dataset/results.jsonl` and writes PNGs into `Instance-Algorithm Datasets/Full Dataset/pngs/`.
  - `train_random_forest_selector.py` – trains the AutoTSP RandomForest selector from instances + results. By default reads the same problems/results as above and writes the model to `Instance-Algorithm Datasets/Full Dataset/random_forest_selector.pkl`. Optionally writes the derived training rows to `rf_training.jsonl`.
  - `Full Dataset/` – results, training rows, and plots for the full instance set.
  - `initial synthetic only/` – results and plots for the initial synthetic‑only subset.
- `visualizer/` – static React visualiser for aggregated benchmarking data:
  - `results_export_for_visualizer.py` – aggregates a results JSONL into grouped statistics consumed by `visualizer/app.jsx`. By default reads `Instance-Algorithm Datasets/Full Dataset/results.jsonl` and writes `visualizer/visualization_data.json`.
  - `index.html`, `app.jsx`, `styles.css`, `pareto.html` – front‑end for interactive metric and Pareto exploration.

## End‑to‑End Pipeline

### 1. Build the instance dataset

- Generate synthetic (and optional TSPLIB) instances:

  ```bash
  python "Instance Datasets/generate_problems.py"
  ```

  This creates/overwrites `Instance Datasets/problems.jsonl`. Use flags such as:

  - `--instances-per-count`
  - `--problem-types`
  - `--tsplib-dir "Instance Datasets/datasets/tsplib"`

- Optionally ingest additional datasets already stored under `Instance Datasets/datasets/`:

  ```bash
  python "Instance Datasets/ingest_instances_from_datasets.py"
  ```

  This appends any new problems into `Instance Datasets/problems.jsonl`.

- Summarise the resulting instance distribution:

  ```bash
  python "Instance Datasets/dataset statistics/dataset_statistics.py"
  ```

  Plots and a `dataset_summary.json` file are written into `Instance Datasets/dataset statistics/`.

### 2. Run TSP solvers on instances

Run all registered AutoTSP solvers on the canonical instance set:

```bash
python "Instance-Algorithm Datasets/run_algorithms.py"
```

By default this:

- reads `Instance Datasets/problems.jsonl`
- writes/extends `Instance-Algorithm Datasets/Full Dataset/results.jsonl`
- enforces a per‑run time budget (default 5 seconds) and optional memory limit.

You can restrict algorithms and adjust budgets with flags such as `--algorithms`, `--time-limit`, and `--memory-limit`.

### 3. Train the AutoTSP RandomForest selector

Train a learned selector that maps features → best solver (within a given time budget):

```bash
python "Instance-Algorithm Datasets/train_random_forest_selector.py" \
  --training-out "Instance-Algorithm Datasets/Full Dataset/rf_training.jsonl"
```

This:

- uses `Instance Datasets/problems.jsonl` and `Instance-Algorithm Datasets/Full Dataset/results.jsonl`
- filters runs that complete within `--time-budget` (default 5s)
- exports:
  - a model file at `Instance-Algorithm Datasets/Full Dataset/random_forest_selector.pkl`
  - optional training rows with features and labels (`best_solver`) if `--training-out` is provided.

The resulting model is used via `AutoTSP.selectors.RandomForestSelector` and the `AutoTSP` pipeline in `AutoTSP/core.py`.

### 4. Visualise solver behaviour

**Static comparison plots**

Produce log‑scaled runtime and tour‑cost plots (overall, per problem type, per origin, and by algorithm family):

```bash
python "Instance-Algorithm Datasets/solver_comparison_plots.py"
```

Outputs are written under `Instance-Algorithm Datasets/Full Dataset/pngs/`.

**Interactive React visualiser**

Aggregate the results into a JSON file consumed by the front‑end:

```bash
python "visualizer/results_export_for_visualizer.py"
```

This reads `Instance-Algorithm Datasets/Full Dataset/results.jsonl` and overwrites `visualizer/visualization_data.json`.

Then serve the repo root and browse the dashboards:

```bash
python -m http.server 8000
```

Visit:

- `http://localhost:8000/visualizer/` for metric trend plots (runtime/cost vs city‑count buckets, with configurable uncertainty bands).
- `http://localhost:8000/visualizer/pareto.html` for Pareto frontiers in runtime–cost space.

## Algorithms and Families

Solvers are implemented in `AutoTSP/solvers/` and exposed via `AutoTSP.get_solver`. They are grouped into families for analysis:

- **Exact** – e.g. branch‑and‑bound, Held–Karp, cutting‑plane, Concorde‑based solvers.
- **Approximation** – e.g. Christofides.
- **Heuristic** – classical constructive and local‑search heuristics.
- **Metaheuristic** – e.g. simulated annealing, genetic algorithms, iterated local search, ant colony.

Each solver implements a `solve(dist_matrix, time_limit=5.0)` method and returns an `AlgorithmResult`.

## Extending AutoTSP

- Add new solvers under `AutoTSP/solvers/` and register them in `AutoTSP/solvers/__init__.py` so they appear in `SOLVER_SPECS`.
- Use the common `AlgorithmResult` dataclass for outputs so all scripts can consume them.
- Extend `PROBLEM_GENERATORS` in `Instance Datasets/generate_problems.py` to introduce new instance structures; ensure each record has a unique `problem_type` and `metric`.
- Extend or swap the selector by implementing a new class under `AutoTSP/selectors/` and exposing it via `AutoTSP.selectors.get_selector`.

### TSPLIB and Other External Datasets

- Place TSPLIB `.tsp`/`.atsp` files (and similar collections) under `Instance Datasets/datasets/` (e.g. `Instance Datasets/datasets/tsplib/`).
- Generate or ingest them into the unified JSONL:
  - via `generate_problems.py` with `--tsplib-dir "Instance Datasets/datasets/tsplib"`, or
  - via `ingest_instances_from_datasets.py` to scan the whole `datasets/` tree.
- Optional deterministic transforms (rotate, jitter, rescale, mirror, etc.) can be applied at generation time and are preserved in the metadata (`origin`, `source_name`, `transformation`) so visualisations can distinguish synthetic vs real instances and different structural variants.
