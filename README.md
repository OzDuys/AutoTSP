# AutoTSP: Instance → Features → Selector → Solver

This repository benchmarks and automates Travelling Salesman Problem (TSP) solving. The pipeline is:

**Instance dataset → Feature extraction → AutoTSP selector → Chosen solver → Solution and benchmarking.**

The layout of the repo mirrors this pipeline.

## Repository Layout

- `AutoTSP/` – Python package containing:
  - `core.py` – end‑to‑end `AutoTSP` pipeline (features → selector → solver).
  - `features.py` – fast instance feature extraction.
  - `solvers/` – solver implementations and taxonomy utilities.
    - `approx/` – `christofides`
    - `exact/` – `branch_and_bound`, `held_karp`, `cutting_plane`, `concorde_exact`
    - `heuristics/` – `simple_nearest_neighbor`, `multi_start_nearest_neighbor`, `two_opt`, `three_opt`, `shinka_spatial_heuristic`
    - `meta/` – `simulated_annealing`, `genetic_algorithm`, `iterated_local_search`, `ant_colony`, `lkh` (handles symmetric and asymmetric)
      - `LKH` – bundled LKH 3.0.13 binary (used by the `lkh`/`lkh_atsp` solvers; no env var required).
  - `selectors/` + `selector.py` – rule‑based and RandomForest selectors.
- `Data/Instance Datasets/` – instance generation, ingestion, and statistics:
  - `generate_synthetic_problems.py` – synthetic (now richer) + TSPLIB generation. Common outputs:
    - `tsp_problems_synth.jsonl` – richer synthetic suite.
    - `tsp_problems_external.jsonl` – ingested external/TSPLIB problems.
    - `tsp_problems_combined.jsonl` – merged synthetic + external set used for benchmarking.
  - `ingest_instances_from_datasets.py` – scans `Data/Instance Datasets/datasets/` (TSPLIB and related collections) and appends instances into a problems JSONL (e.g. `tsp_problems_external.jsonl`).
  - `dataset statistics/dataset_statistics.py` – dataset summary and plots for any problems JSONL.
  - `datasets/` – raw TSPLIB and other benchmark collections.
- `Data/Instance-Algorithm Datasets/` – algorithm runs, selector training, and aggregate outputs:
  - `Full Dataset/` – results/training rows/plots for the canonical set.
  - `initial synthetic only/` – results/plots for the synthetic‑only subset.
- `Scripts/` – runnable entrypoints for the pipeline:
  - `run_algorithms.py` – run solvers over a problems JSONL.
  - `train_random_forest_selector.py` – train the RF selector.
  - `solver_comparison_plots.py` – generate static plots from a results JSONL.
  - `evaluate_selectors.py` – compare selectors and baselines on a results JSONL with failure-aware penalties.
- `visualizer/` – static React visualiser for aggregated benchmarking data:
  - `results_export_for_visualizer.py` – aggregates a results JSONL into grouped statistics consumed by `visualizer/app.jsx`. By default reads `Instance-Algorithm Datasets/Full Dataset/results.jsonl` and writes `visualizer/visualization_data.json`.
  - `index.html`, `app.jsx`, `styles.css`, `pareto.html` – front‑end for interactive metric and Pareto exploration.

## End‑to‑End Pipeline

### 1. Build the instance dataset

- Generate synthetic (and optional TSPLIB) instances:

  ```bash
  python "Data/Instance Datasets/generate_synthetic_problems.py"
  ```

  This creates/overwrites a problems JSONL (e.g. `tsp_problems_synth.jsonl`). Use flags such as:

  - `--instances-per-count`
  - `--problem-types`
  - `--tsplib-dir "Data/Instance Datasets/datasets/tsplib"`

- Optionally ingest additional datasets already stored under `Instance Datasets/datasets/`:

  ```bash
  python "Data/Instance Datasets/ingest_instances_from_datasets.py"
  ```

  This appends any new problems into your chosen target (e.g. `tsp_problems_external.jsonl`).

- Summarise the resulting instance distribution:

  ```bash
  python "Data/Instance Datasets/dataset statistics/dataset_statistics.py" \
    --problems "Data/Instance Datasets/tsp_problems_combined.jsonl"
  ```

  Plots and a `dataset_summary.json` file are written into `Data/Instance Datasets/dataset statistics/`.

### 2. Run TSP solvers on instances

Run all registered AutoTSP solvers on the canonical instance set:

```bash
python "Scripts/run_algorithms.py" \
  --problems "Data/Instance Datasets/tsp_problems_combined.jsonl" \
  --results "Data/Instance-Algorithm Datasets/Full Dataset/results.jsonl"
```

By default this:

- reads the combined problems set (`Data/Instance Datasets/tsp_problems_combined.jsonl`)
- writes/extends `Data/Instance-Algorithm Datasets/Full Dataset/results.jsonl`
- enforces a per‑run time budget (default 10 seconds in the script; override with `--time-limit`) and optional memory limit.

You can restrict algorithms and adjust budgets with flags such as `--algorithms`, `--time-limit`, and `--memory-limit`.

### 3. Train the AutoTSP RandomForest selector

Train a learned selector that maps features → best solver (within a given time budget):

```bash
python "Scripts/train_random_forest_selector.py" \
  --problems "Data/Instance Datasets/tsp_problems_combined.jsonl" \
  --results "Data/Instance-Algorithm Datasets/Full Dataset/results.jsonl" \
  --training-out "Data/Instance-Algorithm Datasets/Full Dataset/rf_training.jsonl"
```

This:

- uses `Data/Instance Datasets/tsp_problems_combined.jsonl` and `Data/Instance-Algorithm Datasets/Full Dataset/results.jsonl`
- filters runs that complete within `--time-budget` (default 5s)
- exports:
  - a model file at `Data/Instance-Algorithm Datasets/Full Dataset/random_forest_selector.pkl`
  - optional training rows with features and labels (`best_solver`) if `--training-out` is provided.

The resulting model is used via `AutoTSP.selectors.RandomForestSelector` and the `AutoTSP` pipeline in `AutoTSP/core.py`.

### 4. Visualise solver behaviour

**Static comparison plots**

Produce log‑scaled runtime and tour‑cost plots (overall, per problem type, per origin, and by algorithm family):

```bash
python "Scripts/solver_comparison_plots.py" \
  --results "Data/Instance-Algorithm Datasets/Full Dataset/results.jsonl" \
  --output-dir "Data/Instance-Algorithm Datasets/Full Dataset/pngs"
```

Outputs are written under `Data/Instance-Algorithm Datasets/Full Dataset/pngs/`.

**Interactive React visualiser**

Aggregate the results into a JSON file consumed by the front‑end:

```bash
python "visualizer/results_export_for_visualizer.py"
```

This reads `Data/Instance-Algorithm Datasets/Full Dataset/results.jsonl` and overwrites `visualizer/visualization_data.json`.

Then serve the repo root and browse the dashboards:

```bash
python -m http.server 8000
```

Visit:

- `http://localhost:8000/visualizer/` for metric trend plots (runtime/cost vs city‑count buckets, with configurable uncertainty bands).
- `http://localhost:8000/visualizer/pareto.html` for Pareto frontiers in runtime–cost space.

## Algorithms and Families

Solvers are implemented in `AutoTSP/solvers/` and exposed via `AutoTSP.get_solver`. They are grouped into families for analysis:

- **Exact** – branch‑and‑bound, Held–Karp, cutting‑plane, Concorde‑based solvers.
- **Approximation** – Christofides.
- **Heuristic** – constructive/local search (e.g. nearest‑neighbour variants, 2‑opt/3‑opt, `shinka_spatial_heuristic`).
- **Metaheuristic** – simulated annealing, genetic algorithms, iterated local search, ant colony, Lin–Kernighan (`lkh`, `lkh_atsp`).

Each solver implements a `solve(dist_matrix, time_limit=5.0)` method and returns an `AlgorithmResult`. The `lkh`/`lkh_atsp` solvers use the bundled LKH binary under `AutoTSP/solvers/meta/LKH` if no system binary is found.

Registered algorithms (via `AutoTSP.solvers`):
- Exact: `branch_and_bound`, `held_karp`, `cutting_plane`, `concorde_exact`
- Approximation: `christofides`
- Heuristic: `simple_nearest_neighbor`, `multi_start_nearest_neighbor`, `two_opt`, `three_opt`, `shinka_spatial_heuristic`
- Metaheuristic: `simulated_annealing`, `genetic_algorithm`, `iterated_local_search`, `ant_colony`, `lkh`, `lkh_atsp`

## Extending AutoTSP

- Add new solvers under `AutoTSP/solvers/` and register them in `AutoTSP/solvers/__init__.py` so they appear in `SOLVER_SPECS`.
- Use the common `AlgorithmResult` dataclass for outputs so all scripts can consume them.
- Extend `PROBLEM_GENERATORS` in `Instance Datasets/generate_synthetic_problems.py` to introduce new instance structures; ensure each record has a unique `problem_type` and `metric`.
- Extend or swap the selector by implementing a new class under `AutoTSP/selectors/` and exposing it via `AutoTSP.selectors.get_selector`.

### TSPLIB and Other External Datasets

- Place TSPLIB `.tsp`/`.atsp` files (and similar collections) under `Data/Instance Datasets/datasets/` (e.g. `Data/Instance Datasets/datasets/tsplib/`).
- Generate or ingest them into the unified JSONL:
  - via `generate_synthetic_problems.py` with `--tsplib-dir "Data/Instance Datasets/datasets/tsplib"`, or
  - via `ingest_instances_from_datasets.py` to scan the whole `datasets/` tree.
- Optional deterministic transforms (rotate, jitter, rescale, mirror, etc.) can be applied at generation time and are preserved in the metadata (`origin`, `source_name`, `transformation`) so visualisations can distinguish synthetic vs real instances and different structural variants.
