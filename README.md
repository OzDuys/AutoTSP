# Travelling Salesman Problem Benchmarking

The repository is structured to explore, extend, and compare travelling salesman problem (TSP) solvers across varying city counts under a strict 5 second execution budget per run.

## Repository Layout
- `code.ipynb` – original exploratory notebook retained for reference.
- `algorithms/` – Python modules implementing each canonical solver (`branch_and_bound`, `held_karp`, `cutting_plane`, `simulated_annealing`, `genetic_algorithm`, `concorde_approx`, `greedy`). New solvers should expose a callable with the signature `solver(dist_matrix, time_limit=5.0)` and return an `AlgorithmResult`.
- `scripts/generate_problems.py` – utility for sampling varied TSP instances and storing them as JSONL (`data/problems.jsonl` by default).
- `scripts/run_algorithms.py` – executes each registered solver against problem instances, skipping runs already captured in the results log and enforcing 5 s CPU plus configurable memory limits (flagging violations as infeasible).
- `scripts/visualize_results.py` – seaborn-based plots (log-scaled with mean ±1σ bands) for runtime and tour cost statistics from JSONL benchmark outputs.
- `data/` – generated artefacts (problem sets, benchmark summaries, plots). Created on demand.

## Typical Workflow
1. Generate benchmark problems (defaults now cover city counts from 5 up to 5000 and use seed `42` for repeatability): `python scripts/generate_problems.py --instances-per-count 25` (use `--seed` to change the sequence).
2. Run algorithm evaluations and append results to JSONL (skipping those already processed): `python scripts/run_algorithms.py --problems data/problems.jsonl --results data/results.jsonl --memory-limit 1024`. Any solver exceeding the 5 s timeout or memory budget is marked as infeasible for that city count.
3. Visualise aggregated performance: `python scripts/visualize_results.py --results data/results.jsonl --figure data/results.png`.

## Extending the Library
- Add a new module under `algorithms/` and register it in `algorithms/__init__.py`.
- Respect the 5 second limit by calling `enforce_time_budget` (see `algorithms/base.py`).
- Return an `AlgorithmResult` to standardise downstream analysis.
