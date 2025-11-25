import json, pathlib, random

root = pathlib.Path(".")
problems_path = root / "Data/Instance Datasets/tsp_problems_combined.jsonl"
results_path  = root / "Data/Instance-Algorithm Datasets/combined_results.jsonl"

# 1) Load all problems
problems = []
with problems_path.open() as fh:
    for line in fh:
        if line.strip():
            problems.append(json.loads(line))

# 2) Collect and split problem_ids
pids = [p["problem_id"] for p in problems if "problem_id" in p]
rng = random.Random(42)
rng.shuffle(pids)
split = int(0.8 * len(pids))
train_ids = set(pids[:split])
test_ids  = set(pids[split:])

# 3) Write train/test problems
def write_filtered_problems(out_path, keep_ids):
    with out_path.open("w") as out:
        for p in problems:
            if p.get("problem_id") in keep_ids:
                out.write(json.dumps(p) + "\n")

problems_train = root / "Data/Instance Datasets/tsp_problems_train.jsonl"
problems_test  = root / "Data/Instance Datasets/tsp_problems_test.jsonl"
write_filtered_problems(problems_train, train_ids)
write_filtered_problems(problems_test, test_ids)

# 4) Split results the same way
def write_filtered_results(out_path, keep_ids):
    with results_path.open() as fh, out_path.open("w") as out:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("problem_id") in keep_ids:
                out.write(json.dumps(row) + "\n")

results_train = root / "Data/Instance-Algorithm Datasets/combined_results_train.jsonl"
results_test  = root / "Data/Instance-Algorithm Datasets/combined_results_test.jsonl"
write_filtered_results(results_train, train_ids)
write_filtered_results(results_test, test_ids)
