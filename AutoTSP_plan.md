# **AutoTSP Implementation Plan**

## **1\. Vision & Objective**

To transform the current benchmarking repository into an intelligent Python package (AutoTSP) that acts as a meta-optimizer. Instead of manually selecting a solver, users pass a problem instance to AutoTSP, which analyzes the graph features and selects the optimal strategy to maximize solution quality within the strict **5-second execution budget**.

## **2\. Proposed Directory Structure**

We will refactor the root-level algorithms/ directory into the AutoTSP package to create a self-contained library.

AutoTSP/  
├── \_\_init\_\_.py          \# Exposes the main AutoTSP class  
├── core.py              \# Main entry point (class AutoTSP)  
├── features.py          \# Feature Extraction Logic ("The Eyes")  
├── selector.py          \# Decision Logic ("The Brain")  
├── solvers/             \# (Refactored from current root /algorithms)  
│   ├── \_\_init\_\_.py  
│   ├── base.py          \# standardized BaseSolver class  
│   ├── exact.py         \# held\_karp, branch\_and\_bound, cutting\_plane  
│   ├── approx.py        \# christofides  
│   ├── heuristics.py    \# nearest\_neighbor (simple & multi-start), two\_opt  
│   └── meta.py          \# simulated\_annealing, genetic, ant\_colony  
└── utils/  
    └── taxonomy.py      \# Enums for AlgorithmFamily (Exact, Meta, etc.)

## **3\. Implementation Phases**

### **Phase 1: Refactoring & Standardization ("The Hands")**

**Goal:** Move existing logic into the package and enforce a uniform interface.

1. **Migration:** Move files from root algorithms/ into AutoTSP/solvers/.  
2. **Standardization (solvers/base.py):**  
   * Ensure all solvers inherit from a BaseSolver.  
   * Standardize the solve() signature:  
     def solve(self, graph, time\_limit=5.0) \-\> AlgorithmResult:  
         \# ...

   * *Crucial:* The time\_limit passed to the solver must be Global Limit \- Feature Extraction Time.

### **Phase 2: Feature Extraction ("The Eyes")**

**Goal:** Implement $O(N)$ or $O(N \\log N)$ analysis of the problem graph. The extraction **must** be extremely fast (aim for \< 0.2s) to leave time for the actual solver.

1. **features.py Implementation:**  
   * **Instant Features (**$O(1)$**):**  
     * n\_nodes (The primary filter)  
     * is\_metric (Boolean check)  
   * **Spatial Features (**$O(N)$**):**  
     * std\_dev\_x, std\_dev\_y (Spread)  
     * bbox\_area (Density proxy)  
     * centroid\_dispersion (Radius of gyration)  
   * **Sampling Features (Approximations):**  
     * landmark\_10\_dist: Avg distance from a random node to 10 others (fast density check).  
     * nn\_probe\_cost: Cost of a greedy path for the first 10% of nodes (clustering check).

### **Phase 3: The Selector ("The Brain")**

**Goal:** Map features to the best solver.

1. **selector.py \- Rule-Based (V1):**  
   * Implement hard-coded logic based on your Thesis research/No Free Lunch observations.  
   * *Example Logic:*  
     * IF n \< 14: Use held\_karp (Dynamic Programming).  
     * IF n \< 25: Use branch\_and\_bound.  
     * IF n \> 2000: Use multi\_start\_nearest\_neighbor (Need speed).  
     * IF is\_clustered: Use genetic\_algorithm.  
     * ELSE: Use simulated\_annealing (Robust default).  
2. **selector.py \- ML-Based (V2 \- Thesis "Meat"):**  
   * Train a Random Forest Classifier using your existing data/results.jsonl.  
   * **Input:** Graph Features.  
   * **Target:** The solver with the best tour\_cost that finished in \< 5s.  
   * *Note:* This justifies the "Meta-Optimisation" angle.

### **Phase 4: The Core Pipeline**

**Goal:** Tie it together in core.py.

class AutoTSP:  
    def solve(self, problem\_data, time\_budget=5.0):  
        start\_time \= time.time()  
          
        \# 1\. Extract Features  
        features \= FeatureExtractor.extract(problem\_data)  
          
        \# 2\. Calculate Remaining Budget  
        overhead \= time.time() \- start\_time  
        remaining\_budget \= time\_budget \- overhead  
          
        \# 3\. Select Algorithm  
        solver\_class \= Selector.predict(features, remaining\_budget)  
          
        \# 4\. Execute  
        result \= solver\_class.solve(problem\_data, time\_limit=remaining\_budget)  
          
        \# 5\. Return result with meta-data (which solver was chosen and why)  
        return result

### **Phase 5: Validation & Benchmarking**

**Goal:** Prove AutoTSP is better than just using one algorithm.

1. **New Script:** scripts/benchmark\_autotsp.py.  
2. **Comparison:** Run AutoTSP against the "Single Best Solver" (e.g., Simulated Annealing) on 100 unseen graphs.  
3. **Pareto Plot:** Show that AutoTSP consistently hugs the Pareto frontier (best quality for the time constraint) across all $N$.

## **4\. Integration with Existing Tools**

* **Data Generation:** Continue using scripts/generate\_problems.py to create training data.  
* **Visualizer:** Update visualizer/ to handle a new algorithm family tag called "AutoTSP" so it can be plotted alongside "Exact" and "Metaheuristic".