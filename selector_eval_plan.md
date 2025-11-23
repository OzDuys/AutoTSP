I’d think of this in two layers:

1. **“Did the selector actually help?”** (vs simple baselines)
2. **“How do the rule-based and RF selectors differ?”** (where/why one wins)

Here’s a clean way to do that.

---

## 1. Set up the comparison properly

For each instance $i$ and algorithm $a$, you already have performance:

* tour length / optimality gap: $q(a,i)$
* runtime: $t(a,i)$

For each selector $S$ (manual rules, RF), define:

* chosen algorithm: $a_S(i)$
* quality: $q_S(i) := q(a_S(i), i)$
* time (including selection overhead):
  $t_S(i) := t(a_S(i), i) + t_{\text{select}}^S(i)$

Then define three baselines:

* **Oracle selector**:
  $a_\star(i) = \arg\min_a q(a,i)$ (or whatever your primary metric is).
  This is an *upper bound* on what any selector could do.
* **Single best algorithm (SBA)**:
  the algorithm $a_{\text{SBA}}$ with best *average* performance across training instances.
* **Random choice** (optional):
  pick an algorithm uniformly at random; nice as a sanity-check baseline.

Every selector (manual rules, RF, SBA, random, oracle) is now just a policy that maps instances to algorithms. You can treat them all the same in evaluation.

---

## 2. Primary scalar metrics (simple and thesis-friendly)

Pick one **main view** that you really focus on, then maybe one secondary.

### Option A: Fixed time budget → compare quality

Choose a time budget $T$ per instance (e.g. 1s or 10s). For each algorithm and selector, measure:

* $q_S(i)$ = best tour found within time $T$.

Then aggregate:

* **Average quality / gap**:
  $$ \overline{q}_S = \frac{1}{|I|}\sum_i q_S(i) $$
  or average relative gap to optimum.

* **Regret vs oracle**:
  $$ r_S(i) = q_S(i) - q_\star(i), \quad
  \overline{r}_S = \frac{1}{|I|}\sum_i r_S(i) $$
  where $q_\star(i)$ is the oracle’s quality for instance $i$.

Then compare:

* manual vs RF vs SBA vs oracle on $\overline{q}_S$ and $\overline{r}_S$.
  If your selectors beat SBA on average, that’s a strong result.

### Option B: Fixed target quality → compare time

Pick a target quality (e.g. “within 1% of optimum”) and measure:

* time to reach the target, or time to fail (capped, with penalty if not reached).

This is nice if your focus is more on speed. You can define:

* **Average (penalised) runtime** per selector:
  $$ \overline{t}_S = \frac{1}{|I|}\sum_i t_S(i) $$
  with a big penalty if the target is not met.

You don’t need both A and B in detail; choose the one that matches your story best and maybe show the other as a supporting view.

---

## 3. Multi-objective view: tie back to Pareto

To keep the Pareto story alive without drowning in indicators:

* For each selector $S$, consider its **cloud of points**
  $$ \{(t_S(i), q_S(i)) : i \in I\} $$
* Plot all selectors on the same $(\text{time}, \text{quality})$ axes (maybe using mean and some quantiles).
* Optionally, compute a simple **hypervolume indicator** or:

  * fraction of instances where selector A’s outcome is Pareto-dominated by selector B’s outcome.

But keep this section light: one or two clear plots where it’s visually obvious which selector dominates more often is enough.

---

## 4. Selector-specific metrics (manual vs random forest)

Because you have an oracle mapping “best algorithm per instance”, you can also view selection as a classification problem.

Let $a_\star(i)$ be the oracle choice (based on your main scalar metric, e.g. tour length at budget $T$).

For each selector $S$:

* **Top-1 accuracy**:
  $\Pr[ a_S(i) = a_\star(i) ]$.
  This tells you how often the selector picks the oracle best algorithm.

* **Cost-sensitive accuracy** (more informative):
  define *regret ratio* per instance:
  $$ \rho_S(i) = \frac{q_S(i) - q_\star(i)}{q_{\text{SBA}}(i) - q_\star(i) + \varepsilon} $$
  where $\varepsilon$ just avoids division by zero if SBA = oracle.
  Then:

  * $\rho_S(i) \approx 0$: close to oracle.
  * $\rho_S(i) \approx 1$: no better than SBA.
  * $\rho_S(i) > 1$: worse than SBA (bad).

Compare distributions of $\rho_S$ for manual vs RF. This captures “near-misses” where the RF picks a slightly worse-but-similar algorithm, which pure accuracy misses.

You can also add:

* **Confusion matrix**: how often each selector picks each algorithm, vs oracle’s choice. Nice for interpretability: maybe the RF over-uses one algorithm on certain instance sizes, whereas the manual rules are more conservative.

---

## 5. Make sure the comparison is fair

A few practical points:

* **Train / test split.**
  Train your random forest on a subset of instances; evaluate both selectors on a disjoint test set.
  The manual rules don’t “train”, but they must be evaluated on the same test set.

* **Include selection overhead.**
  Add feature computation + model inference time to $t_S(i)$.
  It’ll be tiny for the RF in practice, but it’s nice to show you *thought* about it.

* **Statistical significance.**
  If you want to be fancy: use a paired test (e.g. Wilcoxon signed-rank) on per-instance regret to check if RF really beats manual in a statistically meaningful way.

---

## 6. TL;DR recommended setup

If you want one crisp story in the thesis, I’d suggest:

1. Fix a time budget $T$.
2. For each selector (manual, RF, SBA, oracle, maybe random):

   * compute average tour length / gap;
   * compute average regret vs oracle;
   * include selection overhead.
3. Show:

   * a table of these numbers;
   * one or two Pareto-style plots;
   * one small figure comparing regret distributions for manual vs RF.
4. Optionally report:

   * top-1 accuracy vs oracle;
   * which algorithms each selector tends to pick.

That gives you a really clear comparison between the hand-written rules and the random forest, backed by metrics that tie directly into your “multi-objective / Pareto / meta-optimisation” narrative.
