---
name: Iterative Config Search
description: Find an optimal regime-allocation config by running 5–7+ small Monte Carlo batches, analyzing trends in each batch's top performers, and progressively narrowing the parameter space. Use when the user wants the strategy tuned to maximize CAGR while keeping max drawdown within a hard ceiling.
---

# Iterative Config Search

## When to use

The user wants to tune the regime-strategy config (thresholds, allocations, signal-override settings, n_regimes) to:

- **Maximize CAGR** (across many entry-point starts, not just one lucky 1999 start)
- **Cap max drawdown** at a hard ceiling (typically ≤ 40%, ideally ≤ 35%)
- **Avoid excessive turnover** (median rebalances/year stays modest)

A single 500-trial Optuna run is too blunt for a 50-dimensional space. Iterative refinement is the right tool: each batch teaches you something, and you bake that learning into the next batch's parameter constraints.

This methodology is **not** for trivial tuning where one batch is enough. Use it when the user expects this to be an ongoing dialog: run, analyze together, refine, repeat.

## The loop

```
Batch 1 (baseline) ─→ Analyze ─→ Propose constraints ─→ User approves
   ↓
Batch 2 (narrowed) ─→ Analyze ─→ Propose constraints ─→ User approves
   ↓
Batch 3 (narrower) ─→ ...
   ↓
After 5–7+ iterations: a tuned config that the user trusts.
```

Each batch is **50 trials** by default — enough signal to spot trends, fast enough to iterate the same day.

## Per-iteration workflow

### Step 1: Run the batch

```bash
python3 scripts/iterate.py --study iter<N> --trials 50 --entry-points 10 --jobs 2 --dd-floor -0.40
```

If `iter<N-1>` has a saved proposed constraints file, pass it:

```bash
python3 scripts/iterate.py --study iter<N> --constraints iter<N-1>_proposed.json
```

The script forces `FMF_DATA_MODE=frozen` so every trial reads from the committed snapshot. Reproducible across days.

### Step 2: Read the report

The script prints (and the contents are in `optimizer_runs/<study>.db`):

- **Top 10 trials by score** with CAGR, drawdown, rebalances
- **Drawdown profile**: how many trials breached the DD floor
- **Trend summary**: param distributions in top 10 (median, p25, p75)
- **Hypothesis checks**: e.g., "does R2_base TQQQ ≈ 0 in 80% of top trials?"
- **Regime usage**: % time top configs spent in each regime
- **Proposed constraints** for the next iteration

### Step 3: Identify the structural insight

Don't just take the proposed constraints blindly. Read the trend summary and ask:

- **What does the top performer look like?** Don't just look at numbers — look at the *shape* of its config. Which regimes did it use? What allocations? What override behavior?
- **Are the top 10 converging?** If they all share `R2_base_w_tqqq_raw < 0.05`, that's a signal you can pin.
- **Are any params irrelevant?** If a param's distribution in top 10 spans the full range, it doesn't matter — leave it free.
- **What's the worst drawdown in the top 10?** If the best score has -42% DD, the search is gaming the score. Tighten the DD floor (`--dd-floor -0.35`) for the next iter.
- **What's the gap to the user's stated targets?** They said max DD ≤ 40% (ideally 35%). If the best is at 38%, you're close but not done.

### Step 4: Propose constraints for the next batch

Two strategies:

**Strategy A — Pin a structural choice.** When ≥70% of the top trials share a value, pin it. This is the user's "TQQQ should be 0 in R2" hypothesis: if you confirm it in iter 1, lock it for iter 2 onward via `force_zero_params=["R2_base_w_tqqq_raw"]`.

**Strategy B — Narrow a continuous range.** When top trials cluster (small IQR), narrow the bounds to the median ± half-IQR. This is the auto-derived `propose_constraints` default.

Save constraints to `optimizer_runs/constraints/iter<N+1>_constraints.json`. Use the `propose_constraints` helper as a starting point, then edit by hand.

### Step 5: Validate user hypotheses explicitly

The user's domain knowledge is gold. When they say things like "TQQQ should be 0 from 8–20% drawdown", encode it:

1. Add a hypothesis check to `iterate.py`'s checks list, OR call `hypothesis_check(study, predicate)` directly.
2. If it holds in 70%+ of top trials → pin it. If it holds in 30% → it's wrong, leave the params free.
3. Sometimes the user's hypothesis is right but only conditionally (e.g., true when n_regimes=3). Surface that conditionality.

### Step 6: Decide when to stop

Stop iterating when **at least two of these are true**:

- Top trial's worst max DD ≤ user's target (e.g., ≤ -0.35)
- Top trial's median CAGR has plateaued (no improvement vs prior iter)
- Top-10 configs are converging (low variance across params)
- ≥ 5 iterations completed

If only one is true, keep going. If none after 7 iterations, escalate — the parameter space may need restructuring.

### Step 7: Apply the chosen config

When the user picks a winner from the final iteration:

1. Update `config.py` with the chosen thresholds + allocations
2. Update `README.md` and `CLAUDE.md` to reflect the new defaults + brief note on how they were derived
3. Add a locked regression test in `tests/` against the frozen cache (mirror `test_regression_ground_truth.py`'s shape) to catch silent drift

## Tools (where the code lives)

| File | Purpose |
|---|---|
| `optimizer/parameter_space.py` | `IterationConstraints` dataclass + `suggest_config(trial, constraints)` |
| `optimizer/score.py` | Monte Carlo evaluator + score with hard DD-floor penalty |
| `optimizer/runner.py` | Optuna driver; stashes per-run JSON on each trial |
| `optimizer/analysis.py` | `top_configs`, `trend_summary`, `drawdown_profile`, `hypothesis_check`, `regime_usage_summary`, `propose_constraints`, `load_run_details` |
| `scripts/iterate.py` | CLI wrapping the above; prints the per-iteration report |

## What NOT to do

- **Don't run 500 trials in a single Optuna study and call it done.** That wastes compute on regions of the search space the user already ruled out and gives no chance for human-in-the-loop course correction.
- **Don't auto-apply `propose_constraints` blindly.** It's a starting point. Always read the trend summary yourself and decide which constraints to keep.
- **Don't widen DD floor to make a config "win."** If the best score requires -50% DD, the score is wrong; either tighten the floor or increase `SCORE_DD_HARD_PENALTY`.
- **Don't skip the regime-usage summary.** A config that scores 0.7 but only ever uses R1 + R2 means it's effectively a 2-regime strategy with dead weight; pin `n_regimes=2` and re-run.
- **Don't forget to commit cache + constraints between iterations.** Each iteration should be reproducible — commit `optimizer_runs/<study>.db` and `optimizer_runs/constraints/<study>_constraints.json` so future-you can audit the trail.

## Heuristics from prior runs

These are findings the user has shared or that have shown up consistently:

- **R2 (8–20% drawdown) often performs better with TQQQ at 0%.** Pin this aggressively in iter 2+ unless the data argues against it.
- **n_regimes=2 or 3 dominates.** 4–5 regimes rarely earn their complexity. If the top trials are all 2 or 3, drop n_regimes to those choices.
- **drawdown_window_years=1 or 2 dominates.** Longer windows (5y) tend to suppress drawdown signals from recent stress.
- **Calmar ratio (CAGR/|max_dd|) is the right primary objective.** Don't be tempted to add Sharpe or Sortino as an additional objective; they're already correlated.

## Example session

```
$ python3 scripts/iterate.py --study iter1
[...50 trials, ~5 min...]
Top 10:
  trial 23: score=0.42, median_cagr=18.7%, worst_dd=-37.2%, n_regimes=3
  trial 41: score=0.40, median_cagr=21.2%, worst_dd=-42.0%, n_regimes=3   ← BREACHES FLOOR
  ...

Drawdown profile: 50 trials, 12 breached -40% floor (24%)
Trend summary:
  R2_base_w_tqqq_raw   median=0.034, p25=0.012, p75=0.087    ← user hypothesis confirmed
  n_regimes            most_common=3, freq=8/10
  ...

Hypothesis checks:
  R2_base TQQQ ≈ 0 in top 10                     8/10 matched (80%)
  Worst DD better than -35%                      4/10 matched (40%)   ← need tighter floor
  Median CAGR > 12%                              10/10 matched (100%)

Proposed constraints saved to: optimizer_runs/constraints/iter1_proposed.json
  Pin n_regimes to: [3]
  Narrow dd_t1: [0.052, 0.083]
  Narrow dd_t2: [0.156, 0.241]

# I review and edit:
# - keep n_regimes=[3]
# - keep dd_t1 / dd_t2 narrowing
# - ADD force_zero_params=["R2_base_w_tqqq_raw"] (user hypothesis)
# - tighten dd-floor to -0.35

$ python3 scripts/iterate.py --study iter2 \
    --constraints iter1_proposed.json \
    --dd-floor -0.35
[...50 trials...]
```

## Self-check

Before declaring an iteration done, ask:

1. **Did I read every section of the report**, or did I just glance at the top score?
2. **Did I update the `iterate.py` hypothesis checks** to reflect anything new the user said?
3. **Are my proposed constraints traceable** to specific evidence in the report (not vibes)?
4. **Have I told the user** what I learned, what I'm proposing, and why?
