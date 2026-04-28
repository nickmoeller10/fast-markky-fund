---
name: Iterative Config Search
description: Resume the iterative regime-config optimization for fast-markky-fund. Use when the user wants to continue tuning the QQQ/TQQQ/XLU/CASH allocation strategy via Monte Carlo backtests. This is a long-running, multi-iteration process — every iteration tests one structural hypothesis, analyzes the result, and proposes the next.
---

# Iterative Config Search — Resumable Skill

> **For a fresh chat:** read this whole file, then `optimizer_runs/iteration-log.md` (per-iter narrative), then the latest `optimizer_runs/constraints/iter<N>_constraints.json` (active hypothesis). After that you have everything needed to design and launch iter `<N+1>`.

---

## Resume-from-new-chat checklist (the 60-second startup)

1. **Read `CLAUDE.md`** for project orientation (what fast-markky-fund is, file map, regime semantics).
2. **Read this skill** — methodology + heuristics + dead-ends.
3. **Read `optimizer_runs/iteration-log.md`** — running narrative, scroll to the latest iter for what just happened.
4. **List `optimizer_runs/constraints/iter*_constraints.json`** sorted by name to find the highest N. That JSON is the active hypothesis.
5. **Run `python3 scripts/show_best.py --study iter<N>`** to see the latest champion's full allocation + signal overrides.
6. **Design iter `<N+1>`** using the "How to design the next iteration" section below.
7. **Don't stop** even after hitting the target — the user explicitly asked to keep iterating to find better trade-offs.

---

## Current goal and stop conditions

**Primary objective:** find a config with
- **Median CAGR ≥ 30%** (across 11 Monte Carlo entry points; the median, not best/lucky)
- **Worst max drawdown ≥ −35%** (across the same 11 entries; ideally ≥ −30%)
- Median rebalances/year is a **nice-to-have**, NOT a hard constraint — don't sacrifice CAGR/DD to lower it.

**Continue past target.** Per user instruction: even after hitting the goal, keep iterating to explore the Pareto frontier. The skill is iteration, not just achieving a number.

**Escalate to user when:**
- 3 consecutive iterations show no improvement on the active hypothesis axis (we may have hit a real ceiling)
- A new structural mechanism becomes worth exploring that requires code changes (e.g., a new parameter type)
- The user explicitly redirects

---

## The iterative loop in one screen

```
For iteration N:
  1. WRITE hypothesis in iteration-log.md (BEFORE running)
  2. WRITE optimizer_runs/constraints/iterN_constraints.json
  3. RUN  python3 scripts/iterate.py --study iterN --constraints iterN_constraints.json \
            --trials 50 --entry-points 10 --jobs 2 --dd-floor -0.40
        (run in background, ~10-12 min wall-clock)
  4. WAIT for completion notification
  5. ANALYZE  python3 scripts/show_best.py --study iterN
  6. PRESENT  full allocations + signal overrides to user (use the format below)
  7. UPDATE  iteration-log.md with result + lesson
  8. UPDATE  this skill if a new heuristic emerges or a dead-end is confirmed
  9. DESIGN  iter N+1 using "How to design the next iteration" + active hypothesis ladder
  10. GOTO 1
```

Each iteration: **50 trials × (1 full + 10 entry-point) backtests**. With `--jobs 2`, ~10–12 min wall-clock per iteration.

---

## Per-iteration workflow (exact commands)

### Step 1: Write the hypothesis (BEFORE running)

Open `optimizer_runs/iteration-log.md` and append a new section. Example template:

```markdown
### iter N — <one-line summary>
- **Hypothesis:** <what you believe will happen and why, citing prior iterations>
- **Key constraints:** <the 3-5 changes from the prior iteration>
- **Result:** TBD
- **Lesson:** TBD
```

This forces explicit thinking. Don't run an iteration without writing the hypothesis.

### Step 2: Write the constraints JSON

Save to `optimizer_runs/constraints/iterN_constraints.json`. Schema is defined in `optimizer/parameter_space.py` `IterationConstraints` dataclass — see "Constraint JSON reference" below.

### Step 3: Launch in the background

```bash
python3 scripts/iterate.py \
    --study iterN \
    --constraints iterN_constraints.json \
    --trials 50 \
    --entry-points 10 \
    --jobs 2 \
    --dd-floor -0.40 \
    > /tmp/iterN.log 2>&1
```

Use `Bash` tool with `run_in_background=true`. Set up a wait-for-completion monitor:

```bash
until grep -q "REPORT — study 'iterN'\|Traceback" /tmp/iterN.log 2>/dev/null; do sleep 60; done; echo "iterN DONE"
```

### Step 4: Analyze

```bash
python3 scripts/show_best.py --study iterN
```

### Step 5: Always present full allocations to the user

Required format (the user explicitly asked for this every iteration):

```
=== iter N trial X (n-regime, Wy window) ===

Regime thresholds:
  R1   0.00% – X%    drawdown   "Ride High"
  R2   X%    – Y%    drawdown   "Cautious"
  R3   Y%    – 100%  drawdown   "Crisis"

Base allocations:
  R1: TQQQ X% + QQQ X% + XLU X% (+ CASH X%)
  R2: ...
  R3: ...

Signal overrides:
  R1 upside     (signal > +X):  ...
  R1 protection (signal < −X):  ...
  R2 upside     (signal > +X):  ...
  R2 protection (signal < −X):  ...
  R3 upside     (signal > +X):  ...
  R3 protection (signal < −X):  ...

Rebalance behavior:
  R1: <down>/<up>    R2: <down>/<up>    R3: <down>/<up>

Performance:
  Median CAGR:   <%>
  Best entry:    <%>
  Worst entry:   <%>
  p05 / p95:     <%> / <%>
  Worst max DD:  <%>
  Median DD:     <%>
  Rebalances/yr: <#>
  Score:         <#>
  Breaches:      <X/11>
```

### Step 6: Update the iteration log

Replace the "Result: TBD" / "Lesson: TBD" with the actual data and what you learned.

### Step 7: Update the Pareto table (top of iteration-log.md)

Append the new row.

### Step 8: Maybe update this skill

If the iteration revealed:
- A new structural insight that holds → add to "Discovered heuristics"
- A confirmed dead end → add to "Dead ends"
- A new champion config → update "Pareto frontier snapshot"

---

## Constraint JSON reference

Source of truth: `optimizer/parameter_space.py` `IterationConstraints` dataclass.

| Field | Type | Purpose |
|---|---|---|
| `n_regimes_choices` | list[int] | Restrict the search to specific regime counts. Common: `[2]`, `[3]`. `[4]` and `[5]` are dead ends. |
| `drawdown_window_choices` | list[int] | Rolling-peak window in years. `[2]` was old default; `[3]` is current best. `[1]` causes whipsaws. `[5]` untested. |
| `threshold_bounds` | dict | Bounds for `dd_t1`, `dd_t2`, etc. Example: `{"dd_t1": [0.10, 0.14], "dd_t2": [0.18, 0.25]}`. |
| `force_zero_params` | list[str] | Force these raw weights to 0. Pattern: `R<N>_<panel>_w_<ticker>_raw` where panel ∈ `{base, upside, protection}`. |
| `weight_bounds` | dict[str, tuple] | Narrow individual raw weight bounds. Example: `{"R1_base_w_tqqq_raw": [0.85, 1.0]}`. **Critical:** to enforce normalized dominance after simplex normalization, you must ALSO cap the OTHER raw weights (lesson from iter 11). |
| `enable_cash_in_regimes` | list[str] | Add CASH as a 4th simplex coordinate for these regimes. Example: `["R2", "R3"]`. |
| `forced_base_allocations` | dict | Hard-pin a regime's base allocation. Example: `{"R2": {"CASH": 0.5, "XLU": 0.5, "TQQQ": 0, "QQQ": 0}}`. Stricter than weight_bounds — bypasses search entirely. |
| `protection_threshold_bounds` | dict | Per-regime bounds on protection signal threshold. Example: `{"R1": [-2, -2]}` to pin at -2. Sweet spot is -2 to -3. |
| `upside_threshold_bounds` | dict | Same shape, for upside override threshold. Default range is `(1, 5)`. |
| `rebalance_choices` | dict | Pin rebalance flags. Example: `{"R2_rebalance_on_downward": ["hold"], "R2_rebalance_on_upward": ["hold"]}` makes R2 a "passthrough" regime (suppresses rebs while market crosses R2). |
| `notes` | str | Human-readable note describing the hypothesis. **Required** for traceability. |

### Worked examples for common hypothesis archetypes

**(A) Narrow R1 + cash-dominant R3 (defensive baseline):**
```json
{
  "n_regimes_choices": [3], "drawdown_window_choices": [3],
  "threshold_bounds": {"dd_t1": [0.06, 0.10], "dd_t2": [0.14, 0.20]},
  "force_zero_params": [
    "R1_protection_w_tqqq_raw",
    "R2_base_w_tqqq_raw", "R2_protection_w_tqqq_raw",
    "R3_base_w_tqqq_raw", "R3_protection_w_tqqq_raw"
  ],
  "weight_bounds": {
    "R1_base_w_tqqq_raw": [0.85, 1.0],
    "R3_base_w_cash_raw": [0.7, 1.0],
    "R3_base_w_qqq_raw": [0.0, 0.3],
    "R3_base_w_xlu_raw": [0.0, 0.3]
  },
  "enable_cash_in_regimes": ["R2", "R3"],
  "rebalance_choices": {
    "R2_rebalance_on_downward": ["hold"], "R2_rebalance_on_upward": ["hold"],
    "R3_rebalance_on_downward": ["match"], "R3_rebalance_on_upward": ["hold"]
  }
}
```

**(B) Force R3 = 100% CASH (absolute-defense, current champion mechanism):**
```json
{
  "force_zero_params": ["R3_base_w_tqqq_raw", "R3_base_w_qqq_raw", "R3_base_w_xlu_raw"],
  "enable_cash_in_regimes": ["R2", "R3"]
}
```
This pins R3 base to pure CASH (the simplex falls back to 100% CASH when all other raws are zero).

**(C) Signal-driven CASH protection in R1:**
```json
{
  "force_zero_params": ["R1_base_w_cash_raw", "R1_upside_w_cash_raw"],
  "weight_bounds": {
    "R1_protection_w_cash_raw": [0.7, 1.0],
    "R1_protection_w_qqq_raw": [0.0, 0.3],
    "R1_protection_w_xlu_raw": [0.0, 0.3]
  },
  "enable_cash_in_regimes": ["R1", "R2"],
  "protection_threshold_bounds": {"R1": [-2, -2]}
}
```
R1 base stays TQQQ-heavy, but protection mode (signal ≤ -2) sends portfolio to ≥54% normalized CASH.

**(D) 100% TQQQ benchmark (sanity check / recalibrate ceiling):**
```json
{
  "n_regimes_choices": [2],
  "force_zero_params": [
    "R1_base_w_qqq_raw", "R1_base_w_xlu_raw",
    "R1_upside_w_qqq_raw", "R1_upside_w_xlu_raw",
    "R1_protection_w_qqq_raw", "R1_protection_w_xlu_raw",
    "R2_base_w_qqq_raw", "R2_base_w_xlu_raw",
    "R2_upside_w_qqq_raw", "R2_upside_w_xlu_raw",
    "R2_protection_w_qqq_raw", "R2_protection_w_xlu_raw"
  ]
}
```
Useful periodically to verify the absolute upside ceiling. Run with `--dd-floor -0.99` to suppress the DD penalty.

---

## Discovered heuristics (the gold)

Each is a rule-of-thumb earned through ≥1 iteration. Cite the supporting iter when adding new ones.

### Allocation structure
1. **CASH in R3 base ≥ 50% normalized is the single biggest DD breakthrough.** Without it, XLU's own −39% 2008 drawdown leaks through and DD floors at −42%. → iter 10 (first cash-in-R3 result), iter 12 (proper enforcement)
2. **R3 = 100% CASH (absolute defense) is even better.** When market hits dd_t2, holding pure cash stops further bleeding entirely. → iter 25 (best score 0.88)
3. **R2 cash-dominant is required if R3 is also cash-heavy.** If R2 is XLU-heavy while R3 has cash, the strategy bleeds during 5-20% drawdown band when R2 is active. → iter 17 vs iter 18
4. **Signal-driven CASH protection in R1 helps but is not sufficient on its own.** Composite signal lags fast crashes — the leverage decay damage occurs before signal_total reaches -2. → iter 23

### Regime structure
5. **n_regimes = 3 is the natural granularity.** 4-regime configs collapse one regime to a sliver, no real benefit. → iter 14
6. **2-regime gives highest CAGR but highest rebs.** Single dd boundary is crossed often. Use 3-regime + R2 passthrough for low rebs. → iter 13/16 (high rebs), iter 12/25 (low rebs via passthrough)
7. **R2 passthrough (hold/hold) suppresses rebalances.** The strategy crosses into R2 frequently but no trade fires. → iter 12, iter 25

### Drawdown window
8. **3-yr rolling window dominates 1y and 2y.** Smoother regime peak = more decisive transitions; same constraints with 2y window gave DD -36.7%, with 3y gave -25.9%. → iter 18 vs iter 20

### Signal thresholds
9. **R1 protection threshold sweet spot is −2 to −3.** Tighter (−1) causes whipsaws → 7+ rebs/yr and worse DD. Wider (−4 or below) signal lags too much. → iter 24

### R1 / dd_t1 trade-off
10. **Wide R1 (dd_t1 ≥ 14%) = high CAGR but DD floor ~−38 to −42%** regardless of mitigations, because leverage decay during R1→R2 transition window. → iter 21, 22, 23, 24
11. **Narrow R1 (dd_t1 ≤ 10%) = better DD but CAGR caps at ~24%.** Less time in TQQQ-heavy R1 means less compounding. → iter 12, 19
12. **dd_t1 ≈ 11–14% is the productive search zone** when paired with cash-heavy R2/R3 + R2 passthrough. → iter 25 (dd_t1=11.24%, CAGR 28.6%, DD -29.8%)

### Simplex normalization
13. **Raw-weight floors don't enforce normalized dominance** unless you also CAP the other raws. Setting `R3_base_w_cash_raw: [0.5, 1.0]` while leaving QQQ/XLU raws at [0, 1] gave only ~37% normalized cash. Need to also set QQQ/XLU raws to [0, 0.3] each. → iter 11

---

## Dead ends — do NOT retry

- **4-regime configs** — optimizer collapses one regime to a sliver, ~3% wide. Iter 14 confirmed.
- **n_regimes = 5** — same problem as 4-regime, more parameters wasted. Don't bother.
- **Signal threshold pinned at −1** (over-sensitive) — whipsaw rebalances, DD doesn't improve, rebs/y blows up to 7+. Iter 24.
- **R2 cash unconstrained while R3 forced cash** — R2 picks XLU-heavy, leaks during slow grinds (5-20% DD band). Iter 17.
- **R3 cash floor on raw weight without capping QQQ/XLU raws** — normalized cash ends up ~37% even with raw floor 0.5. Iter 11.
- **Alt tickers (TLT, GLD, SPLV, BIL)** — don't backtest to 1999, would bias the Monte Carlo entry-point evaluator. Permanently excluded; CASH (synthetic 4% APY MMF) is the only non-{TQQQ,QQQ,XLU} sleeve.
- **Sending "CASH" to yfinance** — it's a real ticker (Pathward Financial Inc.). `data_loader.SYNTHETIC_TICKERS` filters it out; `tests/test_synthetic_cash_safety.py` enforces. Don't bypass.

---

## Pareto frontier snapshot (as of iter 25)

| Iter | n_reg | window | dd_t1 | CAGR | Worst-DD | rebs/y | Score | Champion of |
|---|---|---|---|---|---|---|---|---|
| 12 | 3 | 2y | 6.7% | 23.6% | −27.0% | 0.59 | 0.81 | (prev) DD champion |
| 16 | 2 | 2y | 15.8% | 27.0% | −26.5% | 2.37 | 0.78 | high CAGR retention (2-reg) |
| 19 | 3 | 2y | 11.5% | 20.7% | **−24.5%** | 0.69 | 0.78 | absolute lowest DD |
| 21 | 2 | 3y | 16.85% | **35.2%** | −40.2% | 2.44 | 0.62 | first to beat 30% CAGR |
| 23 | 2 | 3y | 14.75% | 32.0% | −40.2% | 2.80 | 0.51 | beat 30% with cash protection |
| 24 | 2 | 3y | 10.05% | **36.9%** | −42.0% | 7.16 | 0.06 | top CAGR (whipsaw cost) |
| **25** | **3** | **3y** | **11.24%** | **28.6%** | **−29.8%** | **0.77** | **0.88 🏆** | **CURRENT CHAMPION (best score)** |

**Current champion (iter 25 trial 24):**
- 3-regime, 3-yr window
- R1 (0–11.24%): 81% TQQQ + 7% QQQ + 12% XLU
- R2 (11.24–19.61%): 89% CASH + 6% QQQ + 5% XLU (passthrough)
- R3 (19.61%+): **100% CASH** (forced)
- Median CAGR 28.6%, worst-DD −29.8%, rebs/y 0.77, score 0.88
- p95 CAGR 40.15%, best entry 45.4%

**Gap to user's joint target:** ~1.4 pts on CAGR (28.6 vs 30). DD is 5.2 pts inside target.

---

## Code map (where everything lives)

```
optimizer/parameter_space.py    — IterationConstraints dataclass + suggest_config()
optimizer/score.py              — Monte Carlo scorer; DD-floor penalty
optimizer/runner.py             — Optuna driver; stashes config_json + runs_json on each trial
optimizer/analysis.py           — top_configs, trend_summary, propose_constraints, etc.
optimizer/results.py            — load_results: live SQLite OR post-run parquet
data_loader.py                  — load_price_data + SYNTHETIC_TICKERS handling for CASH/$
data_cache.py                   — yfinance cache (auto/frozen/refresh modes)
scripts/iterate.py              — CLI runner — main entry point
scripts/show_best.py            — pretty-print best config from a study
scripts/freeze_data.py          — refresh the canonical price snapshot

optimizer_runs/<study>.db       — Optuna SQLite (trials + user_attrs)
optimizer_runs/<study>_results.parquet — flat per-trial table (post-run)
optimizer_runs/constraints/iter*_constraints.json — per-iter hypothesis files
optimizer_runs/iteration-log.md — running narrative
data_cache/                     — frozen yfinance pickles + manifest.json
```

Key modules to inspect when something feels off:
- Constraint not having intended effect → `optimizer/parameter_space.py` `_suggest_*` functions
- Score formula questions → `optimizer/score.py` `score_config()`
- Per-run breakdown → load `runs_json` from the SQLite, see iteration entries in `iteration-log.md` for examples

---

## Pitfalls and gotchas

1. **CASH is a real yfinance ticker** (Pathward Financial Inc., a bank stock). Synthetic injection MUST happen via `data_loader.load_price_data` which filters `SYNTHETIC_TICKERS = {"CASH", "$"}` before the yfinance call. Tests in `tests/test_synthetic_cash_safety.py` lock this contract. Symptom: "normalized CASH is going down" → real Pathward data leaked through.
2. **Pickle / numpy version mismatch.** System Python and `.venv` must both be on numpy<2 or pickle deserialization fails with `numpy._core` errors. Memory entry exists for this.
3. **Optuna `n_trials=N` with `load_if_exists=True` runs N MORE trials**, doesn't subtract existing. Re-running iter25 with `--trials 50` doesn't get you back to 50 — it adds 50 more.
4. **Streamlit runs in `auto` mode by default**, can write fresh cache files. Optimizer always uses `frozen` mode to ensure reproducibility.
5. **Cache key includes the full ticker list.** Adding/removing tickers from `config["tickers"]` changes the cache key → may need refresh. The optimizer pre-loads the full panel via `ALL_PANEL_TICKERS` to keep one cache entry.
6. **Simplex normalization can defeat raw-weight floors.** Always cap the OTHER raw weights when you want a specific normalized dominance.

---

## How to design the next iteration

### The framework

1. **Read the latest iter's "Lesson"** in `iteration-log.md`.
2. **Identify which axis lagged most** (CAGR, DD, or rebs).
3. **Pick ONE structural change** — don't change three things at once or you can't isolate the cause.
4. **Write the hypothesis explicitly** in the iteration log BEFORE running — forces clear thinking.
5. **Choose constraints that test the hypothesis cleanly** — see "Constraint JSON reference" worked examples.

### Active hypothesis ladder (ordered by what to try next)

Given the **current champion is iter 25 (CAGR 28.6%, DD -29.8%)**, the gap to target is ~1.4 pts on CAGR. The DD is well inside the target with room to spare. So the next iterations should push CAGR while staying inside the DD ceiling.

**Tier 1 — most likely to close the CAGR gap:**
- **iter 26: widen R1 from 11% to 13-14%** with iter 25's structure intact. Trade ~5 pts of DD headroom for ~3 pts CAGR.
- **iter 27: 5-yr drawdown window** (untested). Could give iter 25's structure more time in R1 → more CAGR.
- **iter 28: aggressive R1 upside override** — when signal_total is strongly positive, override to 100% TQQQ. Capture max upside in clearly-bullish regimes.

**Tier 2 — exploratory:**
- **R3 = 100% CASH but with aggressive upside override** (e.g., 80% TQQQ when signal_total > +4). Buy the panic on signal-confirmed reversals.
- **Hybrid R2: forced 60% CASH but allow R2 upside to go heavy TQQQ** when signal confirms recovery — capture mid-drawdown rebounds without leverage during the drop.
- **Per-regime drawdown windows** (would require code change to `parameter_space.py` to make `drawdown_window_years` per-regime).

**Tier 3 — recalibrate:**
- Re-run the **100% TQQQ benchmark** (iter 15 style) to verify the absolute ceiling hasn't shifted.
- Run a **wider parameter search** (200 trials, more entry points) on the iter 25 structure to see if there's a config in that space we missed.

### Designing constraints

Start from the previous iteration's `optimizer_runs/constraints/iter<N>_constraints.json` as a template. Change ONE thing. Document what changed in the `notes` field of the new JSON.

---

## Update protocol for this skill

This skill is a living document. Update it when:

- **A new heuristic is discovered** (a rule-of-thumb that holds across ≥2 iterations) → add to "Discovered heuristics" with iter citation.
- **A hypothesis is conclusively a dead end** (≥2 iterations refuting it) → add to "Dead ends".
- **A new champion config emerges** → update "Pareto frontier snapshot" and the "current champion" callout.
- **A new `IterationConstraints` field is added to `parameter_space.py`** → add it to "Constraint JSON reference".
- **A new tool/script is added** to the optimizer harness → add to "Code map".

DO NOT:
- Append per-iteration narrative here. That belongs in `optimizer_runs/iteration-log.md`.
- Make this skill describe specific commits or trial numbers — it should describe the meta-process.

When updating, commit the change immediately so the skill stays in sync with the iteration log:
```bash
git add docs/superpowers/methodologies/iterative-config-search.md \
        optimizer_runs/iteration-log.md \
        optimizer_runs/constraints/iter*.json
git commit -m "iter N: <short summary>"
```

---

## Quick command reference

```bash
# Pretty-print best config of any study
python3 scripts/show_best.py --study iter25

# Pretty-print a specific trial
python3 scripts/show_best.py --study iter25 --trial 24

# Compare top trials of any two studies
python3 -c "
import optuna, json
for s in ['iter12', 'iter25']:
    study = optuna.load_study(study_name=s, storage=f'sqlite:///optimizer_runs/{s}.db')
    best = max((t for t in study.trials if t.value is not None), key=lambda t: t.value)
    ua = best.user_attrs
    print(f'{s}: cagr={ua[\"median_cagr\"]:.2%}  dd={ua[\"worst_max_dd\"]:.2%}  rebs/y={ua[\"median_rebalances_per_year\"]:.2f}  score={best.value:.3f}')
"

# Run an iteration
python3 scripts/iterate.py --study iter26 --constraints iter26_constraints.json \
    --trials 50 --entry-points 10 --jobs 2 --dd-floor -0.40 \
    > /tmp/iter26.log 2>&1 &

# Wait for completion
until grep -q "REPORT — study 'iter26'\|Traceback" /tmp/iter26.log; do sleep 60; done; echo DONE

# Re-freeze the data cache (rare — only if you change tickers/dates)
FMF_DATA_MODE=refresh python3 scripts/freeze_data.py
```
