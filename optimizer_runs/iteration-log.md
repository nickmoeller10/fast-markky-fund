# Iteration Log — Config Optimizer

> **Purpose:** running record of every iteration's hypothesis → constraints → result → lesson learned.
> Use to avoid re-trying dead-ends and to spot trends across iterations.
>
> **Goal:** maximize CAGR (median across 11 entry points) + minimize worst-case max drawdown.
> Rebalances/year is a nice-to-have, not a hard constraint.

---

## Pareto frontier across all iterations

| Iter | n_reg | window | dd_t1 | dd_t2 | CAGR | DD | rebs/y | score |
|---|---|---|---|---|---|---|---|---|
| 1  | 3 | 1-3y | 0.10 | 0.18 | 23.0% | -46.2% | — | 0.12 |
| 4  | 3 | 1y   | 0.05 | 0.18 | 20.4% | -45.5% | — | 0.21 |
| 7  | 3 | 1y   | 0.05 | 0.18 | 23.7% | -47.5% | — | 0.21 |
| 8  | 3 | 1y   | 0.07 | 0.12 | 16.9% | -42.1% | — | 0.12 |
| 10 | 3 | 2y   | 0.05 | 0.17 | 22.4% | -32.8% | 0.60 | 0.62 |
| 11 | 3 | 2y   | 0.06 | 0.17 | 25.4% | -37.6% | 0.60 | 0.61 |
| 12 | 3 | 2y   | 0.07 | 0.16 | 23.6% | **-27.0%** | 0.59 | **0.81** |
| 13 | 2 | 1y   | 0.15 | n/a  | 27.4% | -33.9% | 2.27 | 0.58 |
| 14 | 4 | 1y   | 0.07 | 0.15 | 23.8% | -35.0% | 0.53 | 0.63 |
| 15 | 1 (TQQQ) | n/a | n/a | n/a | **43.6%** | -65.2% | 0.00 | reference |
| 16 | 2 | 2y   | 0.16 | n/a  | 27.0% | -26.5% | 2.37 | 0.78 |
| 17 | 3 | 2y   | 0.11 | 0.23 | 21.7% | -33.9% | 0.41 | 0.60 |
| 18 | 3 | 2y   | 0.15 | 0.25 | **26.8%** | -36.7% | **0.39** | 0.69 |
| 19 | 3 | 2y   | 0.11 | 0.18 | 20.7% | -24.5% | 0.69 | 0.78 |
| 20 | 3 | 3y   | 0.14 | 0.18 | 24.2% | -25.9% | 1.35 | 0.80 |
| 21 | 2 | 3y   | 0.17 | n/a  | **35.2%** | -40.2% | 2.44 | 0.62 |
| 22 | 2 | 3y   | 0.14 | n/a  | 28.9% | -38.7% | 2.74 | 0.47 |
| 23 | 2 | 3y   | 0.15 | n/a  | **32.0%** | -40.2% | 2.80 | 0.51 |
| 24 | 2 | 3y   | 0.10 | n/a  | **36.9%** | -42.0% | 7.16 | 0.06 |
| **25** | **3** | **3y** | **0.11** | **0.20** | **28.6%** | **-29.8%** | **0.77** | **0.88 🏆** |

**Goal:** CAGR ≥ 30% AND worst-DD ≥ -35%. **Status:** approaching but unmet.

---

## Iteration entries

### iter 10 — first CASH-in-R3 result (BREAKTHROUGH)
- **Hypothesis:** XLU in R2/R3 was bleeding -39% during 2008. Adding CASH (synthetic 4% APY, zero-DD MMF) as a 4th searchable asset in R2 and R3 should cap that leak.
- **Key constraints:** `enable_cash_in_regimes=["R2","R3"]`, R1 base TQQQ raw [0.6, 1.0], dd_t1 [0.04, 0.07], dd_t2 [0.10, 0.18].
- **Result (trial 44):** CAGR 22.4%, worst-DD -32.8%, rebs/y 0.60. Best worst-DD ever at that point.
- **Lesson:** CASH in R3 cracks the drawdown ceiling. R3 base of 65% CASH stopped the 2008/2022 leak.

### iter 11 — push R1 leverage harder
- **Hypothesis:** Now that R3-cash defends the tail, R1 can be more leveraged for CAGR gain.
- **Key constraints:** R1 base TQQQ raw [0.75, 1.0]. R3 base CASH raw floor at 0.5.
- **Result (trial 44):** CAGR 25.4%, DD -37.6%. CAGR up but DD slipped.
- **Lesson:** A raw-weight floor on CASH (0.5) didn't enforce dominance after simplex normalization — other raws were unbounded. Need to also CAP the others.

### iter 12 — properly enforce R3 cash dominance + ultra R1 (CHAMPION on DD/score)
- **Hypothesis:** Cap R3's QQQ/XLU raws so cash stays dominant after normalization. Push R1 to ULTRA (raw [0.85, 1.0]).
- **Key constraints:** `R3_base_w_cash_raw [0.7, 1.0]` + R3 QQQ/XLU each [0, 0.3]. R1 TQQQ raw [0.85, 1.0]. dd_t1 [0.05, 0.08], dd_t2 [0.13, 0.20].
- **Result (trial 33):** **CAGR 23.6%, DD -27.0% ⭐, rebs/y 0.59, score 0.81 ⭐**. R3 base = 72% CASH + 16% XLU + 12% QQQ.
- **Lesson:** Forcing TRUE cash dominance in R3 (≥54% normalized) is the critical structural win. R3 holds value during 2008/2022 → median DD = worst DD = -27%, all entries clustered.

### iter 13 — try 2-regime simpler structure (CAGR champion)
- **Hypothesis:** A 2-regime structure (just calm vs everything-else) might capture more upside with cleaner mental model.
- **Key constraints:** `n_regimes_choices=[2]`, `drawdown_window_choices=[1,2,3]`, dd_t1 [0.05, 0.15], R1 TQQQ raw [0.7, 1.0], R2 CASH allowed.
- **Result (trial 13):** CAGR **27.4%**, best-entry CAGR **39.2%**, DD -33.9%, rebs/y **2.27**. 1y window. dd_t1 = 14.7% (wide R1).
- **Lesson:** 2-regime + wide R1 + 1y window = highest CAGR, but rebs/y is 4x higher. Single dd boundary gets crossed often → many rebs.

### iter 14 — try 4-regime granularity
- **Hypothesis:** An extra intermediate band might smooth transitions during slow grinds.
- **Key constraints:** `n_regimes_choices=[4]`, R4 cash-dominant, all middle regimes hold/hold.
- **Result (trial 39):** CAGR 23.8%, DD -35.0%, rebs/y 0.53. R3 (only 3% wide, 14.94-18.17%) was vestigial.
- **Lesson:** 4 regimes don't add value — optimizer collapses one to a sliver. 3 regimes is the natural granularity.

### iter 15 — 100% TQQQ buy-and-hold benchmark (REFERENCE)
- **Hypothesis:** Establish the absolute upside ceiling and downside floor.
- **Key constraints:** Force all non-TQQQ raws to zero, n_regimes=2 with both regimes 100% TQQQ.
- **Result:** **Median CAGR 43.6%, worst-entry CAGR 31.4%, worst max DD -65.2%, rebs/y 0.00**.
- **Lesson:** TQQQ alone hits CAGR target on EVERY entry but takes -65% drawdowns. Our regime strategies trade ~35-45% CAGR for ~38-45 pts of DD protection. The user's CAGR≥30% AND DD≥-35% target sits at the tight end of this Pareto frontier.

### iter 16 — 2-regime + 2y window + cash-dominant R2 (CAGR/DD double champion at 2-regime)
- **Hypothesis:** iter 13's 2-regime CAGR with iter 12's slow window and forced R2-cash dominance should keep CAGR but improve DD.
- **Key constraints:** `n_regimes=[2]`, `drawdown_window=[2]`, dd_t1 [0.10, 0.16], R1 TQQQ ultra-aggressive, R2 cash-dominant.
- **Result (trial 29):** CAGR **27.0%**, DD **-26.45% ⭐ (best ever)**, rebs/y 2.37, score 0.78. R1 0-15.8%, R2 base 77% CASH + 23% XLU.
- **Lesson:** 2-regime + cash-dominant R2 gives the best CAGR/DD pair seen so far, BUT 2-regime structure has high rebs/y because the single dd boundary is crossed often during normal volatility.

### iter 17 — try 3-regime wide R1 + passthrough R2 + cash R3 (low-rebs but lost CAGR/DD)
- **Hypothesis:** Wide R1 (iter 13/16) + iter 12's passthrough R2 + cash R3 should combine best of both.
- **Key constraints:** dd_t1 [0.10, 0.16], dd_t2 [0.18, 0.25], R3 cash raw [0.7, 1.0]. R2 unconstrained.
- **Result (trial 42):** CAGR 21.7%, DD -33.9%, rebs/y **0.41 ⭐**. R2 base ended at 84% XLU + 7% CASH (XLU bled during 2008).
- **Lesson:** R2 needs forced cash too — leaving it unconstrained let optimizer pick XLU-heavy which leaks during slow grinds.

### iter 18 — fix R2 cash constraint
- **Hypothesis:** Adding cash dominance to R2 (in addition to R3) should fix iter 17's CAGR/DD regression.
- **Key constraints:** `R2_base_w_cash_raw [0.5, 1.0]` + R2 QQQ/XLU each [0, 0.3]. Other constraints same as iter 17.
- **Result (trial 46):** CAGR **26.8%**, DD -36.7%, rebs/y **0.39 ⭐ (lowest ever in 3-regime)**. R1 dd_t1 = 14.5% (wide). R2 base = 79% CASH + 17% QQQ + 4% XLU.
- **Lesson:** Wide R1 (14.5%) gives CAGR back but blows DD. With ULTRA R1 leverage (87% TQQQ) + dd_t1 of 14.5%, the leverage exposure window is ~14% market drop = ~36% portfolio drop before regime switches. Need narrower R1.

### iter 19 — find sweet-spot R1 width (NEW DD RECORD)
- **Hypothesis:** dd_t1 around 10% should split the iter 12 / iter 18 trade-off — keep CAGR ~25% with DD ~-30% and low rebs.
- **Key constraints:** dd_t1 [0.08, 0.12], dd_t2 [0.16, 0.22], R2 cash raw [0.6, 1.0], R3 cash raw [0.7, 1.0], R1 ultra-aggressive.
- **Result (trial 29):** CAGR **20.7%**, DD **-24.5% ⭐ NEW RECORD**, rebs/y 0.69, score 0.78. dd_t1 = 11.45%. R2 base = 70% CASH, R3 = 73% CASH.
- **Lesson:** Tighter R1 + tighter R2 cash floor (60% vs 50%) bought ~2 pts of DD but cost ~6 pts of CAGR. R2 cash floor of 0.6 normalized to ~70% CASH which is too defensive — strategy under-earns during the 11.5-18% drawdown band where R2 lives. Sweet spot for CAGR/DD trade-off is around dd_t1 = 13-14%, R2 cash ~50% normalized.

### iter 20 — try LONGER drawdown window (3y)
- **Hypothesis:** 3-yr rolling drawdown window smooths the regime signal: market grind-downs don't update the rolling peak immediately, so strategy stays in R1 longer (compounding) AND deep drawdowns are detected with FULL depth (better R3 commit). Wider R1 (10-16%) acceptable because window is slower.
- **Key constraints:** `drawdown_window_choices=[3]`, dd_t1 [0.10, 0.16], dd_t2 [0.18, 0.28], R1 ultra TQQQ, R2 cash [0.5, 1.0], R3 cash [0.7, 1.0].
- **Result (trial 1):** CAGR **24.2%**, DD **-25.9%** (#2 best ever), p95 CAGR **30.9%** (top entries beat 30%!), best_cagr 34.0%, rebs/y 1.35, score **0.80**. dd_t1 = 14.1%. R1 base 81% TQQQ. R2 base 73% CASH. R3 base 70% CASH.
- **Lesson:** **3y window is a STRUCTURAL improvement.** Same constraints with 2y window (iter 18) gave DD -36.7%; with 3y window gave DD -25.9% (+10.8 pts) at only -2.6 pts CAGR cost. The longer window's slower rolling peak means regime transitions happen with more decisive intent. **Promote 3y window to default for iter 21+.**

### iter 21 — 2-regime + 3y window (FIRST TO HIT 30% CAGR)
- **Hypothesis:** iter 16 (2-reg) + iter 20 (3y window) — combine for best CAGR/DD pair.
- **Key constraints:** `n_regimes_choices=[2]`, `drawdown_window_choices=[3]`, dd_t1 [0.10, 0.18], R1 ultra TQQQ, R2 cash [0.6, 1.0].
- **Result (trial 38):** **CAGR 35.15% ⭐⭐ FIRST TO BEAT 30%**, best_cagr **46.29%**, p95 **42.30%**, worst entry CAGR **23.38%**, DD -40.24%, rebs/y 2.44, breach 11/11. dd_t1 = 16.85% (optimizer pushed to upper bound). R1 = 86% TQQQ, R2 = 61% CASH + 18% QQQ + 21% XLU.
- **Lesson:** The 30% CAGR target IS achievable with this structure but DD slips past -35% target. The optimizer pushed dd_t1 to 16.85% (upper bound) — wider R1 = more compounding = more leverage exposure. **Need to find the dd_t1 sweet spot where CAGR is ~30% and DD is ~-32%. Probably dd_t1 around 13-15%.**

### iter 22 — narrow dd_t1 to find CAGR 30% + DD -32% sweet spot (mixed)
- **Hypothesis:** Same iter 21 structure but with dd_t1 narrowed to [0.13, 0.15]. Should drop CAGR from 35% to ~30% while improving DD from -40% to ~-32%.
- **Key constraints:** dd_t1 [0.13, 0.15], everything else same as iter 21.
- **Result (trial 32):** CAGR **28.9%**, DD **-38.7%**, rebs/y 2.74, score 0.47. dd_t1 = 14.0%. R1 = 81% TQQQ, R2 = 66% CASH.
- **Lesson:** **The trade-off between dd_t1 narrowing and DD reduction is poor in 2-regime + 3y window.** Narrowing dd_t1 by 3 pts (16.85% → 14%) cost 6 pts CAGR but only saved 1.5 pts DD. The DD floor for this structure is ~-38 to -40% regardless of dd_t1 — because once a real bear market hits, the strategy holds R1 leverage during the full transition. **To break -35% DD, need a SIGNAL-DRIVEN mechanism that de-risks R1 BEFORE the regime change.** R1 protection mode is the lever — but currently routes to XLU (which has -39% own drawdown). Fix: route R1 protection to CASH directly.

### iter 23 — R1 protection routes to CASH (BEAT 30% CAGR but DD unchanged)
- **Hypothesis:** Force R1 protection mode (signal-driven) to route to CASH-dominant allocation, decoupling DD from dd_t1 width.
- **Key constraints:** `enable_cash_in_regimes=["R1", "R2"]`, R1 base CASH forced 0, R1 protection CASH raw [0.6, 1.0], dd_t1 [0.14, 0.17].
- **Result (trial 38):** CAGR **31.96% ⭐⭐ BEATS 30% target**, best_cagr **41.5%**, p95 **38.0%**, DD **-40.17%** (no improvement), rebs/y 2.80, score 0.51, breaches 7/11. R1 protection = 68% CASH + 22% XLU + 10% QQQ.
- **Lesson:** **Cash protection works structurally (R1 protection IS holding 68% cash when signal fires) but the signal lags the market drop.** Composite signal_total needs L1+L2+L3 alignment; fastest is L1 (VIX z-score) but L2 (MACD) and L3 (MA200 cross) lag. By the time signal hits -2, the leverage decay damage is already done. **The 30% CAGR target is achievable but DD floor sits at ~-40% for any 2-regime structure with wide R1 (≥14%).**

### iter 24 — narrow dd_t1 + most-sensitive protection (-1) (whipsaw)
- **Hypothesis:** Belt-and-suspenders dual de-risk should break -35% DD floor.
- **Key constraints:** dd_t1 [0.09, 0.13], R1 protection threshold pinned at -1, R1 protection cash raw [0.7, 1.0].
- **Result (trial 4):** CAGR **36.85% ⭐ NEW CAGR RECORD**, best_cagr **41.66%**, DD **-42.04%** (worse), rebs/y **7.16** (3x prior), breach 7/11. dd_t1 = 10.05%. R1 upside (>+2) goes to **86% TQQQ** (max leverage on strong signal).
- **Lesson:** **Signal threshold at -1 caused whipsaws**: protection fires on signal=-1 → cash → signal recovers to 0 → unfires → back to TQQQ → market drops → fires again. Each "back to TQQQ" exposed the portfolio to more leverage damage. Threshold at -2 (iter 23) was a better balance. **The signal threshold has a sweet spot around -2 to -3.** Higher CAGR came from the R1 upside override hitting 86% TQQQ on strong signals — capturing more upside in bull markets.

### iter 25 — force R3 to 100% CASH (NEW CHAMPION 🏆)
- **Hypothesis:** XLU's -39% drawdown in 2008 is the irreducible defensive leak. Force R3 base to 100% CASH (no XLU, no QQQ). When market is in deep crisis (>20% DD), strategy holds pure zero-DD asset.
- **Key constraints:** 3-regime, 3y window. R1 ultra TQQQ + cash protection. R2 passthrough cash-dominant. **R3 base FORCED 100% CASH** (force_zero on QQQ + XLU). dd_t1 [0.10, 0.14], dd_t2 [0.18, 0.24].
- **Result (trial 24):** CAGR **28.61%** (close to 30% target), best_cagr **45.40%**, p95 **40.15%**, DD **-29.80%** ⭐ (well inside -35% target), rebs/y **0.77** (low!), score **0.88 ⭐⭐ ALL-TIME RECORD**. R1 = 81% TQQQ. R2 = 89% CASH + 6% QQQ + 5% XLU. R3 = 100% CASH.
- **Lesson:** **"Absolute defense" R3 = 100% CASH is the structural breakthrough.** With R3 at pure cash, the moment market crosses dd_t2 (~20% DD), portfolio stops losing. Combined with R2 passthrough (no rebs while crossing R2 band) and R1 ultra-leverage, this gives the best CAGR/DD pair seen so far. Best CAGR/DD/rebs simultaneously: 28.6% / -29.8% / 0.77. **Active hypothesis: this structure is now the baseline; further iterations refine WITHIN it.**
