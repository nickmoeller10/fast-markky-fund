# Establish Data Provenance + Lock the Production Regression

## Context

The 115-test suite is good but two foundational gaps remain that compromise the project's accuracy and certainty claims:

1. **Backtests are not reproducible.** Every production run fetches live yfinance data with no caching. Yahoo retroactively revises history (dividend corrections, split adjustments, holding-rebalance restatements). Same backtest run on Apr 27 vs Apr 26 can produce different numbers, and you can't tell whether the change came from your code or from Yahoo. The user's stated goal — "I need this to be so secure the data has to be so trusted" — is not achievable while live yfinance is the source of record.

2. **The locked regression test does not cover production code.** `tests/test_regression_ground_truth.py` uses `down_only` strategy, `drawdown_window_enabled=False`, all signal overrides off, synthetic prices. The production config (`config.py`) uses `per_regime` + R2.upward=hold + drawdown_window enabled + 6 signal overrides enabled across R1/R2/R3 + 25 years of real QQQ history through the pre-portfolio walk. **None of these production-only paths have an end-to-end locked test.** A refactor of `apply_per_regime_direction_strategy`, `compute_rolling_ath_and_dd`, the pre-portfolio simulation, or the signal override engine could silently change real backtest results and the test suite would not notice.

These two gaps are coupled: a production-config regression test is fragile without a locked data snapshot, because Yahoo data drift would force constant test updates. Fixing both together is cheaper than fixing either alone.

The existing `test_data_cache/` pattern (`validate_tests.py:27-80`) already proves the approach — hash the args, pickle the result. The work is to promote that pattern to production data loading and add a production-config regression on top of it.

## Outcome

After this work:

- A backtest run on any day produces identical numbers as a backtest run a year later, given the same code.
- The exact production config path (per_regime + drawdown window + signal overrides + pre-portfolio walk + ~27 years of real history) has a locked regression test.
- A future refactor that silently changes any production-relevant number causes a test failure with a specific diff.
- Anyone (including a code reviewer who's never run the project) can clone the repo and reproduce your published backtest results without a network connection.

## Approach

### Part 1 — Production data cache (`data_cache.py`)

A single new module that wraps every yfinance download. Three modes selectable via env var `FMF_DATA_MODE` (default `auto`):

| Mode | Behavior | Use |
|---|---|---|
| `auto` | Use cache if present, else fetch + cache | Default for development runs |
| `frozen` | Use cache only; raise if missing | Tests + CI + reproducible runs |
| `refresh` | Always fetch + overwrite cache | Explicit refresh after a config change |

Cache file layout: `data_cache/{ticker_or_id}__{sha256_short}.parquet` plus a sidecar `data_cache/manifest.json` recording for each file: tickers, date range, fetch timestamp (UTC), yfinance version, content SHA256, row count, first/last date. The manifest is committed to git alongside the parquet files. **Parquet, not pickle** — pickle is Python-version-fragile; parquet is portable and compressible.

Public API (one function):

```python
def cached_yf_download(
    tickers, start, end=None, *, mode=None, **yf_kwargs
) -> pd.DataFrame:
    """Drop-in replacement for yfinance.download with caching + provenance."""
```

Mode resolution: explicit param > `FMF_DATA_MODE` env > `"auto"`.

Integrity guard on load: verify SHA256 of cached parquet against manifest; if mismatch, raise loudly (cache file was modified outside the system).

### Part 2 — Route production data loaders through the cache

Four call sites today:
- `data_loader.py:65` — `load_price_data(tickers, start, end_date=None, ...)` — main panel
- `data_loader.py:168, 171` — `load_spy_series(start, end_excl)` — signals SPY history
- `data_loader.py:200, 205` — `load_vix_series(start, end_excl)` — signals VIX history
- `backtest.py:600` — inline `yf.download` for pre-portfolio QQQ history (~25y)

Replace each `yf.download(...)` call with `cached_yf_download(...)`. The signatures match; the change is mechanical. Keep the existing try/except in `backtest.py:600` so missing cache in `frozen` mode doesn't crash the app — but `tests/` will run with `frozen` and explicit failure.

The validate_tests.py / test_data_cache pickle pattern stays untouched (it predates this work and tests run fine on it). Don't migrate it as part of this plan — out of scope.

### Part 3 — Freeze a canonical snapshot

One-time setup, committed to repo:

1. With `FMF_DATA_MODE=refresh`, run a script (`scripts/freeze_production_data.py`, ~30 lines) that calls every distinct yfinance fetch the production backtest performs, populating the cache.
2. Verify the manifest captures each fetch.
3. `git add data_cache/ && git commit -m "data: freeze canonical price snapshot YYYY-MM-DD"`.
4. Add `data_cache/*.parquet` to git LFS only if the total exceeds 50 MB (estimated ~5 MB for 5 tickers × 30 years × parquet — well under).

### Part 4 — Production regression test (`tests/test_production_regression.py`)

A new locked-anchor test using `FMF_DATA_MODE=frozen`. Imports `CONFIG` from `config.py` directly (no synthetic config), runs the full backtest, and asserts on a frozen ground truth.

Locked assertions (each with a tolerance and a clear "if this changed, here's why" comment):

- `LOCKED_PROD_FINAL_VALUE` — total portfolio value at `end_date`
- `LOCKED_PROD_CAGR`
- `LOCKED_PROD_MAX_DRAWDOWN`
- `LOCKED_PROD_SHARPE` (Sortino too if cheap)
- `LOCKED_PROD_REGIME_TRANSITIONS` — list of (date, from_regime, to_regime) for the first 30 transitions; this is the strongest catch for `apply_per_regime_direction_strategy` regressions
- `LOCKED_PROD_OVERRIDE_ACTIVATIONS` — count of signal-override-driven rebalances over the full history; catches signal_override_engine regressions
- `LOCKED_PROD_INITIAL_REGIME` — the regime determined by the pre-portfolio walk; catches pre-portfolio simulation bugs

Use a module-scoped fixture so the backtest runs once per pytest session.

Generation of the locked values: run the test once with the new cache populated, capture the actual values, paste them as `LOCKED_*` constants. (Same pattern as `test_regression_ground_truth.py`.)

### Part 5 — Documentation

One paragraph in `README.md` explaining the three modes, when to refresh, and what to commit. Keep it short.

## Critical Files

| File | Change |
|---|---|
| `data_cache.py` (new) | Cache wrapper + manifest + integrity check |
| `data_loader.py:49-69, 158-188, 191-222` | Replace 4 `yf.download` calls with `cached_yf_download` |
| `backtest.py:600` | Replace inline `yf.download` with `cached_yf_download` |
| `scripts/freeze_production_data.py` (new) | One-shot freeze script |
| `tests/test_production_regression.py` (new) | Locked anchor for production config |
| `data_cache/` (new) | Bundled parquet snapshots + manifest.json |
| `README.md` | Three-mode doc paragraph |
| `requirements.txt` | Add `pyarrow` (parquet engine) if not already present |

## Existing Code to Reuse

- **Caching key derivation**: `validate_tests.py:34-40` — hash-based filename. Adapt the pattern (use SHA256 + parquet instead of MD5 + pickle).
- **Locked regression structure**: `tests/test_regression_ground_truth.py` — module-scoped fixture, `LOCKED_*` constants with tolerance, descriptive failure messages with "update if intentional" hint. Copy the structure exactly.
- **Yfinance multi-index handling**: `data_loader.py:26 yf_close_to_series` — keep using this; the cache returns the raw yfinance DataFrame, and `yf_close_to_series` continues to do the column unpacking.

## Verification

End-to-end checks the implementation must pass:

1. **All 115 existing tests still pass** with `FMF_DATA_MODE=frozen` set globally.
   ```
   FMF_DATA_MODE=frozen python3 -m pytest tests/ -q
   ```

2. **Production regression locks**: `pytest tests/test_production_regression.py -v` passes on the frozen snapshot.

3. **Determinism check**: run the production backtest twice without touching the cache; assert identical equity curves byte-for-byte.
   ```
   python3 main.py > /tmp/run1.txt
   python3 main.py > /tmp/run2.txt
   diff /tmp/run1.txt /tmp/run2.txt   # must be empty
   ```

4. **Network isolation check**: with the cache populated, run the full backtest with the network disconnected (or `FMF_DATA_MODE=frozen`); it must complete without errors.

5. **Drift detection check**: temporarily change one cache parquet's content; the integrity guard must raise on next load.

6. **Refresh path check**: `FMF_DATA_MODE=refresh python3 -c "from data_cache import cached_yf_download; cached_yf_download(['QQQ'], '2020-01-01', '2020-12-31')"` succeeds, updates the manifest with a new fetch timestamp, and re-running with `auto` uses the fresh data.

## What This Does NOT Address (Explicit Non-Goals)

These are real findings from the test-suite work but are **deliberately excluded from this plan** to keep scope tight:

- Sharpe/Sortino formula standardization (cosmetic, not correctness)
- Duplicate `compute_drawdown_from_ath` in `regime_engine.py` and `backtest.py`
- `dashboard.py` streamlit-at-module-level coupling
- Dividend reinvestment integration tests (production has it disabled)
- Refactoring `run_backtest` to take pre-loaded data as args (this plan addresses the symptom — non-reproducibility — without restructuring the function)

These should be a follow-up plan after this one ships.
