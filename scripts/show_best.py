"""Pretty-print the best config from a completed optimizer study."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import optuna   # noqa: E402


REGIME_LABELS = {
    "R1": "Ride High (calm)",
    "R2": "Cautious (transition)",
    "R3": "Crisis (deep drawdown)",
    "R4": "Deep crisis",
    "R5": "Capitulation",
}


def fmt_alloc(panel: dict) -> str:
    """Format an allocation dict like 'TQQQ 56% + QQQ 23% + XLU 21%'."""
    parts = []
    for ticker in ("TQQQ", "QQQ", "XLU", "CASH", "$"):
        if ticker in panel:
            w = float(panel[ticker])
            if w > 1e-3:
                parts.append(f"{ticker} {w:.0%}")
    return " + ".join(parts) if parts else "(none)"


def pretty_print(study_name: str, trial_number: int | None = None) -> None:
    db = ROOT / "optimizer_runs" / f"{study_name}.db"
    if not db.exists():
        print(f"No study DB at {db}")
        return
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db}")
    complete = [t for t in study.trials if t.state.name == "COMPLETE" and t.value is not None]
    if not complete:
        print(f"No completed trials in study '{study_name}'")
        return

    if trial_number is not None:
        t = next((tt for tt in complete if tt.number == trial_number), None)
        if t is None:
            print(f"Trial {trial_number} not found.")
            return
    else:
        t = max(complete, key=lambda x: x.value)

    ua = t.user_attrs
    cfg = json.loads(ua.get("config_json") or "{}")
    if not cfg:
        print("(trial has no stored config_json)")
        return

    n_regimes = len(cfg["regimes"])
    print()
    print("=" * 78)
    print(f"  OPTIMIZED CONFIG  —  study '{study_name}', trial {t.number}")
    print("=" * 78)

    print(f"\n  Number of regimes:     {n_regimes}")
    print(f"  Drawdown window:       {cfg.get('drawdown_window_years', '?')}-year rolling peak")
    print(f"  Rebalance frequency:   {cfg.get('rebalance_frequency', '?')}")
    print(f"  Rebalance strategy:    {cfg.get('rebalance_strategy', '?')}")
    print(f"  Allocation universe:   {cfg.get('allocation_tickers', '?')}")

    print("\n  Performance:")
    print(f"    Median CAGR:                {ua.get('median_cagr', 0):.2%}")
    print(f"    p05 (worst-5% entry CAGR):  {ua.get('p05_cagr', 0):.2%}")
    print(f"    p95 (best-5% entry CAGR):   {ua.get('p95_cagr', 0):.2%}")
    print(f"    Best entry CAGR:            {ua.get('best_cagr', 0):.2%}")
    print(f"    Worst entry CAGR:           {ua.get('worst_cagr', 0):.2%}")
    print(f"    Worst max drawdown:         {ua.get('worst_max_dd', 0):.2%}")
    print(f"    Median max drawdown:        {ua.get('median_max_dd', 0):.2%}")
    print(f"    Median rebalances/year:     {ua.get('median_rebalances_per_year', 0):.2f}")
    print(f"    Worst rebalances/year:      {ua.get('worst_rebalances_per_year', 0):.2f}")
    print(f"    Entry points breaching DD:  {int(ua.get('dd_floor_breach_count', 0))}/{int(ua.get('n_runs', 0))}")
    print(f"    Optimizer score:            {t.value:.4f}")

    print("\n  Regime thresholds (QQQ drawdown bands):")
    for i in range(n_regimes):
        r = f"R{i + 1}"
        block = cfg["regimes"][r]
        lo = block["dd_low"] * 100
        hi = block["dd_high"] * 100
        label = REGIME_LABELS.get(r, "")
        print(f"    {r}  {lo:5.2f}% – {hi:6.2f}% drawdown    {label}")

    print("\n  Base allocations:")
    for i in range(n_regimes):
        r = f"R{i + 1}"
        block = cfg["regimes"][r]
        print(f"    {r}: {fmt_alloc(block)}")

    print("\n  Signal overrides (composite signal_total ranges from -6 to +6):")
    for i in range(n_regimes):
        r = f"R{i + 1}"
        block = cfg["regimes"][r]
        so = block.get("signal_overrides") or {}
        up = so.get("upside") or {}
        pr = so.get("protection") or {}
        if up.get("enabled"):
            print(f"    {r} upside     (signal > {up.get('threshold', '?'):>2}):  {fmt_alloc(up)}")
        if pr.get("enabled"):
            print(f"    {r} protection (signal < {pr.get('threshold', '?'):>2}):  {fmt_alloc(pr)}")

    print("\n  Rebalance behavior (per-regime):")
    for i in range(n_regimes):
        r = f"R{i + 1}"
        block = cfg["regimes"][r]
        rd = block.get("rebalance_on_downward", "match")
        ru = block.get("rebalance_on_upward", "match")
        print(f"    {r}: on downward = {rd:5s}    on upward = {ru:5s}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Pretty-print the best config from a study.")
    p.add_argument("--study", required=True)
    p.add_argument("--trial", type=int, default=None,
                   help="Specific trial number; defaults to highest-scoring trial.")
    args = p.parse_args()
    pretty_print(args.study, args.trial)


if __name__ == "__main__":
    main()
