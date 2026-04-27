# data_cache.py
# ======================================================================
# Production yfinance download cache.
#
# Three modes (set via FMF_DATA_MODE env var, or `mode=` kwarg):
#   - auto    (default): use cache if present, otherwise fetch + cache.
#   - frozen           : use cache only; raise if missing. Use in tests/CI
#                        and the optimizer to guarantee reproducibility.
#   - refresh          : always fetch + overwrite cache. Use after a
#                        config change to repopulate the snapshot.
#
# Cache layout (files live next to this module):
#   data_cache/
#     <ticker_id>__<sha256_short>.pkl   one file per (tickers, start, end, kwargs) tuple
#     manifest.json                     provenance metadata for every cache file
# ======================================================================

from __future__ import annotations

import hashlib
import json
import os
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf


CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
MANIFEST_PATH = CACHE_DIR / "manifest.json"
ENV_VAR = "FMF_DATA_MODE"
VALID_MODES = ("auto", "frozen", "refresh")


def _resolve_mode(mode: str | None) -> str:
    if mode is not None:
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}; valid: {VALID_MODES}")
        return mode
    env = os.environ.get(ENV_VAR, "auto").strip().lower()
    if env not in VALID_MODES:
        raise ValueError(f"{ENV_VAR}={env!r} invalid; valid: {VALID_MODES}")
    return env


def _normalize_tickers(tickers: Any) -> tuple[str, ...]:
    if isinstance(tickers, str):
        return (tickers,)
    return tuple(sorted(str(t) for t in tickers))


def _cache_key(tickers, start, end, kwargs: dict) -> tuple[str, str]:
    """Returns (ticker_id, sha256_short) for filename construction."""
    tk = _normalize_tickers(tickers)
    payload = json.dumps(
        {
            "tickers": list(tk),
            "start": str(start) if start is not None else None,
            "end": str(end) if end is not None else None,
            "kwargs": {k: str(v) for k, v in sorted(kwargs.items())},
        },
        sort_keys=True,
    ).encode()
    sha = hashlib.sha256(payload).hexdigest()[:12]
    ticker_id = "_".join(t.replace("^", "") for t in tk) or "empty"
    return ticker_id, sha


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            h.update(block)
    return h.hexdigest()


def _load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    try:
        with open(MANIFEST_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def _save_manifest(manifest: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _df_summary(df) -> dict:
    if df is None or (hasattr(df, "empty") and df.empty):
        return {"row_count": 0, "first_date": None, "last_date": None}
    rows = int(len(df))
    first = last = None
    if hasattr(df, "index") and len(df.index) > 0:
        try:
            first = pd.to_datetime(df.index.min()).strftime("%Y-%m-%d")
            last = pd.to_datetime(df.index.max()).strftime("%Y-%m-%d")
        except Exception:
            pass
    return {"row_count": rows, "first_date": first, "last_date": last}


def _load_and_verify(file_path: Path, manifest: dict, file_name: str):
    """Load a cached file and verify its SHA256 against the manifest."""
    expected = manifest.get(file_name, {}).get("sha256")
    if expected:
        actual = _file_sha256(file_path)
        if expected != actual:
            raise RuntimeError(
                f"Cache integrity failure for {file_name}: "
                f"expected sha256 {expected[:12]}…, got {actual[:12]}…"
            )
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _slice_cached_to_range(df, start, end):
    """
    Slice a cached DataFrame's DatetimeIndex to [start, end). yfinance treats
    `end` as exclusive, so we mirror that semantics here.
    """
    if df is None or not hasattr(df, "index") or len(df.index) == 0:
        return df
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        return df
    out = df
    if start is not None:
        start_ts = pd.Timestamp(start)
        out = out[out.index >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        out = out[out.index < end_ts]
    return out


def _try_subsumption(tickers, start, end, kwargs, manifest):
    """
    Look for a cached entry whose date range CONTAINS the requested [start, end).
    If found, load it and slice. Returns None if no subsuming entry exists.

    Subsumption rules:
      - tickers must match exactly (sorted)
      - yf_kwargs must match exactly
      - cached.start <= requested.start  (or cached.start is None)
      - cached.end   >= requested.end    (or cached.end is None)
    """
    target_tickers = list(_normalize_tickers(tickers))
    target_kwargs = {k: str(v) for k, v in sorted(kwargs.items())}
    target_start = pd.Timestamp(start) if start is not None else None
    target_end = pd.Timestamp(end) if end is not None else None

    for fname, entry in manifest.items():
        if entry.get("tickers") != target_tickers:
            continue
        if entry.get("yf_kwargs") != target_kwargs:
            continue

        cached_start = pd.Timestamp(entry["start"]) if entry.get("start") else None
        cached_end = pd.Timestamp(entry["end"]) if entry.get("end") else None

        # Cached entry must START on or before the request start
        if target_start is not None and cached_start is not None and cached_start > target_start:
            continue
        # Cached entry must END on or after the request end (None = open-ended = always covers)
        if target_end is not None and cached_end is not None and cached_end < target_end:
            continue

        path = CACHE_DIR / fname
        if not path.exists():
            continue
        df = _load_and_verify(path, manifest, fname)
        return _slice_cached_to_range(df, target_start, target_end)
    return None


def cached_yf_download(tickers, start=None, end=None, *, mode: str | None = None, **yf_kwargs):
    """
    Drop-in replacement for ``yfinance.download`` with on-disk caching.

    Args mirror yfinance.download. Mode resolution:
        explicit `mode=` arg  >  FMF_DATA_MODE env var  >  "auto"
    """
    resolved_mode = _resolve_mode(mode)
    ticker_id, sha = _cache_key(tickers, start, end, yf_kwargs)
    file_name = f"{ticker_id}__{sha}.pkl"
    file_path = CACHE_DIR / file_name
    manifest = _load_manifest()

    if resolved_mode == "frozen":
        if file_path.exists():
            return _load_and_verify(file_path, manifest, file_name)
        # Subsumption fallback: serve from a wider cached range if one exists
        sliced = _try_subsumption(tickers, start, end, yf_kwargs, manifest)
        if sliced is not None:
            return sliced
        raise RuntimeError(
            f"FMF_DATA_MODE=frozen but cache miss (and no subsuming entry) for "
            f"tickers={tickers} start={start} end={end} kwargs={yf_kwargs}. "
            f"Expected file: {file_path}. "
            f"Run with FMF_DATA_MODE=refresh (or use scripts/freeze_data.py) to populate."
        )

    if resolved_mode == "auto" and file_path.exists():
        try:
            return _load_and_verify(file_path, manifest, file_name)
        except Exception as exc:
            print(f"[data_cache] cache file {file_name} failed verification ({exc}); refetching")

    # auto miss OR refresh: hit the network
    df = yf.download(tickers, start=start, end=end, **yf_kwargs)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        pickle.dump(df, f)

    manifest[file_name] = {
        "tickers": list(_normalize_tickers(tickers)),
        "start": str(start) if start is not None else None,
        "end": str(end) if end is not None else None,
        "yf_kwargs": {k: str(v) for k, v in sorted(yf_kwargs.items())},
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "yfinance_version": getattr(yf, "__version__", "unknown"),
        "python_version": sys.version.split()[0],
        "sha256": _file_sha256(file_path),
        **_df_summary(df),
    }
    _save_manifest(manifest)
    return df


def cache_status() -> dict:
    """Convenience for tooling: count of cached files + manifest summary."""
    if not CACHE_DIR.exists():
        return {"cache_dir": str(CACHE_DIR), "exists": False, "files": 0}
    files = sorted(p.name for p in CACHE_DIR.glob("*.pkl"))
    manifest = _load_manifest()
    return {
        "cache_dir": str(CACHE_DIR),
        "exists": True,
        "files": len(files),
        "manifest_entries": len(manifest),
        "tickers_seen": sorted({t for entry in manifest.values() for t in entry.get("tickers", [])}),
    }
