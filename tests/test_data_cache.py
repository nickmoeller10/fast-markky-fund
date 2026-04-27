"""
Tests for the production data cache wrapper.
Protects: data_cache.cached_yf_download — the three-mode cache, key derivation,
manifest provenance, and integrity check.
"""
import json
import os
import pickle
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Redirect data_cache.CACHE_DIR to a tmp dir for the test."""
    import data_cache

    cache_dir = tmp_path / "data_cache"
    monkeypatch.setattr(data_cache, "CACHE_DIR", cache_dir)
    monkeypatch.setattr(data_cache, "MANIFEST_PATH", cache_dir / "manifest.json")
    monkeypatch.delenv("FMF_DATA_MODE", raising=False)
    return cache_dir


def _fake_df(start="2020-01-02", n=10, value=100.0):
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Close": [value + i for i in range(n)]}, index=idx)


@pytest.mark.unit
def test_auto_mode_fetches_then_caches(isolated_cache):
    from data_cache import cached_yf_download

    fake = _fake_df()
    with patch("data_cache.yf.download", return_value=fake) as mock_dl:
        result = cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31")

    pd.testing.assert_frame_equal(result, fake)
    assert mock_dl.call_count == 1
    assert isolated_cache.exists()
    assert (isolated_cache / "manifest.json").exists()
    pkls = list(isolated_cache.glob("*.pkl"))
    assert len(pkls) == 1


@pytest.mark.unit
def test_auto_mode_second_call_hits_cache(isolated_cache):
    from data_cache import cached_yf_download

    fake = _fake_df()
    with patch("data_cache.yf.download", return_value=fake) as mock_dl:
        first = cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31")
        second = cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31")

    pd.testing.assert_frame_equal(first, second)
    assert mock_dl.call_count == 1, "Second call should not hit yfinance"


@pytest.mark.unit
def test_frozen_mode_raises_on_cache_miss(isolated_cache):
    from data_cache import cached_yf_download

    with pytest.raises(RuntimeError, match="frozen but cache miss"):
        cached_yf_download("QQQ", start="2020-01-01", mode="frozen")


@pytest.mark.unit
def test_frozen_mode_serves_from_cache(isolated_cache):
    from data_cache import cached_yf_download

    fake = _fake_df()
    with patch("data_cache.yf.download", return_value=fake):
        cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31", mode="auto")

    # Now in frozen mode, no network needed
    with patch("data_cache.yf.download") as mock_dl:
        result = cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31", mode="frozen")
    pd.testing.assert_frame_equal(result, fake)
    mock_dl.assert_not_called()


@pytest.mark.unit
def test_refresh_mode_overwrites_cache(isolated_cache):
    from data_cache import cached_yf_download

    fake_v1 = _fake_df(value=100.0)
    fake_v2 = _fake_df(value=200.0)

    with patch("data_cache.yf.download", return_value=fake_v1):
        cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31")

    with patch("data_cache.yf.download", return_value=fake_v2) as mock_dl:
        result = cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31", mode="refresh")
    pd.testing.assert_frame_equal(result, fake_v2)
    assert mock_dl.call_count == 1


@pytest.mark.unit
def test_different_args_create_different_cache_files(isolated_cache):
    from data_cache import cached_yf_download

    with patch("data_cache.yf.download", return_value=_fake_df()):
        cached_yf_download("QQQ", start="2020-01-01")
        cached_yf_download("QQQ", start="2021-01-01")
        cached_yf_download(["QQQ", "TQQQ"], start="2020-01-01")

    pkls = list(isolated_cache.glob("*.pkl"))
    assert len(pkls) == 3, "Each unique call should produce its own cache file"


@pytest.mark.unit
def test_manifest_records_provenance(isolated_cache):
    from data_cache import cached_yf_download

    with patch("data_cache.yf.download", return_value=_fake_df()):
        cached_yf_download("QQQ", start="2020-01-01", end="2020-12-31", auto_adjust=True)

    with open(isolated_cache / "manifest.json") as f:
        manifest = json.load(f)

    assert len(manifest) == 1
    entry = next(iter(manifest.values()))
    assert entry["tickers"] == ["QQQ"]
    assert entry["start"] == "2020-01-01"
    assert entry["end"] == "2020-12-31"
    assert "fetched_at" in entry
    assert "yfinance_version" in entry
    assert "python_version" in entry
    assert "sha256" in entry
    assert entry["row_count"] == 10


@pytest.mark.unit
def test_integrity_failure_when_cache_file_tampered(isolated_cache):
    from data_cache import cached_yf_download

    with patch("data_cache.yf.download", return_value=_fake_df()):
        cached_yf_download("QQQ", start="2020-01-01")

    pkl = next(isolated_cache.glob("*.pkl"))
    # Corrupt the file
    pkl.write_bytes(pkl.read_bytes() + b"corruption")

    # Frozen mode must catch the integrity failure
    with pytest.raises(RuntimeError, match="integrity failure"):
        cached_yf_download("QQQ", start="2020-01-01", mode="frozen")


@pytest.mark.unit
def test_env_var_drives_mode(isolated_cache, monkeypatch):
    from data_cache import cached_yf_download

    monkeypatch.setenv("FMF_DATA_MODE", "frozen")
    with pytest.raises(RuntimeError, match="frozen but cache miss"):
        cached_yf_download("QQQ", start="2020-01-01")


@pytest.mark.unit
def test_invalid_mode_raises():
    from data_cache import cached_yf_download

    with pytest.raises(ValueError, match="Invalid mode"):
        cached_yf_download("QQQ", start="2020-01-01", mode="bogus")
