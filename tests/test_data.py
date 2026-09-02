import pandas as pd
import pytest

from momentum import data


def _fake_ohlcv(start, periods):
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": range(periods),
            "High": range(periods),
            "Low": range(periods),
            "Close": range(periods),
            "Volume": [1000] * periods,
        },
        index=dates,
    )


def test_load_prices_downloads_and_caches_when_no_cache(tmp_path, monkeypatch):
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return _fake_ohlcv("2022-01-03", periods=10)

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(calls) == 1
    assert not result.empty
    assert (tmp_path / "MSFT.parquet").exists()


def test_load_prices_reuses_cache_without_downloading(tmp_path, monkeypatch):
    cached = _fake_ohlcv("2022-01-03", periods=10)
    cached.to_parquet(tmp_path / "MSFT.parquet")

    def _fail_download(ticker, start, end):
        raise AssertionError("should not re-download when cache covers the range")

    monkeypatch.setattr(data, "_download", _fail_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(result) == 10


def test_load_prices_extends_cache_when_range_not_covered(tmp_path, monkeypatch):
    cached = _fake_ohlcv("2022-01-03", periods=5)
    cached.to_parquet(tmp_path / "MSFT.parquet")

    fresh = _fake_ohlcv("2022-01-03", periods=15)
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((start, end))
        return fresh

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-24", tmp_path)

    assert len(calls) == 1
    assert len(result) == 15
