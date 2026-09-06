import pandas as pd
import pytest
import yfinance as yf

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


def _fake_ohlcv_multiindex(start, periods):
    """Create fake OHLCV with MultiIndex columns like yfinance returns."""
    dates = pd.date_range(start, periods=periods, freq="B")
    ohlcv = {
        "Open": range(periods),
        "High": range(periods),
        "Low": range(periods),
        "Close": range(periods),
        "Volume": [1000] * periods,
    }
    df = pd.DataFrame(ohlcv, index=dates)
    # Create MultiIndex columns: (field_name, ticker_symbol)
    df.columns = pd.MultiIndex.from_product([df.columns, ["MSFT"]])
    return df


def _seed_combined(cache_dir, ticker, df):
    """Seed the cache the way `data.py` itself shards it, so these tests can't
    drift from the real layout."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    to_store = df.copy()
    to_store.index.name = "Date"
    to_store = to_store.reset_index()
    to_store.insert(0, "ticker", ticker)

    path = data._shard_path(cache_dir, data._shard_for(ticker))
    if path.exists():
        existing = pd.read_parquet(path)
        existing = existing[existing["ticker"] != ticker]
        to_store = pd.concat([existing, to_store], ignore_index=True)
    to_store.to_parquet(path, index=False)


def _read_cache(cache_dir):
    """Every cached row, across all shards."""
    frames = [
        pd.read_parquet(p)
        for shard in range(data._SHARD_COUNT)
        if (p := data._shard_path(cache_dir, shard)).exists()
    ]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def test_load_prices_downloads_and_caches_when_no_cache(tmp_path, monkeypatch):
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return _fake_ohlcv("2022-01-03", periods=10)

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(calls) == 1
    assert not result.empty
    assert not _read_cache(tmp_path).empty


def test_load_prices_reuses_cache_without_downloading(tmp_path, monkeypatch):
    _seed_combined(tmp_path, "MSFT", _fake_ohlcv("2022-01-03", periods=10))

    def _fail_download(ticker, start, end):
        raise AssertionError("should not re-download when cache covers the range")

    monkeypatch.setattr(data, "_download", _fail_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(result) == 10


def test_load_prices_extends_cache_when_range_not_covered(tmp_path, monkeypatch):
    _seed_combined(tmp_path, "MSFT", _fake_ohlcv("2022-01-03", periods=5))

    fresh = _fake_ohlcv("2022-01-03", periods=15)
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((start, end))
        return fresh

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-24", tmp_path)

    assert len(calls) == 1
    assert len(result) == 15


def test_load_prices_flattens_multiindex_from_download(tmp_path, monkeypatch):
    """Verify _download flattens MultiIndex columns returned by yfinance."""

    def _fake_yf_download(ticker, start, end, progress=False):
        return _fake_ohlcv_multiindex("2022-01-03", periods=10)

    monkeypatch.setattr(yf, "download", _fake_yf_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    # Columns should be flat, not MultiIndex (flattened by _download)
    assert not isinstance(result.columns, pd.MultiIndex)
    # Should contain the expected columns
    assert set(result.columns) == {"Open", "High", "Low", "Close", "Volume"}
    # Data should be usable
    assert len(result) == 10
    assert not result.empty


def test_load_prices_updating_one_ticker_preserves_others(tmp_path, monkeypatch):
    """The combined store holds every ticker; updating one must not drop the rest."""
    _seed_combined(tmp_path, "MSFT", _fake_ohlcv("2022-01-03", periods=10))
    _seed_combined(tmp_path, "AAPL", _fake_ohlcv("2022-01-03", periods=10))

    def _fail_download(ticker, start, end):
        raise AssertionError("should not download when cache covers the range")

    monkeypatch.setattr(data, "_download", _fail_download)

    msft = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)
    aapl = data.load_prices("AAPL", "2022-01-03", "2022-01-14", tmp_path)

    assert len(msft) == 10
    assert len(aapl) == 10

    fresh = _fake_ohlcv("2022-01-03", periods=15)
    monkeypatch.setattr(data, "_download", lambda ticker, start, end: fresh)
    data.load_prices("MSFT", "2022-01-03", "2022-01-24", tmp_path)

    on_disk = _read_cache(tmp_path)
    assert set(on_disk["ticker"]) == {"MSFT", "AAPL"}
    assert (on_disk["ticker"] == "AAPL").sum() == 10
    assert (on_disk["ticker"] == "MSFT").sum() == 15


class _FakeFastInfo(dict):
    pass


class _FakeTicker:
    def __init__(self, shares):
        self.fast_info = _FakeFastInfo(shares=shares)


def test_load_shares_outstanding_downloads_and_caches(tmp_path, monkeypatch):
    calls = []

    def _fake_yf_ticker(ticker):
        calls.append(ticker)
        return _FakeTicker(shares=1_000_000.0)

    monkeypatch.setattr(yf, "Ticker", _fake_yf_ticker)

    result = data.load_shares_outstanding("AAPL", tmp_path)

    assert result == 1_000_000.0
    assert calls == ["AAPL"]
    assert (tmp_path / "shares_outstanding.parquet").exists()


def test_load_shares_outstanding_reuses_cache_without_redownloading(tmp_path, monkeypatch):
    pd.DataFrame([{"ticker": "AAPL", "shares": 2_000_000.0}]).to_parquet(
        tmp_path / "shares_outstanding.parquet"
    )

    def _fail_ticker(ticker):
        raise AssertionError("should not re-fetch a cached ticker")

    monkeypatch.setattr(yf, "Ticker", _fail_ticker)

    result = data.load_shares_outstanding("AAPL", tmp_path)

    assert result == 2_000_000.0


def test_load_shares_outstanding_returns_zero_when_unavailable(tmp_path, monkeypatch):
    def _raise(ticker):
        raise Exception("no data")

    monkeypatch.setattr(yf, "Ticker", _raise)

    result = data.load_shares_outstanding("MISSING", tmp_path)

    assert result == 0.0
