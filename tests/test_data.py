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
    cache_dir.mkdir(parents=True, exist_ok=True)
    to_store = df.copy()
    to_store.index.name = "Date"
    to_store = to_store.reset_index()
    to_store.insert(0, "ticker", ticker)
    to_store.to_parquet(cache_dir / "prices.parquet")


def test_load_prices_downloads_and_caches_when_no_cache(tmp_path, monkeypatch):
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return _fake_ohlcv("2022-01-03", periods=10)

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(calls) == 1
    assert not result.empty
    assert (tmp_path / "prices.parquet").exists()


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
    combined = pd.read_parquet(tmp_path / "prices.parquet")
    aapl_rows = _fake_ohlcv("2022-01-03", periods=10).reset_index().rename(columns={"index": "Date"})
    aapl_rows.insert(0, "ticker", "AAPL")
    pd.concat([combined, aapl_rows], ignore_index=True).to_parquet(tmp_path / "prices.parquet")

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

    on_disk = pd.read_parquet(tmp_path / "prices.parquet")
    assert set(on_disk["ticker"]) == {"MSFT", "AAPL"}
    assert (on_disk["ticker"] == "AAPL").sum() == 10
    assert (on_disk["ticker"] == "MSFT").sum() == 15
