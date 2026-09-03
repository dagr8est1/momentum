from pathlib import Path

import pandas as pd
import yfinance as yf

_COMBINED_FILENAME = "prices.parquet"
_COLUMNS = ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]

# Combined store is loaded once per cache_dir and reused for every ticker in
# a run, instead of re-reading the (potentially large) file on every call.
_combined_cache: dict[str, pd.DataFrame] = {}


def load_prices(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    combined = _load_combined(cache_dir)

    cached = _ticker_slice(combined, ticker)
    if cached is not None and _covers_range(cached, start, end):
        return cached.loc[start:end]

    fresh = _download(ticker, start, end)
    updated_ticker = fresh if cached is None else _merge(cached, fresh)

    _store_ticker(cache_dir, combined, ticker, updated_ticker)
    return updated_ticker.loc[start:end]


def _combined_path(cache_dir: Path) -> Path:
    return cache_dir / _COMBINED_FILENAME


def _load_combined(cache_dir: Path) -> pd.DataFrame:
    path = _combined_path(cache_dir)
    key = str(path)
    if key not in _combined_cache:
        if path.exists():
            _combined_cache[key] = pd.read_parquet(path)
        else:
            _combined_cache[key] = pd.DataFrame(columns=_COLUMNS)
    return _combined_cache[key]


def _ticker_slice(combined: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    rows = combined[combined["ticker"] == ticker]
    if rows.empty:
        return None
    return rows.drop(columns="ticker").set_index("Date").sort_index()


def _store_ticker(
    cache_dir: Path, combined: pd.DataFrame, ticker: str, ticker_df: pd.DataFrame
) -> None:
    to_store = ticker_df.copy()
    to_store.index.name = "Date"
    to_store = to_store.reset_index()
    to_store.insert(0, "ticker", ticker)

    other_tickers = combined[combined["ticker"] != ticker]
    updated = to_store if other_tickers.empty else pd.concat(
        [other_tickers, to_store], ignore_index=True
    )

    path = _combined_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(path)
    _combined_cache[str(path)] = updated


def _covers_range(df: pd.DataFrame, start: str, end: str) -> bool:
    if df.empty:
        return False
    return df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end)


def _merge(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([cached, fresh]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False)
    # yfinance may return MultiIndex columns; flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df
