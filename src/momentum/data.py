from pathlib import Path

import pandas as pd
import yfinance as yf


def load_prices(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ticker}.parquet"

    cached = None
    if cache_path.exists():
        cached = pd.read_parquet(cache_path)
        # Flatten MultiIndex columns if present
        if isinstance(cached.columns, pd.MultiIndex):
            cached.columns = cached.columns.get_level_values(0)

    if cached is not None and _covers_range(cached, start, end):
        return cached.loc[start:end]

    fresh = _download(ticker, start, end)
    combined = fresh if cached is None else _merge(cached, fresh)

    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    return combined.loc[start:end]


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
