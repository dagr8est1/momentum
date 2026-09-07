import zlib
from pathlib import Path

import pandas as pd
import yfinance as yf

_SHARD_FILENAME = "prices_{shard:02d}.parquet"
_SHARD_COUNT = 4
_COLUMNS = ["ticker", "Date", "Open", "High", "Low", "Close", "Volume"]
_SHARES_FILENAME = "shares_outstanding.parquet"
_NO_DATA_FILENAME = "no_data_tickers.parquet"

# Combined store is loaded once per cache_dir and reused for every ticker in
# a run, instead of re-reading the (potentially large) file on every call.
_combined_cache: dict[str, pd.DataFrame] = {}
_shares_cache: dict[str, pd.DataFrame] = {}
_no_data_cache: dict[str, set[str]] = {}


def load_shares_outstanding(ticker: str, cache_dir: Path) -> float:
    """Current shares outstanding for `ticker`, cached indefinitely.

    Used as a market-cap proxy: market_cap(t) = shares_outstanding * close(t).
    Share counts drift slowly (buybacks/issuance) relative to price, so a
    single current snapshot is a reasonable approximation across a
    multi-year backtest — but it does mean this uses information (today's
    share count) that wasn't actually known on historical rebalance dates.
    Returns 0.0 if unavailable, so callers can treat it like "no data".
    """
    cache_dir = Path(cache_dir)
    combined = _load_shares_cache(cache_dir)

    rows = combined[combined["ticker"] == ticker]
    if not rows.empty:
        return float(rows.iloc[0]["shares"])

    shares = _download_shares(ticker)
    if shares:
        _store_shares(cache_dir, combined, ticker, shares)
    return float(shares or 0.0)


def _shares_path(cache_dir: Path) -> Path:
    return cache_dir / _SHARES_FILENAME


def _load_shares_cache(cache_dir: Path) -> pd.DataFrame:
    path = _shares_path(cache_dir)
    key = str(path)
    if key not in _shares_cache:
        if path.exists():
            _shares_cache[key] = pd.read_parquet(path)
        else:
            _shares_cache[key] = pd.DataFrame(columns=["ticker", "shares"])
    return _shares_cache[key]


def _store_shares(cache_dir: Path, combined: pd.DataFrame, ticker: str, shares: float) -> None:
    other = combined[combined["ticker"] != ticker]
    new_row = pd.DataFrame([{"ticker": ticker, "shares": shares}])
    updated = new_row if other.empty else pd.concat([other, new_row], ignore_index=True)

    path = _shares_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    updated.to_parquet(path)
    _shares_cache[str(path)] = updated


def _download_shares(ticker: str) -> float | None:
    try:
        shares = yf.Ticker(ticker).fast_info.get("shares")
    except Exception:
        return None
    return float(shares) if shares else None


def load_prices(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    combined = _load_combined(cache_dir)

    cached = _ticker_slice(combined, ticker)
    if cached is not None and _covers_range(cached, start, end):
        return cached.loc[start:end]

    empty_result = pd.DataFrame(
        columns=["Open", "High", "Low", "Close", "Volume"],
        index=pd.DatetimeIndex([], name="Date"),
    )

    known_empty = _load_no_data(cache_dir)
    if cached is None and ticker in known_empty:
        # A ticker permanently unavailable from yfinance (delisted decades
        # ago, typo, never listed) stays unavailable regardless of the date
        # range requested — skip the network round-trip entirely instead of
        # re-attempting it on every single run.
        return empty_result

    fresh = _download(ticker, start, end)
    updated_ticker = fresh if cached is None else _merge(cached, fresh)

    if not updated_ticker.empty:
        _store_ticker(cache_dir, combined, ticker, updated_ticker)
    elif cached is None:
        _store_no_data(cache_dir, known_empty, ticker)
    return updated_ticker.loc[start:end]


def _no_data_path(cache_dir: Path) -> Path:
    return cache_dir / _NO_DATA_FILENAME


def _load_no_data(cache_dir: Path) -> set[str]:
    path = _no_data_path(cache_dir)
    key = str(path)
    if key not in _no_data_cache:
        if path.exists():
            _no_data_cache[key] = set(pd.read_parquet(path)["ticker"])
        else:
            _no_data_cache[key] = set()
    return _no_data_cache[key]


def _store_no_data(cache_dir: Path, known_empty: set[str], ticker: str) -> None:
    known_empty.add(ticker)
    path = _no_data_path(cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticker": sorted(known_empty)}).to_parquet(path)


def _shard_for(ticker: str) -> int:
    """Which shard a ticker's rows live in.

    Uses `zlib.crc32` rather than the builtin `hash()`, whose string hashing
    is salted per process — a ticker has to land in the same shard on every
    run and on every machine, or the cache silently splits in two.
    """
    return zlib.crc32(ticker.encode()) % _SHARD_COUNT


def _shard_path(cache_dir: Path, shard: int) -> Path:
    return cache_dir / _SHARD_FILENAME.format(shard=shard)


def _load_combined(cache_dir: Path) -> pd.DataFrame:
    """All cached price rows, concatenated across shards.

    The store is split across `_SHARD_COUNT` files rather than kept as one:
    a single combined file grew past GitHub's 100 MB per-file limit once the
    universe covered every historical S&P 500 constituent back to 1990, and
    — more importantly — writing one ticker rewrote the entire file, so each
    commit added a fresh copy of the whole dataset to git history. Sharding
    means a run that touches a few tickers only rewrites the shards those
    tickers live in.
    """
    key = str(cache_dir)
    if key not in _combined_cache:
        frames = [
            pd.read_parquet(path)
            for shard in range(_SHARD_COUNT)
            if (path := _shard_path(cache_dir, shard)).exists()
        ]
        _combined_cache[key] = (
            pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=_COLUMNS)
        )
    return _combined_cache[key]


def _ticker_slice(combined: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    rows = combined[combined["ticker"] == ticker]
    if rows.empty:
        return None
    return rows.drop(columns="ticker").set_index("Date").sort_index()


def _store_ticker(
    cache_dir: Path, combined: pd.DataFrame, ticker: str, ticker_df: pd.DataFrame
) -> None:
    # Keep the canonical OHLCV columns only — yfinance sometimes includes
    # extras (e.g. "Adj Close") that would otherwise leak into the combined
    # store's schema for every ticker via the outer join in pd.concat below.
    to_store = ticker_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    to_store.index.name = "Date"
    to_store = to_store.reset_index()
    to_store.insert(0, "ticker", ticker)
    to_store = _downcast(to_store)

    other_tickers = combined[combined["ticker"] != ticker]
    updated = to_store if other_tickers.empty else pd.concat(
        [other_tickers, to_store], ignore_index=True
    )
    _combined_cache[str(cache_dir)] = updated

    # Only the shard this ticker belongs to needs rewriting.
    shard = _shard_for(ticker)
    shard_rows = updated[updated["ticker"].map(_shard_for) == shard]
    path = _shard_path(cache_dir, shard)
    path.parent.mkdir(parents=True, exist_ok=True)
    shard_rows.to_parquet(path, compression="zstd", index=False)


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Store prices as float32 — the extra float64 precision is well below
    any price's real significance and costs ~35% of the file size."""
    for column in ("Open", "High", "Low", "Close", "Volume"):
        if column in df.columns:
            df[column] = df[column].astype("float32")
    return df


def _covers_range(df: pd.DataFrame, start: str, end: str) -> bool:
    # A ticker whose real trading history starts after `start` (e.g. a recent
    # IPO/spinoff) can never satisfy this, so it gets re-downloaded and
    # re-persisted on every run — a minor, expected source of row reordering
    # / tiny float-precision churn in cache_data/prices.parquet for the
    # handful of tickers this affects. Not worth a "permanently short" cache
    # state for what's a small, cosmetic diff.
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
