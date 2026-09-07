import logging
from dataclasses import asdict
from pathlib import Path

import backtrader as bt
import pandas as pd

from momentum.config import RunConfig, StrategyConfig
from momentum.data import load_prices, load_shares_outstanding
from momentum.strategy import MomentumStrategy
from momentum.universe import load_point_in_time_membership, resolve_universe

logger = logging.getLogger(__name__)

_STARTING_CASH = 100_000.0


def _min_required_bars(strategy: StrategyConfig) -> int:
    """Fewest trading days a data feed needs to ever be scoreable.

    Must match `MomentumStrategy`'s own `min_stock_history` guard in
    `next()` (momentum blend, volatility, skewness, FIP, trend SMA, regime
    SMA). A ticker with fewer bars than this can never pass that guard, so
    it's skipped here before ever being added to `backtrader` — cheaper
    than adding a feed that would sit idle for the entire backtest, and
    (for the benchmark) this is what actually enforces the date-range
    validation, since MomentumStrategy no longer registers per-stock
    indicators that would otherwise make backtrader enforce it implicitly.
    """
    return max(
        max(strategy.lookbacks) + 1,
        strategy.vol_lookback + 1,
        strategy.skewness_lookback,
        strategy.fip_lookback + 1,
        strategy.ts_mom_lookback,
        strategy.regime_ma_period,
    )


def _align_to_calendar(df: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Reindex a ticker's price history onto the benchmark's own trading-day
    calendar, filling every gap, so every feed added to Cerebro has
    identical length regardless of when the stock actually started trading.

    Root cause this works around: with each feed's own (differing) row count
    driving its `len()` in backtrader, a long backtest with hundreds of
    feeds of wildly different lengths caused the whole strategy's `next()`
    to silently not fire its real logic until nearly the end of the run —
    confirmed via direct instrumentation and isolated from several other
    candidate causes (bad ticker data, `runonce` mode, feed count alone,
    backtest length alone). Making every feed the same length sidesteps
    whatever exact internal mechanism caused that, without needing to fully
    reverse-engineer it.

    Both forward AND backward filled — no NaN survives anywhere in the
    result. An earlier version left pre-listing days as NaN and used that
    to detect "not listed yet", which seemed reasonable but silently broke
    backtrader's own portfolio valuation: `BackBroker._get_value()` sums
    `position.size * data.close[0]` over EVERY data feed it has ever seen a
    position object created for (a defaultdict, auto-vivified by any
    `getposition()` call) — not just currently-held ones. `0 * NaN` is NaN
    in IEEE 754, so the very first time any not-yet-listed, NaN-padded
    ticker got touched (e.g. the regime-bearish liquidation loop, which
    checks every stock regardless of membership), the WHOLE portfolio's
    value went permanently NaN until every loaded ticker had real data —
    confirmed via a real 1997-2026 backtest where this made ~26 years of
    returns silently vanish. Real listing dates are tracked separately (see
    `run_backtest`'s `listing_positions`) so eligibility no longer needs
    NaN-detection at all.
    """
    return df.reindex(calendar).ffill().bfill()


def _annualized_turnover(
    total_traded_value: float, portfolio_returns: pd.Series, starting_cash: float
) -> float:
    """Total traded (buy+sell) value, per year of backtest, as a multiple of
    average portfolio value — e.g. 3.0 means the book turned over 3x/year."""
    if portfolio_returns.empty:
        return 0.0
    portfolio_values = starting_cash * (1 + portfolio_returns).cumprod()
    avg_value = portfolio_values.mean()
    years = (portfolio_returns.index.max() - portfolio_returns.index.min()).days / 365.25
    if avg_value <= 0 or years <= 0:
        return 0.0
    return total_traded_value / avg_value / years


def run_backtest(config: RunConfig) -> tuple[pd.Series, pd.Series, dict]:
    cache_dir = Path(config.cache_dir)
    min_bars = _min_required_bars(config.strategy)

    cerebro = bt.Cerebro()

    benchmark_df = load_prices(config.benchmark, config.start_date, config.end_date, cache_dir)
    if benchmark_df.empty:
        raise ValueError(f"No data available for benchmark '{config.benchmark}'")
    if len(benchmark_df) < min_bars:
        raise ValueError(
            f"Benchmark '{config.benchmark}' has only {len(benchmark_df)} trading days "
            f"between {config.start_date} and {config.end_date}; need at least {min_bars} "
            "for the configured strategy lookbacks. Widen the date range or lower the "
            "lookback parameters."
        )
    cerebro.adddata(bt.feeds.PandasData(dataname=benchmark_df, name=config.benchmark))

    tickers = resolve_universe(config.universe)
    added_tickers = []
    listing_positions = {}
    for ticker in tickers:
        if ticker == config.benchmark:
            continue
        df = load_prices(ticker, config.start_date, config.end_date, cache_dir)
        if df.empty:
            logger.warning(
                "No data available for ticker '%s'; skipping and continuing with the "
                "rest of the universe.",
                ticker,
            )
            continue
        if len(df) < min_bars:
            logger.warning(
                "Ticker '%s' has only %d trading days between %s and %s (need at least "
                "%d for the configured strategy lookbacks); skipping and continuing with "
                "the rest of the universe.",
                ticker,
                len(df),
                config.start_date,
                config.end_date,
                min_bars,
            )
            continue
        listing_positions[ticker] = int(benchmark_df.index.searchsorted(df.index.min()))
        df = _align_to_calendar(df, benchmark_df.index)
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=ticker))
        added_tickers.append(ticker)

    shares_outstanding = {}
    if config.strategy.sizing_method == "cap_weighted":
        shares_outstanding = {
            t: load_shares_outstanding(t, cache_dir) for t in added_tickers
        }

    membership = None
    if config.universe.get("source") == "index:sp500_point_in_time":
        membership = load_point_in_time_membership(config.universe["constituents_file"])

    cerebro.addstrategy(
        MomentumStrategy,
        shares_outstanding=shares_outstanding,
        membership=membership,
        listing_positions=listing_positions,
        **asdict(config.strategy),
    )
    cerebro.broker.setcash(_STARTING_CASH)

    results = cerebro.run()
    strategy = results[0]

    dates, values = zip(*strategy.value_history)
    portfolio_value = pd.Series(values, index=pd.to_datetime(dates)).sort_index()
    portfolio_returns = portfolio_value.pct_change().dropna()

    benchmark_returns = benchmark_df["Close"].pct_change().dropna()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

    stats = {
        "turnover": _annualized_turnover(
            strategy.total_traded_value, portfolio_returns, _STARTING_CASH
        ),
    }

    return portfolio_returns, benchmark_returns, stats
