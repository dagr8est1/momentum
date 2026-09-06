import logging
from dataclasses import asdict
from pathlib import Path

import backtrader as bt
import pandas as pd

from momentum.config import RunConfig, StrategyConfig
from momentum.data import load_prices, load_shares_outstanding
from momentum.strategy import MomentumStrategy
from momentum.universe import resolve_universe

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
        cerebro.adddata(bt.feeds.PandasData(dataname=df, name=ticker))
        added_tickers.append(ticker)

    shares_outstanding = {}
    if config.strategy.sizing_method == "cap_weighted":
        shares_outstanding = {
            t: load_shares_outstanding(t, cache_dir) for t in added_tickers
        }

    cerebro.addstrategy(
        MomentumStrategy, shares_outstanding=shares_outstanding, **asdict(config.strategy)
    )
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.broker.setcash(_STARTING_CASH)

    results = cerebro.run()
    strategy = results[0]

    portfolio_returns = pd.Series(strategy.analyzers.timereturn.get_analysis())
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    benchmark_returns = benchmark_df["Close"].pct_change().dropna()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

    stats = {
        "turnover": _annualized_turnover(
            strategy.total_traded_value, portfolio_returns, _STARTING_CASH
        ),
    }

    return portfolio_returns, benchmark_returns, stats
