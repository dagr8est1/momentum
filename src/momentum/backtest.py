import logging
from dataclasses import asdict
from pathlib import Path

import backtrader as bt
import pandas as pd

from momentum.config import RunConfig, StrategyConfig
from momentum.data import load_prices
from momentum.strategy import MomentumStrategy
from momentum.universe import resolve_universe

logger = logging.getLogger(__name__)


def _min_required_bars(strategy: StrategyConfig) -> int:
    """Fewest trading days a data feed needs for every configured indicator.

    Must match the largest `addminperiod` declared across `indicators.py`/
    `strategy.py` (momentum blend, volatility, skewness, FIP, trend SMA,
    regime SMA) — backtrader precomputes every indicator for the feed's
    full history before the strategy's own per-bar checks ever run, so a
    feed shorter than this crashes with an opaque IndexError instead of
    the clear error/skip below.
    """
    return max(
        max(strategy.lookbacks) + 1,
        strategy.vol_lookback,
        strategy.skewness_lookback,
        strategy.fip_lookback + 1,
        strategy.ts_mom_lookback,
        strategy.regime_ma_period,
    )


def run_backtest(config: RunConfig) -> tuple[pd.Series, pd.Series]:
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

    cerebro.addstrategy(MomentumStrategy, **asdict(config.strategy))
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timereturn")
    cerebro.broker.setcash(100_000.0)

    results = cerebro.run()
    strategy = results[0]

    portfolio_returns = pd.Series(strategy.analyzers.timereturn.get_analysis())
    portfolio_returns.index = pd.to_datetime(portfolio_returns.index)

    benchmark_returns = benchmark_df["Close"].pct_change().dropna()
    benchmark_returns.index = pd.to_datetime(benchmark_returns.index)

    return portfolio_returns, benchmark_returns
