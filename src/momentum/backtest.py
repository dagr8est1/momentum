import logging
from dataclasses import asdict
from pathlib import Path

import backtrader as bt
import pandas as pd

from momentum.config import RunConfig
from momentum.data import load_prices
from momentum.strategy import MomentumStrategy
from momentum.universe import resolve_universe

logger = logging.getLogger(__name__)


def run_backtest(config: RunConfig) -> tuple[pd.Series, pd.Series]:
    cache_dir = Path(config.cache_dir)

    cerebro = bt.Cerebro()

    benchmark_df = load_prices(config.benchmark, config.start_date, config.end_date, cache_dir)
    if benchmark_df.empty:
        raise ValueError(f"No data available for benchmark '{config.benchmark}'")
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
