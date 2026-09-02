import logging

import numpy as np
import pandas as pd
import pytest

from momentum.backtest import run_backtest
from momentum.config import RunConfig, StrategyConfig


def _synthetic_ohlcv(start, periods, daily_return, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.001, size=periods)
    closes = 100 * np.exp(np.cumsum(np.full(periods, daily_return) + noise))
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
        index=dates,
    )


def _seed_cache(cache_dir, ticker, df):
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_dir / f"{ticker}.parquet")


def test_run_backtest_end_to_end_with_cached_data(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    n = 400
    start = "2021-01-04"

    _seed_cache(cache_dir, "BENCH", _synthetic_ohlcv(start, n, 0.001, seed=0))
    _seed_cache(cache_dir, "UP", _synthetic_ohlcv(start, n, 0.004, seed=1))
    _seed_cache(cache_dir, "DOWN", _synthetic_ohlcv(start, n, -0.002, seed=2))

    def _fail_download(ticker, start, end):
        raise AssertionError(f"should not hit network for {ticker}; cache should cover it")

    monkeypatch.setattr("momentum.data._download", _fail_download)

    dates = pd.date_range(start, periods=n, freq="B")
    config = RunConfig(
        benchmark="BENCH",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        universe={"source": "static", "tickers": ["UP", "DOWN"]},
        strategy=StrategyConfig(
            lookbacks=[20, 40, 60],
            top_n=1,
            vol_lookback=30,
            skewness_lookback=30,
            fip_lookback=60,
            ts_mom_lookback=60,
            regime_ma_period=60,
            rebalance_frequency="monthly",
        ),
        cache_dir=str(cache_dir),
    )

    portfolio_returns, benchmark_returns = run_backtest(config)

    assert not portfolio_returns.empty
    assert not benchmark_returns.empty
    assert isinstance(portfolio_returns.index, pd.DatetimeIndex)


def test_run_backtest_raises_when_benchmark_data_missing(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"

    def _empty_download(ticker, start, end):
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([]),
        )

    monkeypatch.setattr("momentum.data._download", _empty_download)

    config = RunConfig(
        benchmark="MISSING",
        start_date="2022-01-01",
        end_date="2022-06-01",
        universe={"source": "static", "tickers": []},
        strategy=StrategyConfig(),
        cache_dir=str(cache_dir),
    )

    with pytest.raises(ValueError, match="MISSING"):
        run_backtest(config)


def test_run_backtest_logs_warning_for_ticker_with_no_data(tmp_path, monkeypatch, caplog):
    cache_dir = tmp_path / "cache"
    n = 400
    start = "2021-01-04"

    _seed_cache(cache_dir, "BENCH", _synthetic_ohlcv(start, n, 0.001, seed=0))
    _seed_cache(cache_dir, "UP", _synthetic_ohlcv(start, n, 0.004, seed=1))

    def _empty_download(ticker, start, end):
        return pd.DataFrame(
            columns=["Open", "High", "Low", "Close", "Volume"],
            index=pd.DatetimeIndex([]),
        )

    monkeypatch.setattr("momentum.data._download", _empty_download)

    dates = pd.date_range(start, periods=n, freq="B")
    config = RunConfig(
        benchmark="BENCH",
        start_date=str(dates[0].date()),
        end_date=str(dates[-1].date()),
        universe={"source": "static", "tickers": ["UP", "MISSING"]},
        strategy=StrategyConfig(
            lookbacks=[20, 40, 60],
            top_n=1,
            vol_lookback=30,
            skewness_lookback=30,
            fip_lookback=60,
            ts_mom_lookback=60,
            regime_ma_period=60,
            rebalance_frequency="monthly",
        ),
        cache_dir=str(cache_dir),
    )

    with caplog.at_level(logging.WARNING, logger="momentum.backtest"):
        run_backtest(config)

    assert "MISSING" in caplog.text
