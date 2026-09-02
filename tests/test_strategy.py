import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from momentum.strategy import MomentumStrategy


def _make_feed(prices, name):
    dates = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    df = pd.DataFrame(
        {
            "Open": prices,
            "High": prices,
            "Low": prices,
            "Close": prices,
            "Volume": [1_000] * len(prices),
        },
        index=dates,
    )
    return bt.feeds.PandasData(dataname=df, name=name)


def _uptrend(n, start=100.0, daily_return=0.003, seed=0):
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, 0.001, size=n)
    return list(start * np.exp(np.cumsum(np.full(n, daily_return) + noise)))


def _flat(n, level=100.0):
    return [level] * n


def _run(market_prices, stock_prices_by_name, strategy_cls=MomentumStrategy, **strategy_params):
    cerebro = bt.Cerebro()
    cerebro.adddata(_make_feed(market_prices, "MARKET"))
    for name, prices in stock_prices_by_name.items():
        cerebro.adddata(_make_feed(prices, name))
    cerebro.addstrategy(strategy_cls, **strategy_params)
    cerebro.broker.setcash(100_000.0)
    results = cerebro.run()
    return results[0]


class _PositionTrackingStrategy(MomentumStrategy):
    """MomentumStrategy subclass that records total stock position size
    after every bar, so tests can inspect pre-liquidation state instead of
    only the final state."""

    def __init__(self):
        super().__init__()
        self.position_history = []

    def next(self):
        super().next()
        self.position_history.append(
            sum(self.getposition(d).size for d in self.stocks)
        )


def test_bearish_regime_liquidates_all_positions():
    # Market rises for long enough to clear the 200-day regime SMA warmup
    # (and get the stock scored/selected/bought), then crashes hard so the
    # regime flips bearish and the liquidation branch must fire.
    n_up = 260
    n_crash = 200
    uptrend = _uptrend(n_up, start=200.0, daily_return=0.004, seed=11)
    crash = list(np.linspace(uptrend[-1], 20.0, n_crash))[1:]
    market = uptrend + crash
    n = len(market)

    stocks = {"UP": _uptrend(n, start=100.0, daily_return=0.004, seed=12)}

    strategy = _run(
        market,
        stocks,
        strategy_cls=_PositionTrackingStrategy,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
    )

    # A position was actually opened at some point during the bullish phase...
    assert max(strategy.position_history) > 0
    # ...and the bearish-regime liquidation branch closed it out by the end.
    assert strategy.position_history[-1] == 0
    for data in strategy.stocks:
        assert strategy.getposition(data).size == 0


def test_bullish_regime_selects_uptrending_stock_over_flat_one():
    n = 300
    market = _uptrend(n, daily_return=0.001)
    stocks = {
        "UP": _uptrend(n, daily_return=0.004, seed=1),
        "FLAT": _flat(n),
    }

    strategy = _run(
        market,
        stocks,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
    )

    up_data = next(d for d in strategy.stocks if d._name == "UP")
    flat_data = next(d for d in strategy.stocks if d._name == "FLAT")
    assert strategy.getposition(up_data).size > 0
    assert strategy.getposition(flat_data).size == 0


def test_periodic_rebalance_fires_without_membership_change():
    n = 400
    market = _uptrend(n, daily_return=0.001)
    stocks = {"UP": _uptrend(n, daily_return=0.004, seed=2)}

    strategy = _run(
        market,
        stocks,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency="monthly",
    )

    # Same single stock is the only candidate the whole time (no membership
    # change possible), but monthly rebalances should still have occurred.
    assert strategy.rebalance_count > 1


def test_membership_only_rebalance_does_not_repeat_monthly():
    n = 400
    market = _uptrend(n, daily_return=0.001)
    stocks = {"UP": _uptrend(n, daily_return=0.004, seed=3)}

    strategy = _run(
        market,
        stocks,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
    )

    assert strategy.rebalance_count == 1
