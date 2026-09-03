import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from momentum.indicators import DownsideDeviation, FrogInThePan, RollingSkewness
from momentum.scoring import downside_deviation, fip_score, skewness_score


def _make_feed(prices, name="TEST"):
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


class _CaptureStrategy(bt.Strategy):
    params = dict(period=None, indicator_cls=None, line_name=None)

    def __init__(self):
        self.indicator = self.p.indicator_cls(self.data, period=self.p.period)
        self.captured = []

    def next(self):
        self.captured.append(getattr(self.indicator.lines, self.p.line_name)[0])


def _run(prices, indicator_cls, period, line_name):
    cerebro = bt.Cerebro()
    cerebro.adddata(_make_feed(prices))
    cerebro.addstrategy(
        _CaptureStrategy, period=period, indicator_cls=indicator_cls, line_name=line_name
    )
    results = cerebro.run()
    return results[0].captured


def test_rolling_skewness_matches_scoring_function():
    rng = np.random.default_rng(42)
    prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=120))))
    period = 90
    captured = _run(prices, RollingSkewness, period, "skewness")

    expected_last = skewness_score(prices[-period:])
    assert captured[-1] == pytest.approx(expected_last)


def test_frog_in_the_pan_matches_scoring_function():
    rng = np.random.default_rng(7)
    prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=260))))
    period = 252
    captured = _run(prices, FrogInThePan, period, "fip_score")

    window = prices[-(period + 1):]
    returns = np.diff(window) / np.asarray(window[:-1])
    expected_last = fip_score(returns)
    assert captured[-1] == pytest.approx(expected_last)


def test_downside_deviation_indicator_matches_scoring_function():
    rng = np.random.default_rng(3)
    prices = list(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=150))))
    period = 126
    captured = _run(prices, DownsideDeviation, period, "downside_dev")

    window = prices[-(period + 1):]
    returns = np.diff(window) / np.asarray(window[:-1])
    expected_last = downside_deviation(returns)
    assert captured[-1] == pytest.approx(expected_last)
