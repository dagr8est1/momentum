import backtrader as bt
import numpy as np

from momentum.scoring import downside_deviation, fip_score, skewness_score


class RollingSkewness(bt.Indicator):
    lines = ("skewness",)
    params = (("period", 90),)

    def __init__(self):
        self.addminperiod(self.p.period)

    def next(self):
        window = np.array(self.data.get(size=self.p.period))
        self.lines.skewness[0] = skewness_score(window)


class FrogInThePan(bt.Indicator):
    lines = ("fip_score",)
    params = (("period", 252),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        window = np.array(self.data.get(size=self.p.period + 1))
        returns = np.diff(window) / window[:-1]
        self.lines.fip_score[0] = fip_score(returns)


class DownsideDeviation(bt.Indicator):
    lines = ("downside_dev",)
    params = (("period", 126),)

    def __init__(self):
        self.addminperiod(self.p.period + 1)

    def next(self):
        window = np.array(self.data.get(size=self.p.period + 1))
        returns = np.diff(window) / window[:-1]
        self.lines.downside_dev[0] = downside_deviation(returns)
