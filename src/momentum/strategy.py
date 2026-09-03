import backtrader as bt
import numpy as np

from momentum.indicators import FrogInThePan, RollingSkewness
from momentum.scoring import combined_score, inverse_vol_weights, momentum_blend


class MomentumStrategy(bt.Strategy):
    params = dict(
        lookbacks=[60, 120, 252],
        top_n=5,
        vol_lookback=126,
        skewness_lookback=90,
        fip_lookback=252,
        ts_mom_lookback=200,
        regime_ma_period=200,
        momentum_weight=0.5,
        fip_weight=0.5,
        skewness_penalty=0.5,
        rebalance_frequency="monthly",
    )

    def __init__(self):
        self.market = self.datas[0]
        self.stocks = self.datas[1:]
        self._max_momentum_lookback = max(self.p.lookbacks)

        self.regime_ma = bt.indicators.SimpleMovingAverage(
            self.market.close, period=self.p.regime_ma_period
        )

        self.indicators = {}
        for d in self.stocks:
            self.indicators[d._name] = {
                "volatility": bt.indicators.StdDev(d.close, period=self.p.vol_lookback),
                "skewness": RollingSkewness(d, period=self.p.skewness_lookback),
                "fip": FrogInThePan(d, period=self.p.fip_lookback),
                "ts_mom": bt.indicators.SimpleMovingAverage(
                    d.close, period=self.p.ts_mom_lookback
                ),
            }

        self.last_rebalanced_stocks = []
        self.last_rebalance_date = None
        self.rebalance_count = 0
        self.total_traded_value = 0.0

        max_overall_lookback = max(
            self.p.regime_ma_period,
            self._max_momentum_lookback,
            self.p.vol_lookback,
            self.p.skewness_lookback,
            self.p.fip_lookback,
            self.p.ts_mom_lookback,
        )
        self.addminperiod(max_overall_lookback)

    def notify_order(self, order):
        if order.status == order.Completed:
            self.total_traded_value += abs(order.executed.value)

    def _momentum_score(self, d):
        window = np.array(d.close.get(size=self._max_momentum_lookback + 1))
        return momentum_blend(window, self.p.lookbacks)

    def _is_rebalance_due(self, current_date):
        if self.p.rebalance_frequency is None:
            return False
        if self.last_rebalance_date is None:
            return True
        if self.p.rebalance_frequency == "monthly":
            return (current_date.year, current_date.month) != (
                self.last_rebalance_date.year,
                self.last_rebalance_date.month,
            )
        raise ValueError(f"Unsupported rebalance_frequency: {self.p.rebalance_frequency}")

    def next(self):
        current_date = self.datetime.date()

        if len(self.market) < self.p.regime_ma_period:
            return

        if self.market.close[0] < self.regime_ma[0]:
            for d in self.stocks:
                if self.getposition(d).size:
                    self.close(data=d)
            self.last_rebalanced_stocks = []
            return

        min_stock_history = max(
            self.p.regime_ma_period,
            self.p.fip_lookback,
            self.p.ts_mom_lookback,
            self._max_momentum_lookback + 1,
        )

        scores = []
        for d in self.stocks:
            if len(d) < min_stock_history:
                continue
            if d.close[0] <= self.indicators[d._name]["ts_mom"][0]:
                continue

            mom = self._momentum_score(d)
            if mom <= 0:
                continue

            fip = self.indicators[d._name]["fip"].fip_score[0]
            skewness = self.indicators[d._name]["skewness"].skewness[0]
            score = combined_score(
                mom, fip, skewness,
                self.p.momentum_weight, self.p.fip_weight, self.p.skewness_penalty,
            )
            scores.append({"data": d, "score": score})

        scores.sort(key=lambda x: x["score"], reverse=True)
        new_top = [x["data"] for x in scores[: self.p.top_n]]

        membership_changed = sorted(d._name for d in new_top) != sorted(
            d._name for d in self.last_rebalanced_stocks
        )
        if not membership_changed and not self._is_rebalance_due(current_date):
            return

        self.last_rebalanced_stocks = new_top
        self.last_rebalance_date = current_date
        self.rebalance_count += 1

        current_positions = [
            d for d, pos in self.broker.positions.items()
            if pos.size != 0 and d in self.stocks
        ]
        for d in current_positions:
            if d not in new_top:
                self.close(data=d)

        if not new_top:
            return

        vols = {d._name: self.indicators[d._name]["volatility"][0] for d in new_top}
        weights = inverse_vol_weights(vols)
        # A small safety buffer is subtracted from total portfolio value before
        # sizing: order_target_value sizes against today's close, but the
        # default broker fills market orders at the *next* bar's open. For a
        # fast-appreciating holding sized at (close to) 100% of portfolio
        # value, that one-bar price drift can push the real fill cost above
        # available cash, causing a spurious margin rejection that silently
        # leaves the position at zero. Reserving 1% of value absorbs that
        # drift without materially changing the inverse-vol weighting.
        total_value = self.broker.getvalue() * 0.99
        for d in new_top:
            target_value = total_value * weights[d._name]
            self.order_target_value(data=d, target=target_value)
