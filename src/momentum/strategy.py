import backtrader as bt
import numpy as np
import pandas as pd

from momentum.scoring import (
    cap_weighted_inverse_vol_weights,
    combined_score,
    downside_deviation,
    fip_score,
    inverse_vol_weights,
    momentum_blend,
    skewness_score,
)


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
        sizing_method="inverse_vol",
        shares_outstanding={},
        membership=None,
        listing_positions={},
    )

    def __init__(self):
        self.market = self.datas[0]
        self.stocks = self.datas[1:]
        self._max_momentum_lookback = max(self.p.lookbacks)

        self.regime_ma = bt.indicators.SimpleMovingAverage(
            self.market.close, period=self.p.regime_ma_period
        )

        self.last_rebalance_date = None
        self.rebalance_count = 0
        self.total_traded_value = 0.0
        self.value_history = []

        # Deliberately NOT attaching per-stock bt.Indicator objects (as an
        # earlier version did) — backtrader computes a Strategy's warmup
        # period as the max over every registered indicator, evaluated on
        # each indicator's own data feed. One recently-listed stock with a
        # short history (e.g. a spinoff with barely enough total bars to
        # pass the min-history filter in backtest.py) would then silently
        # freeze next() for every stock until that single feed's own
        # indicators became ready — which can be almost the entire backtest
        # window if the universe spans decades. Per-stock values are instead
        # computed on demand in next() from raw .get() windows (same pattern
        # _momentum_score already used safely), gated by the min_stock_history
        # check below rather than backtrader's global minperiod machinery.
        self.addminperiod(self.p.regime_ma_period)

    def notify_order(self, order):
        if order.status == order.Completed:
            self.total_traded_value += abs(order.executed.value)

    def _momentum_score(self, d):
        window = np.array(d.close.get(size=self._max_momentum_lookback + 1))
        return momentum_blend(window, self.p.lookbacks)

    def _fip_score(self, d):
        window = np.array(d.close.get(size=self.p.fip_lookback + 1))
        returns = np.diff(window) / window[:-1]
        return fip_score(returns)

    def _skewness_score(self, d):
        window = np.array(d.close.get(size=self.p.skewness_lookback))
        return skewness_score(window)

    def _volatility(self, d):
        window = np.array(d.close.get(size=self.p.vol_lookback + 1))
        returns = np.diff(window) / window[:-1]
        return downside_deviation(returns)

    def _trend_sma(self, d):
        return float(np.mean(d.close.get(size=self.p.ts_mom_lookback)))

    def _current_members(self, current_date):
        """Tickers eligible to be traded as of `current_date`, or None if no
        point-in-time membership data was supplied (i.e. every loaded feed is
        eligible, as with a static/current-list universe)."""
        if self.p.membership is None:
            return None
        ts = pd.Timestamp(current_date)
        pos = self.p.membership.index.searchsorted(ts, side="right") - 1
        if pos < 0:
            return frozenset()
        return self.p.membership.iloc[pos]

    def _has_full_history(self, d, lookback):
        """Whether `d` has `lookback` bars of REAL (non-placeholder) history.

        Data feeds are pre-aligned onto the benchmark's own trading-day
        calendar before being added to Cerebro (see `backtest.py`), so every
        feed reports the same `len(d)` from day one regardless of when the
        stock actually started trading — `len(d)` alone can no longer detect
        "not listed yet". Pre-listing days are backward-filled with the
        stock's own first real price rather than left NaN (a NaN close on a
        never-held position was found to poison backtrader's own
        broker.getvalue() for the whole portfolio via a `0 * NaN` in its
        valuation loop — see `_align_to_calendar`'s docstring), so real
        listing position is tracked separately via `listing_positions`
        instead. Defaults to 0 when not provided (e.g. in tests that build
        short feeds directly, without this pre-alignment), reducing to a
        plain `len(d) >= lookback` check.
        """
        listing_position = self.p.listing_positions.get(d._name, 0)
        return (len(d) - listing_position) >= lookback

    def _is_rebalance_due(self, current_date):
        # The very first rebalance always fires, regardless of frequency --
        # otherwise a portfolio never enters a single position under
        # rebalance_frequency=None, now that membership changes no longer
        # trigger a rebalance on their own (see next()).
        if self.last_rebalance_date is None:
            return True
        if self.p.rebalance_frequency is None:
            return False
        if self.p.rebalance_frequency == "monthly":
            return (current_date.year, current_date.month) != (
                self.last_rebalance_date.year,
                self.last_rebalance_date.month,
            )
        if self.p.rebalance_frequency == "quarterly":
            return (current_date.year, (current_date.month - 1) // 3) != (
                self.last_rebalance_date.year,
                (self.last_rebalance_date.month - 1) // 3,
            )
        raise ValueError(f"Unsupported rebalance_frequency: {self.p.rebalance_frequency}")

    def next(self):
        current_date = self.datetime.date()
        # Recorded unconditionally, before any early return below, so this
        # is the source of truth for portfolio returns instead of
        # bt.analyzers.TimeReturn (which computes from the same
        # broker.getvalue(), so it isn't actually a more independent
        # source — but tracking it ourselves made the getvalue()-poisoning
        # bug below visible and debuggable).
        self.value_history.append((current_date, self.broker.getvalue()))

        if len(self.market) < self.p.regime_ma_period:
            return

        if self.market.close[0] < self.regime_ma[0]:
            for d in self.stocks:
                if self.getposition(d).size:
                    self.close(data=d)
            return

        if not self._is_rebalance_due(current_date):
            return

        min_stock_history = max(
            self.p.regime_ma_period,
            self.p.fip_lookback + 1,
            self.p.vol_lookback + 1,
            self.p.skewness_lookback,
            self.p.ts_mom_lookback,
            self._max_momentum_lookback + 1,
        )

        members = self._current_members(current_date)

        scores = []
        for d in self.stocks:
            if members is not None and d._name not in members:
                continue
            if not self._has_full_history(d, min_stock_history):
                continue
            if d.close[0] <= self._trend_sma(d):
                continue

            mom = self._momentum_score(d)
            if mom <= 0:
                continue

            fip = self._fip_score(d)
            skewness = self._skewness_score(d)
            score = combined_score(
                mom, fip, skewness,
                self.p.momentum_weight, self.p.fip_weight, self.p.skewness_penalty,
            )
            scores.append({"data": d, "score": score})

        scores.sort(key=lambda x: x["score"], reverse=True)
        new_top = [x["data"] for x in scores[: self.p.top_n]]

        current_positions = [
            d for d, pos in self.broker.positions.items()
            if pos.size != 0 and d in self.stocks
        ]
        if not new_top and not current_positions:
            # Nothing eligible and nothing held -- a no-op, not a real
            # rebalance. Leaving last_rebalance_date untouched means
            # rebalance_frequency=None keeps checking every day until a
            # candidate actually appears, instead of permanently consuming
            # its one allowed rebalance on an empty day.
            return

        self.last_rebalance_date = current_date
        self.rebalance_count += 1

        for d in current_positions:
            if d not in new_top:
                self.close(data=d)

        if not new_top:
            return

        vols = {d._name: self._volatility(d) for d in new_top}
        if self.p.sizing_method == "cap_weighted":
            market_caps = {
                d._name: self.p.shares_outstanding.get(d._name, 0.0) * d.close[0]
                for d in new_top
            }
            weights = cap_weighted_inverse_vol_weights(market_caps, vols)
        else:
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
