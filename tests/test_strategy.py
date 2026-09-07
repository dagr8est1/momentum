import math

import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from momentum.scoring import downside_deviation
from momentum.strategy import MomentumStrategy


def _make_feed(prices, name, start="2020-01-01"):
    dates = pd.date_range(start, periods=len(prices), freq="B")
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


class _VolatilityRecordingStrategy(MomentumStrategy):
    """Records the volatility indicator's value at every bar, so the final
    (last-bar) value can be checked against a manual reference calculation."""

    def __init__(self):
        super().__init__()
        self.last_volatility = None

    def next(self):
        super().next()
        d = self.stocks[0]
        if len(d) >= self.p.vol_lookback + 1:
            self.last_volatility = self._volatility(d)


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


def test_total_traded_value_accumulates_from_filled_orders():
    n = 300
    market = _uptrend(n, daily_return=0.001)
    stocks = {"UP": _uptrend(n, daily_return=0.004, seed=1)}

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

    assert strategy.getposition(strategy.stocks[0]).size > 0
    assert strategy.total_traded_value > 0


def test_volatility_indicator_uses_downside_deviation_not_full_stdev():
    n = 300
    market = _uptrend(n, daily_return=0.001)
    prices = _uptrend(n, daily_return=0.004, seed=5)
    vol_lookback = 126

    strategy = _run(
        market,
        {"UP": prices},
        strategy_cls=_VolatilityRecordingStrategy,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=vol_lookback,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
    )

    prices_arr = np.array(prices)
    daily_returns = np.diff(prices_arr) / prices_arr[:-1]
    expected = downside_deviation(daily_returns[-vol_lookback:])
    # A full (upside-and-downside) stdev over the same window would be
    # noticeably larger for this steadily-uptrending series, since it also
    # counts the (larger, more frequent) up days — asserting closeness to
    # the downside-only figure rules out a silent regression back to full
    # standard deviation.
    assert strategy.last_volatility == pytest.approx(expected, rel=1e-6)
    full_stdev = np.std(daily_returns[-vol_lookback:])
    assert strategy.last_volatility < full_stdev


def test_cap_weighted_sizing_favors_larger_market_cap():
    n = 300
    market = _uptrend(n, daily_return=0.001)
    # Identical price paths -> identical momentum/volatility/FIP/skew, so
    # any weight difference must come from the market-cap tilt, not scoring.
    prices = _uptrend(n, daily_return=0.004, seed=9)
    stocks = {"BIGCAP": prices, "SMALLCAP": list(prices)}

    strategy = _run(
        market,
        stocks,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=2,
        rebalance_frequency=None,
        sizing_method="cap_weighted",
        shares_outstanding={"BIGCAP": 300.0, "SMALLCAP": 100.0},
    )

    big = next(d for d in strategy.stocks if d._name == "BIGCAP")
    small = next(d for d in strategy.stocks if d._name == "SMALLCAP")
    big_size = strategy.getposition(big).size
    small_size = strategy.getposition(small).size

    assert big_size > 0 and small_size > 0
    # Same price series -> position value ratio == position size ratio;
    # 3x the shares outstanding at an identical price is 3x the market cap.
    assert big_size / small_size == pytest.approx(3.0, rel=0.05)


def test_default_sizing_ignores_shares_outstanding():
    n = 300
    market = _uptrend(n, daily_return=0.001)
    prices = _uptrend(n, daily_return=0.004, seed=9)
    stocks = {"A": prices, "B": list(prices)}

    strategy = _run(
        market,
        stocks,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=2,
        rebalance_frequency=None,
        # sizing_method left at its "inverse_vol" default; shares_outstanding
        # is supplied anyway to prove it's a no-op outside cap_weighted mode.
        shares_outstanding={"A": 300.0, "B": 100.0},
    )

    a = next(d for d in strategy.stocks if d._name == "A")
    b = next(d for d in strategy.stocks if d._name == "B")
    a_size = strategy.getposition(a).size
    b_size = strategy.getposition(b).size

    assert a_size > 0 and b_size > 0
    assert a_size / b_size == pytest.approx(1.0, rel=0.05)


class _NamedPositionTrackingStrategy(MomentumStrategy):
    """Records one named stock's position size after every bar, so a test
    can check when trading actually started rather than only the end state."""

    def __init__(self):
        super().__init__()
        self.tracked_position_history = []

    def next(self):
        super().next()
        tracked = next(d for d in self.stocks if d._name == "OLD")
        self.tracked_position_history.append(self.getposition(tracked).size)


def test_late_starting_stock_does_not_freeze_other_stocks():
    """Regression test for a real bug found via a live 25-year backtest: a
    per-stock bt.Indicator attached to every stock in __init__ makes
    backtrader compute the Strategy's global warmup as the max over every
    indicator's own feed. A stock added late in the backtest - e.g. a real
    spinoff (Veralto/VLTO, listed 2023-10-04) with enough total history to
    pass the per-ticker minimum-bars filter, but starting long after the
    backtest's actual start date - silently froze next() for every OTHER
    stock too, until that one stock's own indicators became ready (which
    landed 2024-10-04, VLTO's exact 253rd trading day - confirmed by
    replaying the real cache). A 3-year backtest never exposed this because
    every included ticker's own history happened to reach back far enough;
    a 25-year one did. The fix computes per-stock values on demand in
    next() instead of via registered indicators, so a late-starting stock
    is excluded by the min_stock_history guard without blocking anyone else.
    """
    n = 1200
    market = _uptrend(n, daily_return=0.001)
    old_stock = _uptrend(n, daily_return=0.004, seed=4)
    # NEW starts 900 bars into the backtest and has 210 bars of its own -
    # enough to eventually clear every lookback in this test (all <= 200),
    # just very late, mirroring VLTO's real shape (enough history overall,
    # but arriving too late for its warmup to matter until near the end).
    new_stock_start = pd.bdate_range("2020-01-01", periods=n)[900]
    new_stock = _uptrend(210, daily_return=0.004, seed=5)

    cerebro = bt.Cerebro()
    cerebro.adddata(_make_feed(market, "MARKET"))
    cerebro.adddata(_make_feed(old_stock, "OLD"))
    cerebro.adddata(_make_feed(new_stock, "NEW", start=new_stock_start))
    cerebro.addstrategy(
        _NamedPositionTrackingStrategy,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
    )
    cerebro.broker.setcash(100_000.0)
    results = cerebro.run()
    strategy = results[0]

    # OLD has qualifying history from bar ~200 onward - centuries (in
    # backtest terms) before NEW's window opens at bar 900. If OLD only
    # ever traded near the very end (or never), NEW silently starved
    # everyone else, exactly as it did in the real 25-year run.
    assert any(size > 0 for size in strategy.tracked_position_history[:400])


def test_calendar_aligned_late_starting_stock_does_not_freeze_other_stocks():
    """Regression test for a second, distinct freeze bug found via a real
    1997-2026 backtest, after the per-stock-indicator freeze above was
    already fixed. With ~700+ feeds of widely varying real lengths (some
    stocks starting decades apart), `next()`'s real trading logic silently
    didn't fire until nearly the end of the 29.5-year run - confirmed by
    direct instrumentation, and isolated (via several ruled-out hypotheses:
    bad ticker data, `runonce` mode, feed count alone, backtest length
    alone) to something about backtrader's handling of many feeds whose OWN
    `len()` differs drastically. `backtest.py`'s `_align_to_calendar`
    reindexes every ticker onto the benchmark's own calendar before adding
    it as a feed, so every feed has identical length regardless of when the
    stock actually started trading. This test reproduces that exact shape -
    not the shorter/differently-lengthed feeds the test above uses - and
    checks the freeze is gone.

    Pre-listing days are backward-filled with the stock's own first real
    price (not left NaN - an earlier version did that, but a NaN close on a
    position backtrader has ever created an entry for, even at zero size,
    poisons its own broker.getvalue() for the whole portfolio via a
    `0 * NaN`). Real listing position is passed in separately via
    `listing_positions` instead, matching what `backtest.py` does.
    """
    from momentum.backtest import _align_to_calendar

    n = 1200
    market_prices = _uptrend(n, daily_return=0.001)
    old_prices = _uptrend(n, daily_return=0.004, seed=4)
    new_real_prices = _uptrend(300, daily_return=0.004, seed=5)

    calendar = pd.bdate_range("2020-01-01", periods=n)
    market_df = pd.DataFrame(
        {"Open": market_prices, "High": market_prices, "Low": market_prices,
         "Close": market_prices, "Volume": [1_000] * n},
        index=calendar,
    )
    old_df = pd.DataFrame(
        {"Open": old_prices, "High": old_prices, "Low": old_prices,
         "Close": old_prices, "Volume": [1_000] * n},
        index=calendar,
    )
    new_raw_df = pd.DataFrame(
        {"Open": new_real_prices, "High": new_real_prices, "Low": new_real_prices,
         "Close": new_real_prices, "Volume": [1_000] * 300},
        index=calendar[900:],
    )
    new_df = _align_to_calendar(new_raw_df, calendar)
    assert len(new_df) == len(market_df) == len(old_df)  # identical length, unlike the test above
    assert not new_df["Close"].isna().any()  # no NaN anywhere - backward-filled instead
    assert (new_df["Close"].iloc[:900] == new_real_prices[0]).all()  # pre-listing = first real price

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=market_df, name="MARKET"))
    cerebro.adddata(bt.feeds.PandasData(dataname=old_df, name="OLD"))
    cerebro.adddata(bt.feeds.PandasData(dataname=new_df, name="NEW"))
    cerebro.addstrategy(
        _NamedPositionTrackingStrategy,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
        listing_positions={"NEW": 900},
    )
    cerebro.broker.setcash(100_000.0)
    results = cerebro.run()
    strategy = results[0]

    assert any(size > 0 for size in strategy.tracked_position_history[:400])


def test_bearish_regime_does_not_poison_value_with_unlisted_stock():
    """Regression test for a real bug found via a 1997-2026 backtest:
    backtrader's own BackBroker._get_value() sums `position.size *
    data.close[0]` over EVERY data feed it has ever created a position
    entry for (a defaultdict, auto-vivified by any getposition() call) -
    not just currently-held ones. `0 * NaN` is NaN in IEEE 754, so the
    regime-bearish liquidation loop (which calls getposition() on every
    stock regardless of eligibility) permanently NaN'd the whole
    portfolio's value the first time it ran while any not-yet-listed,
    NaN-padded stock existed among the loaded feeds - silently vanishing
    ~26 years of returns in the real backtest. Fixed by backward-filling
    pre-listing days (see _align_to_calendar) so no close price is ever
    NaN; eligibility now tracked separately via listing_positions.
    """
    from momentum.backtest import _align_to_calendar

    n_up = 260
    n_crash = 200
    n = n_up + n_crash - 1
    uptrend = _uptrend(n_up, start=200.0, daily_return=0.004, seed=11)
    crash = list(np.linspace(uptrend[-1], 20.0, n_crash))[1:]
    market_prices = uptrend + crash
    old_prices = _uptrend(n, start=100.0, daily_return=0.004, seed=12)
    # NEW hasn't listed yet by the time the crash happens (starts at bar
    # n - 10, i.e. only 10 real bars exist by the end of this test).
    new_real_prices = _uptrend(10, daily_return=0.003, seed=5)

    calendar = pd.bdate_range("2020-01-01", periods=n)
    market_df = pd.DataFrame(
        {"Open": market_prices, "High": market_prices, "Low": market_prices,
         "Close": market_prices, "Volume": [1_000] * n},
        index=calendar,
    )
    old_df = pd.DataFrame(
        {"Open": old_prices, "High": old_prices, "Low": old_prices,
         "Close": old_prices, "Volume": [1_000] * n},
        index=calendar,
    )
    new_raw_df = pd.DataFrame(
        {"Open": new_real_prices, "High": new_real_prices, "Low": new_real_prices,
         "Close": new_real_prices, "Volume": [1_000] * 10},
        index=calendar[-10:],
    )
    new_df = _align_to_calendar(new_raw_df, calendar)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=market_df, name="MARKET"))
    cerebro.adddata(bt.feeds.PandasData(dataname=old_df, name="OLD"))
    cerebro.adddata(bt.feeds.PandasData(dataname=new_df, name="NEW"))
    cerebro.addstrategy(
        _PositionTrackingStrategy,
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
        rebalance_frequency=None,
        listing_positions={"NEW": n - 10},
    )
    cerebro.broker.setcash(100_000.0)
    results = cerebro.run()
    strategy = results[0]

    # The crash triggers regime-bearish liquidation while NEW is still
    # unlisted (NaN-free only because of the backward-fill fix) - this is
    # exactly the moment the real bug corrupted portfolio value forever.
    assert not math.isnan(strategy.broker.getvalue())
    assert max(strategy.position_history) > 0  # OLD was actually bought at some point
    assert strategy.position_history[-1] == 0  # and liquidated by the crash


def test_null_frequency_rebalances_only_once():
    """rebalance_frequency=None has no periodic trigger, and (since the
    membership-change trigger was removed) no other trigger either -- so a
    stable single-candidate scenario should enter its one position and then
    never rebalance again."""
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


def test_membership_exit_does_not_force_rebalance_before_calendar_date():
    """Core behavior of the calendar-only trigger: a held stock dropping out
    of the point-in-time membership mid-month must NOT be liquidated until
    the next scheduled monthly rebalance -- unlike the old dual-trigger
    design, where any membership change fired a rebalance immediately."""
    n = 300
    market = _uptrend(n, daily_return=0.001)
    stocks = {"ONLY": _uptrend(n, daily_return=0.004, seed=4)}
    calendar = pd.bdate_range("2020-01-01", periods=n)

    # ONLY is the sole member from day 0 until day 220 (2020-11-04), when it
    # drops out of the index -- a date deliberately NOT aligned with any
    # monthly rebalance boundary (those fall on 2020-10-07, 2020-11-02, and
    # 2020-12-01 for this scenario).
    membership = pd.Series(
        [frozenset({"ONLY"}), frozenset()],
        index=[calendar[0], calendar[220]],
    )

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
        rebalance_frequency="monthly",
        membership=membership,
    )

    # Position opened during the initial (October) rebalance. position_history
    # is indexed from backtrader's first next() call, which (via
    # addminperiod(regime_ma_period)) is bar 199 of the feed, not bar 0 -- so
    # feed-calendar index k maps to position_history[k - 199].
    assert strategy.position_history[10] > 0
    # Membership drops at day 220, but the next monthly boundary isn't until
    # December 1st (day 239) -- the position must still be held in between.
    assert strategy.position_history[220 - 199] > 0
    # By the end (well past the December rebalance + one bar to fill the
    # close), the now-ineligible position has been liquidated.
    assert strategy.position_history[-1] == 0
    # Exactly three rebalances: initial entry (Oct), monthly resize (Nov,
    # membership hadn't dropped yet), and the exit (Dec) -- the membership
    # drop itself did not create a fourth, out-of-band event.
    assert strategy.rebalance_count == 3


def test_point_in_time_membership_excludes_non_member_stocks():
    n = 300
    market = _uptrend(n, daily_return=0.001)
    stocks = {
        "WINNER": _uptrend(n, daily_return=0.006, seed=2),  # best momentum, never a member
        "MEMBER": _uptrend(n, daily_return=0.003, seed=3),  # weaker momentum, the only member
    }
    membership = pd.Series(
        [frozenset({"MEMBER"})],
        index=[pd.Timestamp("2020-01-01")],
    )

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
        membership=membership,
    )

    winner_data = next(d for d in strategy.stocks if d._name == "WINNER")
    member_data = next(d for d in strategy.stocks if d._name == "MEMBER")
    assert strategy.getposition(winner_data).size == 0
    assert strategy.getposition(member_data).size > 0


def test_quarterly_rebalance_fires_less_often_than_monthly():
    """Same membership-stable scenario, only the calendar trigger differs.
    Quarterly buckets three months per rebalance instead of one, so it must
    fire strictly less often over the same window."""
    n = 400
    market = _uptrend(n, daily_return=0.001)
    stocks = {"UP": _uptrend(n, daily_return=0.004, seed=3)}

    kwargs = dict(
        regime_ma_period=200,
        ts_mom_lookback=200,
        fip_lookback=200,
        lookbacks=[60, 120, 200],
        vol_lookback=126,
        skewness_lookback=90,
        top_n=1,
    )
    monthly = _run(market, stocks, rebalance_frequency="monthly", **kwargs)
    quarterly = _run(market, stocks, rebalance_frequency="quarterly", **kwargs)

    assert quarterly.rebalance_count < monthly.rebalance_count
    assert quarterly.rebalance_count >= 1  # still fires at least the initial rebalance
