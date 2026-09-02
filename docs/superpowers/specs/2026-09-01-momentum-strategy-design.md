# Momentum Investing Strategy — Design

## Purpose

Turn the prototype in `raw/` (a single-file `backtrader` script, duplicated in a
notebook) into a proper, testable research-backtesting package. This is a
research tool for iterating on and evaluating the strategy — not a live or
paper-trading system.

## Source material

`raw/main.py` and `raw/Momentum_Investing.ipynb` implement the same
cross-sectional + dual-momentum stock rotation strategy on ~140 hardcoded
tickers:

- Regime filter: benchmark (SPY) price vs. its 200-day SMA — flat to cash
  market-wide when bearish.
- Stock filter: price above its own 200-day SMA.
- Scoring: mean of 60/120/252-day momentum, blended with a "Frog in the Pan"
  score (fraction of positive-return days) and return skewness, via fixed
  weights.
- Sizing: top 5 scorers, weighted by inverse volatility.
- Rebalance trigger: fires only when top-5 *membership* changes.
- Reporting: hand-rolled Excel output (transaction log, monthly per-stock
  performance, portfolio weights, summary metrics).

## Goals

- Installable package (`src/momentum`) with a CLI entrypoint, replacing the
  top-level script that executes on import.
- Strategy scoring math (`scoring.py`) is pure, framework-agnostic, and unit
  testable without spinning up `backtrader`.
- Backtest parameters (dates, universe, strategy weights, benchmark) are
  driven by a YAML config file, not hardcoded.
- Universe supports both a static ticker list and dynamic resolution (e.g.
  current index constituents).
- Market data is cached locally (parquet) between runs.
- Reporting uses `quantstats` (already a declared dependency, currently
  unused) instead of the custom Excel pipeline.

## Non-goals

- Live or paper trading execution.
- Walk-forward optimization / parameter search tooling.
- Multi-strategy portfolio composition.

## Architecture

```
momentum/
  pyproject.toml, README.md          (existing, keep)
  configs/
    default.yaml                     # sample run config
  src/momentum/
    __init__.py
    config.py                        # RunConfig/StrategyConfig dataclasses + YAML loader
    universe.py                      # static list + dynamic index-constituent resolution
    data.py                          # yfinance fetch + local parquet cache
    scoring.py                       # pure functions: momentum, FIP, skewness, combined score, inverse-vol weights
    indicators.py                    # backtrader Indicator wrappers calling into scoring.py
    strategy.py                      # MomentumStrategy(bt.Strategy)
    backtest.py                      # orchestration: build Cerebro, run, hand off to reporting
    reporting.py                     # quantstats tearsheet generation
    cli.py                           # `momentum run configs/default.yaml`
  tests/
    test_scoring.py                  # unit tests, no backtrader involved
    test_config.py
    test_backtest_smoke.py           # small end-to-end run, few tickers, short range
  .cache/                            # gitignored local data cache
```

## Components

- **config.py** — `RunConfig` (date range, benchmark ticker, universe spec,
  cache dir) and `StrategyConfig` (lookbacks, top_n, vol_lookback,
  skewness_lookback, fip_lookback, ts_mom_lookback, regime_ma_period,
  momentum_weight, fip_weight, skewness_penalty, rebalance_frequency).
  Loads and validates a YAML file into these dataclasses.
- **universe.py** — resolves a ticker list from either a `static` list inline
  in the config or a dynamic source (e.g. `index:sp500`, fetched at runtime).
- **data.py** — fetches OHLCV per ticker via `yfinance` for the requested
  range; checks a local `.cache/<ticker>.parquet` for range coverage first,
  fetching and merging only what's missing/stale.
- **scoring.py** — pure functions, no `backtrader` dependency:
  - `momentum_blend(prices, lookbacks) -> float`
  - `fip_score(returns, period) -> float`
  - `skewness_score(returns) -> float`
  - `combined_score(mom, fip, skew, weights) -> float`
  - `inverse_vol_weights(vols: dict) -> dict` (zero-vol entries excluded)
- **indicators.py** — thin `bt.Indicator` wrappers (`RollingSkewness`,
  `FrogInThePan`) that call into `scoring.py` for the actual math.
- **strategy.py** — `MomentumStrategy(bt.Strategy)`: wires indicators,
  applies the regime/trend filters, calls `scoring.combined_score` and
  `scoring.inverse_vol_weights`, and rebalances on membership change or on
  the configured `rebalance_frequency`, whichever comes first.
- **backtest.py** — builds `Cerebro`, adds the benchmark as `datas[0]` and
  the rest of the universe as `datas[1:]`, attaches the strategy with
  `StrategyConfig` params, runs it, and extracts the `TimeReturn` analyzer's
  daily returns series.
- **reporting.py** — calls `quantstats.reports.html(returns, benchmark=...)`
  to produce a tearsheet.
- **cli.py** — `momentum run <config.yaml>`: loads config, runs the full
  pipeline above, prints summary stats and the tearsheet path.

### Example config

```yaml
benchmark: SPY
start_date: "2022-01-01"
end_date: "2024-12-31"
cache_dir: .cache

universe:
  source: static        # or "index:sp500"
  tickers: [MSFT, AAPL, NVDA, GOOGL, AMZN]

strategy:
  lookbacks: [60, 120, 252]
  top_n: 5
  vol_lookback: 126
  skewness_lookback: 90
  fip_lookback: 252
  ts_mom_lookback: 200
  regime_ma_period: 200
  momentum_weight: 0.5
  fip_weight: 0.5
  skewness_penalty: 0.5
  rebalance_frequency: monthly   # in addition to membership-change rebalances
```

## Data flow

```
CLI (momentum run configs/default.yaml)
  → config.py: parse YAML → RunConfig
  → universe.py: resolve ticker list (static or dynamic)
  → data.py: fetch OHLCV per ticker (+ benchmark), using local cache
  → backtest.py: build Cerebro, add feeds, attach MomentumStrategy, run()
  → strategy.py / indicators.py: per-bar scoring via scoring.py
  → backtest.py: pull TimeReturn analyzer's daily returns
  → reporting.py: quantstats HTML tearsheet, benchmark-relative
  → CLI: print summary + tearsheet path
```

The benchmark ticker (hardcoded `"SPY"` today) becomes a config field, so the
regime filter can point at any benchmark. The Excel reporting pipeline
(`generate_reports`/`save_reports_to_excel` in `raw/main.py`, ~230 lines) is
dropped entirely.

## Strategy logic: carried over vs. changed

Carried over as-is: regime filter, stock-level trend filter, momentum blend
across three lookbacks, FIP score, skewness score, combined weighted score,
inverse-volatility sizing, top-N selection — all with today's defaults,
overridable via config.

Two changes:

1. **Rebalance trigger** — today, rebalancing fires only when top-5
   membership changes, so inverse-vol target weights never refresh while the
   same names stay on top even as their relative volatility drifts. Add a
   periodic rebalance (default monthly, via `rebalance_frequency` in
   `StrategyConfig`) alongside the membership-change trigger; a membership
   change still forces an immediate rebalance outside that schedule.
2. **Warmup guard** — `raw/main.py` line 132 has a commented-out NaN guard
   on `regime_ma` before its warmup completes. Replace the dead comment with
   an explicit guard (`if len(self.market) < regime_ma_period: return`)
   rather than relying implicitly on `addminperiod`.

## Error handling

- Universe resolution failure (e.g. dynamic constituent fetch fails): raise
  and abort — never fall back to an empty/partial universe silently.
- Per-ticker data gaps: log and skip that ticker, proceed with the rest (as
  today).
- Benchmark data missing: hard failure naming the configured benchmark — the
  regime filter cannot function without it.
- Cache staleness: coverage-based, not time-based — `data.py` checks whether
  the cached range covers the requested range and re-fetches only if not.
- Zero/degenerate volatility: excluded from inverse-vol sizing (as today's
  `1/v if v > 0 else 0` guard), preserved in `scoring.inverse_vol_weights`.

## Testing

- `test_scoring.py` — combined-score weighting, FIP ratio on known
  positive/negative-day sequences, skewness on a known return series,
  inverse-vol weight normalization including the zero-volatility case. No
  `backtrader` involved.
- `test_config.py` — YAML loads into the expected dataclass shape; invalid/
  missing fields raise clear errors.
- `test_backtest_smoke.py` — one small end-to-end run (benchmark + 3-5
  tickers, short date range, fixture/cached data, no live network call in
  CI) asserting the run completes, produces a non-empty returns series, and
  writes a tearsheet file. Regression guard for cross-module wiring.

## Open questions

None outstanding — all sections were reviewed and approved during
brainstorming.
