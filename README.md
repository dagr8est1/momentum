# momentum

A momentum investing strategy backtester built on `backtrader`, with
config-driven runs, cached price data, and `quantstats` tearsheet reporting.

📖 [Strategy overview & architecture guide](https://dagr8est1.github.io/momentum/strategy-guide.html)

## Setup

```bash
uv sync
```

Or, with [`just`](https://github.com/casey/just):

```bash
just venv
```

## Running a backtest

```bash
uv run momentum run configs/default.yaml
```

Price data for the configured universe and benchmark is read from the
sharded `cache_data/prices_00..03.parquet` store (checked into the repo)
when it's already covered there; anything missing is downloaded and merged
into the relevant shard, so a fresh clone works offline for any ticker/date
range already fetched by someone else. The strategy then runs and writes an HTML performance
tearsheet to `tearsheet.html`. Pass `--output <path>` to change the
tearsheet location.

## Configuring a run

See `configs/default.yaml` for the full set of options: date range,
benchmark, universe (a static ticker list, or `source: index:sp500` for the
current S&P 500 constituents), and strategy parameters (lookbacks, top_n,
scoring weights, rebalance frequency).

## Running tests

```bash
uv run pytest
```

## Project layout

- `src/momentum/scoring.py` — pure momentum/FIP/skewness/inverse-vol math
- `src/momentum/indicators.py` — backtrader indicator wrappers around scoring.py
- `src/momentum/strategy.py` — the MomentumStrategy backtrader strategy
- `src/momentum/config.py`, `universe.py`, `data.py` — config, universe, and cached data loading
- `src/momentum/backtest.py`, `reporting.py`, `cli.py` — orchestration, reporting, CLI

`raw/` contains the original prototype notebook/script this package was
built from.
