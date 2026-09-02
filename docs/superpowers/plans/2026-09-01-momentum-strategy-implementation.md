# Momentum Strategy Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn `raw/main.py` (a single-file `backtrader` script) into an installable `momentum` package with a CLI, config-driven runs, a cached data layer, a dynamic/static universe resolver, testable pure scoring math, and `quantstats` reporting.

**Architecture:** `backtrader` remains the execution engine (order/position/commission simulation); the strategy's scoring math (momentum blend, FIP, skewness, combined score, inverse-vol weights) is extracted into plain functions in `scoring.py` that the `backtrader` `Indicator`/`Strategy` classes call into. Config, universe resolution, data caching, backtest orchestration, and reporting are separate modules wired together by a thin CLI.

**Tech Stack:** Python 3.12+, `backtrader`, `pandas`, `numpy`, `scipy`, `yfinance`, `quantstats`, `pyyaml`, `pyarrow` (parquet cache), `requests` + `lxml` (dynamic universe fetch), `pytest`, managed via `uv`.

**Spec:** [docs/superpowers/specs/2026-09-01-momentum-strategy-design.md](../specs/2026-09-01-momentum-strategy-design.md)

## Global Constraints

- Python `>=3.12` (per existing `pyproject.toml`).
- Package lives under `src/momentum/` (src layout); build backend is `hatchling`.
- No live network calls in any test — mock/monkeypatch `yfinance`, `requests`, and cache reads/writes.
- `scoring.py` has zero `backtrader` imports — it must be importable and testable standalone.
- Drop `openpyxl` and the custom Excel reporting pipeline entirely; `quantstats` is the only reporting path.
- All commands run through `uv` (`uv sync`, `uv run pytest`, `uv run momentum ...`).

---

### Task 1: Project scaffolding

**Files:**
- Modify: `pyproject.toml`
- Create: `src/momentum/__init__.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: an importable `momentum` package (empty) and a `momentum` console script pointing at `momentum.cli:main` (created in Task 10 — the entry is declared now, the module arrives later).

- [ ] **Step 1: Update `pyproject.toml`**

Replace the file with:

```toml
[project]
name = "quant"
version = "0.1.0"
description = "Momentum investing strategy backtester"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "backtrader>=1.9.78.123",
    "numpy>=2.3.0",
    "pandas>=2.3.0",
    "pyarrow>=17.0.0",
    "pyyaml>=6.0",
    "quantstats>=0.0.64",
    "requests>=2.32.0",
    "lxml>=5.0.0",
    "scipy>=1.15.3",
    "yfinance>=0.2.63",
]

[dependency-groups]
dev = ["pytest>=8.3.0"]

[project.scripts]
momentum = "momentum.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/momentum"]
```

- [ ] **Step 2: Create the package init**

Create `src/momentum/__init__.py`:

```python
```

(empty file — just marks `momentum` as a package)

- [ ] **Step 3: Create `.gitignore`**

```
.cache/
__pycache__/
*.pyc
.venv/
*.egg-info/
tearsheet.html
```

- [ ] **Step 4: Sync and verify**

Run: `uv sync`
Expected: resolves and installs all dependencies with no errors.

Run: `uv run pytest`
Expected: `no tests ran` (exit code 5) — no test files exist yet, this just confirms `pytest` is wired up.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml uv.lock src/momentum/__init__.py .gitignore
git commit -m "chore: scaffold momentum package with src layout"
```

---

### Task 2: Scoring math (`scoring.py`)

**Files:**
- Create: `src/momentum/scoring.py`
- Test: `tests/test_scoring.py`

**Interfaces:**
- Produces:
  - `momentum_blend(closes: Sequence[float], lookbacks: Sequence[int]) -> float`
  - `fip_score(returns: Sequence[float]) -> float`
  - `skewness_score(closes: Sequence[float]) -> float`
  - `combined_score(momentum: float, fip: float, skewness: float, momentum_weight: float, fip_weight: float, skewness_penalty: float) -> float`
  - `inverse_vol_weights(volatilities: dict[str, float]) -> dict[str, float]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scoring.py`:

```python
import numpy as np
import pytest

from momentum.scoring import (
    combined_score,
    fip_score,
    inverse_vol_weights,
    momentum_blend,
    skewness_score,
)


def test_momentum_blend_averages_price_diffs_across_lookbacks():
    closes = list(range(1, 254))  # 1..253, strictly increasing by 1 each day
    result = momentum_blend(closes, lookbacks=[60, 120, 252])
    # closes[-1] = 253; diffs are 253-193=60, 253-133=120, 253-1=252
    assert result == pytest.approx((60 + 120 + 252) / 3)


def test_fip_score_is_fraction_of_positive_return_days():
    returns = [0.01, -0.02, 0.03, 0.04, -0.01]
    assert fip_score(returns) == pytest.approx(3 / 5)


def test_fip_score_all_negative_is_zero():
    assert fip_score([-0.01, -0.02, -0.03]) == 0.0


def test_skewness_score_zero_for_symmetric_returns():
    # prices whose log returns are symmetric around zero
    closes = [100, 110, 100, 110, 100, 110, 100]
    assert skewness_score(closes) == pytest.approx(0.0, abs=1e-9)


def test_skewness_score_positive_for_right_skewed_returns():
    closes = np.exp(np.cumsum([0.01, 0.01, 0.01, 0.01, 0.5, -0.01, -0.01])).tolist()
    assert skewness_score(closes) > 0


def test_combined_score_applies_weights():
    result = combined_score(
        momentum=10.0,
        fip=0.6,
        skewness=0.2,
        momentum_weight=0.5,
        fip_weight=0.3,
        skewness_penalty=0.1,
    )
    assert result == pytest.approx(0.5 * 10.0 + 0.3 * 0.6 + 0.1 * 0.2)


def test_inverse_vol_weights_favors_lower_volatility():
    weights = inverse_vol_weights({"low_vol": 1.0, "high_vol": 4.0})
    assert weights["low_vol"] > weights["high_vol"]
    assert weights["low_vol"] == pytest.approx(0.8)
    assert weights["high_vol"] == pytest.approx(0.2)


def test_inverse_vol_weights_excludes_zero_volatility():
    weights = inverse_vol_weights({"zero_vol": 0.0, "normal": 2.0})
    assert weights["zero_vol"] == 0.0
    assert weights["normal"] == pytest.approx(1.0)


def test_inverse_vol_weights_all_zero_returns_all_zero():
    weights = inverse_vol_weights({"a": 0.0, "b": 0.0})
    assert weights == {"a": 0.0, "b": 0.0}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.scoring'`

- [ ] **Step 3: Implement `scoring.py`**

Create `src/momentum/scoring.py`:

```python
from typing import Sequence

import numpy as np
from scipy.stats import skew


def momentum_blend(closes: Sequence[float], lookbacks: Sequence[int]) -> float:
    """Mean price difference between the latest close and each lookback's close."""
    closes = np.asarray(closes, dtype=float)
    latest = closes[-1]
    diffs = [latest - closes[-1 - lb] for lb in lookbacks]
    return float(np.mean(diffs))


def fip_score(returns: Sequence[float]) -> float:
    """Fraction of days in the window with a positive return."""
    returns = np.asarray(returns, dtype=float)
    return float(np.sum(returns > 0) / len(returns))


def skewness_score(closes: Sequence[float]) -> float:
    """Skewness of the log returns implied by a window of closing prices."""
    closes = np.asarray(closes, dtype=float)
    log_returns = np.diff(np.log(closes))
    return float(skew(log_returns))


def combined_score(
    momentum: float,
    fip: float,
    skewness: float,
    momentum_weight: float,
    fip_weight: float,
    skewness_penalty: float,
) -> float:
    """Weighted blend of momentum, frog-in-the-pan, and skewness scores."""
    return (
        momentum_weight * momentum
        + fip_weight * fip
        + skewness_penalty * skewness
    )


def inverse_vol_weights(volatilities: dict[str, float]) -> dict[str, float]:
    """Portfolio weights inversely proportional to each name's volatility.

    Names with non-positive volatility get a weight of 0 and are excluded
    from the normalization.
    """
    inv_vols = {name: (1 / vol if vol > 0 else 0.0) for name, vol in volatilities.items()}
    total = sum(inv_vols.values())
    if total <= 0:
        return {name: 0.0 for name in volatilities}
    return {name: inv_vol / total for name, inv_vol in inv_vols.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scoring.py -v`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/scoring.py tests/test_scoring.py
git commit -m "feat: add pure scoring functions for momentum strategy"
```

---

### Task 3: Backtrader indicator wrappers (`indicators.py`)

**Files:**
- Create: `src/momentum/indicators.py`
- Test: `tests/test_indicators.py`

**Interfaces:**
- Consumes: `momentum.scoring.skewness_score`, `momentum.scoring.fip_score` (Task 2)
- Produces:
  - `RollingSkewness(bt.Indicator)` — line `skewness`, param `period` (default 90)
  - `FrogInThePan(bt.Indicator)` — line `fip_score`, param `period` (default 252)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_indicators.py`:

```python
import backtrader as bt
import numpy as np
import pandas as pd
import pytest

from momentum.indicators import FrogInThePan, RollingSkewness
from momentum.scoring import fip_score, skewness_score


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_indicators.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.indicators'`

- [ ] **Step 3: Implement `indicators.py`**

Create `src/momentum/indicators.py`:

```python
import backtrader as bt
import numpy as np

from momentum.scoring import fip_score, skewness_score


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_indicators.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/indicators.py tests/test_indicators.py
git commit -m "feat: add backtrader indicator wrappers around scoring functions"
```

---

### Task 4: Config loading (`config.py`)

**Files:**
- Create: `src/momentum/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `StrategyConfig` dataclass with fields: `lookbacks: list[int]`, `top_n: int`, `vol_lookback: int`, `skewness_lookback: int`, `fip_lookback: int`, `ts_mom_lookback: int`, `regime_ma_period: int`, `momentum_weight: float`, `fip_weight: float`, `skewness_penalty: float`, `rebalance_frequency: str | None` (defaults matching `raw/main.py`, `rebalance_frequency` default `"monthly"`)
  - `RunConfig` dataclass with fields: `benchmark: str`, `start_date: str`, `end_date: str`, `universe: dict`, `strategy: StrategyConfig`, `cache_dir: str` (default `".cache"`)
  - `load_config(path: str | Path) -> RunConfig`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_config.py`:

```python
import pytest

from momentum.config import RunConfig, StrategyConfig, load_config

VALID_YAML = """
benchmark: SPY
start_date: "2022-01-01"
end_date: "2024-12-31"
cache_dir: .cache
universe:
  source: static
  tickers: [MSFT, AAPL]
strategy:
  top_n: 3
  momentum_weight: 0.6
"""

MISSING_FIELD_YAML = """
start_date: "2022-01-01"
end_date: "2024-12-31"
universe:
  source: static
  tickers: [MSFT]
"""

UNKNOWN_STRATEGY_FIELD_YAML = """
benchmark: SPY
start_date: "2022-01-01"
end_date: "2024-12-31"
universe:
  source: static
  tickers: [MSFT]
strategy:
  not_a_real_field: 1
"""


def test_load_config_parses_valid_yaml(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_YAML)

    config = load_config(config_path)

    assert isinstance(config, RunConfig)
    assert config.benchmark == "SPY"
    assert config.start_date == "2022-01-01"
    assert config.universe == {"source": "static", "tickers": ["MSFT", "AAPL"]}
    assert isinstance(config.strategy, StrategyConfig)
    assert config.strategy.top_n == 3
    assert config.strategy.momentum_weight == 0.6
    # untouched fields keep their defaults
    assert config.strategy.fip_weight == 0.5
    assert config.strategy.lookbacks == [60, 120, 252]


def test_load_config_uses_default_cache_dir_when_omitted(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID_YAML.replace("cache_dir: .cache\n", ""))

    config = load_config(config_path)

    assert config.cache_dir == ".cache"


def test_load_config_raises_on_missing_required_field(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(MISSING_FIELD_YAML)

    with pytest.raises(ValueError, match="benchmark"):
        load_config(config_path)


def test_load_config_raises_on_unknown_strategy_field(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(UNKNOWN_STRATEGY_FIELD_YAML)

    with pytest.raises(ValueError, match="not_a_real_field"):
        load_config(config_path)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.config'`

- [ ] **Step 3: Implement `config.py`**

Create `src/momentum/config.py`:

```python
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class StrategyConfig:
    lookbacks: list[int] = field(default_factory=lambda: [60, 120, 252])
    top_n: int = 5
    vol_lookback: int = 126
    skewness_lookback: int = 90
    fip_lookback: int = 252
    ts_mom_lookback: int = 200
    regime_ma_period: int = 200
    momentum_weight: float = 0.5
    fip_weight: float = 0.5
    skewness_penalty: float = 0.5
    rebalance_frequency: str | None = "monthly"


@dataclass
class RunConfig:
    benchmark: str
    start_date: str
    end_date: str
    universe: dict
    strategy: StrategyConfig
    cache_dir: str = ".cache"


_REQUIRED_FIELDS = ["benchmark", "start_date", "end_date", "universe"]


def load_config(path: str | Path) -> RunConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)

    missing = [key for key in _REQUIRED_FIELDS if key not in raw]
    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    strategy_raw = raw.get("strategy", {})
    try:
        strategy = StrategyConfig(**strategy_raw)
    except TypeError as exc:
        raise ValueError(f"Invalid strategy config: {exc}") from exc

    return RunConfig(
        benchmark=raw["benchmark"],
        start_date=raw["start_date"],
        end_date=raw["end_date"],
        universe=raw["universe"],
        strategy=strategy,
        cache_dir=raw.get("cache_dir", ".cache"),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/config.py tests/test_config.py
git commit -m "feat: add YAML config loading for backtest runs"
```

---

### Task 5: Universe resolution (`universe.py`)

**Files:**
- Create: `src/momentum/universe.py`
- Test: `tests/test_universe.py`

**Interfaces:**
- Produces: `resolve_universe(spec: dict) -> list[str]`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_universe.py`:

```python
import pytest

from momentum import universe


def test_resolve_universe_static_returns_ticker_list():
    result = universe.resolve_universe({"source": "static", "tickers": ["MSFT", "AAPL"]})
    assert result == ["MSFT", "AAPL"]


def test_resolve_universe_unknown_source_raises():
    with pytest.raises(ValueError, match="unknown_source"):
        universe.resolve_universe({"source": "unknown_source"})


def test_resolve_universe_sp500_parses_wikipedia_table(monkeypatch):
    fake_html = """
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th></tr>
      <tr><td>MSFT</td><td>Microsoft</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td></tr>
    </table>
    """

    class _FakeResponse:
        text = fake_html

        def raise_for_status(self):
            pass

    def _fake_get(url, headers=None, timeout=None):
        return _FakeResponse()

    monkeypatch.setattr(universe.requests, "get", _fake_get)

    result = universe.resolve_universe({"source": "index:sp500"})

    assert "MSFT" in result
    assert "BRK-B" in result  # dots normalized to dashes for yfinance compatibility
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_universe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.universe'`

- [ ] **Step 3: Implement `universe.py`**

Create `src/momentum/universe.py`:

```python
import pandas as pd
import requests

_SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def resolve_universe(spec: dict) -> list[str]:
    source = spec.get("source", "static")
    if source == "static":
        return list(spec["tickers"])
    if source == "index:sp500":
        return _fetch_sp500_constituents()
    raise ValueError(f"Unknown universe source: {source}")


def _fetch_sp500_constituents() -> list[str]:
    response = requests.get(
        _SP500_WIKIPEDIA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
    )
    response.raise_for_status()
    table = pd.read_html(response.text, attrs={"id": "constituents"})[0]
    return [ticker.replace(".", "-") for ticker in table["Symbol"].tolist()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_universe.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/universe.py tests/test_universe.py
git commit -m "feat: add static and dynamic S&P 500 universe resolution"
```

---

### Task 6: Data fetching with local cache (`data.py`)

**Files:**
- Create: `src/momentum/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Produces: `load_prices(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame` — a DataFrame with a `DatetimeIndex` and columns `Open, High, Low, Close, Volume`, covering `[start, end]`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_data.py`:

```python
import pandas as pd
import pytest

from momentum import data


def _fake_ohlcv(start, periods):
    dates = pd.date_range(start, periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": range(periods),
            "High": range(periods),
            "Low": range(periods),
            "Close": range(periods),
            "Volume": [1000] * periods,
        },
        index=dates,
    )


def test_load_prices_downloads_and_caches_when_no_cache(tmp_path, monkeypatch):
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((ticker, start, end))
        return _fake_ohlcv("2022-01-03", periods=10)

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(calls) == 1
    assert not result.empty
    assert (tmp_path / "MSFT.parquet").exists()


def test_load_prices_reuses_cache_without_downloading(tmp_path, monkeypatch):
    cached = _fake_ohlcv("2022-01-03", periods=10)
    cached.to_parquet(tmp_path / "MSFT.parquet")

    def _fail_download(ticker, start, end):
        raise AssertionError("should not re-download when cache covers the range")

    monkeypatch.setattr(data, "_download", _fail_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-14", tmp_path)

    assert len(result) == 10


def test_load_prices_extends_cache_when_range_not_covered(tmp_path, monkeypatch):
    cached = _fake_ohlcv("2022-01-03", periods=5)
    cached.to_parquet(tmp_path / "MSFT.parquet")

    fresh = _fake_ohlcv("2022-01-03", periods=15)
    calls = []

    def _fake_download(ticker, start, end):
        calls.append((start, end))
        return fresh

    monkeypatch.setattr(data, "_download", _fake_download)

    result = data.load_prices("MSFT", "2022-01-03", "2022-01-24", tmp_path)

    assert len(calls) == 1
    assert len(result) == 15
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.data'`

- [ ] **Step 3: Implement `data.py`**

Create `src/momentum/data.py`:

```python
from pathlib import Path

import pandas as pd
import yfinance as yf


def load_prices(ticker: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ticker}.parquet"

    cached = pd.read_parquet(cache_path) if cache_path.exists() else None
    if cached is not None and _covers_range(cached, start, end):
        return cached.loc[start:end]

    fresh = _download(ticker, start, end)
    combined = fresh if cached is None else _merge(cached, fresh)

    cache_dir.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path)
    return combined.loc[start:end]


def _covers_range(df: pd.DataFrame, start: str, end: str) -> bool:
    if df.empty:
        return False
    return df.index.min() <= pd.Timestamp(start) and df.index.max() >= pd.Timestamp(end)


def _merge(cached: pd.DataFrame, fresh: pd.DataFrame) -> pd.DataFrame:
    combined = pd.concat([cached, fresh]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def _download(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, progress=False)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: all 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/data.py tests/test_data.py
git commit -m "feat: add cached price data fetching"
```

---

### Task 7: Momentum strategy (`strategy.py`)

**Files:**
- Create: `src/momentum/strategy.py`
- Test: `tests/test_strategy.py`

**Interfaces:**
- Consumes: `momentum.scoring.{momentum_blend, combined_score, inverse_vol_weights}` (Task 2), `momentum.indicators.{RollingSkewness, FrogInThePan}` (Task 3)
- Produces: `MomentumStrategy(bt.Strategy)` with `params` matching `StrategyConfig` field names (`lookbacks, top_n, vol_lookback, skewness_lookback, fip_lookback, ts_mom_lookback, regime_ma_period, momentum_weight, fip_weight, skewness_penalty, rebalance_frequency`)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_strategy.py`:

```python
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


def _run(market_prices, stock_prices_by_name, **strategy_params):
    cerebro = bt.Cerebro()
    cerebro.adddata(_make_feed(market_prices, "MARKET"))
    for name, prices in stock_prices_by_name.items():
        cerebro.adddata(_make_feed(prices, name))
    cerebro.addstrategy(MomentumStrategy, **strategy_params)
    cerebro.broker.setcash(100_000.0)
    results = cerebro.run()
    return results[0]


def test_bearish_regime_liquidates_all_positions():
    n = 300
    # Market trends down hard -> price stays below its 200-day SMA
    market = list(np.linspace(200, 50, n))
    stocks = {"UP": _uptrend(n)}

    strategy = _run(market, stocks, regime_ma_period=200, top_n=1)

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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_strategy.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.strategy'`

- [ ] **Step 3: Implement `strategy.py`**

Create `src/momentum/strategy.py`:

```python
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

        max_overall_lookback = max(
            self.p.regime_ma_period,
            self._max_momentum_lookback,
            self.p.vol_lookback,
            self.p.skewness_lookback,
            self.p.fip_lookback,
            self.p.ts_mom_lookback,
        )
        self.addminperiod(max_overall_lookback)

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
        total_value = self.broker.getvalue()
        for d in new_top:
            target_value = total_value * weights[d._name]
            self.order_target_value(data=d, target=target_value)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_strategy.py -v`
Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/strategy.py tests/test_strategy.py
git commit -m "feat: add MomentumStrategy with regime filter and rebalance scheduling"
```

---

### Task 8: Backtest orchestration + end-to-end smoke test (`backtest.py`)

**Files:**
- Create: `src/momentum/backtest.py`
- Test: `tests/test_backtest_smoke.py`

**Interfaces:**
- Consumes: `momentum.config.RunConfig` (Task 4), `momentum.universe.resolve_universe` (Task 5), `momentum.data.load_prices` (Task 6), `momentum.strategy.MomentumStrategy` (Task 7)
- Produces: `run_backtest(config: RunConfig) -> tuple[pd.Series, pd.Series]` returning `(portfolio_returns, benchmark_returns)`, both `pd.Series` indexed by date. Raises `ValueError` if the configured benchmark has no data.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest_smoke.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest_smoke.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.backtest'`

- [ ] **Step 3: Implement `backtest.py`**

Create `src/momentum/backtest.py`:

```python
from dataclasses import asdict
from pathlib import Path

import backtrader as bt
import pandas as pd

from momentum.config import RunConfig
from momentum.data import load_prices
from momentum.strategy import MomentumStrategy
from momentum.universe import resolve_universe


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest_smoke.py -v`
Expected: both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/backtest.py tests/test_backtest_smoke.py
git commit -m "feat: add backtest orchestration wiring config/universe/data/strategy"
```

---

### Task 9: Reporting (`reporting.py`)

**Files:**
- Create: `src/momentum/reporting.py`
- Test: `tests/test_reporting.py`

**Interfaces:**
- Produces: `generate_tearsheet(returns: pd.Series, benchmark: pd.Series, output_path: str) -> None`

- [ ] **Step 1: Write the failing test**

Create `tests/test_reporting.py`:

```python
import numpy as np
import pandas as pd

from momentum.reporting import generate_tearsheet


def test_generate_tearsheet_writes_html_file(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    returns = pd.Series(rng.normal(0.0005, 0.01, size=120), index=dates)
    benchmark = pd.Series(rng.normal(0.0003, 0.008, size=120), index=dates)

    output_path = tmp_path / "tearsheet.html"
    generate_tearsheet(returns, benchmark, str(output_path))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_reporting.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.reporting'`

- [ ] **Step 3: Implement `reporting.py`**

Create `src/momentum/reporting.py`:

```python
import pandas as pd
import quantstats as qs


def generate_tearsheet(returns: pd.Series, benchmark: pd.Series, output_path: str) -> None:
    qs.reports.html(returns, benchmark=benchmark, output=output_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_reporting.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/reporting.py tests/test_reporting.py
git commit -m "feat: add quantstats tearsheet reporting"
```

---

### Task 10: CLI entrypoint (`cli.py`)

**Files:**
- Create: `src/momentum/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `momentum.config.load_config`, `momentum.backtest.run_backtest`, `momentum.reporting.generate_tearsheet`
- Produces: `main(argv: list[str] | None = None) -> int`, invoked as `momentum run <config_path> [--output PATH]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_cli.py`:

```python
import pandas as pd

from momentum import cli


def test_main_run_command_wires_config_backtest_and_reporting(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "benchmark: SPY\nstart_date: '2022-01-01'\nend_date: '2022-06-01'\n"
        "universe:\n  source: static\n  tickers: [MSFT]\n"
    )
    output_path = tmp_path / "out.html"

    fake_returns = pd.Series([0.01, -0.02], index=pd.date_range("2022-01-03", periods=2))
    fake_benchmark = pd.Series([0.005, -0.01], index=pd.date_range("2022-01-03", periods=2))
    calls = {}

    def _fake_run_backtest(config):
        calls["config"] = config
        return fake_returns, fake_benchmark

    def _fake_generate_tearsheet(returns, benchmark, output_path):
        calls["returns"] = returns
        calls["benchmark"] = benchmark
        calls["output_path"] = output_path

    monkeypatch.setattr(cli, "run_backtest", _fake_run_backtest)
    monkeypatch.setattr(cli, "generate_tearsheet", _fake_generate_tearsheet)

    exit_code = cli.main(["run", str(config_path), "--output", str(output_path)])

    assert exit_code == 0
    assert calls["config"].benchmark == "SPY"
    assert calls["output_path"] == str(output_path)
    assert str(output_path) in capsys.readouterr().out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'momentum.cli'`

- [ ] **Step 3: Implement `cli.py`**

Create `src/momentum/cli.py`:

```python
import argparse

from momentum.backtest import run_backtest
from momentum.config import load_config
from momentum.reporting import generate_tearsheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="momentum")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("config_path")
    run_parser.add_argument("--output", default="tearsheet.html")

    args = parser.parse_args(argv)

    config = load_config(args.config_path)
    returns, benchmark_returns = run_backtest(config)
    generate_tearsheet(returns, benchmark_returns, args.output)
    print(f"Tearsheet written to {args.output}")
    return 0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cli.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/momentum/cli.py tests/test_cli.py
git commit -m "feat: add momentum CLI entrypoint"
```

---

### Task 11: Sample config, README, and manual verification

**Files:**
- Create: `configs/default.yaml`
- Modify: `README.md`

**Interfaces:**
- None (integration/documentation task; no new library code).

- [ ] **Step 1: Create the sample config**

Create `configs/default.yaml`, porting the original `raw/main.py` ticker universe as a static list:

```yaml
benchmark: SPY
start_date: "2022-01-01"
end_date: "2024-12-31"
cache_dir: .cache

universe:
  source: static
  tickers:
    [
      MSFT, AAPL, NVDA, GOOGL, AMZN, META, AVGO, TSLA, LLY, UNH, JNJ, MRK, ABBV,
      PFE, TMO, DHR, MDT, "BRK-B", JPM, V, MA, BAC, WFC, GS, BLK, AXP, COST, WMT,
      HD, PG, KO, PEP, MCD, NKE, SBUX, LOW, XOM, CVX, CAT, UNP, GE, BA, LMT, DE,
      SMCI, TSM, ORCL, ADBE, CRM, AMD, INTC, QCOM, IBM, LIN, NFLX, DIS, VZ, CMCSA,
      ABT, ACN, CSCO, TMUS, TXN, HON, GILD, BKNG, C, SPG, PLD, AMT, EQIX, NOW,
      PLTR, UBER, PYPL, QQQ, VTI, DIA, IWM, XLK, XLV, XLF, XLY, XLC, XLE, XLI,
      XLP, XLB, XLU, XLRE, VEA, VWO, EFA, EWJ, EWG, INDA, MCHI, EEM, ACWI, MTUM,
      QUAL, USMV, VLUE, VIG, SOXX, HACK, ICLN, BOTZ, ARKK, TAN, IBB, FDN, PAVE,
      URA, AGG, TLT, LQD, HYG, SHY, GLD, SLV, DBC, VNQ, USO, FEZ, FXI,
    ]

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
  rebalance_frequency: monthly
```

(Note: SPY is dropped from the tickers list since it's the configured `benchmark`; the NSE-suffixed tickers from `raw/main.py` are dropped too, since `yfinance` and the S&P-500-oriented dynamic universe don't mix well with them — they can be added back into a separate config if needed.)

- [ ] **Step 2: Update `README.md`**

Replace the file with:

```markdown
# momentum

A momentum investing strategy backtester built on `backtrader`, with
config-driven runs, cached price data, and `quantstats` tearsheet reporting.

## Setup

```bash
uv sync
```

## Running a backtest

```bash
uv run momentum run configs/default.yaml
```

This downloads (and locally caches under `.cache/`) price data for the
configured universe and benchmark, runs the momentum strategy, and writes an
HTML performance tearsheet to `tearsheet.html`. Pass `--output <path>` to
change the tearsheet location.

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
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests across all modules PASS

- [ ] **Step 4: Manual end-to-end verification (real network, not part of CI)**

Run:

```bash
uv run momentum run configs/default.yaml --output /tmp/tearsheet.html
```

Expected: downloads data for the configured universe, prints
`Tearsheet written to /tmp/tearsheet.html`, and the file opens in a browser
showing a quantstats performance tearsheet benchmarked against SPY.

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml README.md
git commit -m "docs: add sample config and usage instructions"
```
