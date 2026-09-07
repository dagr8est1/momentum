from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class StrategyConfig:
    lookbacks: list[int] = field(default_factory=lambda: [60, 120, 252])
    momentum_skip: int = 0
    top_n: int = 10
    vol_lookback: int = 126
    skewness_lookback: int = 90
    fip_lookback: int = 252
    ts_mom_lookback: int = 200
    regime_ma_period: int = 200
    momentum_weight: float = 0.5
    fip_weight: float = 0.5
    skewness_penalty: float = 0.5
    rebalance_frequency: str | None = "monthly"
    sizing_method: str = "inverse_vol"


@dataclass
class RunConfig:
    benchmark: str
    start_date: str
    end_date: str
    universe: dict
    strategy: StrategyConfig
    cache_dir: str = "cache_data"


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

    if strategy.rebalance_frequency not in (None, "monthly", "quarterly"):
        raise ValueError(
            f"Invalid rebalance_frequency: {strategy.rebalance_frequency!r} "
            "(must be None, 'monthly', or 'quarterly')"
        )

    if strategy.sizing_method not in ("inverse_vol", "cap_weighted"):
        raise ValueError(
            f"Invalid sizing_method: {strategy.sizing_method!r} "
            "(must be 'inverse_vol' or 'cap_weighted')"
        )

    return RunConfig(
        benchmark=raw["benchmark"],
        start_date=raw["start_date"],
        end_date=raw["end_date"],
        universe=raw["universe"],
        strategy=strategy,
        cache_dir=raw.get("cache_dir", "cache_data"),
    )
