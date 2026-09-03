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

INVALID_REBALANCE_FREQUENCY_YAML = """
benchmark: SPY
start_date: "2022-01-01"
end_date: "2024-12-31"
universe:
  source: static
  tickers: [MSFT]
strategy:
  rebalance_frequency: weekly
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

    assert config.cache_dir == "cache_data"


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


def test_load_config_raises_on_invalid_rebalance_frequency(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(INVALID_REBALANCE_FREQUENCY_YAML)

    with pytest.raises(ValueError, match="rebalance_frequency"):
        load_config(config_path)
