import pandas as pd
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


def _write_constituents_file(path, rows):
    pd.DataFrame(rows, columns=["FactorDate", "Ticker"]).to_parquet(path)
    return str(path)


def test_resolve_universe_point_in_time_returns_superset_of_all_tickers(tmp_path):
    path = _write_constituents_file(
        tmp_path / "constituents.parquet",
        [
            (pd.Timestamp("2020-01-31"), "AAPL"),
            (pd.Timestamp("2020-01-31"), "MSFT"),
            (pd.Timestamp("2020-02-29"), "AAPL"),
            (pd.Timestamp("2020-02-29"), "GOOG"),  # AAPL stays, MSFT drops, GOOG joins
        ],
    )

    result = universe.resolve_universe(
        {"source": "index:sp500_point_in_time", "constituents_file": path}
    )

    assert result == ["AAPL", "GOOG", "MSFT"]


def test_load_point_in_time_membership_groups_tickers_by_date(tmp_path):
    path = _write_constituents_file(
        tmp_path / "constituents.parquet",
        [
            (pd.Timestamp("2020-01-31"), "AAPL"),
            (pd.Timestamp("2020-01-31"), "MSFT"),
            (pd.Timestamp("2020-02-29"), "AAPL"),
            (pd.Timestamp("2020-02-29"), "GOOG"),
        ],
    )

    membership = universe.load_point_in_time_membership(path)

    assert membership.index.is_monotonic_increasing
    assert membership.loc[pd.Timestamp("2020-01-31")] == frozenset({"AAPL", "MSFT"})
    assert membership.loc[pd.Timestamp("2020-02-29")] == frozenset({"AAPL", "GOOG"})
