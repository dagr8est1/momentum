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
