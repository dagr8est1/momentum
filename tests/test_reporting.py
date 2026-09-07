import re
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


def test_generate_tearsheet_injects_summary_statistics(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=260, freq="B")
    returns = pd.Series(rng.normal(0.0008, 0.01, size=260), index=dates)
    benchmark = pd.Series(rng.normal(0.0003, 0.008, size=260), index=dates)

    output_path = tmp_path / "tearsheet.html"
    generate_tearsheet(returns, benchmark, str(output_path), turnover=2.35)

    html = output_path.read_text()
    assert "Summary Statistics" in html
    assert "Backtest period" in html
    assert "Annualized return (CAGR)" in html
    assert "Sharpe ratio" in html
    assert "Annualized turnover" in html
    assert "2.35x" in html


def test_generate_tearsheet_omits_turnover_row_when_not_provided(tmp_path):
    rng = np.random.default_rng(0)
    dates = pd.date_range("2022-01-03", periods=120, freq="B")
    returns = pd.Series(rng.normal(0.0005, 0.01, size=120), index=dates)
    benchmark = pd.Series(rng.normal(0.0003, 0.008, size=120), index=dates)

    output_path = tmp_path / "tearsheet.html"
    generate_tearsheet(returns, benchmark, str(output_path))

    html = output_path.read_text()
    assert "Summary Statistics" in html
    assert "Annualized turnover" not in html


def test_summary_compares_benchmark_over_the_strategys_own_window(tmp_path):
    """The strategy's series starts after the regime-SMA warmup; the benchmark
    must be clipped to that same window or it gets credited with returns the
    strategy never had the chance to earn."""
    dates = pd.date_range("2020-01-01", periods=400, freq="B")
    benchmark = pd.Series(0.002, index=dates)  # steady riser over the FULL window
    returns = pd.Series(0.002, index=dates[200:])  # strategy starts halfway in

    output = tmp_path / "tearsheet.html"
    generate_tearsheet(returns, benchmark, str(output))
    html = output.read_text()

    # Identical daily returns over the matched window => no relative gap.
    match = re.search(r"Relative total return.*?([+-][\d.]+)%", html, re.S)
    assert match is not None
    assert abs(float(match.group(1))) < 0.5
