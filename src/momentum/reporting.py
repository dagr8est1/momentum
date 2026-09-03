import re

import pandas as pd
import quantstats as qs


def generate_tearsheet(
    returns: pd.Series,
    benchmark: pd.Series,
    output_path: str,
    turnover: float | None = None,
) -> None:
    qs.reports.html(returns, benchmark=benchmark, output=output_path)
    summary_html = _render_summary(returns, benchmark, turnover)
    _inject_after_body(output_path, summary_html)


def _render_summary(returns: pd.Series, benchmark: pd.Series, turnover: float | None) -> str:
    start = returns.index.min()
    end = returns.index.max()
    years = (end - start).days / 365.25

    total_return = qs.stats.comp(returns)
    benchmark_total_return = qs.stats.comp(benchmark)
    cagr = qs.stats.cagr(returns)
    benchmark_cagr = qs.stats.cagr(benchmark)
    sharpe = qs.stats.sharpe(returns)
    benchmark_sharpe = qs.stats.sharpe(benchmark)

    rows = [
        ("Backtest period", f"{start:%Y-%m-%d} to {end:%Y-%m-%d} (~{years:.1f} years)", ""),
        ("Total return (absolute)", f"{total_return:.1%}", f"benchmark: {benchmark_total_return:.1%}"),
        (
            "Relative total return",
            f"{total_return - benchmark_total_return:+.1%}",
            "strategy − benchmark",
        ),
        ("Annualized return (CAGR)", f"{cagr:.1%}", f"benchmark: {benchmark_cagr:.1%}"),
        ("Relative CAGR", f"{cagr - benchmark_cagr:+.1%}", "strategy − benchmark"),
        ("Sharpe ratio", f"{sharpe:.2f}", f"benchmark: {benchmark_sharpe:.2f}"),
    ]
    if turnover is not None:
        rows.append(("Annualized turnover", f"{turnover:.2f}x", "traded value ÷ avg. portfolio value ÷ year"))

    row_html = "\n".join(
        f'<tr><td style="padding:6px 16px 6px 0;color:#666;white-space:nowrap">{label}</td>'
        f'<td style="padding:6px 16px 6px 0;font-weight:600;white-space:nowrap">{value}</td>'
        f'<td style="padding:6px 0;color:#999">{note}</td></tr>'
        for label, value, note in rows
    )

    return f"""
<div style="font-family:-apple-system,Helvetica,Arial,sans-serif;max-width:900px;margin:24px auto;padding:20px 24px;border:1px solid #ddd;border-radius:10px;background:#fafafa">
  <h2 style="margin:0 0 12px;font-size:18px">Summary Statistics</h2>
  <table style="border-collapse:collapse;font-size:14px">
    {row_html}
  </table>
</div>
"""


def _inject_after_body(output_path: str, snippet: str) -> None:
    with open(output_path, encoding="utf-8") as f:
        html = f.read()
    html, count = re.subn(r"(<body\b[^>]*>)", r"\1\n" + snippet.replace("\\", "\\\\"), html, count=1)
    if count == 0:
        raise ValueError(f"Could not find a <body> tag to inject the summary into: {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
