import pandas as pd
import quantstats as qs


def generate_tearsheet(returns: pd.Series, benchmark: pd.Series, output_path: str) -> None:
    qs.reports.html(returns, benchmark=benchmark, output=output_path)
