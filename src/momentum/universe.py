import io

import pandas as pd
import requests

_SP500_WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def resolve_universe(spec: dict) -> list[str]:
    source = spec.get("source", "static")
    if source == "static":
        return list(spec["tickers"])
    if source == "index:sp500":
        return _fetch_sp500_constituents()
    if source == "index:sp500_point_in_time":
        return _superset_tickers(spec["constituents_file"])
    raise ValueError(f"Unknown universe source: {source}")


def _fetch_sp500_constituents() -> list[str]:
    response = requests.get(
        _SP500_WIKIPEDIA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
    )
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})[0]
    return [ticker.replace(".", "-") for ticker in table["Symbol"].tolist()]


def _superset_tickers(constituents_file: str) -> list[str]:
    """Every ticker that has EVER been a member, per the point-in-time file.

    Used only to decide which data feeds to load — actual per-date
    eligibility is enforced separately by `load_point_in_time_membership`,
    consulted by `MomentumStrategy` at each rebalance.
    """
    df = pd.read_parquet(constituents_file, columns=["Ticker"])
    return sorted(df["Ticker"].unique())


def load_point_in_time_membership(constituents_file: str) -> pd.Series:
    """Month-end FactorDate -> frozenset of tickers that were S&P 500
    members as of that date, sorted by date for as-of lookups."""
    df = pd.read_parquet(constituents_file)
    return df.groupby("FactorDate")["Ticker"].apply(frozenset).sort_index()
