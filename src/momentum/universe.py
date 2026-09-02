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
    raise ValueError(f"Unknown universe source: {source}")


def _fetch_sp500_constituents() -> list[str]:
    response = requests.get(
        _SP500_WIKIPEDIA_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=10
    )
    response.raise_for_status()
    table = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})[0]
    return [ticker.replace(".", "-") for ticker in table["Symbol"].tolist()]
