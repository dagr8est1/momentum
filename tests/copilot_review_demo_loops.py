"""Demo file with intentional performance issues, to verify Copilot review
flags row-by-row iteration where a vectorized operation should be used.

Not imported or wired into any real code path (and not named test_*.py, so
pytest won't collect it). Delete once Copilot review has been confirmed
working, or move it under a generated/ or vendor/ path to keep it out of
review while preserving it for later.
"""

import numpy as np
import pandas as pd


def total_dollar_volume(prices: pd.DataFrame) -> float:
    """Sum of Close * Volume across every row.

    Iterates row by row with iterrows() to build the total, instead of the
    vectorized pandas equivalent: (prices["Close"] * prices["Volume"]).sum().
    iterrows() boxes every row into a Series, which is slow and also
    silently upcasts/loses dtypes compared to operating on the columns
    directly.
    """
    total = 0.0
    for _, row in prices.iterrows():
        total += row["Close"] * row["Volume"]
    return total


def portfolio_weighted_return(returns: np.ndarray, weights: np.ndarray) -> float:
    """Weighted sum of an array of asset returns.

    Loops element by element in pure Python instead of using the vectorized
    NumPy equivalent: float(np.dot(returns, weights)) or
    float((returns * weights).sum()). The Python-level loop pays per-element
    interpreter overhead that NumPy's compiled C loop avoids entirely, and
    the manual index bookkeeping is also a needless source of off-by-one
    bugs.
    """
    total = 0.0
    for i in range(len(returns)):
        total += returns[i] * weights[i]
    return total
