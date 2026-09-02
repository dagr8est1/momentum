from typing import Sequence

import numpy as np
from scipy.stats import skew


def momentum_blend(closes: Sequence[float], lookbacks: Sequence[int]) -> float:
    """Mean price difference between the latest close and each lookback's close."""
    closes = np.asarray(closes, dtype=float)
    latest = closes[-1]
    diffs = [latest - closes[-1 - lb] for lb in lookbacks]
    return float(np.mean(diffs))


def fip_score(returns: Sequence[float]) -> float:
    """Fraction of days in the window with a positive return."""
    returns = np.asarray(returns, dtype=float)
    return float(np.sum(returns > 0) / len(returns))


def skewness_score(closes: Sequence[float]) -> float:
    """Skewness of the log returns implied by a window of closing prices."""
    closes = np.asarray(closes, dtype=float)
    log_returns = np.diff(np.log(closes))
    return float(skew(log_returns))


def combined_score(
    momentum: float,
    fip: float,
    skewness: float,
    momentum_weight: float,
    fip_weight: float,
    skewness_penalty: float,
) -> float:
    """Weighted blend of momentum, frog-in-the-pan, and skewness scores."""
    return (
        momentum_weight * momentum
        + fip_weight * fip
        + skewness_penalty * skewness
    )


def inverse_vol_weights(volatilities: dict[str, float]) -> dict[str, float]:
    """Portfolio weights inversely proportional to each name's volatility.

    Names with non-positive volatility get a weight of 0 and are excluded
    from the normalization.
    """
    inv_vols = {name: (1 / vol if vol > 0 else 0.0) for name, vol in volatilities.items()}
    total = sum(inv_vols.values())
    if total <= 0:
        return {name: 0.0 for name in volatilities}
    return {name: inv_vol / total for name, inv_vol in inv_vols.items()}
