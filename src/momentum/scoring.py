from typing import Sequence

import numpy as np
from scipy.stats import skew


def momentum_blend(closes: Sequence[float], lookbacks: Sequence[int]) -> float:
    """Mean percentage return from each lookback's close to the latest close."""
    closes = np.asarray(closes, dtype=float)
    latest = closes[-1]
    returns = [(latest - closes[-1 - lb]) / closes[-1 - lb] for lb in lookbacks]
    return float(np.mean(returns))


def fip_score(returns: Sequence[float]) -> float:
    """Fraction of days in the window with a positive return."""
    returns = np.asarray(returns, dtype=float)
    return float(np.sum(returns > 0) / len(returns))


def downside_deviation(returns: Sequence[float], floor: float = 1e-4) -> float:
    """Standard deviation computed only over negative returns.

    Unlike plain standard deviation, upside moves aren't treated as risk —
    a stock that only ever has big up days scores as low-risk here, not
    high-risk. `floor` guards against a zero result (e.g. no down days in
    the window) being treated as "no data" by inverse_vol_weights, which
    would otherwise exclude a genuinely low-risk stock from sizing.
    """
    returns = np.asarray(returns, dtype=float)
    downside = np.minimum(returns, 0.0)
    return float(max(np.sqrt(np.mean(downside**2)), floor))


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


def cap_weighted_inverse_vol_weights(
    market_caps: dict[str, float], volatilities: dict[str, float]
) -> dict[str, float]:
    """Portfolio weights proportional to market_cap / volatility.

    A market-cap tilt on top of the existing inverse-vol risk-parity tilt:
    for the same volatility, a larger (more liquid, more established) name
    gets more weight. Names with non-positive volatility or a missing/
    non-positive market cap get a weight of 0 and are excluded from the
    normalization, matching inverse_vol_weights' convention.
    """
    scores = {
        name: (market_caps.get(name, 0.0) / vol if vol > 0 and market_caps.get(name, 0.0) > 0 else 0.0)
        for name, vol in volatilities.items()
    }
    total = sum(scores.values())
    if total <= 0:
        return {name: 0.0 for name in volatilities}
    return {name: score / total for name, score in scores.items()}
