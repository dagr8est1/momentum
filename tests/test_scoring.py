import numpy as np
import pytest

from momentum.scoring import (
    combined_score,
    fip_score,
    inverse_vol_weights,
    momentum_blend,
    skewness_score,
)


def test_momentum_blend_averages_percentage_returns_across_lookbacks():
    closes = [100.0] * 253
    closes[0] = 100.0  # 252-day-ago reference
    closes[132] = 120.0  # 120-day-ago reference
    closes[192] = 100.0  # 60-day-ago reference
    closes[252] = 150.0  # latest close

    result = momentum_blend(closes, lookbacks=[60, 120, 252])
    # returns: (150-100)/100=0.50, (150-120)/120=0.25, (150-100)/100=0.50
    assert result == pytest.approx((0.50 + 0.25 + 0.50) / 3)


def test_fip_score_is_fraction_of_positive_return_days():
    returns = [0.01, -0.02, 0.03, 0.04, -0.01]
    assert fip_score(returns) == pytest.approx(3 / 5)


def test_fip_score_all_negative_is_zero():
    assert fip_score([-0.01, -0.02, -0.03]) == 0.0


def test_skewness_score_zero_for_symmetric_returns():
    # prices whose log returns are symmetric around zero
    closes = [100, 110, 100, 110, 100, 110, 100]
    assert skewness_score(closes) == pytest.approx(0.0, abs=1e-9)


def test_skewness_score_positive_for_right_skewed_returns():
    closes = np.exp(np.cumsum([0.01, 0.01, 0.01, 0.01, 0.5, -0.01, -0.01])).tolist()
    assert skewness_score(closes) > 0


def test_combined_score_applies_weights():
    result = combined_score(
        momentum=10.0,
        fip=0.6,
        skewness=0.2,
        momentum_weight=0.5,
        fip_weight=0.3,
        skewness_penalty=0.1,
    )
    assert result == pytest.approx(0.5 * 10.0 + 0.3 * 0.6 + 0.1 * 0.2)


def test_inverse_vol_weights_favors_lower_volatility():
    weights = inverse_vol_weights({"low_vol": 1.0, "high_vol": 4.0})
    assert weights["low_vol"] > weights["high_vol"]
    assert weights["low_vol"] == pytest.approx(0.8)
    assert weights["high_vol"] == pytest.approx(0.2)


def test_inverse_vol_weights_excludes_zero_volatility():
    weights = inverse_vol_weights({"zero_vol": 0.0, "normal": 2.0})
    assert weights["zero_vol"] == 0.0
    assert weights["normal"] == pytest.approx(1.0)


def test_inverse_vol_weights_all_zero_returns_all_zero():
    weights = inverse_vol_weights({"a": 0.0, "b": 0.0})
    assert weights == {"a": 0.0, "b": 0.0}
