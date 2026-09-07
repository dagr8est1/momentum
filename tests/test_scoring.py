import numpy as np
import pytest

from momentum.scoring import (
    cap_weighted_inverse_vol_weights,
    combined_score,
    downside_deviation,
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


def test_momentum_blend_skip_measures_to_an_earlier_close_not_the_latest():
    closes = [100.0] * 20
    closes[10] = 80.0  # 5 trading days before the skip point
    closes[15] = 100.0  # the "skip=4" reference close (index 20-1-4-1=15)
    closes[19] = 200.0  # latest close -- must be ignored when skip=4

    result = momentum_blend(closes, lookbacks=[5], skip=4)
    # measures from index 15-5=10 (80.0) to index 15 (100.0), NOT to 19 (200.0)
    assert result == pytest.approx((100.0 - 80.0) / 80.0)


def test_momentum_blend_skip_zero_matches_default_behavior():
    closes = [100.0, 110.0, 90.0, 130.0]
    assert momentum_blend(closes, lookbacks=[2], skip=0) == momentum_blend(
        closes, lookbacks=[2]
    )


def test_downside_deviation_ignores_positive_returns():
    returns = [0.05, 0.03, -0.02, 0.04, -0.04]
    # downside-only: [0, 0, -0.02, 0, -0.04] -> sqrt(mean([0, 0, 0.0004, 0, 0.0016]))
    expected = np.sqrt((0.0004 + 0.0016) / 5)
    assert downside_deviation(returns) == pytest.approx(expected)


def test_downside_deviation_all_positive_returns_hits_floor():
    returns = [0.01, 0.02, 0.03]
    assert downside_deviation(returns, floor=1e-4) == pytest.approx(1e-4)


def test_downside_deviation_larger_for_choppier_downside():
    calm = downside_deviation([0.01, -0.005, 0.01, -0.005])
    volatile = downside_deviation([0.01, -0.05, 0.01, -0.05])
    assert volatile > calm


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


def test_cap_weighted_inverse_vol_favors_larger_cap_at_equal_volatility():
    weights = cap_weighted_inverse_vol_weights(
        market_caps={"big": 200.0, "small": 100.0},
        volatilities={"big": 1.0, "small": 1.0},
    )
    assert weights["big"] == pytest.approx(2 / 3)
    assert weights["small"] == pytest.approx(1 / 3)


def test_cap_weighted_inverse_vol_still_penalizes_higher_volatility():
    # Same market cap, but "volatile" has 4x the volatility of "calm" ->
    # cap/vol scores are 100 and 25, i.e. calm should get 4x the weight.
    weights = cap_weighted_inverse_vol_weights(
        market_caps={"calm": 100.0, "volatile": 100.0},
        volatilities={"calm": 1.0, "volatile": 4.0},
    )
    assert weights["calm"] == pytest.approx(0.8)
    assert weights["volatile"] == pytest.approx(0.2)


def test_cap_weighted_inverse_vol_excludes_missing_market_cap():
    weights = cap_weighted_inverse_vol_weights(
        market_caps={"has_cap": 100.0},
        volatilities={"has_cap": 1.0, "no_cap": 1.0},
    )
    assert weights["has_cap"] == pytest.approx(1.0)
    assert weights["no_cap"] == 0.0


def test_cap_weighted_inverse_vol_all_zero_returns_all_zero():
    weights = cap_weighted_inverse_vol_weights(
        market_caps={"a": 0.0, "b": 0.0}, volatilities={"a": 1.0, "b": 1.0}
    )
    assert weights == {"a": 0.0, "b": 0.0}
