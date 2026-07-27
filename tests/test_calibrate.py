"""Split-conformal recalibration of the quantile band (PYQ-248)."""

import numpy as np
import pytest

from pyquant.analysis import metrics
from pyquant.analysis.calibrate import (
    ConformalOffset,
    apply_conformal_offset,
    conformity_scores,
    fit_conformal_offset,
)


def _band(lower, median, upper):
    """(n_samples, 1, 3) predictions from three same-length sequences."""
    return np.stack([np.asarray(lower), np.asarray(median), np.asarray(upper)], axis=-1)[:, None, :]


def test_conformity_score_is_negative_inside_the_band_and_positive_outside():
    """The CQR score is signed distance to the band, which is what lets one
    formula both widen a too-narrow band and narrow a too-wide one."""
    scores = conformity_scores(
        np.array([10.0, 30.0, -5.0]), np.array([0.0, 0.0, 0.0]), np.array([20.0, 20.0, 20.0])
    )
    assert scores[0] == pytest.approx(-10.0)  # inside, 10 to spare either side
    assert scores[1] == pytest.approx(10.0)  # 10 above the upper edge
    assert scores[2] == pytest.approx(5.0)  # 5 below the lower edge


def test_calibration_narrows_a_deliberately_overwide_band_toward_nominal():
    """The PYQ-117 pathology, reproduced and then fixed.

    A band covering ~100% of outcomes at a nominal 80% is nearly uninformative.
    Conformal calibration must pull it in, and coverage must land near nominal
    rather than near 1.0.
    """
    rng = np.random.default_rng(7)
    actuals = rng.normal(0.0, 1.0, (600, 1))
    # p10/p90 at +-8 sigma: absurdly wide, exactly the shape PYQ-117 measured.
    overwide = np.tile(np.array([[-8.0, 0.0, 8.0]]), (600, 1, 1))

    before = metrics.calibration_coverage(actuals, overwide[:, :, 0], overwide[:, :, -1])
    assert before == 1.0  # the pathology

    offset = fit_conformal_offset(actuals[:300], overwide[:300], [0.1, 0.5, 0.9])
    calibrated = apply_conformal_offset(overwide[300:], offset)
    after = metrics.calibration_coverage(actuals[300:], calibrated[:, :, 0], calibrated[:, :, -1])

    assert offset.offset < 0  # it narrowed
    assert abs(after - 0.8) < 0.05, f"coverage {after:.1%} is not near the nominal 80%"
    assert after < before


def test_calibration_widens_a_deliberately_narrow_band_toward_nominal():
    """The mirror case, so the fix is not accidentally one-directional."""
    rng = np.random.default_rng(11)
    actuals = rng.normal(0.0, 1.0, (600, 1))
    narrow = np.tile(np.array([[-0.05, 0.0, 0.05]]), (600, 1, 1))

    offset = fit_conformal_offset(actuals[:300], narrow[:300], [0.1, 0.5, 0.9])
    calibrated = apply_conformal_offset(narrow[300:], offset)
    after = metrics.calibration_coverage(actuals[300:], calibrated[:, :, 0], calibrated[:, :, -1])

    assert offset.offset > 0
    assert abs(after - 0.8) < 0.05


def test_calibration_leaves_the_median_untouched():
    """CQR calibrates the *interval*. Moving the median would change the
    reported direction with no evidence for doing so."""
    predictions = _band([1.0, 2.0], [5.0, 6.0], [9.0, 10.0])
    calibrated = apply_conformal_offset(predictions, ConformalOffset(-2.0, 0.8, 100))
    np.testing.assert_allclose(calibrated[:, :, 1], predictions[:, :, 1])


def test_calibration_keeps_the_band_monotonic_even_when_it_collapses():
    """A large negative offset can pull the outer edges inside the median, and
    every consumer downstream assumes monotonic input (PYQ-124)."""
    predictions = _band([1.0], [5.0], [9.0])
    calibrated = apply_conformal_offset(predictions, ConformalOffset(-10.0, 0.8, 50))
    assert np.all(np.diff(calibrated, axis=-1) >= 0)


def test_offset_uses_the_finite_sample_corrected_quantile_level():
    """The ceil((n+1)*coverage)/n correction is what buys the marginal coverage
    guarantee; plain np.quantile(scores, coverage) does not. At small n it is
    not a rounding detail."""
    rng = np.random.default_rng(3)
    actuals = rng.normal(0, 1, (20, 1))
    predictions = np.tile(np.array([[-1.0, 0.0, 1.0]]), (20, 1, 1))

    offset = fit_conformal_offset(actuals, predictions, [0.1, 0.5, 0.9])
    scores = conformity_scores(actuals, predictions[:, :, 0], predictions[:, :, -1]).reshape(-1)
    corrected = float(np.quantile(scores, np.ceil((20 + 1) * 0.8) / 20, method="higher"))

    assert offset.offset == pytest.approx(corrected)
    assert offset.n_calibration == 20
    assert offset.nominal_coverage == pytest.approx(0.8)


def test_offset_round_trips_through_its_dict_form():
    """It is persisted in meta.json so `forecast` applies the same correction
    the reported coverage was computed under."""
    original = ConformalOffset(offset=-0.25, nominal_coverage=0.8, n_calibration=42)
    assert ConformalOffset.from_dict(original.to_dict()) == original


def test_applying_no_offset_is_the_identity():
    """A bundle trained without a calibration slice must be unaffected."""
    predictions = _band([1.0, 2.0], [5.0, 6.0], [9.0, 10.0])
    np.testing.assert_allclose(apply_conformal_offset(predictions, None), predictions)
