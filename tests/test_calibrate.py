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

    assert offset.offset[0] < 0  # it narrowed
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

    assert offset.offset[0] > 0
    assert abs(after - 0.8) < 0.05


def test_calibration_leaves_the_median_untouched():
    """CQR calibrates the *interval*. Moving the median would change the
    reported direction with no evidence for doing so."""
    predictions = _band([1.0, 2.0], [5.0, 6.0], [9.0, 10.0])
    calibrated = apply_conformal_offset(predictions, ConformalOffset([-2.0], 0.8, 100))
    np.testing.assert_allclose(calibrated[:, :, 1], predictions[:, :, 1])


def test_calibration_keeps_the_band_monotonic_even_when_it_collapses():
    """A large negative offset can pull the outer edges inside the median, and
    every consumer downstream assumes monotonic input (PYQ-124)."""
    predictions = _band([1.0], [5.0], [9.0])
    calibrated = apply_conformal_offset(predictions, ConformalOffset([-10.0], 0.8, 50))
    assert np.all(np.diff(calibrated, axis=-1) >= 0)


def test_offset_uses_the_finite_sample_corrected_quantile_level():
    """The ceil((n+1)*coverage)/n correction is what buys the marginal coverage
    guarantee; plain np.quantile(scores, coverage) does not. At small n it is
    not a rounding detail. horizon=1 here, so effective_sample_size(20, 1) ==
    20 -- the raw and corrected sample sizes coincide and this is purely a
    single-step regression test for the correction formula itself."""
    rng = np.random.default_rng(3)
    actuals = rng.normal(0, 1, (20, 1))
    predictions = np.tile(np.array([[-1.0, 0.0, 1.0]]), (20, 1, 1))

    offset = fit_conformal_offset(actuals, predictions, [0.1, 0.5, 0.9])
    scores = conformity_scores(actuals, predictions[:, :, 0], predictions[:, :, -1]).reshape(-1)
    corrected = float(np.quantile(scores, np.ceil((20 + 1) * 0.8) / 20, method="higher"))

    assert offset.offset == pytest.approx([corrected])
    assert offset.n_calibration == 20
    assert offset.nominal_coverage == pytest.approx(0.8)


def test_offset_round_trips_through_its_dict_form():
    """It is persisted in meta.json so `forecast` applies the same correction
    the reported coverage was computed under."""
    original = ConformalOffset(offset=[-0.25, 0.1, 0.4], nominal_coverage=0.8, n_calibration=42)
    assert ConformalOffset.from_dict(original.to_dict()) == original


def test_offset_from_dict_accepts_a_legacy_scalar():
    """Bundles calibrated before PYQ-144 stored a bare scalar in meta.json;
    loading one must not error, and it should behave as a single-step offset."""
    legacy = {"offset": -0.25, "nominal_coverage": 0.8, "n_calibration": 42}
    assert ConformalOffset.from_dict(legacy).offset == [-0.25]


def test_applying_no_offset_is_the_identity():
    """A bundle trained without a calibration slice must be unaffected."""
    predictions = _band([1.0, 2.0], [5.0, 6.0], [9.0, 10.0])
    np.testing.assert_allclose(apply_conformal_offset(predictions, None), predictions)


def test_applying_a_legacy_scalar_offset_broadcasts_to_every_step():
    """apply_conformal_offset must still accept a bare float/single-element
    offset (a pre-PYQ-144 bundle) and widen every horizon step identically."""
    predictions = np.tile(np.array([[-1.0, 0.0, 1.0]]), (1, 3, 1))  # horizon=3
    calibrated = apply_conformal_offset(predictions, 0.5)
    np.testing.assert_allclose(calibrated[0, :, 0], [-1.5, -1.5, -1.5])
    np.testing.assert_allclose(calibrated[0, :, -1], [1.5, 1.5, 1.5])


def test_applying_offset_of_wrong_length_raises():
    predictions = np.tile(np.array([[-1.0, 0.0, 1.0]]), (1, 3, 1))  # horizon=3
    with pytest.raises(ValueError, match="horizon"):
        apply_conformal_offset(predictions, ConformalOffset([0.1, 0.2], 0.8, 10))


def test_fit_conformal_offset_produces_different_offsets_when_dispersion_varies_by_step():
    """PYQ-144 acceptance criterion: horizon-varying synthetic dispersion must
    yield per-step offsets that differ, and each step's post-calibration
    coverage must land closer to nominal than a single pooled offset's."""
    rng = np.random.default_rng(42)
    n_cal, n_test, horizon = 2000, 4000, 3
    # True distribution is iid N(0, 1) at every step; the predicted band is
    # deliberately mis-calibrated *differently* per step: too narrow at h=1,
    # about right at h=2, too wide at h=3.
    half_widths = np.array([0.3, 1.2816, 3.0])
    band_template = np.stack([-half_widths, np.zeros(horizon), half_widths], axis=-1)

    def sample(n):
        actuals = rng.normal(0.0, 1.0, (n, horizon))
        predictions = np.tile(band_template, (n, 1, 1))
        return actuals, predictions

    cal_actuals, cal_predictions = sample(n_cal)
    test_actuals, test_predictions = sample(n_test)

    per_step = fit_conformal_offset(cal_actuals, cal_predictions, [0.1, 0.5, 0.9])
    assert len(per_step.offset) == horizon
    assert len(set(round(o, 3) for o in per_step.offset)) == horizon  # all differ

    # The old (pre-PYQ-144) behaviour: one offset pooled across every step.
    pooled_scores = conformity_scores(
        cal_actuals, cal_predictions[:, :, 0], cal_predictions[:, :, -1]
    ).reshape(-1)
    n_pooled = pooled_scores.size
    pooled_level = min(1.0, np.ceil((n_pooled + 1) * 0.8) / n_pooled)
    pooled_offset = float(np.quantile(pooled_scores, pooled_level, method="higher"))

    per_step_calibrated = apply_conformal_offset(test_predictions, per_step)
    pooled_calibrated = apply_conformal_offset(test_predictions, pooled_offset)

    for h in range(horizon):
        per_step_coverage = metrics.calibration_coverage(
            test_actuals[:, h], per_step_calibrated[:, h, 0], per_step_calibrated[:, h, -1]
        )
        pooled_coverage = metrics.calibration_coverage(
            test_actuals[:, h], pooled_calibrated[:, h, 0], pooled_calibrated[:, h, -1]
        )
        assert abs(per_step_coverage - 0.8) < abs(pooled_coverage - 0.8), (
            f"step {h}: per-step coverage {per_step_coverage:.3f} is not closer to "
            f"nominal than pooled coverage {pooled_coverage:.3f}"
        )
