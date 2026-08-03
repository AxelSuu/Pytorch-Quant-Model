"""Split-conformal recalibration of a quantile band (PYQ-248).

PYQ-117 measured 99.3% empirical coverage on a nominal 80% (p10-p90) band. A
band that wide is close to uninformative: it tells a reader almost nothing, and
``scan``'s "is the whole band on one side of zero" guard (PYQ-206) can never
fire, silently collapsing BUY/SELL into permanent HOLD.

PYQ-227 diagnoses which side is at fault. Nothing fixed it. Retraining with
different hyperparameters is an indirect and unreliable route to a calibrated
interval; direct recalibration is the standard one.

This module implements **conformalized quantile regression** (CQR, Romano et
al. 2019): on a calibration slice disjoint from both training and test, score
each point by how far outside the band it fell,

    score = max(q_lo - y, y - q_hi)

(negative when the point is comfortably inside), take the appropriate empirical
quantile of those scores, and shift both edges of the band outward by it. A
*negative* offset is the case that matters here -- it means the band was too
wide and gets pulled in.

Properties that make this the right tool:

- distribution-free, with a finite-sample marginal coverage guarantee;
- no retraining, no change to the model or the loss;
- composes with the existing quantile head and with the PYQ-247 return target,
  because it operates on whatever units the quantiles are already in.

The guarantee assumes exchangeability, which financial time series violate. So
the offset is fitted on a slice that is itself out-of-sample and the *achieved*
coverage is reported on a genuinely held-out period (PYQ-250's purged splits),
rather than relying on the theoretical guarantee. Nothing here reports a
coverage number; it only produces the offset.

This module deliberately imports neither torch nor pytorch-forecasting: it works
on plain arrays, so it stays usable from `analysis/` under the layering rule.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from pyquant.analysis.metrics import effective_sample_size

logger = logging.getLogger(__name__)


@dataclass
class ConformalOffset:
    """A per-horizon-step additive widening (or narrowing) of a quantile band.

    ``offset`` holds one value per decoder step, in the target's own units --
    dollars for a ``close`` target, log-return units for ``log_return``
    (PYQ-247). Each is added to that step's upper edge and subtracted from its
    lower. Negative narrows.

    Before PYQ-144 this was a single scalar pooled across every horizon step.
    Forecast dispersion grows with horizon (PYQ-142's own sqrt(h) measurement),
    so a pooled offset over-narrowed near steps (already closest to correct)
    and under-widened far ones -- the bug PYQ-144 fixes. ``from_dict`` still
    accepts a bare scalar to load a bundle calibrated before this fix, applying
    it identically to every step (its pre-fix behaviour).
    """

    offset: list[float]
    nominal_coverage: float
    n_calibration: int

    def to_dict(self) -> dict[str, list[float] | float | int]:
        return {
            "offset": list(self.offset),
            "nominal_coverage": self.nominal_coverage,
            "n_calibration": self.n_calibration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConformalOffset:
        raw_offset = data["offset"]
        offset = (
            [float(raw_offset)]
            if isinstance(raw_offset, int | float)
            else [float(x) for x in raw_offset]
        )
        return cls(
            offset=offset,
            nominal_coverage=float(data["nominal_coverage"]),
            n_calibration=int(data["n_calibration"]),
        )


def conformity_scores(actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """CQR score per point: how far outside the band the actual fell.

    Positive means the point missed the band by that much; negative means it was
    inside with that much room to spare. Taking a high quantile of these scores
    therefore *widens* a band that is too narrow and *narrows* one that is too
    wide, with the same formula and no special-casing.
    """
    actuals = np.asarray(actuals, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return np.maximum(lower - actuals, actuals - upper)


def fit_conformal_offset(
    actuals: np.ndarray,
    predictions: np.ndarray,
    quantiles: list[float],
) -> ConformalOffset:
    """Fit a per-horizon-step CQR offset on a calibration slice.

    ``predictions`` is ``(n_samples, horizon, n_quantiles)``; the first and last
    quantile columns are the band, matching ``evaluate_predictions``.

    Fits one offset **per horizon step** (PYQ-144) rather than pooling across
    it: forecast dispersion grows with horizon (PYQ-142's own sqrt(h)
    measurement), so a single scalar correction over-narrows h=1, where the
    band is already closest to correct, and under-widens h=5, where it needs
    the most correction.

    The quantile level taken is the finite-sample corrected
    ``ceil((n + 1) * coverage) / n``, which is what buys the marginal guarantee
    rather than plain ``np.quantile(scores, coverage)``. With few calibration
    points the correction is not a rounding detail: at n = 20 and 80% nominal it
    is the difference between the 80th and the 84th percentile. ``n`` is
    ``effective_sample_size(n_samples, horizon)`` (PYQ-251), not the raw
    ``n_samples``: the calibration windows are overlapping sliding windows, so
    raw ``n_samples`` overstates the independent evidence backing the
    correction by roughly the horizon factor -- PYQ-144's second bug.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 3:
        raise ValueError(
            f"predictions must be (n_samples, horizon, n_quantiles), got {predictions.shape}"
        )
    if len(quantiles) < 2:
        raise ValueError("at least two quantiles are needed to form a band")

    n_samples, horizon, _ = predictions.shape
    if n_samples == 0:
        raise ValueError("no calibration points supplied")

    nominal = float(quantiles[-1] - quantiles[0])
    # scores[:, h] is horizon step h's conformity scores across calibration windows.
    scores = conformity_scores(actuals, predictions[:, :, 0], predictions[:, :, -1])
    n_eff = effective_sample_size(n_samples, horizon)
    level = min(1.0, np.ceil((n_eff + 1) * nominal) / n_eff)

    offsets = [float(np.quantile(scores[:, h], level, method="higher")) for h in range(horizon)]
    narrowed_steps = [h + 1 for h, o in enumerate(offsets) if o < 0]
    if narrowed_steps:
        logger.info(
            "Conformal calibration is narrowing the band at horizon step(s) %s "
            "(nominal %.0f%%, %d calibration window(s), %d effective): the "
            "predicted interval was wider than its nominal coverage required there.",
            narrowed_steps,
            nominal * 100,
            n_samples,
            n_eff,
        )
    return ConformalOffset(offset=offsets, nominal_coverage=nominal, n_calibration=n_eff)


def apply_conformal_offset(
    predictions: np.ndarray, offset: ConformalOffset | float | list[float] | None
) -> np.ndarray:
    """Widen (or narrow) the outer band of ``predictions`` by the fitted offset.

    ``predictions``'s horizon axis is second-to-last -- either
    ``(horizon, n_quantiles)`` from a single forecast or ``(n_samples, horizon,
    n_quantiles)`` from validation -- and ``offset`` (PYQ-144) holds one value
    per horizon step, broadcasting against it positionally. A scalar (or
    single-element list) broadcasts identically to every step -- the pre-PYQ-144
    behaviour, kept so a bundle calibrated before that fix still applies.

    Only the outermost quantiles move. Interior quantiles -- the median above
    all -- are left alone, because CQR calibrates the *interval*, not the point
    forecast, and shifting the median would change the reported direction
    without any evidence for doing so.

    The result is re-sorted along the quantile axis: a large negative offset can
    in principle pull the band inside an interior quantile, and every consumer
    downstream assumes monotonicity (PYQ-124).
    """
    if offset is None:
        return np.asarray(predictions, dtype=float)
    raw_delta = offset.offset if isinstance(offset, ConformalOffset) else offset
    out = np.array(predictions, dtype=float, copy=True)
    if out.shape[-1] < 2:
        return out
    horizon = out.shape[-2]
    delta = np.atleast_1d(np.asarray(raw_delta, dtype=float))
    if delta.size == 1:
        delta = np.full(horizon, float(delta[0]))
    elif delta.size != horizon:
        raise ValueError(
            f"conformal offset has {delta.size} step(s) but predictions have horizon {horizon}"
        )
    out[..., 0] -= delta
    out[..., -1] += delta
    return np.sort(out, axis=-1)
