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

logger = logging.getLogger(__name__)


@dataclass
class ConformalOffset:
    """An additive widening (or narrowing) of a quantile band.

    ``offset`` is in the target's own units -- dollars for a ``close`` target,
    log-return units for ``log_return`` (PYQ-247) -- and is added to the upper
    edge and subtracted from the lower. Negative narrows.
    """

    offset: float
    nominal_coverage: float
    n_calibration: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "offset": self.offset,
            "nominal_coverage": self.nominal_coverage,
            "n_calibration": self.n_calibration,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ConformalOffset:
        return cls(
            offset=float(data["offset"]),
            nominal_coverage=float(data["nominal_coverage"]),
            n_calibration=int(data["n_calibration"]),
        )


def conformity_scores(
    actuals: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
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
    """Fit the CQR offset on a calibration slice.

    ``predictions`` is ``(n_samples, horizon, n_quantiles)``; the first and last
    quantile columns are the band, matching ``evaluate_predictions``.

    The quantile level taken is the finite-sample corrected
    ``ceil((n + 1) * coverage) / n``, which is what buys the marginal guarantee
    rather than plain ``np.quantile(scores, coverage)``. With few calibration
    points the correction is not a rounding detail: at n = 20 and 80% nominal it
    is the difference between the 80th and the 84th percentile.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.ndim != 3:
        raise ValueError(f"predictions must be (n_samples, horizon, n_quantiles), got {predictions.shape}")
    if len(quantiles) < 2:
        raise ValueError("at least two quantiles are needed to form a band")

    nominal = float(quantiles[-1] - quantiles[0])
    scores = conformity_scores(actuals, predictions[:, :, 0], predictions[:, :, -1]).reshape(-1)
    n = scores.size
    if n == 0:
        raise ValueError("no calibration points supplied")

    level = min(1.0, np.ceil((n + 1) * nominal) / n)
    offset = float(np.quantile(scores, level, method="higher"))
    if offset < 0:
        logger.info(
            "Conformal calibration is narrowing the band by %.6g (nominal %.0f%%, "
            "%d calibration points): the predicted interval was wider than its "
            "nominal coverage required.",
            -offset,
            nominal * 100,
            n,
        )
    return ConformalOffset(offset=offset, nominal_coverage=nominal, n_calibration=n)


def apply_conformal_offset(
    predictions: np.ndarray, offset: ConformalOffset | float | None
) -> np.ndarray:
    """Widen (or narrow) the outer band of ``predictions`` by the fitted offset.

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
    delta = offset.offset if isinstance(offset, ConformalOffset) else float(offset)
    out = np.array(predictions, dtype=float, copy=True)
    if out.shape[-1] < 2:
        return out
    out[..., 0] -= delta
    out[..., -1] += delta
    return np.sort(out, axis=-1)
