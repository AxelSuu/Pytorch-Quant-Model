"""PYQ-239: does the training pipeline learn anything, and does it avoid learning noise?

``test_train_load_predict_roundtrip`` (test_tft.py) only asserts structural properties --
bundle files exist, ``n_features > 0``, metrics land in ``[0, 1]``. A model that always
predicts a constant would pass every one of those. Nothing else in the suite asserts that
training can learn *anything*, which matters more than usual here because the project's
headline skill number is small and could equally be explained by "the target genuinely
isn't forecastable" or "something in the wiring is silently broken" -- these two tests
discriminate between those explanations directly rather than by inference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pyquant.data.prices import add_technical_indicators
from pyquant.models import tft


def _signal_panel(
    n: int = 400,
    k: float = 0.08,
    noise_std: float = 0.002,
    seed: int = 7,
    learnable: bool = True,
) -> pd.DataFrame:
    """A synthetic OHLCV panel plus a ``Signal`` feature.

    When ``learnable``, day t's log-return is ``k * Signal[t-1] + noise`` -- a
    one-day-lagged, deterministic-up-to-noise relationship that a model with any
    encoder history at all should be able to recover. When not, ``Signal`` carries no
    information about the returns whatsoever.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    signal = rng.choice([-1.0, 1.0], size=n)

    if learnable:
        log_returns = k * np.roll(signal, 1) + rng.normal(0, noise_std, n)
        log_returns[0] = 0.0
    else:
        log_returns = rng.normal(0, noise_std, n)
        rng.shuffle(signal)  # decouple from returns even by construction accident

    close = 100 * np.exp(np.cumsum(log_returns))
    df = pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.0005, n)),
            "High": close * (1 + np.abs(rng.normal(0, 0.001, n))),
            "Low": close * (1 - np.abs(rng.normal(0, 0.001, n))),
            "Close": close,
            "Volume": np.abs(rng.normal(0, 1, n)) * 1_000_000,
            "Signal": signal,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


@pytest.fixture
def learnability_settings(tmp_path, settings):
    """A one-day-ahead config: the target is exactly the lagged signal's effect.

    ``max_prediction_length=1`` is deliberate: ``Signal`` is an *unknown* real (like
    every other feature here), so it is only ever visible to the encoder, never to the
    decoder. A multi-step horizon would require the model to predict steps whose
    driving Signal value hasn't been observed yet -- genuinely unlearnable by
    construction, not a wiring bug -- which would confound this test's purpose. One
    step keeps the learnable relationship unambiguous.
    """
    settings.checkpoint_dir = tmp_path / "checkpoints"
    settings.training.target = "log_return"
    settings.training.max_encoder_length = 15
    settings.training.max_prediction_length = 1
    settings.training.validation_days = 60
    settings.training.batch_size = 32
    settings.training.max_epochs = 25
    settings.training.early_stopping_patience = 8
    settings.tft.hidden_size = 8
    settings.tft.hidden_continuous_size = 4
    return settings


def test_model_recovers_an_injected_learnable_signal(monkeypatch, learnability_settings):
    """skill_vs_baseline must be clearly positive when the target is a deterministic
    (plus light noise) function of an observable feature at a one-day lag.
    """
    panel = add_technical_indicators(_signal_panel(learnable=True)).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("SIGNAL", learnability_settings, progress=False)

    assert result.evaluation.n_samples >= 20, (
        "too few validation windows to trust the skill estimate"
    )
    assert result.evaluation.skill_vs_baseline > 0.3, (
        "expected clear positive skill on an injected learnable signal, got "
        f"{result.evaluation.skill_vs_baseline:.3f} -- the training pipeline may be "
        "silently broken (normalisation, feature ordering, target scaling)"
    )


def test_model_does_not_find_skill_in_pure_noise(monkeypatch, learnability_settings):
    """The degenerate control: skill must not be implausibly positive when the target
    is genuinely unpredictable. A pipeline that finds skill in noise has a leak
    somewhere in the split/scaling machinery, and this test catches it without needing
    to know where.
    """
    panel = add_technical_indicators(_signal_panel(learnable=False)).dropna()
    monkeypatch.setattr(tft, "build_panel", lambda *a, **k: panel)

    result = tft.train("NOISE", learnability_settings, progress=False)

    assert result.evaluation.n_samples >= 20, (
        "too few validation windows to trust the skill estimate"
    )
    assert result.evaluation.skill_vs_baseline < 0.15, (
        "expected no meaningful skill on pure noise, got "
        f"{result.evaluation.skill_vs_baseline:.3f} -- possible leak in the split/scaling path"
    )
