"""Is ceil(windows/horizon) the right effective_n discount for walk-forward windows? (#202)

`EvaluationMetrics.effective_n_samples` (`effective_sample_size`, PYQ-251) discounts a raw
sample count by `ceil(n / horizon)`, calibrated for `train()`'s validation split: many
overlapping decode origins from *one* fit, where origin t and origin t+1 share horizon-1 of
their horizon scored days, and therefore highly correlated errors from the same model.

`walk_forward_backtest` windows are a different situation. PYQ-127 guarantees no two
windows' *decoded* days overlap -- so the correlation the formula was built to discount
does not exist in the walk-forward case. But #193 raised a real counter-argument the
methodology.md fix deliberately left open rather than deciding by fiat: at the default
`step = horizon`, two neighboring windows' *training* sets differ by only `step` days out
of however many are available -- near-identical models, not independent ones. That is a
genuine correlation source, just a different one (model-level, from shared training data)
that `ceil(n/horizon)` was never built to measure and does not correctly size either.

This script does not decide the question (per the ticket's own scope -- a synthetic
experiment or a literature read, not a quick code change, and *not* the one that gets to
pick whichever answer is cheaper for #190's compute budget, per non-negotiable #1's
spirit). It quantifies the mismatch: an AR(1) model of "how much would the walk-forward
mean-skill estimator's true sampling variance differ from what each of the two candidate
formulas implies," parameterized by (a) the real per-window training-overlap fraction at
PyQuant's default `step = horizon` and (b) a swept range for the one thing that can't be
derived analytically -- the ratio of shared-model variance to idiosyncratic decode-noise
variance, which is an empirical unknown until measured on real data.

Usage:
    uv run python scripts/simulate_walk_forward_effective_n.py
"""

from __future__ import annotations

import math

import numpy as np

N_TRIALS = 200_000
N_WINDOWS = 5
HORIZON = 5
SEED = 42


def training_overlap_fraction(n_train_rows: int, step: int) -> float:
    """Fraction of two neighboring windows' training rows that are identical.

    Each walk-forward window trains from scratch on every row available up to its own
    cutoff (models/tft.py's `walk_forward_backtest`); consecutive cutoffs are `step` days
    apart. Out of `n_train_rows` rows in the later window's training set, all but the
    newest `step` are also in the earlier window's -- this is exact, not a simulation
    input, for any panel long enough to run the backtest at all.
    """
    return max(0.0, (n_train_rows - step) / n_train_rows)


def ar1_mean_variance_multiplier(rho: float, n: int) -> float:
    """Var(mean of n AR(1)-correlated unit-variance draws) / (1/n).

    The closed-form inflation factor: 1 at rho=0 (independent, matches naive 1/n), and
    grows toward n as rho -> 1 (fully correlated draws don't average down at all, so the
    mean's variance stops shrinking with n -- the textbook "effective n collapses to 1"
    case, not a distinctive walk-forward property, whenever windows are that correlated).
    """
    if n <= 1:
        return 1.0
    return 1.0 + 2.0 * sum((1 - k / n) * rho**k for k in range(1, n))


def simulate_true_variance(
    rho: float, shared_frac: float, n_windows: int, n_trials: int, seed: int
) -> float:
    """Monte Carlo the mean-skill estimator's variance under the AR(1) model directly.

    Independent cross-check of `ar1_mean_variance_multiplier`'s closed form -- generating
    actual AR(1) sample paths rather than trusting the algebra alone.
    """
    rng = np.random.default_rng(seed)
    shared = np.empty((n_trials, n_windows))
    innovation_std = math.sqrt(max(1e-12, 1 - rho**2)) if rho < 1 else 0.0
    shared[:, 0] = rng.normal(size=n_trials)
    for i in range(1, n_windows):
        shared[:, i] = rho * shared[:, i - 1] + innovation_std * rng.normal(size=n_trials)
    idio = rng.normal(size=(n_trials, n_windows))
    draws = math.sqrt(shared_frac) * shared + math.sqrt(1 - shared_frac) * idio
    means = draws.mean(axis=1)
    return float(means.var(ddof=1))


def main() -> None:
    # PyQuant's actual default: step = horizon (models/tft.py `walk_forward_backtest`).
    # n_train_rows swept across a realistic range for the project's default `period`
    # (~1-3y of daily bars, minus encoder/purge/embargo/selection overhead) rather than
    # assumed -- the overlap fraction is what matters, and it saturates fast regardless.
    print(f"walk-forward default: step = horizon = {HORIZON}, n_windows = {N_WINDOWS}\n")
    print("Training-data overlap between neighboring windows, by available history:")
    for n_train_rows in (100, 250, 500, 1000):
        frac = training_overlap_fraction(n_train_rows, HORIZON)
        print(f"  {n_train_rows:5d} training rows -> {frac:.1%} shared with the next window")
    print(
        "\nAt any realistic history length this project runs with, neighboring windows'\n"
        "training sets overlap 95%+ -- treating them as independent models is the wrong\n"
        "prior on its face. rho (the AR(1) autocorrelation of *model quality* across\n"
        "windows) is swept below rather than derived from the overlap fraction directly:\n"
        "no closed-form link between 'fraction of rows shared' and 'correlation of the\n"
        "fitted model's generalization error' exists without real data to calibrate it.\n"
    )

    naive_n = N_WINDOWS
    current_formula_n = effective_sample_size(N_WINDOWS, HORIZON)
    print(f"Candidate effective_n for {N_WINDOWS} windows at horizon {HORIZON}:")
    print(f"  naive (full independence):        {naive_n}")
    print(f"  current ceil(windows/horizon):    {current_formula_n}\n")

    print(
        f"{'rho':>5} {'shared_frac':>12} {'true_mult':>10} "
        f"{'naive_implied_n':>16} {'current_implied_n':>18} {'true_effective_n':>17}"
    )
    for rho in (0.0, 0.3, 0.6, 0.8, 0.95, 0.99):
        for shared_frac in (0.25, 0.5, 0.75, 1.0):
            closed_form_mult = ar1_mean_variance_multiplier(rho, N_WINDOWS)
            # Combine the correlated "shared" component and the independent
            # "idiosyncratic decode noise" component (PYQ-127's non-overlap guarantee
            # means this half really is independent across windows, unlike the shared
            # half): Var(mean) = shared_frac * mult/n + (1 - shared_frac) * 1/n.
            true_var_over_1_over_n = shared_frac * closed_form_mult + (1 - shared_frac)
            true_effective_n = N_WINDOWS / true_var_over_1_over_n
            print(
                f"{rho:5.2f} {shared_frac:12.2f} {true_var_over_1_over_n:10.2f} "
                f"{naive_n:16d} {current_formula_n:18d} {true_effective_n:17.2f}"
            )

    print(
        "\nMonte Carlo cross-check (should match true_effective_n's column above closely):"
    )
    for rho, shared_frac in ((0.8, 0.5), (0.95, 0.75), (0.99, 1.0)):
        mc_var = simulate_true_variance(rho, shared_frac, N_WINDOWS, N_TRIALS, SEED)
        mc_effective_n = 1.0 / mc_var
        print(
            f"  rho={rho:.2f} shared_frac={shared_frac:.2f}: "
            f"Monte Carlo effective_n = {mc_effective_n:.2f}"
        )

    print(
        "\nReading: at low rho (windows behave almost independently despite shared\n"
        "training data -- plausible if the newest few days dominate what the model\n"
        "learns) true_effective_n approaches the naive count, and the current formula's\n"
        "discount to 1 is far too conservative. At high rho *and* a training-data-driven\n"
        "component dominating over decode noise, true_effective_n collapses toward 1 or\n"
        "below -- in that regime the current formula, if anything, is not conservative\n"
        "enough. Which regime PyQuant is actually in is not resolvable from priors alone;\n"
        "it depends on shared_frac and rho, neither of which this script can derive from\n"
        "first principles. Recommendation in the issue comment: use\n"
        "`walk_forward_backtest_multi_seed` (already built, PYQ-265) to measure the\n"
        "idiosyncratic (same-cutoff, different-seed) variance directly and compare it to\n"
        "the across-cutoff variance -- that decomposition is exactly shared_frac, measured\n"
        "on real data instead of assumed, and is cheap relative to #190's sweep."
    )


def effective_sample_size(n_samples: int, horizon: int) -> int:
    """Mirrors `pyquant.analysis.metrics.effective_sample_size` (not imported: this
    script is a standalone what-if simulation, deliberately not exercising the package
    under test -- see the module docstring)."""
    return math.ceil(n_samples / horizon)


if __name__ == "__main__":
    main()
