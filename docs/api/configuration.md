# Configuration reference

Every tunable in PyQuant lives on one of four pydantic models. The project convention
(CLAUDE.md, *Config over constants*) is that anything a user might reasonably change
belongs here with a comment explaining its default — hardcoded literals in
`models/tft.py` have generated four separate tickets.

**Precedence, highest first:** CLI flags → environment variables → `.env` →
YAML experiment file → built-in defaults.

The YAML source sits deliberately *below* the environment, so a checked-in experiment
config stays overridable per-run, and explicit CLI flags — applied after `load_settings()`
— win over everything. Nested values use a double-underscore delimiter, e.g.
`PYQUANT_TRAINING__MAX_EPOCHS=50` maps to `settings.training.max_epochs`.

```bash
uv run pyquant train AAPL --config configs/aapl_baseline.yaml
PYQUANT_CONFIG=configs/aapl_baseline.yaml uv run pyquant backtest AAPL
```

A `--config` path that does not exist is an error, not a silent fallback to defaults: the
underlying settings source treats a missing file as "no values to contribute", so a typo
used to train a completely different experiment than the one asked for — and record it in
`meta.json` as such.

:::{note}
The **Description** column below is read out of `pyquant/config.py` itself — an explicit
`Field(description=...)` if one is set, otherwise the `#` comment the source attaches to
that field. Fields whose description is blank have no explanation in the source; that gap
is reported here rather than papered over with invented prose.
:::

## Settings

```{eval-rst}
.. automodule:: pyquant.config
   :no-members:

.. pyquant-config-model:: pyquant.config.Settings
```

Secrets are optional, and their absence simply disables the corresponding enrichment. Note
that key *presence* is recorded in cache fingerprints and `meta.json`; key **values** never
are — not in logs, not in `runs.jsonl`, not in a fingerprint.

## TrainingConfig

```{eval-rst}
.. pyquant-config-model:: pyquant.config.TrainingConfig
```

The three fields most worth understanding before running an experiment:

`validation_days`
: Length of the scored holdout, in trading days. The number of validation windows is
  `validation_days - max_prediction_length + 1`, so a holdout of exactly one horizon yields
  a *single* window — five points driving every reported metric plus early stopping and
  checkpoint selection. That is where the project's discarded "directional accuracy 100.0%"
  came from. See {ref}`Methodology <sample-size>`.

`purge_horizon` / `embargo_days`
: Look-ahead control around every split. They shrink the *training* slice only, never the
  scored window. See {ref}`invariant 10 <invariant-purge-embargo>`.

`target`
: `"close"` predicts the price level; `"log_return"` predicts per-step log returns and
  reconstructs a price path for display. The default is deliberately conservative — see
  {ref}`Methodology <negative-result>` for why the
  choice matters more than any hyperparameter here.

## TFTConfig

```{eval-rst}
.. pyquant-config-model:: pyquant.config.TFTConfig
```

`quantiles` must be sorted ascending, and this is validated rather than assumed: the first
and last entries are used as the lower and upper calibration bounds, so an unsorted list
such as `[0.9, 0.1, 0.5]` would silently invert the band with no error. `0.5` must be
present, because every consumer of a forecast reads a median.

## DataConfig

```{eval-rst}
.. pyquant-config-model:: pyquant.config.DataConfig
```

Each enrichment flag is a *request*, not an assertion. A source only activates if it is
both enabled here and has the credentials and data it needs; otherwise it degrades
gracefully and its features are dropped with a logged notice. That contract holds for
training and deliberately stops at predict time — see
{ref}`Architecture <graceful-degradation>`.

`use_options` gates a display-only fetch. The options snapshot is never a model feature.

```{eval-rst}
.. autofunction:: pyquant.config.load_settings
```
