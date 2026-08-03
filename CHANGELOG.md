# Changelog

All notable changes to this project are documented here, in [Keep a
Changelog](https://keepachangelog.com/en/1.1.0/) form. This is a single-maintainer,
personally-run project rather than a distributed package (see `NORTH_STAR.md`'s
non-goals), so entries are grouped by pass rather than promised on any release cadence, and
`version` moves when a pass is judged worth marking, not on a fixed schedule.

This file **summarizes**; it does not duplicate. Per-ticket detail — the reasoning, the
decision made, the before/after numbers — lives in the resolution note on that ticket's
GitHub Issue (issues migrated from the old `backlog/*.md` files carry a `[PYQ-NNN]` title
prefix and a `Migrated from:` footer, so an old cross-reference still resolves). The full
prose account of every pass through 2026-07-29 is `backlog/_archive/README.md`'s History
section; this file starts a shorter, forward-looking version of the same record.

## [Unreleased]

## [0.3.0] - 2026-08-03

Everything recorded in `backlog/_archive/README.md`'s History since `0.2.0`, condensed.
`0.2.0` was never itself tagged or released — see "0.2.0" below.

### Added

- `pyquant/api/` (PYQ-261): a FastAPI service layer over the same `analysis`/`models` core
  the CLI uses, additive per the layering rule in `CLAUDE.md` — `forecast`/`explain`/`scan`
  (sync), `train`/`backtest` (async via a job queue), API-key auth, and per-bundle
  prediction locks. `GET /symbols` and `GET /metrics/{symbol}` (PYQ-283) followed, so a
  caller can discover what's trained instead of guessing a symbol and reading a 404.
- `pyquant tune` (PYQ-253): an Optuna-backed hyperparameter search.
- `TrainingConfig.target = "log_return"` (PYQ-247): an alternative to the default
  price-level target. Measured, on one symbol, to move skill from -59.5% to +2.4% — real,
  but on a sample too small to flip the default (non-negotiable #1 in `CLAUDE.md`).
- The pre-registered multi-symbol sweep harness and its decision rule (PYQ-268, PYQ-322,
  `docs/methodology.md`) — the measurement apparatus `NORTH_STAR.md` names as this
  project's actual deliverable, rather than a forecasting edge.
- `TrainingConfig.seeds` / `pyquant backtest --seeds N` (PYQ-265): seed-to-seed variance,
  measured rather than reported as if one draw were the model.
- `compare_backtests()` (PYQ-266): a paired, moving-block-bootstrap significance test
  between two configurations scored on the same walk-forward windows.
- Per-horizon-step metrics (PYQ-267) and MAE against baselines beyond persistence
  (PYQ-275).
- `pyquant --as-of` point-in-time simulation.
- A `PriceProvider` protocol with a licensed Tiingo implementation alongside yfinance
  (PYQ-258).
- `pyquant doctor` (PYQ-263), a pipeline health check — filed because PYQ-139 (below)
  demonstrated a total vendor failure could otherwise go unnoticed.
- Split-conformal calibration (PYQ-248), shipped but defaulted off once PYQ-247 showed the
  coverage pathology it targets was largely a symptom of the target formulation instead.
- Executable doctests on `analysis/metrics.py`/`analysis/forecast.py` (PYQ-237), run in CI
  as a documentation-drift guard.
- A tag-triggered release workflow and this file (PYQ-274).

### Changed

- Ticket tracking migrated from `backlog/*.md` + `scripts/backlog.py` to GitHub Issues;
  the old files are archived at `backlog/_archive/`, no longer read or enforced by
  anything, including CI.
- `with_retry`'s default no longer retries a definitively non-retryable 4xx (401 bad key,
  404 unknown symbol) — only connection/timeout/429/5xx are, with jitter added to the
  backoff so concurrent callers don't retry in lockstep (PYQ-151).
- CI moved to a 3.10-3.12 matrix with a frozen install, a lockfile-sync check, coverage
  reporting, and a nightly live-vendor smoke job.

### Fixed

- Six look-ahead leaks across `data/` and split geometry (PYQ-101, 103, 115, 116, 123,
  127) — see `CLAUDE.md`'s non-negotiable #2. One (PYQ-129) remains open.
- Reported metrics now come from the best checkpoint, not the live post-fit model
  (PYQ-109).
- PYQ-139 (Critical): the FRED vintage fetch failed against the real API three separate
  ways once exercised live rather than against its own mocks — an unbounded realtime
  window, a `NaT` on a market holiday, and a future `realtime_end` whenever the caller's
  clock is ahead of FRED's. Every macro feature had silently vanished from every panel;
  graceful degradation reduced a total vendor loss to one log line, but the loss itself
  was invisible until this was found.
- A MACD front-of-panel error (5.66% -> 0.08% of its own magnitude, 7.2% of rows),
  traced to truncation rather than the EMA seed itself (PYQ-137).

## [0.2.0] - unreleased marker

The version this project carried from its first commit through the passes summarized
above, never itself tagged. Covers the leak-audited daily panel (Yahoo
Finance/FRED/Finnhub/sector ETFs), the TFT train/backtest/forecast/explain pipeline, and
the Rich terminal UI — the full account is `backlog/_archive/README.md`'s History section.
