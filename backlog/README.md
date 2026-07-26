# PyQuant backlog

Three files, one per ticket type. IDs never move between files and tickets are
never relocated within a file when they resolve — only their `Status:` line
changes. This means an ID always tells you the type (`1xx` bug, `2xx`
feature, `3xx` investigation) and a ticket's location in file/ID-order never
changes, so links like "see PYQ-109" stay valid forever.

- [`bugs.md`](bugs.md) — PYQ-1xx — concrete, reproducible defects.
- [`features.md`](features.md) — PYQ-2xx — things to build.
- [`investigations.md`](investigations.md) — PYQ-3xx — open questions that
  need reasoning/experimentation before they become a bug or feature ticket
  (or get answered and closed as-is).

## Format

Each file opens with a scan table (ID / Priority / Status / Title) for a
5-second overview, then full detail below in ID order. When you touch a
ticket, update *both* the table row and the detail block — `scripts/backlog.py
check` catches it if you forget one.

Status is one of: `Open`, `Resolved (<commit>, <date>)`, `Answered (<date>)`
(investigations only — the question was answered but didn't need a code
change), or `Superseded by PYQ-XXX`.

Priority: `Critical > High > Medium > Low`.

## Tooling

```bash
uv run python scripts/backlog.py list                       # all open tickets, sorted by priority
uv run python scripts/backlog.py list --type bug --priority critical,high
uv run python scripts/backlog.py check                      # table/detail consistency + duplicate-ID check
```

`check` runs in CI (PYQ-311) and is cheap enough to run before every backlog
edit lands.

## Now

A hand-curated shortlist, not auto-generated — re-pick this after every
review pass. Full context for each is in its file. (Re-picked 2026-07-26 after
the external review pass below, which argued the prior #1 pick (PYQ-211)
optimises inside a formulation that may be near-unbeatable by construction —
see PYQ-247. Weighted toward what needs no GPU sweep and no product decision,
since that's what actually blocked most of the old list.)

1. **PYQ-238** (feature, High) — `tests/test_invariants.py`. Six leaks found
   so far (PYQ-101/103/115/116/123/127) share one shape: correct in every file,
   wrong across files. `backlog/README.md`'s own recorded lesson after the
   third pass was that this backlog "optimised local correctness ticket by
   ticket while the invariants that span the pipeline went unstated" — this is
   the structural fix. No hardware, no keys, no product decision.
2. **PYQ-247** (feature, High) — forecast log-returns instead of price levels.
   `TARGET = "Close"` means the baseline (predict the last close) is
   near-optimal by construction for a near-random-walk level series, which
   means PYQ-117's −23.5% skill may be close to what this formulation predicts
   *a priori*, largely independent of hyperparameters. The change most likely
   to move the headline number, and — unlike LR tuning — needs no GPU sweep to
   try.
3. **PYQ-129** (bug, Critical) — sentiment is joined to the UTC calendar date
   a headline was published on, so post-close headlines (the most
   market-moving ones) leak into the same day's training row. Same class as
   PYQ-101, small and self-contained, and the last known member of that leak
   family.
4. **PYQ-248** (feature, High) — conformal calibration of the quantile band.
   PYQ-117 measured 99.3% coverage on a nominal 80% band — the interval is so
   wide it is close to uninformative, and nothing currently fixes that
   (PYQ-227 only diagnoses it). Split-conformal is distribution-free, needs no
   retraining, and is roughly 80 lines.
5. **PYQ-232** (feature, High) — Sphinx + autodoc site. 79% docstring
   coverage already exists and is unusually good (cites ticket IDs, explains
   *why*); none of it is rendered anywhere. The expensive part is done.
6. **PYQ-301** (investigation, Medium) — how much of the training window has
   non-neutral sentiment? Was blocked on a `FINNHUB_API_KEY`; both that key
   and a `FRED_API_KEY` are now configured locally (2026-07-26), so this and
   any other key-gated ticket can actually be run rather than just reasoned
   about. Promoted from "blocked" rather than ranked purely on value.

**PYQ-211** (LR tuning) is demoted out of Now — see its 2026-07-26 update for
why, and PYQ-253 for the ticket likely to supersede it.

## History

This backlog started as a single `pyquant_backlog.md` (2026-07-23 code
review, then a 2026-07-24 research + review pass) that was never committed.
It was split into this `backlog/` structure on 2026-07-24, before its first
commit, once the project moved to an almost-fully-agentic workflow motivated
a queryable, per-ticket-status format over a single growing file with a
manual prose archive. No ticket content was lost in the split — every ID from
the original file has a matching entry here. A second review pass the same
day (2026-07-24), specifically targeting CLI/data/models/tests/CI/docs,
added PYQ-110..114, PYQ-218..223, and PYQ-311 — bringing the total to 48
tickets, `scripts/backlog.py check`-clean.

A closeout pass (2026-07-24) then resolved 16 tickets in one batch — every
open bug (PYQ-109..114), six features (PYQ-210, 213, 216, 218, 219, 222, 223),
and three investigations (PYQ-306 answered, PYQ-309/311 resolved). Highlights:
the critical PYQ-109 (metrics now come from the deployed best checkpoint, not
the discarded final-epoch model) and the PYQ-213 FastAPI design note
([`docs/api-design.md`](../docs/api-design.md)). The change added 15 tests
(90 → 105 passing), a `LICENSE`, and a CI backlog-check step; still
`check`-clean. 13 tickets remain open (the larger/experiment-heavy ones:
PYQ-209/211/212/214/215/217/220 and investigations PYQ-301/302/303/308/310).

A second closeout pass (2026-07-24) resolved 6 more: PYQ-209 (YAML experiment
configs, layered below env via pydantic-settings' YAML source, with two example
`configs/`), PYQ-212 (`--format json`/`--quiet` + a reusable
`analysis/serialize.py`), PYQ-215 (a dependency-free retry helper on
`fetch_prices`/`fetch_news`), PYQ-221 (`pyquant cache list/prune/rm-pin`), and
investigations PYQ-308 (offline FinBERT-mapping fixture; real-model CI job
declined) and PYQ-310 (mypy surveyed — nothing real at a cheap setting, so
local-only `[tool.mypy]`, no CI gate). Added 17 tests (105 → 122 passing) and
`pyyaml` as a dependency; still `check`-clean. 7 tickets remain open —
PYQ-211/214/217/220 (features) and PYQ-301/302/303 (investigations) — all
needing real API keys, GPU hardware, Docker, or a product decision to close
responsibly.

A third review pass (2026-07-26) audited models/data/analysis/CLI/config/
tooling/project-management together and added PYQ-115..128 (bugs) and
PYQ-224..231 (features) — 24 tickets, bringing the total to 70. It then closed
18 of them, including all four defects that decided whether the tool did what
it claimed:

- **PYQ-115** (Critical) — `forecast` was predicting the last five
  *already-observed* days rather than the next five, because `predict=True`
  anchors the decoder to the end of the frame it is handed. Every number
  `forecast`/`scan`/`explain` printed was affected; `expected_return_pct` was a
  residual on known prices presented as a prediction. Live before/after on the
  same NVO bundle: `+2.73%` off medians for 2026-07-17..23 became `−5.31%` off
  medians for 2026-07-24..30.
- **PYQ-117** (High) — every reported metric, plus `EarlyStopping` and
  `ModelCheckpoint`, rested on a single 5-point validation sample, which is
  where "directional accuracy 100.0%" came from. Now 56 windows / 280 points at
  the default config, and the honest numbers are much worse (57.5% direction,
  −23.5% skill). Superseded investigations.md#pyq-303.
- **PYQ-127** (High) — `backtest --windows 5` trained five models and scored
  all five on the *same* final five days, so the walk-forward never walked.
- **PYQ-116** (Critical) — pooled training computed its cutoff from a global
  `time_idx` while numbering each symbol's rows from zero independently, so a
  late-listing symbol's validation window sat inside the training slice.

Two of those (PYQ-115, PYQ-127) were invisible from inside any single file —
they only surface if you ask what `decoder_time_idx` actually contains, which
is why the previous two passes found neither. The lesson recorded here: this
backlog was optimising local correctness ticket by ticket while the invariants
that span the pipeline went unstated. PYQ-115/117/127 now each ship a test that
asserts one of those invariants directly.

Also closed: PYQ-118/119 (schema drift now a clear error naming the missing
source, and bundles record the config they were trained with, answering
investigations.md#pyq-302), PYQ-121 (`RSI_14` was a simple moving average, not
Wilder's RSI), PYQ-123 (a `bfill` back-filling future values into leading
rows), PYQ-124 (crossed quantile bands reaching `scan`'s BUY/SELL guards),
PYQ-120/128/231 (CLI failure paths — writing the missing tests is what found
both defects), and PYQ-122/125/126/224/225/226.

The change added 47 tests (122 → 169 passing), still `check`-clean and
`ruff`-clean. Nine tickets remain open — PYQ-211/214/217/220/227/228/229/230
and investigation PYQ-301 — every one needing real API keys, GPU hardware,
Docker, or a product decision to close responsibly.

An external review pass (2026-07-26, same day, separate session) audited the repo cold —
full source read, backlog audit, git-history audit, partial test execution, and ecosystem
research — and scored it 7.5/10 overall: "a 9/10 software-engineering artifact wrapped
around a 5/10 quantitative result." Its output landed as three root-level files
(`backlog_adds.md`, `review.md`, `systems_research.md`) and was merged wholesale into this
structure as **49 new tickets** — PYQ-129..136 (bugs), PYQ-232..263 (features), PYQ-312..320
(investigations) — bringing the total to **119**, still `check`-clean.

The headline finding: `dataset.TARGET = "Close"` predicts the price **level**, and for a
near-random-walk series the persistence baseline is close to unbeatable on that
formulation by construction — so PYQ-117's −23.5% skill may be largely explained by the
target choice rather than by hyperparameters. That reframes PYQ-211 (learning-rate tuning,
the previous #1 "Now" pick) as optimising inside a near-uninformative formulation, and
promotes PYQ-247 (log-return target) above it; PYQ-211 is downgraded to Low in place and
cross-referenced to PYQ-253 (Optuna search, which proposes absorbing its scope) rather than
closed outright, since nobody has actually run the comparison yet. Other recurring threads:
the six-times-repeated "correct in one file, wrong across files" leak shape now has a
proposed structural fix (PYQ-238) instead of a seventh regression test; the 99.3%-on-80%
coverage figure has a proposed direct fix (PYQ-248, conformal calibration) rather than only
a diagnosis (PYQ-227); and one more look-ahead leak was found in the one source PYQ-305's
convention was never extended to (PYQ-129, sentiment joined by UTC calendar date rather
than publication time).

Also during this session: `FRED_API_KEY` and `FINNHUB_API_KEY` were both configured
locally for the first time, which is why PYQ-301 moved from "blocked" to the re-picked
`## Now` list above — it was the one existing open ticket whose stated blocker (a Finnhub
key) is now gone. Neither key's *value* was read or logged anywhere, per this file's own
non-negotiable on secrets; only presence was checked.

No ticket content from the external review was filtered out in the merge — all 49 are
recorded as proposed (`Open`), to be triaged and worked in priority order like PYQ-115..128
were, not treated as already-decided. The one substantive edit to an existing ticket was
PYQ-211's priority/cross-reference, above.
