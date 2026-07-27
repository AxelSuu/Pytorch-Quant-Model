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

A hand-curated shortlist, not auto-generated — re-pick this after every review
pass. (Re-picked 2026-07-27 after a second implementation pass closed 16 more
tickets — PYQ-140/238/239/241/243/246/254/255/259/253/261 resolved,
PYQ-314/315/316/319 answered, PYQ-211 superseded by PYQ-253. Only **5 tickets
remain open in the entire backlog**, so this list is now close to exhaustive
rather than a curated subset of a much larger open set.)

1. **A multi-symbol repeat of PYQ-247's comparison**, now joined by a
   multi-symbol repeat of investigations.md#pyq-315/#pyq-316's feature and
   pooling findings. All three are the same shape of open question — one
   symbol, tens of points, one run said "the effect looks real," and none of
   them are yet at the sample size this project's own non-negotiable #1
   requires before changing a default (`TrainingConfig.target`,
   `DataConfig.use_sentiment`, and whether pooling is on by default). Still no
   ticket, because each is a *run* rather than a code change.
2. **PYQ-249** (feature, Medium) — time-series foundation-model baseline.
   Explicitly deferred to design-plus-stub in the 2026-07-27 pass (a genuinely
   heavy new dependency, judged not worth the install/runtime risk in that
   session) — `baselines.py`'s actual `chronos_baseline()` integration is
   still unwritten. The CLI plumbing this needs is otherwise ready.
3. **PYQ-217** (feature, Low) — Dockerfile. Deprioritised from Medium by an
   explicit user call the same pass; also genuinely blocked on verification —
   no sandbox used so far in this project's history has had a Docker CLI to
   confirm `docker build`/`docker run` against.
4. PYQ-237/242/245 (Low) — doctests, property-based tests, mutation testing on
   `analysis/metrics.py`. Never picked up across three passes; not urgent, but
   the only genuinely untouched items left besides the two above.

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

An implementation pass (2026-07-27) worked the 49 open tickets in priority order and closed
**28**, taking the backlog to 123 tickets with 21 open. It also filed three new bugs, two of
them found only by running the pipeline against live vendors rather than against its own
mocks.

The headline is **PYQ-247**. Switching the target from price level to log-return, on one
pinned dataset with the seed, epoch budget and window count held fixed, moved skill from
**−59.5% to +2.4%** (+3.8% with PYQ-250's purged splits) and calibration coverage from
**52% to 76–80%** against a nominal 80% band — the latter with no conformal correction at
all. The external review's central claim was right: the −23.5% headline was substantially a
property of the formulation, not of the hyperparameters. Two honest counterweights are
recorded with it. Directional accuracy *falls* from 80% to 52–56%, because "direction versus
the last close" is nearly free on a level target and a genuine coin-flip on returns — the
lower number is the true one, and it suggests the README's 57.5% was flattered the same way.
And the sample is one symbol, 25 predictions, effective n≈5, so the default target was
**deliberately left unchanged**; flipping it on this evidence is the move non-negotiable #1
forbids. investigations.md#pyq-312 records the reframing: the old negative number was mostly
measurement, the new near-zero number is probably real.

The two live-vendor bugs are the pass's other lesson. **PYQ-139** (Critical): PYQ-257's
ALFRED vintage fetch, shipped Resolved with a passing test, failed against the real FRED API
three separate ways — an unbounded realtime window rejected for exceeding 2000 vintages, a
`NaT` in the value column that took a whole series down on one market holiday, and a
`realtime_end` in the future whenever the caller's clock is ahead of FRED's. Every FRED
macro feature had silently vanished from every panel; only `VIX` survived, and graceful
degradation reduced a total vendor loss to one log line. **PYQ-140** (High, open): Finnhub's
free tier ignores `from` and returns ~6 days of news, not the ~365 the module documents, so
`Sentiment` is 99.7% structural zeros — investigations.md#pyq-301 had estimated 80%. Both
are the case features.md#pyq-243 argues: mocking at our own function boundary verifies our
logic against our own assumptions, which is half a test. **PYQ-138** (Low) was a third: a CLI
test that passed or failed depending on whether stdout was a terminal.

Also closed: **PYQ-137** reversed its own ticket's premise on measurement — `adjust=True`
does *not* remove the EMA seed bias, and against a full-history reference is 1.3–1.6x worse
than the status quo; truncation rather than the seed is the real error source, so the fix is
a four-span warm-up (MACD front-of-panel error 5.66% → 0.08% of its own magnitude, for 7.2%
of rows). **PYQ-248** shipped split-conformal calibration, verified to pull a 100%-coverage
band to within 5 points of nominal *and* to widen a too-narrow one, but defaulted **off**
because PYQ-247 showed the pathology was largely a symptom of the target. **PYQ-258** added a
`PriceProvider` protocol with a licensed Tiingo implementation and an executable schema
contract. **PYQ-263** added `pyquant doctor`, which exists because PYQ-139 was invisible.
**PYQ-232/233/234/235** rendered the docs (18 module pages, 47 pandas and 6
pytorch-forecasting intersphinx links, warning-clean under `-W`). **PYQ-229/230/236/244**
took CI to a 3.10–3.12 matrix with a frozen install, lockfile check, coverage reporting and
a nightly live-vendor smoke job.

What did **not** get done is worth stating as plainly. PYQ-238's invariant module, PYQ-239's
learnability test and PYQ-246's determinism test were all started and abandoned mid-flight;
PYQ-261 and PYQ-217 likewise. PYQ-252 landed CRPS, Winkler and PIT as numbers but the PIT
*histogram* is not rendered. PYQ-233's "CI fails on a deliberately broken cross-reference"
half was never executed, and PYQ-234's hosted build has no connected Read the Docs project.
PYQ-320 is a dated recorded judgement, not the clause-level terms review it asks for, and
PYQ-318's `neuralforecast` spike was not run. Each of those is recorded in its own ticket as
verified-in-part rather than claimed whole.

Final state: 251 tests passing (was 204), `ruff check` clean with the `D` ruleset newly
enabled, `scripts/backlog.py check` clean, `uv lock --check` clean, `pre-commit run
--all-files` clean across nine hooks, and the docs building warning-free.
