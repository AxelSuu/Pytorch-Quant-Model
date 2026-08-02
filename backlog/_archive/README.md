# PyQuant backlog — ARCHIVED 2026-08-02

**Superseded by GitHub Issues + a Project (v2) board.** This file and the three ticket
files beside it are historical record only — nothing here is live, and CI no longer runs
`backlog.py check` against them. Every ticket below was migrated to a GitHub Issue with
its original `PYQ-NNN` ID preserved in the issue title and a `Migrated from:` footer in
the body, so old cross-references still resolve (search Issues for the ID). See
`CLAUDE.md`'s "GitHub Issues + Projects are the source of truth for work" and
`docs/autonomous-loop-plan.md` §2 for the migration this reflects.

The rest of this file is preserved unedited below — it's the real history of how this
backlog worked and evolved, not something to keep maintaining.

---

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

A hand-curated shortlist, not auto-generated — re-pick this after every review pass.
Re-picked 2026-07-29. **163 tickets, 24 open, zero Critical/High.**

### The decision this list encodes

The project reached a fork: keep hardening what exists, or start adding vendors, features
and integrations. It is worth recording that the answer came from the backlog itself rather
than from taste.

investigations.md#pyq-312 already settled what this project *is*. Its recorded conclusion —
"the deliverable should be reframed around the measurement apparatus… the honest headline is
now 'no detectable edge after fixing the formulation'" — chose the measurement apparatus and
a rigorous negative result over another repo claiming edge. It then named the single thing
blocking that reframing: **the multi-symbol repeat.**

That item carried at #1 across three passes without starting. This pass (2026-07-29)
finished the entire measurement apparatus it was blocked on — every Critical/High ticket
that stood between "stalled" and "able to run the repeat and mean it" is now closed. What is
left is the repeat itself, which is a run, not a ticket, and bugs.md#pyq-141 (demoted to the
top Medium item, see below).

**Adding data vendors is still the worst available move right now.** investigations.md#pyq-312
puts the mainstream prior at no edge for 5-day single-name direction from public daily data.
investigations.md#pyq-316 measured that adding a *fourth* source made things worse, not
better. investigations.md#pyq-321 (below) measured the seed-noise floor and found PYQ-316's
effect survives it, but with less margin than PYQ-247's — not a reason to add a fifth source
chasing an edge the project's own analysis still says is probably absent.

### Finished this pass (2026-07-29) — the full measurement apparatus

All nine Critical/High tickets closed in one session:

- **investigations.md#pyq-321** (Critical) — the seed-variance floor. Measured, not just
  built: AAPL, 10 seeds, smoke-scale config. `log_return` skill sd ≈ 0.0114 — PYQ-247's
  effect survives it 54x over, investigations.md#pyq-316's sentiment delta survives it 2.4x
  (per the ticket's own stated bar), and investigations.md#pyq-315's **AAPL** pooling delta
  does *not* clear one sd — flagged as a real caveat on that specific number, not a
  supersession. `target=close` (the actual default) showed 18x more skill variance and
  *zero* variance in directional accuracy across all ten seeds — a smoke-scale artifact
  worth a closer look, not evidence about the full-scale published headline. Full numbers in
  the ticket and in `docs/methodology.md`'s "Seed variance" section.
- **bugs.md#pyq-142/143/144** (High) — three interacting corrections to what a `log_return`
  bundle displays, what `train()`/`walk_forward_backtest()` select checkpoints against, and
  how the conformal offset (PYQ-248) is fit. All three change what a re-measured headline
  would read as "more honest," not "better" — expect every number to look worse once
  re-measured, per non-negotiable #1.
- **features.md#pyq-265/266/267** (High) — multi-seed backtest reporting
  (`pyquant backtest --seeds N`), paired significance testing (`compare_backtests`), and the
  per-horizon-step breakdown. The instruments PYQ-321 needed, and needed them first.
- **features.md#pyq-275** (High) — `analysis/baselines.py`: four comparators beyond
  persistence (random-walk-with-drift, seasonal-naive, climatological, a hand-rolled AR(1) —
  `statsmodels`' ARIMA/ETS declined per non-negotiable #5).
- **features.md#pyq-268 + investigations.md#pyq-322** (High) — the sweep harness
  (`pyquant sweep`, `pyquant/experiments/sweep.py`) and the pre-registered rule for what
  evidence flips a default (`docs/methodology.md`'s "What it takes to flip a default"):
  N≥10 symbols/3 sectors, per-symbol `effective_n_samples`≥10, K≥5 seeds, a per-symbol
  paired interval excluding zero, an explicit "helped 11/hurt 4" resolution, and a higher
  bar for `target` (blast radius) than for `use_sentiment`.

None of the three pending repeats (PYQ-247's target comparison at multi-symbol scale,
investigations.md#pyq-315/#pyq-316's pooling and feature findings) were actually re-run —
that remains a run, not a ticket, and needs live vendor-data access at a scale (N≥10 symbols
x K≥5 seeds) this pass's smoke-scale investigations did not attempt. It is, for the first
time, fully unblocked: the harness, the seed tooling, the paired test, the baselines, and the
decision rule all now exist.

### What is left

1. **bugs.md#pyq-141** (Medium) — `backtest`'s headline skill and the per-window column
   beneath it are different estimators that already disagree by 71 points in
   `docs/methodology.md`. The one item that was already Phase-1-adjacent and is still open;
   fix before running the sweep in earnest, since the sweep will generate a lot more of both
   numbers.
2. **Run the sweep**, against the decision rule, at the scale it specifies. This is what
   everything above was for.
3. **features.md#pyq-276** — execute PYQ-312's reframing in `README.md`/`docs/index.md`.
   Explicitly gated on (2): rewriting the README on pre-sweep evidence is the same move
   non-negotiable #1 forbids for `TrainingConfig.target`.
4. **features.md#pyq-249** (Medium) — foundation-model baseline, sharing PYQ-275's baseline
   interface. Worth running once (2) exists to say whether it beat anything.
5. Structural and coverage work, none of it urgent and none of it blocking:
   features.md#pyq-269 (split the 1075-line `models/tft.py`), #pyq-272 (dedicated tests for
   `serialize`/`doctor`/`provenance`/`charts`), #pyq-273 (the four PYQ-139/140 failure-mode
   regressions on PYQ-243's existing harness), #pyq-271 (`/backtest` endpoint), #pyq-270
   (interval on the headline skill), #pyq-277 (Tiingo selectability), bugs.md#pyq-149
   (signal-scoring calibration mismatch), investigations.md#pyq-323 (`Settings` coupling),
   #pyq-324 (does the forecast band fan or translate?).
6. **PYQ-217** (Low) — Dockerfile. Deprioritised by explicit user call in the 2026-07-27
   pass, and blocked on verification — no sandbox in this project's history has had a Docker
   CLI to confirm `docker build`/`docker run` against.
7. Remaining Low tickets (PYQ-151/153/154/159/237/242/245/274) — doctests, property-based
   tests, mutation testing, a CHANGELOG/release workflow, and a handful of small bugs. Never
   picked up across five passes; not urgent.

### Not now, and why

**New data vendors, alternative data, fundamentals, options-implied history.** See the
reasoning above. The gate is Phase 2: once a sweep can say "this source moved skill by X,
and the seed floor is Y," adding a source becomes a measurable proposition instead of a
hopeful one. Until then it is unfalsifiable work. PYQ-254 is already accumulating options
snapshots against the day this changes, which is the correct shape for vendor work right
now — cheap, additive, and costing nothing while the question is unanswerable.

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

A review pass (2026-07-28) added **15 tickets** — bugs.md#pyq-141,
features.md#pyq-265..274, investigations.md#pyq-321..324 — taking the backlog to
**139 tickets, 20 open**, still `check`-clean. It closed nothing: it was a
gap-finding pass over the source, the docs, the existing backlog and a generated
knowledge graph of the repo, not an implementation pass.

Its organising observation is that this project has a well-developed apparatus
for *reporting* numbers honestly and almost none for *deciding whether a
difference is real*. The two are not the same discipline, and the second is now
the binding constraint. Concretely: `TrainingConfig.seed` is fixed at 42, so
every headline the project has published is one draw from one seed and the
seed-to-seed spread has never been measured (investigations.md#pyq-321).
Configurations are compared by eyeballing two point estimates, with no paired
test on the windows they share (features.md#pyq-266). Skill — the number in the
README, in `docs/methodology.md` and in `explain`'s warning banner — carries no
confidence interval, while directional accuracy, the metric PYQ-247 showed to be
*flattered* by the level target, does (features.md#pyq-270). And every metric is
a mean over h=1..5, so a model whose skill rises with horizon and one whose skill
collapses at h=5 produce the same headline (features.md#pyq-267).

That reframes the standing `## Now` #1. Two passes carried "a multi-symbol repeat
of PYQ-247/#pyq-315/#pyq-316" with the note that it had no ticket "because each
is a *run* rather than a code change." That reasoning is why it never started —
it is not purely a run, because no tool performs it. `scripts/ablate_features.py`
and `scripts/compare_pooling.py` are both self-described one-off scripts, each
wired to one question and between them to one or two symbols; repeating either
across fifteen symbols means editing a script and reconciling output by hand,
which is exactly why three findings that each name a multi-symbol repeat as
their prerequisite have all sat un-repeated. It is now features.md#pyq-268 (the
harness), #pyq-266 and #pyq-265 (the statistics), and investigations.md#pyq-322
— which asks the project to write down *what result would flip a default* before
running the sweep rather than after. An unspecified threshold is a deferred
decision, not a conservative one, and it fails in both directions: never met, or
met retrospectively by whichever run looks convincing.

One real defect was found by reading `analysis/metrics.py` against `cli/app.py`.
**bugs.md#pyq-141** (Medium): `backtest` prints an aggregate skill computed as a
ratio of `n_points`-weighted pooled MAEs, and directly beneath it a per-window
table whose skill column is a mean of per-window ratios. The two can diverge
without limit, and already do — `docs/methodology.md` records the level-target
per-window skills as `[+0.28, +0.47, +0.35, −2.71, −3.13]`, mean **−94.8%**,
beside a **−23.5%** headline, with nothing on screen or on the page reconciling
them. This is PYQ-136 one level up: that ticket fixed numerator and denominator
being computed two ways *inside* the aggregate; this is the aggregate and its own
detail rows being computed two ways.

The remaining tickets are structural and were found by inventory rather than
inference. `models/tft.py` is 1075 lines holding train, backtest, tune, predict,
interpret and the window geometry that produced PYQ-115/116/127/250 — the
project's four most expensive bugs — and the containment rule that causes the
accretion is worth keeping, so the fix is a package rather than a relaxation
(features.md#pyq-269). Four modules have no dedicated test file, and they are the
wrong four: `serialize.py` (the machine-readable contract), `doctor.py` (which
exists *because* PYQ-139 was invisible), `provenance.py` (PYQ-134 was a
provenance function resolving against the wrong directory) and `charts.py` (the
only leg of invariant 8 with no direct test) — features.md#pyq-272. Vendor tests
still patch at our own function boundary rather than replaying the recorded
payloads in `tests/fixtures/`, which is the half of the PYQ-139/140 lesson that
was diagnosed but never built (features.md#pyq-273). The API has no `/backtest`,
which is the one capability that would actually test `docs/architecture.md`'s
two-front-ends-one-core claim (features.md#pyq-271).

Two tickets came from a generated knowledge graph of the repository rather than
from reading it. `Settings` is the highest-betweenness node in the codebase
(0.092, bridging 28 communities) — investigations.md#pyq-323 asks whether passing
it whole costs more than it saves, and is deliberately framed as a question, not
a refactor, since PYQ-310's precedent is that declining on evidence is a valid
outcome. And an automated read of the committed `nvo.png` reports the forecast
median opening ~10% below the last observed close with a band that translates
rather than fans; that reading is unverified and possibly pre-PYQ-115, but the
figure is the README's most-viewed claim about the project's own output, so
investigations.md#pyq-324 asks for it to be confirmed or regenerated.

No existing ticket's status, priority or content was changed by this pass.

A follow-up pass the same day (2026-07-28) answered the question the previous entry left
implicit — *solidify or expand?* — and re-picked `## Now` as a gated three-phase plan
rather than a flat list. It added two tickets (features.md#pyq-275, #pyq-276), corrected
one filed hours earlier on a false premise, and changed no other ticket. 139 → 141
tickets, 22 open.

The answer came from the backlog rather than from taste. investigations.md#pyq-312 is
Answered and already chose the deliverable: "the deliverable should be reframed around the
measurement apparatus… the honest headline is now 'no detectable edge after fixing the
formulation', not 'negative skill'." It then named the one blocker — the multi-symbol
repeat — which is the same item `## Now` carried at #1 across two passes without starting.
So the project had committed to a deliverable and stalled on its single prerequisite, and
the correct move was to unstall that rather than to open a new front.

The case against expanding is the project's own evidence, and is recorded in `## Now` so it
does not have to be re-argued: #pyq-312 puts the mainstream prior at no edge for 5-day
single-name direction from public daily data; #pyq-316 measured that adding a *fourth*
source made things worse; and #pyq-321 is unanswered, so nothing here can currently
distinguish a +0.027 effect from seed noise. A fifth vendor would be an input nobody can
evaluate. Vendor work is gated, not forbidden — PYQ-254's accumulating options snapshots
are the right shape for it meanwhile: cheap, additive, costing nothing while the question
is unanswerable.

**features.md#pyq-275** is the pass's substantive addition and follows directly from
#pyq-312's choice. `analysis/` has no `baselines.py`; `persistence_baseline_mae()` is the
only comparator in the codebase. A negative result is a claim about the baselines it was
measured against, and persistence is near-optimal on a random-walk level series by
construction — so failing to beat it is weak evidence, and "does not beat persistence" is a
much weaker publishable claim than "does not beat anything a competent practitioner would
try." Which baselines the model beats and which it does not is more diagnostic than the
single signed number the project reports today. This is the groundwork the reframed
deliverable needs, and it is worth more than any new data source.

**features.md#pyq-276** exists because #pyq-312's reframing was tracked nowhere. An
Answered investigation records a conclusion, not an open action, so the project's decision
about what it *is* lived only inside a closed ticket while `README.md` continued to lead
with "Probabilistic equity forecasting with a Temporal Fusion Transformer." The ticket is
explicitly gated on the sweep: rewriting the README on n≈5 is the same move non-negotiable
#1 forbids for `TrainingConfig.target`.

**features.md#pyq-273 was corrected in place rather than deleted.** As filed hours earlier
it asked for boundary-level vendor replay tests on the reading that vendor tests still
patch at our own `fetch_*` boundary — but PYQ-243 shipped exactly that and is Resolved, with
six tests mocking at `yf.Ticker`/`yf.download`/`fredapi.Fred`/`requests.get` against
recorded fixtures. The residual is narrower and real: PYQ-243 landed in the same pass that
later produced PYQ-139 and PYQ-140, four live-vendor failure modes its happy-path
recordings could not exercise. The ticket now asks for those four as named regressions on
the harness that already exists. Recorded as a correction, per this file's own convention
that a wrong premise is superseded in place and explained rather than quietly rewritten.

A second external review pass (2026-07-28) audited the repo cold, structured around three
severity tiers rather than a full-source read, and its 20-ish findings were verified against
current source (not trusted at face value — several of its line numbers and one of its
priority claims had already drifted) before being merged as **22 new tickets** —
bugs.md#pyq-142..159, features.md#pyq-277..280 — bringing the total to **163, 44 open**.
Every finding was independently confirmed by direct code reading or a background research
pass before filing; one claim (`apply_conformal_offset`'s re-sort "contradicting" its own
docstring) was checked and refuted — the sort is deliberate, documented, and cites PYQ-124's
precedent — and was not filed. Nothing was filed on the review's word alone.

The two headline findings are both **High**, not Critical, calibrated against this
backlog's own bar (PYQ-115/116/139's total, silently-wrong output) rather than the
review's: **bugs.md#pyq-142** — `log_returns_to_prices`'s `cumsum` compounds each
quantile column independently down the horizon axis, so the *displayed* band for
`target="log_return"` bundles is ~√h too wide (verified by a 400k-path simulation matching
√1..√5 almost exactly, and by the fact that `tests/test_forecast.py
::test_log_return_price_round_trip` currently locks the buggy behavior in as intended). It
does not touch any number this project has actually published — `_evaluate_validation`
scores the correct raw per-step arrays, so PYQ-247's 76-80% coverage figure is unaffected —
only what `forecast`/`scan`/`explain` show and what `backtest --signals` scores. And
**bugs.md#pyq-143** — `train()` and `walk_forward_backtest()` both monitor
`EarlyStopping`/`ModelCheckpoint` against the identical loader that later becomes the
reported `EvaluationMetrics`, the exact "every trial is a selection event" bias
`TuneResult`'s own docstring already names and guards against one function over. Worst in
the backtest, where `predict=True` makes it a 5-point window doing both jobs at once.

Two more (both High) are security/correctness gaps in the `pyquant/api/` scaffold added by
PYQ-261: **bugs.md#pyq-145** — `_bundle_dir` joins an unvalidated `symbol`/`bundle_name`
straight into `mkdir`/`torch.load(weights_only=False)`, reachable from `POST /train` and
`POST /scan` with no path-escape guard; and **bugs.md#pyq-146** — `load_settings()`'s
module-global `_active_yaml_file` races under FastAPI's threadpool, letting one concurrent
request silently drop another's `--config` layer, the PYQ-128 failure mode reintroduced one
level up. **bugs.md#pyq-150** (High) is a secrets-non-negotiable violation: Finnhub's key
travels in the query string and a retried failure logs the full URL at WARNING, reaching CI
logs where the key is a real secret.

One finding corrects the backlog's own record rather than the code:
**features.md#pyq-277** — PYQ-258 is marked Resolved with acceptance criteria stating Tiingo
is "implemented **and selectable**," but no `DataConfig` field, CLI flag, or `build_panel`
argument makes it reachable outside a Python REPL. Per this file's convention, PYQ-258's
content and status were left untouched — a new ticket records the gap rather than
retroactively editing a closed one. **features.md#pyq-280** turns that same discovery into
a proposed extension of `scripts/backlog.py check`, so a Resolved ticket's named tests can
be verified to exist rather than taken on faith.

The remaining 16 are smaller, independently confirmed defects and hardening gaps spread
across `analysis/calibrate.py` (conformal offset pooled across horizon steps, and not
reusing PYQ-251's own `effective_sample_size` for the finite-sample correction — a fresh
instance of the "correct in one file, not applied in the analogous one" shape this backlog
keeps finding), `data/prices.py` (RSI reads 100, not 50, on a flat series), `analysis
/metrics.py` (PIT values clamp at the outer quantiles), `data/cache.py` (non-atomic writes),
`data/trading_calendar.py`, `config.py` (`extra="ignore"`, an unenforced `0.5 in quantiles`
invariant), `tests/conftest.py` (an `options_history_dir` hermeticity gap), and CI's
formatting-drift check (33 files now adrift, not the 20-22 baseline it was scoped against).
Full detail, evidence and acceptance criteria are in each ticket; none were resolved by this
pass — it was a verify-and-file pass, same discipline as the 2026-07-26 review's merge.

No existing ticket's status, priority or content was changed. The `## Now` list above was
not re-picked against these additions — several (PYQ-142/143/145/146/150) are plausible
candidates for it on a future pass, but that call is deliberately left to one, rather than
made as a side effect of filing.

A ninth pass (2026-07-29) closed the entire Critical/High backlog left after the previous
review — nine tickets in one session: bugs.md#pyq-142/143/144 (the forecast band's √h-wide
compounding, checkpoint selection scored against its own reported window, and the pooled
conformal offset — three bugs that composed with each other and were fixed together),
features.md#pyq-265/266/267/268/275 (multi-seed backtest reporting, paired significance
testing, per-horizon breakdown, the multi-symbol sweep harness, and baselines beyond
persistence — the full measurement apparatus PYQ-312 named as the blocker on its own
reframing), investigations.md#pyq-322 (a pre-registered rule for what evidence flips a
default, written before any run rather than after), and investigations.md#pyq-321 (the
seed-variance floor, Critical) — which, after an initial network check suggested it was
blocked, turned out not to be: `curl`'s default User-Agent was drawing an HTTP 429 from
Yahoo Finance, not the sandbox itself, and the project's actual `yfinance` client worked
fine. A real 10-seed, smoke-scale sweep against live AAPL data measured a `log_return` skill
sd of ≈0.0114 — PYQ-247's target-change effect survives it 54x over,
investigations.md#pyq-316's sentiment finding survives it (2.4x, per PYQ-321's own stated
bar) with less margin than PYQ-247, and investigations.md#pyq-315's AAPL-specific pooling
delta does *not* clear one sd, flagged as a caveat on that number rather than a
supersession. Every resolution note states plainly what real-data limitation applied and
what was deliberately deferred (multi-seed `train()`, `--seeds` support inside `pyquant
sweep`'s cells, PYQ-205 pin support in `walk_forward_backtest`) rather than rushed. 65 new
tests; `pyquant/experiments/` and `pyquant/analysis/baselines.py` are new modules. What
remains is bugs.md#pyq-141 (Medium) and then the sweep itself — a run, not a ticket, now
fully unblocked. The `## Now` list above was re-picked to reflect this.
