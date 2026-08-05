# CLAUDE.md — PyQuant agent operating manual

Place this at the repo root. Claude Code reads it automatically at session start.

---

## What this project is

A probabilistic equity forecasting harness. It assembles a leak-audited daily panel from
four vendors (Yahoo Finance, FRED, Finnhub, sector ETFs), trains a Temporal Fusion
Transformer via pytorch-forecasting / Lightning, and serves p10/p50/p90 forecasts plus
feature-importance interpretation from a Rich terminal UI.

**The forecaster does not currently beat a naive persistence baseline** (−23.5% skill,
99.3% coverage on a nominal 80% band, measured over 280 predictions). This is known,
deliberately reported, and is the central open problem. Do not "fix" it by tuning numbers
into looking better — see *Non-negotiables* below.

Python 3.10–3.12, `uv`-managed, MIT.

**`NORTH_STAR.md`** (repo root) is the short, human-maintained statement of what "good"
looks like this quarter and what's explicitly out of scope. Read it before picking work —
this file describes *how* to work; `NORTH_STAR.md` says *what's worth working on*.

---

## Commands

```bash
uv sync --extra dev                        # install
uv run pytest -q                           # full suite (network-free, ~169 tests)
uv run ruff check .                        # lint — must stay clean
uv run ruff format --check .               # formatting — a separate CI gate from lint
gh issue list --label status:ready         # open, ready-to-pick-up tickets

uv run pyquant train AAPL                  # train → checkpoints/AAPL/
uv run pyquant forecast AAPL               # 5-day quantile forecast
uv run pyquant explain AAPL                # importances + attention
uv run pyquant backtest AAPL --windows 5   # walk-forward
uv run pyquant --format json forecast AAPL # machine-readable
```

`ruff check` and `ruff format --check` are two separate CI jobs (`.github/workflows/ci.yml`)
— run both before proposing a commit. `ruff check --fix` does not fix formatting; a
`ruff check`-clean tree can still fail the `Format` step, run `ruff format .` separately.

---

## Architecture (and why it is shaped this way)

```
pyquant/
  config.py          pydantic-settings. Precedence: CLI flags > env > .env > YAML > defaults.
  data/
    prices.py        yfinance OHLCV + technical indicators
    macro.py         FRED series (with publication lags) + VIX
    sectors.py       sector ETF returns
    sentiment.py     Finnhub headlines → local FinBERT
    options.py       options snapshot (DISPLAY ONLY — not a model input)
    dataset.py       build_panel → panel_to_long → make_dataset
    cache.py         TTL panel cache + TTL-exempt named pins
    retry.py         dependency-free exponential backoff
  models/tft.py      ALL pytorch-forecasting / Lightning calls live here
  analysis/          forecast.py, interpret.py, metrics.py, serialize.py — library-agnostic
  cli/app.py         Typer + Rich. A thin caller over analysis/ and models/.
```

Two structural rules that the codebase depends on — do not violate them:

1. **pytorch-forecasting and Lightning are confined to `models/tft.py` and
   `data/dataset.py`.** Nothing in `analysis/` or `cli/` may import them. This is what
   makes a FastAPI layer additive rather than a rewrite (`docs/api-design.md`).
2. **`analysis/` and `models/` never import Typer or Rich.** The CLI is one of two planned
   front-ends.

---

## GitHub Issues (labels are the source of truth for status)

Ticket-level work lives in GitHub Issues, migrated 2026-08-02 from the old
`backlog/{bugs,features,investigations}.md` (see `docs/autonomous-loop-plan.md` §2 for
the migration design). The old markdown files and `scripts/backlog.py` are archived at
`backlog/_archive/` — historical record only, no longer live or enforced by CI.

Original `PYQ-NNN` IDs are preserved: each migrated issue's title is prefixed
`[PYQ-NNN]` and its body ends with a `Migrated from: bugs.md#pyq-nnn`-style footer, so an
old cross-reference like "see PYQ-109" still resolves — search Issues for the ID
(`gh issue list --search "PYQ-109 in:title"`).

Labels (rules enforced by convention, not a script — there's no `backlog.py check`
anymore):

- Type: `type:bug`, `type:feature`, `type:investigation`
- Priority: `P0` (was Critical) … `P3` (was Low)
- Status: `status:backlog`, `status:ready`, `status:in-progress`, `status:blocked`
- `needs-human`: a routine or a person flagging something only Axel can decide

**Labels are the only source of truth for a ticket's status.** There is also a GitHub
Project (v2) board (`https://github.com/users/AxelSuu/projects/4`), but it is a
human-facing view only — nothing reads or writes it. Do not move cards on it, do not
treat its Status field as authoritative, and do not let it drift out of sync with labels
by trying to keep both updated. Query status with labels: `gh issue list --label
status:ready`, `gh issue edit <n> --add-label status:in-progress --remove-label
status:ready`, etc.

### Resolution notes are the point

A resolution note is not a changelog entry. Look at PYQ-111, PYQ-115, PYQ-117 for the
standard. A good one states:

- what was actually changed, in terms of behaviour rather than diff;
- **the decision made**, when the ticket left one open (PYQ-111 chose a behaviour rather
  than just patching a symptom);
- verification evidence — real before/after numbers where possible (PYQ-115 quotes live
  NVO figures either side of the fix);
- the names of the tests that now guard it;
- anything the fix invalidates elsewhere (PYQ-121 noted it redefines a feature, so older
  bundles are no longer comparable).

Write these. They are the most valuable artifact in the repo.

### When work reveals something new

File a new GitHub Issue rather than fixing it inline and moving on. If an investigation's
premise turns out to be wrong, close it with a comment explaining why and linking the
issue that supersedes it — PYQ-303 (now a migrated, closed issue) is the model for this.
Do not silently delete or rewrite history.

---

## Non-negotiables

**1. Never make a metric look better without making the model better.**
The project's credibility rests entirely on having replaced "directional accuracy 100.0%"
with an honest "57.5% on 280 points." If a change improves a reported number, verify the
improvement is real before reporting it. If it turns out a past number was wrong, say so
loudly and supersede the tickets that relied on it — that is what PYQ-307 did.

**2. Look-ahead leakage is the top risk in this codebase.**
Six leaks have been found and fixed (PYQ-101, 103, 115, 116, 123, 127) and one is
outstanding (PYQ-129). Every one was *correct in each individual file and wrong across
files*. Before any change to `data/` or to split geometry, ask explicitly: **could a row
at time t now see information that did not exist at time t?** If touching splits, ask what
`decoder_time_idx` actually contains — twice, that question was the entire bug.

**3. Do not weaken a test to make it pass.**
If a test starts failing, the default assumption is that the code broke. PYQ-120 exists
because 18 CLI tests all asserted `exit_code == 0` and therefore tested nothing about
failure. A test that cannot fail is worse than no test.

**4. Every claim in the README must be measured, not expected.**
The README currently states pooling's benefit as a rationale rather than a result
(PYQ-315 is about fixing that). Do not add more of these.

**5. Do not add a dependency without justifying it against doing nothing.**
PYQ-310 declined mypy in CI on evidence. PYQ-308 declined a real-FinBERT CI job on
evidence. That disposition is a feature. New dependencies need a recorded reason.

---

## Autonomous loop (Routines) and human approval gates

This repo is operated partly by two unattended Claude Code Routines, in addition to
interactive sessions. Full design: `docs/autonomous-loop-plan.md`. Summary, because this
is the part that must be self-enforced with no human in the loop at run time:

**Role split.** Routine A ("dev") reads open work, implements a bounded batch on a
`claude/`-prefixed branch, runs the test suite, and opens a PR — it never merges to
`main`. Routine B ("PM + report") grooms backlog/Issues and posts a daily report — it
never touches code or opens PRs. Keep these separate; do not let a PM-phase run start
editing files, and do not let a dev-phase run redefine priorities beyond what
`NORTH_STAR.md` already states.

**Per-run ticket budget (Routine A).** Up to 5–10 tickets per run, preferring P0/P1 and
issues already marked ready. This is a ceiling driven by session length, not a target —
stop and open the PR once the batch is done rather than stretching to hit the number.

**The five human approval gates — non-negotiable, apply to every routine run:**

1. Merging any PR to `main`.
2. Adding any new external dependency, API, or vendor.
3. Anything that would cost money.
4. Anything touching secrets, credentials, or `.env`-style config.
5. Any request to change branch protection or the routine's own permissions.

If a run's work would require crossing one of these, stop, label the relevant Issue
`needs-human`, explain why in the Issue body, and move on to the next item in the batch
rather than guessing or waiting idle.

**Source of truth for tickets.** GitHub Issues + Projects, migrated 2026-08-02 — see
"GitHub Issues + Projects are the source of truth for work" above.
`backlog/*.md`/`scripts/backlog.py` are archived at `backlog/_archive/` and are no longer
read or enforced by anything, including CI.

---

## Testing conventions

- **Network-free.** Every external call is mocked. `pytest` must pass offline.
- **Behaviour-named.** `test_walk_forward_window_validation_targets_its_own_origin`, not
  `test_backtest_2`. The name states the property being asserted.
- **Test-first for bug fixes.** Write the failing test, confirm it fails against the
  unfixed code, then fix. PYQ-231 found two real defects (PYQ-120, PYQ-128) purely by
  writing failure-path tests — that is the argument for the practice.
- **Assert the invariant, not the output.** Prefer "every predicted step is beyond the
  last observed bar" over "the first value is 45.43."
- Shared fixtures are in `tests/conftest.py`: `sample_ohlcv_df` (400 synthetic business
  days, seeded) and `settings` (all enrichments off, small windows, cache disabled).
- Torch-dependent files: `test_tft`, `test_dataset`, `test_cli`, `test_forecast`,
  `test_interpret`. The rest run without the ML stack.

---

## Invariants the pipeline must satisfy

These are currently guarded by scattered regression tests; PYQ-238 proposes consolidating
them. Treat this list as normative regardless.

1. No training row contains information dated after that row's own timestamp.
2. Indicator warm-up rows are genuinely NaN and are dropped, never filled.
3. `predict=True` decodes timesteps strictly *after* the last observed bar.
4. The prediction encoder ends on the last observed bar.
5. Pooled symbols share one calendar: the same `Date` ⇒ the same `time_idx`.
6. Every validation decoder index exceeds the training cutoff, for every group.
7. Consecutive walk-forward origins evaluate disjoint windows, each starting at
   `cutoff + 1`.
8. Forecast dates in the table, the JSON, the PNG and the appended rows are one set.
9. A `Forecast` cannot exist with a crossed quantile band (`Forecast.__post_init__`).
10. Reported metrics come from the *best* checkpoint, not the live post-fit model.

---

## Conventions

- **Comments explain why, not what.** The existing style cites ticket IDs inline (68
  references across `pyquant/`). Keep doing this — it is how the code and the backlog stay
  connected.
- **Graceful degradation is a contract**, but only for *training*. A source that is
  missing, rate-limited, or disabled is dropped with a logged notice. At *predict* time a
  missing trained feature is a hard, clearly-worded error (`FeatureSchemaMismatch`),
  because a model cannot run without the columns it was trained on.
- **Config over constants.** Anything a user might tune belongs on `TrainingConfig` /
  `TFTConfig` / `DataConfig` with a comment explaining the default. Hardcoded literals in
  `models/tft.py` have generated four tickets (PYQ-218, 223, 224, and PYQ-211).
- **Reproducibility has three legs:** seed (PYQ-210) + pinned data (PYQ-205) + code
  version (PYQ-225). A change that breaks any one of them needs a ticket. Note the
  outstanding hole in PYQ-133.
- **Secrets never enter `meta.json`, `runs.jsonl`, logs, or cache fingerprints.** Key
  *presence* is fingerprinted; key values are not.
- Line length 100. Ruff rules `E, F, I, UP, B`, ignoring `E501` and `B008`.

---

## Working on a ticket

1. Read the ticket in full, plus every ticket it cross-references. They are dense and
   usually contain the reasoning that makes the fix obvious.
2. Reproduce the problem first. Several tickets include a reproduction (PYQ-115 and
   PYQ-127 both print the exact arrays). Confirm you see the same thing before changing
   anything.
3. Write the failing test.
4. Fix it.
5. Run `ruff check .`, `ruff format --check .`, `pytest -q`. (`scripts/backlog.py check` is
   archived along with `backlog/*.md` — see "GitHub Issues" above — and no longer applies.)
6. Update the ticket: `Status:` line, the scan-table row, and a resolution note to the
   standard above.
7. If the fix changed a model input or a reported metric, say so explicitly in the note
   and check whether it invalidates any other ticket's evidence.

**Do not close a ticket you cannot verify.** Several open tickets require a GPU, a real
API key, or a product decision — the correct action there is to say so and leave them
open, which is what previous passes did.

---

## Current state (2026-07-27, uncommitted pass on top of commit `90afcf8`)

- 124 tickets, 119 closed (Resolved/Answered/Superseded), **5 open**: PYQ-249
  (foundation-model baseline, deliberately deferred — see below), PYQ-217 (Dockerfile,
  deprioritised to Low by explicit user call, not attempted), and three Low-priority
  test-hardening tickets (PYQ-237 doctests, PYQ-242 property-based tests, PYQ-245
  mutation testing).
- Docs now live at <https://axelsuu.github.io/Pytorch-Quant-Model/> (PYQ-264): rebuilt on
  every push to `main` (`.github/workflows/docs.yml`) and nightly against live upstream
  intersphinx inventories (`nightly.yml`'s `docs-drift` job). `methodology.md` carries all
  three configurations now — the default, PYQ-247's log-return comparison, and a freshly
  measured `backtest --windows 5` walk-forward number (+36.2%, but 5 origins/25 points,
  deliberately not promoted over the other two) — rather than the single stale figure a
  previous pass left it with.
- 320 tests passing (was 169 two passes ago, 251 at the start of this one), ruff-clean,
  backlog-check-clean.
- Model quality is now genuinely mixed rather than uniformly bad, and every number below
  carries the same caveat: **one symbol, tens of points, one run** — directional evidence
  at the "smoke-scale" depth this pass was deliberately scoped to, not a production
  verdict. Treat all of it the way PYQ-247 treats its own +2.4%: real, but not yet enough
  to move a default.
  - Pooling (investigations.md#pyq-315): measured *worse* than solo training on an
    AAPL+ARM comparison — the README's old "meaningfully more data" rationale is now a
    measured, corrected claim.
  - Feature ablation (investigations.md#pyq-316): technicals added ~nothing over
    price-only; macro and sectors each helped; **sentiment measurably hurt** (skill
    +0.045 → +0.018 with it added), consistent with bugs.md#pyq-140's mechanism
    (Finnhub's free tier delivers ~6 days of news, not 365 — `Sentiment` is 99.7%
    structural zeros). Both tickets recommend flipping `use_sentiment`'s default, both
    stop short of doing it on this sample size — a multi-symbol repeat is the
    prerequisite, same as PYQ-247's own target-format change was.
  - `explain`'s interpretation (investigations.md#pyq-314): permutation importance
    agrees with the TFT's own variable-selection weights on the single top feature, but
    only weakly (Spearman ρ ≈ 0.3) beyond it. `explain` now prints a caveat when the
    bundle's own recorded skill is non-positive.
  - Forecast latency (investigations.md#pyq-319): a cold call is ~98% vendor
    fetch/panel-build (~65s); the actual forward pass is under a second either way. This
    is now the empirical basis for `pyquant/api/`'s concurrency design, not a guess.
- `pyquant/api/` (PYQ-261) and `pyquant tune` (PYQ-253, Optuna) both landed this pass —
  see their resolution notes in `backlog/features.md` for what each deliberately did not
  build (a real job queue/object storage for the API; anything beyond a smoke-scale
  search for tuning).
- `FRED_API_KEY` and `FINNHUB_API_KEY` are configured locally (values never read, only
  presence checked, per the secrets non-negotiable).

An external review (2026-07-26) proposed 49 further tickets, merged into `backlog/` as
PYQ-129..136 (bugs), PYQ-232..263 (features), PYQ-312..320 (investigations) — see
`backlog/README.md`'s History for the full merge and implementation-pass summaries, and
the review's own reasoning in `review.md`/`systems_research.md`. Almost all of that batch
is now closed; `backlog/README.md`'s `## Now` list is the current hand-picked shortlist of
what is actually left, and is a shorter, more targeted read than this section for "what
should I work on next."