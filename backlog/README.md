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
the third closeout pass below. Every bug is closed; what remains needs
hardware, real API keys, or a product decision.)

1. **PYQ-211** (feature, Medium) — learning-rate tuning. Now the most
   valuable open ticket rather than a nice-to-have: PYQ-117 showed the default
   config scores −23.5% skill against a persistence baseline on 280 real
   validation points, and PYQ-127 made the walk-forward actually walk, so an
   lr sweep can finally be judged on trustworthy numbers. Needs real data +
   GPU.
2. **PYQ-227** (feature, Medium) — per-quantile calibration + pinball loss.
   Directly motivated by what PYQ-117 exposed: 99.3% coverage on a nominal 80%
   band says the interval is far too wide, and a single band-coverage number
   cannot say *which* side is at fault.
3. **PYQ-220** (feature, Medium) — absolute bundle/cache paths; a stated
   prerequisite in the PYQ-213 design note before anything server-side. Needs
   a location decision (platformdirs XDG vs. project-root anchor).
4. **PYQ-301** (investigation, Medium) — how much of the training window has
   non-neutral sentiment? The last open question about whether a feature the
   model is being fed is worth its place. Needs a `FINNHUB_API_KEY`.
5. **PYQ-217** (feature, Medium) — Dockerfile; pairs with the PYQ-213 design
   note for deploying the eventual API.

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
