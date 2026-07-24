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

`check` is cheap enough to run before every backlog edit lands; there's no CI
hook for it yet (see PYQ-311 in investigations.md for whether that's worth
adding).

## Now

A hand-curated shortlist, not auto-generated — re-pick this after every
review pass. Full context for each is in its file. (Re-picked 2026-07-24 after
the second closeout pass below — PYQ-209/212 landed and dropped off; only the
experiment-heavy and decision-gated tickets remain.)

1. **PYQ-302** (investigation, High) — confirm what actually happens on
   schema drift between train-time and predict-time panels (the top blocker
   for trusting the PYQ-213 API against live data).
2. **PYQ-303** (investigation, High) — is a single 5-day validation window
   reliable enough to drive early stopping / model selection? Now unblocked:
   PYQ-210 (`seed_everything`) landed and PYQ-109 (best-checkpoint eval) is
   fixed, so the seed-variance comparison it asks for can finally be run on
   correctly-evaluated models.
3. **PYQ-220** (feature, Medium) — absolute bundle/cache paths; a stated
   prerequisite in the PYQ-213 design note before anything server-side. Needs
   a location decision (platformdirs XDG vs. project-root anchor).
4. **PYQ-211** (feature, Medium) — learning-rate tuning; re-run its backtest
   comparison now that PYQ-109 evaluates the right checkpoint (needs real data
   + GPU).
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
