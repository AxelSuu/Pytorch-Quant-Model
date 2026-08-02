# North Star

One page. Read this before picking work; don't redefine it as a side effect of a run.

---

## What this project is for, right now

PyQuant asks one question honestly: does a Temporal Fusion Transformer extract any
predictive signal from public daily equity data, at 5-day horizon, beyond a naive
persistence baseline? The project's credibility is the honesty of the answer, not the
size of the number. Current state: switching from a price-level to a log-return target
moved skill from **−23.5% to roughly +2.4%** — real, but measured on one symbol, ~25
predictions (effective n≈5). Not yet a result. `investigations.md#pyq-312` already named
the deliverable this project is actually building: a measurement apparatus that can tell
a real effect from seed noise, not another repo claiming an edge.

## What "good" looks like this quarter (through ~2026-10)

1. **Run the pre-registered sweep** (`features.md#pyq-268`, decision rule in
   `docs/methodology.md`) at the scale it specifies: N≥10 symbols across ≥3 sectors, K≥5
   seeds, a per-symbol paired interval excluding zero. The harness, seed tooling, paired
   test, baselines, and decision rule all exist as of 2026-07-29 — this is now a run, not
   a build.
2. **Fix `bugs.md#pyq-141`** (headline skill vs. per-window skill are different
   estimators) before trusting sweep output — the sweep will generate a lot more of both
   numbers.
3. **Act on the sweep's own decision rule.** Flip `use_sentiment` / `target` defaults
   only if the pre-registered bar is met. An unspecified threshold met after the fact is
   not evidence, per `investigations.md#pyq-322`.
4. **Rewrite `README.md` / `docs/index.md`** (`features.md#pyq-276`) to reflect whatever
   the sweep actually finds — explicitly gated on (1)–(3), the same discipline
   non-negotiable #1 already applies to `TrainingConfig.target`.
5. **Land the backlog → GitHub Issues migration and the supervised two-routine loop**
   (see `docs/autonomous-loop-plan.md`) without introducing pay-per-token spend or a new
   vendor credential.

## Non-goals (explicitly out of scope right now)

- **A fifth data vendor, alternative data, fundamentals, or options-implied history.**
  The project's own evidence (`investigations.md#pyq-312/#pyq-316/#pyq-321`) puts the
  mainstream prior at "no edge" and shows a fourth source already made things worse. A
  fifth input nobody can evaluate is not how this gets resolved. Revisit only once the
  sweep exists as a measured result.
- **Turning this into a product or a service.** Personal use, single maintainer. No
  external users, no SLA, no commercial framing.
- **Weakening non-negotiable #1** (`CLAUDE.md`): no metric improves without the model
  improving. This applies to the autonomous loop's own output exactly as much as to a
  human's.
- **API-billed automation.** The autonomous loop authenticates through Axel's Claude
  subscription (Routines) only — never an `ANTHROPIC_API_KEY`. See
  `docs/autonomous-loop-plan.md` constraint #1.
- **Full autonomy.** The loop proposes PRs and grooms the backlog; a human merges to
  `main` and approves anything touching secrets, spend, new vendors, or scope. See the
  five approval gates in `CLAUDE.md`.

## Where the detail lives

- Ticket-level reasoning: GitHub Issues (migrated from `backlog/*.md` — see
  `backlog/_archive/README.md` once the migration lands for where the old format went).
- Methodology and what would flip a default: `docs/methodology.md`.
- The autonomous-loop design and its constraints: `docs/autonomous-loop-plan.md`.
