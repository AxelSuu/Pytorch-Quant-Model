# PyQuant autonomous backlog loop — plan and decisions

**Repo:** `AxelSuu/Pytorch-Quant-Model`
**Owner:** Axel Sundqvist (personal use only — not a commercial product)
**Status:** Design confirmed 2026-08-02. Build not yet executed — see "Build checklist"
for what's actually landed vs. still manual.

This is the working spec for an unattended, subscription-only agentic loop that grooms
this repo's backlog, implements tickets, runs tests/CI, and reports back to Axel. It
supersedes nothing in `CLAUDE.md`'s existing non-negotiables — it is bound by them.

---

## 0. Constraints

1. **Subscription billing only.** Every piece of automation here authenticates through
   Axel's Claude subscription (OAuth), never through an `ANTHROPIC_API_KEY`.
2. Personal use, not commercial. No legal/liability surface beyond what already applies
   to a personal research repo.
3. **Supervised autonomy, not full autonomy.** The loop runs unattended for routine work
   (bug fixes, features, investigations, tests, docs, backlog grooming) but stops and
   asks a human for anything touching secrets, external services/vendors, spend, or scope
   changes.

---

## 1. Which Claude features to use, and how

### 1.1 Claude Code Routines — the core of the loop

Routines are Anthropic's built-in "put Claude Code on autopilot" feature: a saved
prompt + one or more repos + triggers, executed on Anthropic-managed cloud
infrastructure. No VM, no server, nothing to host. Currently in research preview.

- Create/manage at `claude.ai/code/routines`, from the Claude Desktop app ("Routines" in
  the sidebar → New routine → **Remote**, not Local), or from the CLI with `/schedule`.
- **Billing:** routines draw down the same subscription usage as an interactive Claude
  Code session (the rolling 5-hour + weekly window) — no separate API bill. There is a
  daily cap on routine-run starts; if that or the usage window is hit, a run is skipped,
  not billed extra.
- **Two routines:**
  1. A **schedule-triggered "dev" routine** on this repo — see confirmed cadence below.
  2. A **schedule-triggered "PM + report" routine**, nightly, code-untouched.
- Each run is a full autonomous session — no permission prompts mid-run. The cloud
  environment for the routine must never contain real secrets (section 6).
- Routines can also carry a **GitHub trigger** (fires on PR opened/closed, labeled,
  etc.) — worth adding later for "react when CI fails on a PR" without waiting for the
  next scheduled run.

### 1.2 The Claude GitHub App (repo access, not an MCP connector)

Not installed yet. This is a GitHub App, distinct from an MCP "connector" — Anthropic's
infrastructure needs it installed on the repo so Routines/Claude Code on the web can
clone it, push `claude/`-prefixed branches, and (for GitHub-trigger routines) receive
webhook events.

- Install it the first time a routine or GitHub trigger is set up — the setup flow
  prompts for it. `/web-setup` in the CLI can also grant it, but that path alone does
  **not** enable webhook delivery for GitHub triggers.
- Scope the App to only this repo, not "all repositories."

### 1.3 Claude Code CLI, locally, for supervision

Keep using the interactive CLI (or Cowork) for anything needing human judgment: reviewing
PRs, updating `NORTH_STAR.md`, tuning `CLAUDE.md`, and the one-time GitHub Issues
migration in section 2. Not automated.

### 1.4 What NOT to use for this project, and why

- **Claude Agent SDK.** Right tool for hand-written custom orchestration, but its
  supported auth path is an `ANTHROPIC_API_KEY` — pay-per-token billing, which violates
  constraint #1. Revisit only as a deliberate, budgeted decision if Routines prove too
  limited.
- **Claude Code GitHub Action with `CLAUDE_CODE_OAUTH_TOKEN`.** Can run on subscription
  billing (`claude setup-token` on a Pro/Max login produces a long-lived OAuth token), and
  is worth adding later for `@claude` mentions in PR/issue comments — but it shares the
  *same* 5-hour/weekly usage window as interactive sessions. Don't wire it into the main
  loop; keep it optional and low-frequency.

---

## 2. Backlog migration: `backlog/*.md` → GitHub Issues + Projects

Current: `backlog/bugs.md`, `backlog/features.md`, `backlog/investigations.md` (163
tickets as of 2026-07-29, most Resolved), plus `scripts/backlog.py`.
Target: GitHub Issues as the single source of truth, a Projects (v2) board for status,
Milestones for grouping.

### 2.1 Schema — **confirmed as proposed, 2026-08-02**

| Old file | New representation |
|---|---|
| `bugs.md` entries | Issues with label `type:bug` |
| `features.md` entries | Issues with label `type:feature` |
| `investigations.md` entries | Issues with label `type:investigation` |

Additional labels:
- Priority: `P0` (Critical), `P1` (High), `P2` (Medium), `P3` (Low)
- Status: `status:backlog`, `status:ready`, `status:in-progress`, `status:blocked`
  (only applied to still-open tickets; Resolved/Answered/Superseded tickets are migrated
  as closed issues so history isn't lost, with no `status:*` label)
- `needs-human` — the loop's way of flagging "a person needs to decide something"

### 2.2 Project board

One GitHub Project (v2): `Backlog → Ready → In progress → In review → Done`, plus a
"Priority" field mirroring `P0–P3`.

### 2.3 Milestones

Loose, time-boxed groupings (e.g. "the multi-symbol sweep") — not hard sprint deadlines.

### 2.4 Migration mechanics — **confirmed 2026-08-02: archive, don't delete**

1. `scripts/migrate_backlog_to_issues.py` (added alongside this plan) parses the three
   `.md` files with the same table/detail regexes as `scripts/backlog.py`, plus the full
   per-ticket body, and calls `gh issue create` — one issue per ticket, `Migrated from:
   bugs.md#PYQ-NNN` preserved in the body, closed immediately (with the original
   resolution note as a closing comment) if the ticket's status isn't Open.
2. Run it once, interactively, with Axel watching — `gh` needs to be installed and
   authenticated in whatever shell runs it. **This cannot be run from a Cowork sandbox
   session** (no `gh` auth, no reliable shell against this repo's mount as of 2026-08-02
   — see the session note in this file's git history / PR that added it). Run it from a
   local terminal or an interactive Claude Code session instead.
3. After verifying the issues look right, move `backlog/bugs.md`, `backlog/features.md`,
   `backlog/investigations.md`, and `scripts/backlog.py` to `backlog/_archive/` with a
   short `README.md` explaining they're superseded by GitHub Issues. **Do this only after
   migration is verified** — do not archive speculatively.
4. Update `CLAUDE.md` to point at GitHub Issues + Projects instead of `backlog/*.md` once
   (3) is done.

---

## 3. The feedback loop, in detail

Two routines, two jobs, kept separate so each run's diff and reasoning stays reviewable.

### 3.1 Routine A — "dev"

**Cadence — revised 2026-08-02:** the original confirmed cadence (2–3 runs/day, 6–7h
apart) turned out to structurally conflict with a second goal Axel raised the same day —
keeping routine runs from eating into the same rolling 5-hour subscription usage window
he wants available for his own interactive daytime use. Three runs spaced 6–7h apart
spans 12–14h, which cannot fit inside typical overnight hours without spilling into the
morning. Resolution: **2 runs/day, 6h apart, both overnight** (e.g. 01:00 and 07:00
local) as the safer default, with an explicit escalation path back to 3x/day once a week
of `/usage` / the routines page's daily-cap headroom confirms there's room. Budget
remains 5–10 tickets per run, a ceiling not a target.

**Verified working 2026-08-02:** a live run (via "Run now") produced PR #188 from
`claude/blissful-feynman-6sdv36` — picked 5 tickets (no `status:ready` queue existed yet,
so it fell back to hand-picking by priority, see the Routine B fix below), correctly
declined 2 out-of-budget tickets with stated reasoning, added 5 tests (475 passing),
relabeled addressed issues to `status:in-progress`, opened the PR without merging. Axel
merged it himself; all 3 required CI checks passed. The mechanics work end to end.

1. **Trigger fires** (schedule). Fresh cloud session, repo cloned at `main`.
2. **PM phase (read-only):** read open Issues, `NORTH_STAR.md`, last CI run. Pick up to
   the run's ticket budget, preferring P0/P1 and `status:ready`.
3. **Dev phase:** implement on a `claude/`-prefixed branch (leave the GitHub App's
   branch restriction on). Run the existing test suite locally before pushing.
4. **Push + PR:** open a PR referencing the Issue(s), relabel them `status:in-progress`.
5. **CI:** existing GitHub Actions (pytest, sphinx build, nightly smoke test, vendor
   tests) run unchanged.
6. **If CI fails:** the next dev-routine run reads the failure and either fixes it or
   opens a `needs-human`-labeled Issue rather than guessing repeatedly. (No GitHub-trigger
   routine for this: the current GitHub-trigger event surface only covers Pull request and
   Release events, not a direct "check failed" event, so faster reaction comes from
   cadence, not a third routine.)
   **Filing a new issue from a scratch file (added 2026-08-07, `#212`):** when an issue's
   body is written to a scratchpad file first, pass it with `gh issue create --body-file
   <path>` (or the REST API's equivalent of reading the file's contents into the `body`
   field) — never `gh issue create --body "@<path>"`. `@file` expansion is specific to
   flags documented per-command (`--body-file`, `-F key=@file`); `--body` has no such
   expansion, so `--body "@<path>"` posts the literal path string as the issue body. #210
   was filed this way and ended up with a body consisting of nothing but the unexpanded
   path, discovered only because PR #209 happened to discuss #210 at length elsewhere.
7. **If CI passes:** PR sits ready for Axel's review. The loop never merges to `main`
   itself.

### 3.2 Routine B — "PM + report"

Once daily, code-untouched, scheduled after Routine A's last nightly run finishes (e.g.
08:30 if A's last run is 07:00) so the report reflects settled state.

1. Reads Issues opened/closed since last run, PRs merged/opened, CI history.
2. Grooms the backlog: re-labels stale priority, closes irrelevant issues with a
   comment, opens new issues for things noticed (failing tests without a ticket, TODOs,
   coverage gaps) — always scoped by `NORTH_STAR.md`, never inventing new direction.
   **Promotes well-scoped `status:backlog` issues to `status:ready`** — added
   2026-08-02 after finding zero issues had ever carried `status:ready`, which meant
   Routine A had no groomed queue to draw from and always fell back to hand-picking.
3. Flags anything needing Axel: new external dependency, cost/vendor decision, anything
   needing a secret it doesn't have. Labels the Issue `needs-human`, surfaces it
   prominently.
4. Writes the daily report (3.3) and posts it — creating the "Daily Reports" Discussion
   thread itself on first run if it doesn't exist yet (added 2026-08-02 after finding the
   thread had never been created, and the original prompt had no fallback for that).
5. Never pushes code, never opens PRs.

### 3.3 The report — **destination confirmed 2026-08-02: pinned GitHub Discussion**

One thread, one comment per day. Contents:
- What shipped (merged PRs) and what's open (PRs awaiting review)
- CI health (pass/fail trend, anything flaky)
- Backlog delta (opened/closed/reprioritized, with the *why* for anything non-obvious)
- Anything labeled `needs-human`, listed explicitly
- One honest line on direction: does the week's activity match `NORTH_STAR.md`, or is it
  drifting into busywork?

---

## 4. Axel's role and the approval gates

Axel moves from operator to **director + reviewer**:
- Maintains `NORTH_STAR.md` — the loop reads it every run and works *within* it, never
  redefines it.
- Reviews PRs before merging to `main`.
- Reads the daily report, especially `needs-human` items.
- Approves or rejects anything flagged.
- Periodically tunes `CLAUDE.md` / routine prompts based on transcripts.

**Human approval gates (non-negotiable — also stated in `CLAUDE.md` so the agent
self-enforces them):**

1. Merging any PR to `main`
2. Adding any new external dependency, API, or vendor
3. Anything that would cost money
4. Anything touching secrets, credentials, or `.env`-style config
5. Any request to change branch protection or the routine's own permissions

---

## 5. Security

- **No `ANTHROPIC_API_KEY` anywhere in this project.** If one shows up in a `.env`, a
  GitHub secret, or a routine's environment variables, remove it.
- **Optional `@claude`-in-comments Action:** token from `claude setup-token`, stored only
  as an encrypted Actions secret named `CLAUDE_CODE_OAUTH_TOKEN`, never committed, never
  echoed in logs.
- **Routine cloud environment:** only what tests need. No broker/exchange keys, payment
  provider keys, or production DB credentials — the project stays unable to touch real
  money or real user data even in principle.
- **Branch protection on `main`:** require PR review + existing CI checks, keep the
  GitHub App restricted to `claude/`-prefixed branches.
- **GitHub App scope:** this repository only.
- **Secret scanning:** confirm it's enabled so an accidental credential gets caught
  before merge.

---

## 6. Build checklist — status as of 2026-08-02

1. ☑ Audit `backlog/*.md`, design label/Project/Milestone schema — done, see §2.1.
2. ☑ Write + run the one-time migration script; verify Issues; archive old backlog files
   — **done 2026-08-02** (all tickets migrated, verified on GitHub, old files moved to
   `backlog/_archive/`). `CLAUDE.md` and `ci.yml`'s "Backlog consistency" CI step (which
   would otherwise fail now that `scripts/backlog.py` no longer exists at that path) were
   both updated to match.
3. ☑ Write `NORTH_STAR.md` — done.
4. ☑ Update `CLAUDE.md` with the routine role split, ticket budget, approval gates,
   pointer to `NORTH_STAR.md`, and the post-migration Issues/Projects pointer — done.
5. ☐ Install the Claude GitHub App on this repo only — **manual, Axel**. Happens
   automatically as part of step 7 (connecting the repo to a routine prompts for it).
6. ☐ Branch protection on `main`, confirm secret scanning + push protection, enable
   Discussions — **manual, Axel**, mostly via `gh` — see chat for exact commands (not
   duplicated here to avoid drift between two copies).
7. ☐ Create Routine A ("dev") via `claude.ai/code/routines` or Desktop "Routines" —
   **manual, Axel**. Prompt ready at `docs/routines/dev-routine-prompt.md`. Cadence:
   2–3x/day, 6–7h apart, 5–10 tickets/run.
8. ☐ Create Routine B ("PM + report") — **manual, Axel**. Prompt ready at
   `docs/routines/pm-report-routine-prompt.md`. Nightly, posts to a pinned GitHub
   Discussion.
9. ☐ Let it run two weeks untouched except PR review + report reading.
10. ☐ Revisit: tune prompts, decide on `@claude` Action / higher frequency / GitHub
    triggers.

---

## Reference

- Routines: `https://code.claude.com/docs/en/routines.md`
- Claude Code costs & subscription usage windows: `https://code.claude.com/docs/en/costs.md`
- GitHub Actions integration: `https://code.claude.com/docs/en/github-actions.md`
- Claude Code on the web / GitHub auth: `https://code.claude.com/docs/en/claude-code-on-the-web.md`
- Agent SDK overview (not used here, kept for reference): `https://code.claude.com/docs/en/agent-sdk/overview.md`
