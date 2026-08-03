# Development

Everything here runs from a clone with `uv`. Nothing needs a GPU, an API key, or network
access — the test suite is offline by construction, and the two CI gates below are the same
commands a contributor runs locally.

```bash
uv sync --extra dev                        # install
uv run pytest -q                           # full suite, network-free
uv run ruff check .                        # lint — must stay clean
uv run python scripts/backlog.py check     # backlog consistency — must stay clean
```

`ruff check` and `backlog.py check` both gate CI. Line length is 100; the enabled rule sets
are `E, F, I, UP, B`, ignoring `E501` and `B008`.

## Testing conventions

**Network-free.** Every external call is mocked, so `pytest` passes offline. Real vendor
payloads are checked in as fixtures instead — see `record_fixtures.py` below.

**Behaviour-named.** `test_walk_forward_window_validation_targets_its_own_origin`, not
`test_backtest_2`. The name states the property being asserted, so a failure report reads
as a claim that stopped holding.

**Test-first for bug fixes.** Write the failing test, confirm it fails against the *unfixed*
code, then fix. This is not ceremony: writing failure-path tests for the CLI found two real
defects that success-path tests could not have, because 18 of those tests asserted
`exit_code == 0` and therefore tested nothing about failure.

**Assert the invariant, not the output.** Prefer "every predicted step is beyond the last
observed bar" over "the first value is 45.43." A test pinned to a number breaks on every
legitimate change and catches nothing structural. The full list of what must hold is
[Leakage invariants](invariants.md).

**Never weaken a test to make it pass.** If a test starts failing, the default assumption
is that the code broke. A test that cannot fail is worse than no test.

Shared fixtures live in `tests/conftest.py`: `sample_ohlcv_df` (400 synthetic business days,
seeded) and `settings` (enrichments off, small windows, cache disabled). `test_tft`,
`test_dataset`, `test_cli`, `test_forecast` and `test_interpret` need torch; `test_api`
needs the `api` extra and skips cleanly without it; the rest run with neither installed.

## Research scripts

`scripts/` holds the tooling that produced the measured numbers quoted throughout this
site. They are deliberately *not* part of the test suite — most need network, some need
hours — but they are how each result is reproduced rather than taken on trust.

| Script | Answers | Reported in |
|---|---|---|
| `compare_pooling.py SYM1 SYM2 …` | Is pooling helping? Scores a pooled model on each symbol's own validation slice separately, which `train()`'s aggregated metric cannot do. | `investigations.md#pyq-315` |
| `ablate_features.py [SYMBOL]` | Which of the 25+ features earn their place? Walk-forward backtests across cumulative feature groups, plus a correlation matrix naming the redundant technicals. | `investigations.md#pyq-316`, `bugs.md#pyq-140` |
| `profile_forecast.py [SYMBOL]` | Where does forecast latency actually go? Times a cold and a warm call, split into bundle-load / fetch+panel-build / predict. | `investigations.md#pyq-319`, [the API design note](api-design.md) |
| `runs.py compare` | Which of my last 30 runs scored best, and what did they have in common? Reads every bundle's `runs.jsonl` and lines the fields up. | `features.md#pyq-259` |
| `record_fixtures.py` | Refreshes the checked-in vendor payloads the contract tests read. Article text is replaced with placeholders — the tests care about response *shape*, not copyrighted copy. | `features.md#pyq-243` |
| `backlog.py check\|list` | Backlog consistency, and the open tickets priority-sorted. No dependencies; runs from a bare clone. | — |

`runs.py` is also an argument in itself: it is the ~100-line alternative that was shipped
instead of taking on `mlflow` as a dependency. New dependencies need a recorded reason —
mypy in CI and a real-FinBERT CI job were both declined on measured evidence, and that
disposition is a feature.

### `pyquant sweep`: the multi-symbol repeat these scripts couldn't do (PYQ-268)

`compare_pooling.py` and `ablate_features.py` are each hard-wired to one symbol (or two);
repeating either across fifteen symbols meant editing the script and reconciling the output
by hand — which is why the multi-symbol repeat both investigations name as their own
prerequisite sat un-run across two backlog review passes. `pyquant sweep --symbols
A,B,C --arm target=close --arm target=log_return --windows 5` (`pyquant/experiments/sweep.py`)
is the reusable instrument: it walk-forward backtests every symbol against every named
config-override "arm", reports per-symbol and pooled skill, a "helped N of M symbols"
summary, and a paired significance comparison (`compare_backtests`) between the first two
arms per symbol — and a symbol that fails for one arm degrades to a recorded gap rather than
taking the sweep down. Neither existing script was rewritten over it: `ablate_features.py`'s
correlation-matrix analysis has no sweep-harness equivalent, and `compare_pooling.py`'s
pooled model is one bundle trained once and sliced per symbol, not `N` independent
per-symbol walk-forward runs — a different measurement shape the harness does not produce.
Running the harness against the three pending repeats is separate, later work; this landed
the tool, not a result (see `features.md#pyq-268`'s resolution note).

## The backlog

`backlog/` is the source of truth for work: `bugs.md` (`PYQ-1xx`), `features.md`
(`PYQ-2xx`), `investigations.md` (`PYQ-3xx`). Read `backlog/README.md` before editing any
of them — the rules below are enforced by `backlog.py check`, not stylistic.

- **IDs never move between files, and tickets never move within a file.** Only the
  `Status:` line changes. That is what makes "see PYQ-109" a permanent link.
- **Every ticket edit touches both the scan-table row and the detail block.**
- Status is exactly one of `Open`, `Resolved (<commit>, <date>)`, `Answered (<date>)`
  (investigations only), or `Superseded by PYQ-XXX`.

A resolution note is not a changelog entry. It states what changed in terms of *behaviour*,
the decision made when the ticket left one open, verification evidence with real
before/after numbers, the names of the tests that now guard it, and anything the fix
invalidates elsewhere. These notes are the most valuable artifact in the repo, and they are
why the docs can cite a ticket instead of restating an argument.

When work reveals something new, file a ticket rather than fixing it inline. If an
investigation's premise turns out to be wrong, mark it superseded and explain why in place
— history is not silently rewritten.

## Building these docs

```bash
uv sync --group docs --extra api
uv run --group docs --extra api sphinx-build -W --keep-going -b html docs docs/_build/html
uv run --group docs --extra api sphinx-autobuild docs docs/_build/html   # live reload
```

The `-W` is not decoration. A docs site not built with warnings-as-errors rots within two
refactors: a renamed module, a dead `:func:` reference or an autodoc import failure all
degrade silently otherwise. CI gates this exact command on every pull request, and a
separate workflow rebuilds and publishes on merge to `main`. A nightly job re-resolves the
intersphinx inventories against live upstream documentation, so a cross-reference that
upstream removes is caught by a scheduled run rather than by a reader.

`--extra api` is required, not optional: `pyquant/api/` imports fastapi at module level,
autodoc imports every module it documents, and `fail_on_warning` would fail the whole build
over one section most readers never open.

Colour tokens live in `docs/conf.py`'s theme options and structure lives in
`docs/_static/custom.css`, so a value is defined once per theme rather than once per rule.

## Releasing

`pyproject.toml`'s `version` only asserts something about a bundle's provenance (PYQ-225,
PYQ-133) if it actually points at a commit. To cut a release:

1. Update `CHANGELOG.md` (Keep a Changelog format) and bump `version` in `pyproject.toml`
   to match, in the same PR.
2. Once that PR is merged to `main`, tag the merge commit and push the tag:
   ```bash
   git tag v0.3.0
   git push origin v0.3.0
   ```
3. `.github/workflows/release.yml` runs on any `v*` tag push: it verifies the tag matches
   `pyproject.toml`'s version (failing loudly if they disagree, PYQ-274), runs the full
   test suite, and cuts a GitHub release with auto-generated notes.

There is no PyPI publish step. This project has no external consumers, so a release channel
with nobody downstream is exactly the unjustified addition non-negotiable #5 is about — the
tag + GitHub release is the whole point: a reproducible pointer from a version string to a
commit, nothing more.

## Things that will get a change rejected

**Never make a metric look better without making the model better.** This project's
credibility rests on having replaced "directional accuracy 100.0%" with an honest "57.5% on
280 points." If a change improves a reported number, verify the improvement is real before
reporting it. If a past number turns out to be wrong, say so loudly and supersede the
tickets that relied on it.

**Look-ahead leakage is the top risk here.** Every leak found so far was correct in each
individual file and wrong across files. Before any change to `data/` or to split geometry,
ask explicitly: *could a row at time t now see information that did not exist at time t?*
If you are touching splits, ask what `decoder_time_idx` actually contains — twice, that
question was the entire bug.

**Every claim in the README and in these docs must be measured, not expected.** A rationale
for why something should help is not a result. Pooling was described that way once; it is
now a measured — and corrected — claim.

**Comments explain why, not what**, and cite ticket IDs inline. That convention is how the
code and the backlog stay connected, and there are dozens of such references across
`pyquant/`.
