# PyQuant

```{eval-rst}
.. raw:: html

   <div class="pq-hero">
     <p class="pq-hero__lede">A probabilistic equity forecasting harness: a leak-audited
     daily panel from four vendors, a Temporal Fusion Transformer, and p10/p50/p90
     forecasts you can trace back to the ticket and the test that guard them.</p>

     <svg class="pq-hero__fan" viewBox="0 0 480 156" role="img"
          aria-label="A price history ending at the last observed bar, followed by a
                      p10-p90 forecast band that widens with each step ahead.">
       <path d="M300 72 C360 62 410 44 470 26 L470 124 C410 102 360 86 300 72 Z"
             fill="var(--color-brand-primary)" fill-opacity="0.13"/>
       <path d="M300 72 C360 62 410 44 470 26" fill="none"
             stroke="var(--color-brand-primary)" stroke-opacity="0.5" stroke-width="1.5"/>
       <path d="M300 72 C360 86 410 102 470 124" fill="none"
             stroke="var(--color-brand-primary)" stroke-opacity="0.5" stroke-width="1.5"/>
       <path d="M300 72 C360 68 410 66 470 64" fill="none"
             stroke="var(--color-brand-primary)" stroke-width="2.5"
             stroke-linecap="round"/>
       <path d="M300 14 L300 138" fill="none" stroke="var(--color-foreground-muted)"
             stroke-opacity="0.55" stroke-width="1.2" stroke-dasharray="3 4"/>
       <polyline points="10,104 30,96 50,108 70,88 90,95 110,78 130,86 150,70 170,80
                         190,64 210,74 230,60 250,68 270,58 300,72"
                 fill="none" stroke="var(--color-foreground-primary)" stroke-width="2"
                 stroke-linejoin="round" stroke-linecap="round"/>
       <g fill="var(--color-foreground-secondary)" font-size="11"
          font-family="ui-monospace, SFMono-Regular, Menlo, monospace">
         <text x="192" y="150">last observed bar</text>
         <text x="446" y="20">p90</text>
         <text x="446" y="60">p50</text>
         <text x="446" y="140">p10</text>
       </g>
     </svg>

     <div class="pq-result">
       <div class="pq-result__col">
         <p class="pq-result__eyebrow">default &mdash; target = "close"</p>
         <p class="pq-result__figure">&minus;23.5%</p>
         <p class="pq-result__metric">skill vs. a naive persistence baseline</p>
         <p class="pq-result__denominator">
           <b>56</b> walk-forward windows<br>
           <b>280</b> predictions, one symbol<br>
           57.5% directional &middot; 99.3% coverage on a nominal 80% band
         </p>
       </div>
       <div class="pq-result__col">
         <p class="pq-result__eyebrow">PYQ-247 &mdash; target = "log_return"</p>
         <p class="pq-result__figure">+2.4%</p>
         <p class="pq-result__metric">skill, everything else held fixed</p>
         <p class="pq-result__denominator">
           <b>5</b> walk-forward windows<br>
           <b>25</b> predictions, one symbol, effective n&thinsp;&asymp;&thinsp;5<br>
           52&ndash;56% directional &middot; 76&ndash;80% coverage, uncalibrated
         </p>
       </div>
     </div>

     <p class="pq-verdict">Neither number is promoted over the other, and the default has
     not been changed on the strength of the second one: n&thinsp;&asymp;&thinsp;5 is not
     this project's bar for changing what every user gets. Both are reported here for the
     same reason the first one is reported at all.</p>
   </div>
```

The default forecaster does not beat "predict no change" — that is the project's central
open problem, stated rather than tuned out of sight. For a near-random-walk series,
persistence is close to unbeatable on that formulation *by construction*; see
{ref}`negative-result` for why, {ref}`sample-size` for what these denominators do and do
not license, and {ref}`related-open-questions` for what the same discipline found when it
was applied to pooling, to the feature set, and to `explain`'s own interpretation.

```{eval-rst}
.. raw:: html

   <hr class="pq-band">
```

## Quickstart

PyQuant is managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync                          # core install
uv sync --extra sentiment        # + FinBERT news sentiment
uv sync --extra api              # + the FastAPI service

uv run pyquant train AAPL        # train a TFT   -> checkpoints/AAPL/
uv run pyquant forecast AAPL     # 5-day p10/p50/p90 forecast + fan chart
uv run pyquant explain AAPL      # which features and which days drove it
uv run pyquant backtest AAPL --windows 5
uv run pyquant --format json forecast AAPL
```

Defaults: 5 years of daily bars, a 60-day encoder, a 5-day horizon, p10/p50/p90, up to 30
epochs with early stopping, and a 60-trading-day validation holdout. Every one of those is
a field on {py:class}`~pyquant.config.Settings` — see the
[configuration reference](api/configuration.md).

All API keys are optional. PyQuant runs on Yahoo Finance OHLCV alone; each key lights up
one more source, and a source that cannot be reached is dropped with a logged notice
rather than failing the run. That contract, and the one place it deliberately stops, is
described in {ref}`Architecture <graceful-degradation>`.

## Where to go

::::{container} pq-cards

:::{container} pq-card
[Leakage invariants](invariants.md)

The most useful page here. Every look-ahead leak found in this pipeline was correct in
each individual file and wrong across files. Each is stated as a falsifiable claim, with
its ticket and the test that now guards it.
[15 invariants]{.pq-card__count}
:::

:::{container} pq-card
[Methodology](methodology.md)

What is measured, the split geometry it is measured under, and an honest reading of every
number above.
[3 protocols, 3 configurations]{.pq-card__count}
:::

:::{container} pq-card
[Architecture](architecture.md)

What each layer owns, and why pytorch-forecasting is confined to two modules.
[6 layers]{.pq-card__count}
:::

:::{container} pq-card
[CLI reference](cli.md)

Every command, flag and exit code, plus what `--format json` emits.
[9 commands]{.pq-card__count}
:::

:::{container} pq-card
[HTTP API](http-api.md)

Running the FastAPI service: endpoints, auth, the concurrency model, and what v1
deliberately does not do.
[6 endpoints]{.pq-card__count}
:::

:::{container} pq-card
[Configuration](api/configuration.md)

Every tunable, its default, and the source comment explaining why the default is what it
is.
[4 pydantic models]{.pq-card__count}
:::

:::{container} pq-card
[API reference](api/index.md)

Autodoc over every public module. The docstrings cite ticket IDs and record decisions —
they are worth reading directly.
[29 modules]{.pq-card__count}
:::

:::{container} pq-card
[Development](development.md)

Testing conventions, the research scripts behind every measured number here, and what will
get a change rejected.
[2 CI gates, 6 scripts]{.pq-card__count}
:::

::::

```{toctree}
:caption: Concepts
:maxdepth: 2
:hidden:

architecture
invariants
methodology
```

```{toctree}
:caption: Using PyQuant
:maxdepth: 2
:hidden:

cli
http-api
api/configuration
```

```{toctree}
:caption: Reference
:maxdepth: 2
:hidden:

api/index
```

```{toctree}
:caption: Project
:maxdepth: 2
:hidden:

development
api-design
autonomous-loop-plan
routines/dev-routine-prompt
routines/pm-report-routine-prompt
```

## Tickets

Ticket-level work lives in
[GitHub Issues](https://github.com/AxelSuu/Pytorch-Quant-Model/issues), migrated
2026-08-02 from the old `backlog/{bugs,features,investigations}.md`, and is deliberately
*not* mirrored here — their value is that a resolution note sits one comment from the
commit that earned it. Original `PYQ-NNN` IDs are preserved as a title prefix, so a
reference such as `PYQ-115` still resolves (search Issues for the ID). Status is tracked
with labels (`type:*`, `P0`–`P3`, `status:*`) — see `CLAUDE.md` for the full scheme. The
pre-migration files are archived at
[`backlog/_archive/`](https://github.com/AxelSuu/Pytorch-Quant-Model/tree/main/backlog/_archive)
as a historical record, no longer live or enforced by CI.
