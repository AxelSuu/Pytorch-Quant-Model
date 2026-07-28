# Graph Report - .  (2026-07-28)

## Corpus Check
- 99 files · ~133,985 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1659 nodes · 3104 edges · 179 communities (96 shown, 83 thin omitted)
- Extraction: 96% EXTRACTED · 4% INFERRED · 0% AMBIGUOUS · INFERRED: 109 edges (avg confidence: 0.72)
- Token cost: 775,000 input · 52,000 output

## Community Hubs (Navigation)
- Forecast Metrics & Calibration
- CLI Output Formatting & Options Snapshot
- TimeSeriesDataSet & Window Geometry
- Panel Assembly & Leak Tests
- Docs, Invariants & CI Contract
- FRED Macro & VIX Ingestion
- News Sentiment & FinBERT
- API Job Registry
- API Dependency Injection & Bundle Cache
- Feature Schema Mismatch & API Tests
- Split-Conformal Recalibration
- TFT Bundle, Interpret & Predict
- Trading Signals & P&L
- Panel Cache & Pins
- NYSE Trading Calendar
- Model Construction & Purged Cutoff
- Technical Indicator Tests
- Backtest Aggregation
- Typer CLI Application
- Forecast Entry Point & Invariant Tests
- Forecast Dataclass & Serialization
- Long-Format Reshape & Latency Profiling
- CLI Smoke Tests
- Pydantic Settings & Config
- Doctor Health Checks
- Log-Return Reconstruction
- API Response Schemas
- Pluggable Price Providers
- Pyquant Config Cluster
- Test Config Cluster
- Test Providers Cluster
- Test Sectors Cluster
- Investigations Cluster
- Cli Cluster
- Test Prices Cluster
- Test Tft Cluster
- Test Config Cluster
- Charts Cluster
- Tft Cluster
- Prices Cluster
- App Cluster
- Test Dataset Cluster
- Interpret Cluster
- Cache Cluster
- Provenance Cluster
- Investigations Cluster
- Test Retry Cluster
- Api-Design Cluster
- Development Cluster
- App Cluster
- Readme Cluster
- Record Fixtures Cluster
- Test Cli Cluster
- Test Learnability Cluster
- Api-Design Cluster
- Backlog Cluster
- Configuration Cluster
- Nvo Cluster
- Cache Cluster
- Runs Cluster
- Investigations Cluster
- Investigations Cluster
- Api-Design Cluster
- Test Forecast Cluster
- Health Cluster
- Manifest Cluster
- Manifest Cluster
- Readme Cluster
- Features Cluster
- Logo Cluster
- Http-Api Cluster
- Conf Cluster
- Test Interpret Cluster
- Test Sentiment Cluster
- Manifest Cluster
- Investigations Cluster
- Forecast Cluster
- Test Dataset Cluster
- Ablate Features Cluster
- Conftest Cluster
- Manifest Cluster
- Manifest Cluster
- Investigations Cluster
- Providers Cluster
- Manifest Cluster
- Manifest Cluster
- Test Cli Cluster
- Investigations Cluster
- Development Cluster
- Test Cli Cluster
- Compare Pooling Cluster
- Test Cli Cluster
- Test Cli Cluster
- Cli Cluster
- Cache Cluster
- Readme Cluster
-   Init   Cluster
-   Init   Cluster
-   Init   Cluster
-   Init   Cluster
-   Init   Cluster
-   Init   Cluster
-   Init   Cluster
- Test Cli Cluster
- Test Cli Cluster
- Test Cli Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Features Cluster
- Pyproject Cluster

## God Nodes (most connected - your core abstractions)
1. `add_technical_indicators()` - 77 edges
2. `Settings` - 58 edges
3. `build_panel()` - 58 edges
4. `train()` - 53 edges
5. `panel_to_long()` - 39 edges
6. `EvaluationMetrics` - 36 edges
7. `Forecast` - 30 edges
8. `load()` - 27 edges
9. `evaluate_predictions()` - 24 edges
10. `_patch_prices()` - 24 edges

## Surprising Connections (you probably didn't know these)
- `_FakeBundleCache` --uses--> `Forecast`  [INFERRED]
  tests/test_api.py → pyquant/analysis/forecast.py
- `_FakeBundleCache` --uses--> `Interpretation`  [INFERRED]
  tests/test_api.py → pyquant/analysis/interpret.py
- `_FakeBundleCache` --uses--> `EvaluationMetrics`  [INFERRED]
  tests/test_api.py → pyquant/analysis/metrics.py
- `_FakeBundleCache` --uses--> `JobRegistry`  [INFERRED]
  tests/test_api.py → pyquant/api/jobs.py
- `test_fmt_bytes_formats_every_unit()` --calls--> `_fmt_bytes()`  [EXTRACTED]
  tests/test_cli.py → pyquant/cli/app.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Smoke-scale findings that recommend but deliberately do not flip a default** — backlog_investigations_pyq_312, backlog_investigations_pyq_315, backlog_investigations_pyq_316, backlog_features_pyq_247, backlog_readme_multi_symbol_repeat, backlog_investigations_smoke_scale_evidence_discipline [EXTRACTED 1.00]
- **The predict=True / TimeSeriesDataSet window-semantics bug cluster** — backlog_bugs_pyq_109, backlog_bugs_pyq_115, backlog_bugs_pyq_117, backlog_bugs_pyq_127, backlog_features_pyq_250, backlog_investigations_window_semantics_concentration, backlog_readme_correct_in_one_file_wrong_across_files [EXTRACTED 1.00]
- **Sentiment coverage failure chain: vendor limit to feature harm** — backlog_investigations_pyq_301, backlog_bugs_pyq_140, backlog_features_pyq_256, backlog_investigations_pyq_316, backlog_investigations_sentiment_structural_zeros, backlog_investigations_train_serve_shift [EXTRACTED 1.00]
- **The seven fixed look-ahead leaks and the invariants that guard them** — docs_invariants_macro_publication_lag, docs_invariants_warmup_nan, docs_invariants_predict_decodes_after_last_bar, docs_invariants_shared_calendar, docs_invariants_no_backfill, docs_invariants_disjoint_walk_forward_windows, docs_invariants_post_close_news_next_session [EXTRACTED 1.00]
- **Three measured configurations, three sample sizes, one honesty discipline** — docs_methodology_default_close_target_result, docs_methodology_log_return_target_result, docs_methodology_backtest_five_window_result, docs_methodology_sample_size_gate [EXTRACTED 1.00]
- **Layer containment: ML stack below, library-agnostic analysis above, two front-ends** — docs_architecture_ml_stack_containment, docs_architecture_two_frontends_one_core, docs_architecture_panel_build_pipeline, readme_cli_command_surface [INFERRED 0.85]
- **Two front-ends over one serializer/dataclass implementation** — docs_cli_cli_reference, docs_http_api_http_api, docs_cli_thin_front_end, docs_api_design_additive_not_rewrite, docs_cli_json_output, docs_api_design_shared_serializers, docs_http_api_scan_endpoint [EXTRACTED 1.00]
- **Measured-not-expected discipline across docs** — docs_development_metric_honesty, docs_development_measured_not_expected, docs_development_research_scripts, docs_cli_pooling_not_free_win, docs_cli_rate_carries_denominator, docs_api_design_latency_profile, docs_api_configuration_validation_days [INFERRED 0.85]
- **Single-instance serving scaffold and its graduation triggers** — docs_http_api_lru_bundle_cache, docs_http_api_per_bundle_lock, docs_api_design_in_process_job_registry, docs_api_design_bundle_storage, docs_http_api_v1_limits, docs_api_design_queue_graduation_triggers, docs_http_api_bundle_cache_eviction [EXTRACTED 1.00]
- **Forecast geometry depicted by the logo: history, boundary, median, band** — docs__static_logo_price_history_polyline, docs__static_logo_last_observed_bar_rule, docs__static_logo_median_forecast_line, docs__static_logo_quantile_fan [EXTRACTED 1.00]
- **Three Plot Layers Forming One Quantile Forecast View** — nvo_price_history_series, nvo_p10_p90_band, nvo_median_path, nvo_quantile_visual_legend [EXTRACTED 1.00]
- **Visual Evidence of Forecast Geometry Invariants (post-last-bar decode, uncrossed band, single date set)** — nvo_five_day_horizon, nvo_p10_p90_band, nvo_median_path, nvo_price_history_series [INFERRED 0.85]

## Communities (179 total, 83 thin omitted)

### Community 0 - "Forecast Metrics & Calibration"
Cohesion: 0.06
Nodes (59): Guarantee a monotonic band, and record whether one had to be imposed.…, calibration_coverage(), crps_from_quantiles(), directional_hit_rate(), effective_sample_size(), evaluate_predictions(), model_mae(), moving_block_bootstrap_interval() (+51 more)

### Community 1 - "CLI Output Formatting & Options Snapshot"
Cohesion: 0.08
Nodes (44): dict, parametrize, _Output, Global output preferences set by the callback (PYQ-212)., True when ``--format json`` was requested, i.e. Rich output is suppressed., append_snapshot(), fetch_options_snapshot(), load_snapshot_history() (+36 more)

### Community 2 - "TimeSeriesDataSet & Window Geometry"
Cohesion: 0.07
Nodes (48): align_time_index(), make_dataset(), TimeSeriesDataSet, Re-map ``time_idx`` onto one calendar shared by every symbol. panel_to_long()…, Build a TimeSeriesDataSet for training from a long df. If ``training_cutoff``…, add_technical_indicators(), Append technical indicators to an OHLCV DataFrame (in place-safe copy).…, Invariant 7, PYQ-127: each rolling origin must decode its own out-of-sample… (+40 more)

### Community 3 - "Panel Assembly & Leak Tests"
Cohesion: 0.09
Nodes (44): build_panel(), Fetch + join all enabled data sources into one date-indexed panel. ``pin``, if…, _epoch_et(), _patch_prices(), Tests for panel assembly and TimeSeriesDataSet construction., `pyquant train SPY` must not get a SEC_SPY feature that duplicates the target., The leak PYQ-129 fixed was only visible across two files. fetch_sentiment picks…, Epoch seconds for ``hour`` exchange-local time on ``date``. (+36 more)

### Community 4 - "Docs, Invariants & CI Contract"
Cohesion: 0.07
Nodes (42): CI workflow: pytest, ruff, backlog check, Network-free test suite (every vendor call mocked), Nightly workflow incl. docs-drift against live intersphinx, Architecture: five layers, one hard rule, Graceful degradation applies to training only; predict-time schema mismatch is fatal, The ML stack does not leak upwards (Lightning confined to models/ and data/dataset), Options snapshot is display-only, never a model input, build_panel -> panel_to_long -> make_dataset (+34 more)

### Community 5 - "FRED Macro & VIX Ingestion"
Cohesion: 0.08
Nodes (38): DateOffset, NamedTuple, _fetch_fred(), fetch_macro(), _fetch_vix(), _FredSeriesSpec, _period_to_offset(), DataFrame (+30 more)

### Community 6 - "News Sentiment & FinBERT"
Cohesion: 0.08
Nodes (37): fetch_news(), fetch_sentiment(), _finbert(), Timestamp, News sentiment features (Finnhub headlines + local FinBERT scoring). Builds a…, Map a FinBERT top_k result to a signed score in [-1, 1]., Score a batch of headlines into signed sentiment values., The first session date that may legitimately use a headline published at… (+29 more)

### Community 7 - "API Job Registry"
Cohesion: 0.09
Nodes (26): BackgroundTasks, get_job_registry(), JobRecord, JobRegistry, In-process training-job registry (v1), per docs/api-design.md #2. `tft.train()`…, One training job's status, and its result or error once it finishes., Thread-safe in-process store of JobRecords, keyed by job id., Register a new queued job and return its id. (+18 more)

### Community 8 - "API Dependency Injection & Bundle Cache"
Cohesion: 0.09
Nodes (26): Lock, BundleCache, get_bundle_cache(), get_prediction_lock(), get_settings(), _PredictionLocks, Settings, FastAPI dependencies: settings, the bundle LRU cache, per-bundle locks, API-key… (+18 more)

### Community 9 - "Feature Schema Mismatch & API Tests"
Cohesion: 0.11
Nodes (30): FeatureSchemaMismatch, RuntimeError, A bundle's trained features are not all present in the freshly built panel.…, _bypass_auth(), _clear_dependency_overrides(), _fake_forecast(), _FakeBundleCache, _override_settings_and_cache() (+22 more)

### Community 10 - "Split-Conformal Recalibration"
Cohesion: 0.11
Nodes (29): apply_conformal_offset(), ConformalOffset, conformity_scores(), fit_conformal_offset(), ndarray, Split-conformal recalibration of a quantile band (PYQ-248). PYQ-117 measured…, Fit the CQR offset on a calibration slice. ``predictions`` is ``(n_samples,…, Widen (or narrow) the outer band of ``predictions`` by the fitted offset. Only… (+21 more)

### Community 11 - "TFT Bundle, Interpret & Predict"
Cohesion: 0.10
Nodes (30): bundle_conformal_offset(), _check_feature_schema(), interpret(), ModelBundle, permutation_importance(), predict_quantiles(), _prediction_dataset(), _provenance() (+22 more)

### Community 12 - "Trading Signals & P&L"
Cohesion: 0.11
Nodes (26): classify_signal(), _compound(), evaluate_signals(), PYQ-255: does `scan`'s BUY/SELL/HOLD signal actually make money? The project…, BUY/SELL/HOLD from an expected move plus a whole-band-on-one-side guard. The…, P&L accounting for a signal series scored against its realized returns., Compound a sequence of percent returns into one cumulative percent., Score a signal series against what actually happened.… (+18 more)

### Community 13 - "Panel Cache & Pins"
Cohesion: 0.16
Nodes (25): DataFrame, Return the cached panel for ``key`` if present and not past its TTL., Persist ``panel`` under ``key``, timestamped for TTL expiry., Return a pinned dataset snapshot, ignoring any TTL -- exact reproducibility., read_cache(), read_pin(), write_cache(), _df() (+17 more)

### Community 14 - "NYSE Trading Calendar"
Cohesion: 0.12
Nodes (22): AbstractHolidayCalendar, exchange_holidays(), next_sessions(), NYSEHolidayCalendar, DatetimeIndex, Timestamp, The exchange calendar the forecast horizon is laid out on (PYQ-130).…, NYSE market holidays. Deliberately *not* ``USFederalHolidayCalendar``, which… (+14 more)

### Community 15 - "Model Construction & Purged Cutoff"
Cohesion: 0.13
Nodes (24): build_model(), _build_pooled_long_df(), _bundle_dir(), purged_training_cutoff(), Path, Settings, Construct a TFT sized to the dataset and config., Fetch + join each symbol's panel, then pool into one long df. TimeSeriesDataSet… (+16 more)

### Community 16 - "Technical Indicator Tests"
Cohesion: 0.09
Nodes (23): _full_history_ema(), Tests for price data + technical indicators., Textbook Wilder RSI, written the slow obvious way as an independent check. SMA…, min_periods=1 used to emit a value from row 2 off a one-row window; those…, A halted/gapped session reported as Volume=0 makes the *next* row's pct_change…, Belt-and-braces: the whole indicator block is inf-free even on degenerate…, EMA_``span`` computed with ``n_prior`` extra rows of history in front. This is…, From the first surviving panel row onward, EMA_26 must agree with an EMA that… (+15 more)

### Community 17 - "Backtest Aggregation"
Cohesion: 0.13
Nodes (22): aggregate_metrics(), EvaluationMetrics, Model quality vs. a naive baseline, direction, and quantile calibration.…, Relative MAE improvement over the persistence baseline (positive = better)., Pool metrics across multiple windows (e.g. a walk-forward backtest). The sample…, BacktestResult, PYQ-255: --signals wires walk_forward_backtest's per-window signals through…, test_backtest_command_reports_aggregated_metrics() (+14 more)

### Community 18 - "Typer CLI Application"
Cohesion: 0.14
Nodes (21): callback, _add_metric_rows(), backtest(), _band_label(), _color_pct(), _configure_logging(), _directional_interval(), _forecast_table() (+13 more)

### Community 19 - "Forecast Entry Point & Invariant Tests"
Cohesion: 0.10
Nodes (20): High-level forecasting: turn a trained bundle into a structured forecast., invariants_settings(), fixture, PYQ-238: the pipeline-spanning invariants, asserted directly.…, Invariant 1, PYQ-123 shape: a source with no data before day N must not leak…, Invariant 1, PYQ-129 shape: a headline published after the exchange close must…, Invariant 2, PYQ-103/132 shape: the panel's first surviving row must be…, Invariant 5, PYQ-116: the same calendar Date must map to the same time_idx for… (+12 more)

### Community 20 - "Forecast Dataclass & Serialization"
Cohesion: 0.13
Nodes (19): Forecast, A multi-horizon quantile forecast plus context for display., Percent change from current price to the final-day median forecast., backtest_to_dict(), evaluation_to_dict(), forecast_to_dict(), interpretation_to_dict(), Any (+11 more)

### Community 21 - "Long-Format Reshape & Latency Profiling"
Cohesion: 0.11
Nodes (21): panel_to_long(), Convert a wide panel to the long format TimeSeriesDataSet expects., _count_calls(), main(), _profile_call(), Profile a real `forecast`/`explain` call, broken down by phase (PYQ-319).…, Wrap every vendor call site so one profiled call's request count is visible., One forecast-shaped call, phase-timed. Returns {"timings": ..., "requests":… (+13 more)

### Community 22 - "CLI Smoke Tests"
Cohesion: 0.15
Nodes (17): _fake_forecast(), _has_ignore_filter(), CLI smoke tests using Typer's CliRunner (network-free, mocked forecasts)., `--format json` must emit valid JSON with no ANSI escape codes (PYQ-212)., test_debug_flag_enables_debug_logging_and_lightning_chatter(), test_debug_flag_restores_default_warning_filters(), test_default_logging_level_is_warning(), test_default_run_suppresses_user_and_deprecation_warnings() (+9 more)

### Community 23 - "Pydantic Settings & Config"
Cohesion: 0.14
Nodes (16): field_validator, _anchor(), DataConfig, project_root(), BaseModel, Path, Typed configuration for PyQuant. Settings load from environment variables and…, Data sourcing and enrichment toggles. Each enrichment flag is a *request*. A… (+8 more)

### Community 24 - "Doctor Health Checks"
Cohesion: 0.13
Nodes (17): _bundle_health(), BundleHealth, DoctorReport, Any, Path, Settings, Environment and bundle health check (PYQ-263). The project has a lot of…, Read one bundle's meta.json and check its features are still buildable. The… (+9 more)

### Community 25 - "Log-Return Reconstruction"
Cohesion: 0.13
Nodes (18): log_returns_to_prices(), ndarray, Reconstruct a price path from per-step log-return quantiles., Forecast path for a given quantile (must be one of self.quantiles)., ndarray, The (signal, realized_return_pct) scan() would have shown for one window.…, _window_signal(), _make_forecast() (+10 more)

### Community 26 - "API Response Schemas"
Cohesion: 0.13
Nodes (20): EvaluationResponse, FeatureImportance, ForecastResponse, InterpretationResponse, BaseModel, Pydantic response models, built directly from analysis/serialize.py's dicts.…, 202 response to POST /train: the job to poll via GET /train/{job_id}., Response for GET /train/{job_id}. (+12 more)

### Community 27 - "Pluggable Price Providers"
Cohesion: 0.17
Nodes (19): assert_ohlcv_contract(), get_provider(), normalize_ohlcv(), PriceProviderError, RuntimeError, Pluggable OHLCV price providers (PYQ-258, the concrete half of PYQ-214).…, Construct the configured price provider by name., A provider could not return usable OHLCV data. (+11 more)

### Community 28 - "Pyquant Config Cluster"
Cohesion: 0.13
Nodes (17): _block_above(), _cell(), _class_source(), _comments_by_line(), _default_expression(), _describe_fields(), PyQuantConfigModel, Render a pydantic settings model as a field reference (PYQ-232). Why this… (+9 more)

### Community 29 - "Test Config Cluster"
Cohesion: 0.14
Nodes (19): load_settings(), Load settings from environment + .env, optionally layering a YAML config.…, Training and windowing settings., TrainingConfig, Tests for typed configuration validation., Defaults keep today's behavior: fixed seed, single-process loading, fp32., A YAML config overrides built-in defaults for the fields it sets (PYQ-209)., PYQ-117: a holdout of exactly one horizon yields a single validation window. (+11 more)

### Community 30 - "Test Providers Cluster"
Cohesion: 0.12
Nodes (15): _period_start(), First calendar date covered by a yfinance-style period, as YYYY-MM-DD.…, Yahoo Finance via the unofficial ``yfinance`` client (the default)., Tiingo's licensed daily EOD API -- a real vendor with real terms. Chosen as the…, ``session`` is injectable so tests drive the real parsing offline., TiingoProvider, YFinanceProvider, _FakeTiingoSession (+7 more)

### Community 31 - "Test Sectors Cluster"
Cohesion: 0.14
Nodes (16): extend_for_prediction(), feature_columns(), DataFrame, Assemble enrichment sources into a unified panel and a TimeSeriesDataSet. Flow:…, Append ``horizon`` future rows per symbol so a prediction decoder covers the…, Dynamic real feature columns present in the long df (excludes target)., fetch_sector_returns(), DataFrame (+8 more)

### Community 32 - "Investigations Cluster"
Cohesion: 0.18
Nodes (17): PYQ-109: reported metrics came from the final-epoch model, not the best checkpoint, PYQ-129: sentiment joined by UTC calendar date rather than publication time, PYQ-211: Learning-rate tuning (superseded by PYQ-253), PYQ-227: Per-quantile calibration + pinball loss, PYQ-240: Regression test that predictions/actuals/last_observed share units, PYQ-247: Forecast log-returns instead of price levels, PYQ-248: Conformal / split-calibration of the quantile band, PYQ-253: Optuna hyperparameter search (absorbs PYQ-211's scope) (+9 more)

### Community 33 - "Cli Cluster"
Cohesion: 0.13
Nodes (17): TFTConfig.quantiles must be sorted and contain 0.5, TrainingConfig.validation_days, Panel cache hit rate decides perceived service latency, pyquant backtest command (walk-forward), pyquant cache list/prune/rm-pin, Per-trade round-trip cost in basis points, pyquant explain command, pyquant forecast command (+9 more)

### Community 34 - "Test Prices Cluster"
Cohesion: 0.12
Nodes (17): fetch_prices(), _normalize_index(), DataFrame, Make the index a tz-naive, normalized DatetimeIndex named 'Date'., Fetch OHLCV history for ``symbol`` with optional technical indicators. Returns…, Passing only `start` must use the date range, not silently fall back to…, A single transient yfinance failure must be retried, not hard-fail the whole…, PYQ-243: every other test here mocks at our own function boundary, which… (+9 more)

### Community 35 - "Test Tft Cluster"
Cohesion: 0.12
Nodes (17): load(), Load a trained bundle for ``symbol``., Invariants 3 and 4, both PYQ-115: predict=True must decode strictly future…, test_prediction_decoder_starts_after_and_encoder_ends_on_the_last_observed_bar(), PYQ-115: attention_to_series() labels attention with the *last* observed panel…, PYQ-119: the feature schema is a function of the data toggles, so the bundle…, PYQ-225: seed + pinned data only reproduce a run if the code version is known…, calibration_days defaults to 0, so nothing changes for existing bundles and no… (+9 more)

### Community 36 - "Test Config Cluster"
Cohesion: 0.13
Nodes (15): BaseSettings, PydanticBaseSettingsSource, Top-level settings, composed from sub-configs plus secrets/paths., Settings, `pyquant cache` list/prune wire the helpers to the CLI (PYQ-221)., PYQ-254: `pyquant snapshot SYMBOL` -- the only way to ever accumulate a…, test_cache_list_and_prune_commands(), test_snapshot_command_json_output() (+7 more)

### Community 37 - "Charts Cluster"
Cohesion: 0.14
Nodes (15): explain(), forecast(), Forecast SYMBOL with p10/p50/p90 uncertainty bands., Explain SYMBOL's forecast: feature importance + temporal attention., attention_chart(), export_fan_chart(), fan_chart(), importance_chart() (+7 more)

### Community 38 - "Tft Cluster"
Cohesion: 0.17
Nodes (16): Settings, Configured model target, retaining ``Close`` for legacy bundles., target_column(), _evaluate_best_checkpoint(), _evaluate_validation(), _load_best_checkpoint(), Train/evaluate across many rolling origins (walk-forward validation). Unlike…, The best-epoch checkpoint, falling back to the live model. EarlyStopping does… (+8 more)

### Community 39 - "Prices Cluster"
Cohesion: 0.17
Nodes (15): compute_bollinger_bands(), compute_macd(), compute_rsi(), Series, Price data and technical indicators (Yahoo Finance). Returns a date-indexed…, MACD line, signal line, histogram. Each ``ewm`` carries ``min_periods =…, Bollinger band width and %B., Wilder's smoothed average: SMA seed, then ``((n-1)*prev + new) / n``. Not the… (+7 more)

### Community 40 - "App Cluster"
Cohesion: 0.18
Nodes (15): command, cache_list(), cache_prune(), cache_rm_pin(), doctor(), _emit_json(), _fmt_bytes(), Record today's options snapshot for SYMBOL into its accumulated history.… (+7 more)

### Community 41 - "Test Dataset Cluster"
Cohesion: 0.16
Nodes (14): DatetimeIndex, The dates each forecast step is for -- the business days after ``last_date``.…, future_business_dates(), DatetimeIndex, Timestamp, The ``horizon`` trading sessions that follow ``last_date``. Single source of…, Timestamp, 2026-07-04 is a Saturday, so NYSE closes Friday 2026-07-03. pd.bdate_range… (+6 more)

### Community 42 - "Interpret Cluster"
Cohesion: 0.19
Nodes (12): attention_to_series(), _bundle_skill(), Interpretation, Series, Interpretability: which features and which past days drove the forecast., What the model attended to when producing a forecast. ``feature_importance`` is…, Return the ``n`` highest-weighted features as ``(name, weight)``, descending., Recompute skill_vs_baseline from the bundle's recorded evaluation. Not read… (+4 more)

### Community 43 - "Cache Cluster"
Cohesion: 0.24
Nodes (14): cache_stats(), _entry_files(), list_pins(), _meta_path(), prune_expired(), Path, Local cache for assembled data panels. Two complementary mechanisms: - a TTL…, TTL cache entry pickles (excludes the pins/ subdirectory). (+6 more)

### Community 44 - "Provenance Cluster"
Cohesion: 0.17
Nodes (14): code_version(), _git(), git_sha(), package_version(), Path, Which code produced an artifact — version and git sha. PYQ-225's thesis is that…, Installed distribution version, or ``"unknown"`` from a bare source tree., Run a git command, returning its stdout or ``None`` if it is unusable. ``None``… (+6 more)

### Community 45 - "Investigations Cluster"
Cohesion: 0.19
Nodes (14): PYQ-140: Finnhub free tier returns ~6 days of news, not the documented ~365, PYQ-214: Broaden and harden external data providers, PYQ-232: Sphinx + autodoc API documentation site, PYQ-256: has_sentiment_data indicator column, scripts/ablate_features.py, Permutation importance's blind spot with collinear features, Interpreting a model that does not beat its baseline is interpreting noise, Permutation importance as a model-agnostic importance check (+6 more)

### Community 46 - "Test Retry Cluster"
Cohesion: 0.18
Nodes (12): BaseException, Tiny retry helper for flaky external calls (no extra dependency). A single…, Call ``func`` up to ``attempts`` times with exponential backoff. Returns the…, with_retry(), T, _no_sleep(), fixture, Tests for the shared retry helper (PYQ-215). (+4 more)

### Community 47 - "Api-Design Cluster"
Cohesion: 0.23
Nodes (13): autosummary module template, Additive second front-end, not a rewrite, Deployment shape (CPU image for serving, CUDA for training), FastAPI service layer design note (PYQ-213), pyquant/api/ package layout, API reference index (autosummary), Intersphinx upstream type resolution, Documented layer map (config/data/models/analysis/cli/api) (+5 more)

### Community 48 - "Development Cluster"
Cohesion: 0.17
Nodes (13): nitpicky is deliberately off (PYQ-233), scripts/ablate_features.py, Assert the invariant, not the output, Behaviour-named tests, New dependencies need a recorded reason, Development guide, Network-free test suite, Never weaken a test to make it pass (+5 more)

### Community 49 - "App Cluster"
Cohesion: 0.19
Nodes (13): _build_settings(), _fail(), Exception, Path, Settings, Resolve settings for one command invocation, applying CLI flags last.…, Train a Temporal Fusion Transformer for SYMBOLS (pooled if more than one)., Optuna hyperparameter search for SYMBOL (PYQ-253); writes the winner to… (+5 more)

### Community 50 - "Readme Cluster"
Cohesion: 0.17
Nodes (12): PYQ-217: Dockerfile for reproducible training/serving (Open), PYQ-237: Executable doctests for metrics and forecast APIs (Open), PYQ-242: Property-based tests for analysis/metrics.py (Open), PYQ-245: Mutation testing on metrics and indicator modules (Open), PYQ-249: Time-series foundation model as a zero-shot baseline (Open), PYQ-311: Should scripts/backlog.py check run in CI?, scripts/backlog.py check consistency tool, PyQuant backlog: three files, one per ticket type (+4 more)

### Community 51 - "Record Fixtures Cluster"
Cohesion: 0.39
Nodes (11): main(), Settings, Record real vendor payloads as checked-in fixtures (PYQ-243). One-off, run…, _record(), record_finnhub(), record_fred(), record_options(), record_prices() (+3 more)

### Community 52 - "Test Cli Cluster"
Cohesion: 0.20
Nodes (12): _bundle(), _doctor_settings(), Write a minimal meta.json bundle the way train() would., Secrets never enter logs or output -- presence only., The genuinely useful check (PYQ-263): a bundle trained with sentiment cannot be…, FRED-derived features are unbuildable without the key, whatever the toggle says., A fresh install is healthy, not broken., test_doctor_exits_non_zero_when_a_bundles_schema_can_no_longer_be_satisfied() (+4 more)

### Community 53 - "Test Learnability Cluster"
Cohesion: 0.20
Nodes (11): learnability_settings(), DataFrame, fixture, PYQ-239: does the training pipeline learn anything, and does it avoid learning…, The degenerate control: skill must not be implausibly positive when the target…, A synthetic OHLCV panel plus a ``Signal`` feature. When ``learnable``, day t's…, A one-day-ahead config: the target is exactly the lagged signal's effect.…, skill_vs_baseline must be clearly positive when the target is a deterministic… (+3 more)

### Community 54 - "Api-Design Cluster"
Cohesion: 0.18
Nodes (11): Training as a background job (202 + job id), Recommended API build order, Bundle storage backend decision (local disk vs object storage), In-process job registry (v1 scaffold), Triggers to graduate to a real job queue, Per-key rate limiting (deferred), Pydantic response models at the API boundary, Shared serializers between CLI and API (PYQ-212 first) (+3 more)

### Community 55 - "Backlog Cluster"
Cohesion: 0.38
Nodes (10): Pattern, cmd_check(), cmd_list(), _leading_keyword(), load_all(), main(), parse_file(), Namespace (+2 more)

### Community 56 - "Configuration Cluster"
Cohesion: 0.24
Nodes (10): Configuration reference, Field descriptions read out of pyquant/config.py, A missing --config path is an error, not a silent fallback, Settings precedence chain, purge_horizon / embargo_days, TrainingConfig.target (close vs log_return), Why pyquant.config gets a hand-built page, pyquant train command (+2 more)

### Community 57 - "Nvo Cluster"
Cohesion: 0.36
Nodes (10): Downward-Drifting Uncertainty Band, Five Business-Day Forecast Horizon, NVO 5-Day Quantile Forecast Chart, Level Discontinuity Between Last Close and Forecast Median, p50 Median Forecast Path, p10-p90 Prediction Interval Band, NVO Observed Price History (2026-03-15 to 2026-07-27), Probabilistic (Non-Point) Forecast Output Convention (+2 more)

### Community 58 - "Cache Cluster"
Cohesion: 0.24
Nodes (10): _entry_path(), Save a named, TTL-exempt dataset snapshot for later exact reuse., Delete a named pin; return True if it existed., Path of the pickled panel for ``key``., remove_pin(), write_pin(), A pin is TTL-exempt and permanent; nothing recorded what computed it., test_pin_metadata_records_the_version_and_the_column_list() (+2 more)

### Community 59 - "Runs Cluster"
Cohesion: 0.31
Nodes (9): cmd_compare(), load_runs(), main(), Any, Namespace, Path, Compare training runs recorded in runs.jsonl, across every bundle (PYQ-259).…, Every recorded run across every bundle directory under checkpoint_dir. (+1 more)

### Community 60 - "Investigations Cluster"
Cohesion: 0.33
Nodes (9): PYQ-115: forecast predicted already-observed days (predict=True anchoring), PYQ-117: every reported metric rested on a single 5-point validation sample, PYQ-127: every backtest origin scored the same final window, PYQ-202: Rolling / walk-forward backtest command, PYQ-210: seed_everything + recorded seed for reproducibility, PYQ-251: Report effective sample size and block-bootstrap intervals, PYQ-303: Is a single 5-day validation window reliable for model selection?, Accumulated subtlety of predict=True / TimeSeriesDataSet window selection (+1 more)

### Community 61 - "Investigations Cluster"
Cohesion: 0.22
Nodes (9): PYQ-118: validate meta features and raise a clear schema-mismatch error, PYQ-119: forecast/explain/scan rebuilt panels from defaults, not bundle config, PYQ-213: Design a FastAPI service layer alongside the CLI, PYQ-261: Scaffold pyquant/api/ per the PYQ-213 design note, Fetch/panel-build is 98% of cold forecast latency, Panel cache and vendor quota, not GPU throughput, are the API's constraint, scripts/profile_forecast.py, PYQ-302: Schema drift between train-time and predict-time panels (+1 more)

### Community 62 - "Api-Design Cluster"
Cohesion: 0.28
Nodes (9): In-process LRU cache of ModelBundle (design), Measured forecast latency profile (PYQ-319), Per-bundle prediction lock (design), Pooling is not a free win (measured worse), scripts/compare_pooling.py, Every claim must be measured, not expected, Shipped concurrency model, LRU cache of 8 loaded bundles (shipped) (+1 more)

### Community 63 - "Test Forecast Cluster"
Cohesion: 0.22
Nodes (9): generate_forecast(), Settings, Build a forecast for ``symbol`` using its trained bundle. ``pin`` replays a…, Train with --no-sectors then forecast without flags: the panel must still be…, test_generate_forecast_forwards_pin_to_build_panel(), test_generate_forecast_orchestration(), test_generate_forecast_rebuilds_the_panel_with_the_bundles_recorded_config(), Invariant 8, PYQ-115/130: the table, the JSON payload, the chart and the… (+1 more)

### Community 64 - "Health Cluster"
Cohesion: 0.25
Nodes (7): FastAPI app instance: router mounting (PYQ-261, docs/api-design.md). uv sync…, healthz(), get, GET /healthz -- liveness, no auth (docs/api-design.md)., Liveness check: always 200 if the process is up., HealthResponse, Response for GET /healthz.

### Community 65 - "Manifest Cluster"
Cohesion: 0.39
Nodes (9): Close, Dividends, High, Low, Open, Stock Splits, Volume, columns (+1 more)

### Community 66 - "Manifest Cluster"
Cohesion: 0.22
Nodes (8): SPY, XLK, yfinance_sectors.pkl, library_version, recorded_at, rows, symbols, vendor

### Community 67 - "Readme Cluster"
Cohesion: 0.25
Nodes (8): PYQ-101: FRED publication-lag look-ahead leak, PYQ-138: CLI test outcome depended on stdout being a terminal, PYQ-139: ALFRED vintage fetch failed three ways against the real FRED API, PYQ-243: Recorded-payload contract tests for every external vendor, PYQ-257: Use FRED/ALFRED vintages instead of a fixed publication lag, PYQ-263: pyquant doctor — environment and bundle health check, PYQ-305: Documented publication-lag convention for macro sources, Mocking at our own function boundary is half a test

### Community 68 - "Features Cluster"
Cohesion: 0.25
Nodes (8): PYQ-228: Bound dependency majors; pass auto_adjust explicitly, PYQ-238: tests/test_invariants.py — assert pipeline-spanning invariants, PYQ-239: Learnability test — inject a known signal and assert recovery, PYQ-250: Purge + embargo around every walk-forward split, PYQ-252: CRPS, Winkler score and a PIT histogram, pytorch-forecasting/Lightning confined to two modules, PYQ-318: pytorch-forecasting vendor risk vs neuralforecast / Darts, Recording work as verified-in-part rather than claimed whole

### Community 69 - "Logo Cluster"
Cohesion: 0.54
Nodes (8): Logo Depicts Emitted Output, Not Decoration, Last Observed Bar (dashed decode boundary), p50 Median Forecast Line, Pipeline Invariants 3 & 4 (decode strictly after last observed bar), Observed Price History Polyline, PyQuant Logo (fan-chart mark), p10-p90 Quantile Fan, Teal #0f6b73 Forecast Stroke Convention

### Community 70 - "Http-Api Cluster"
Cohesion: 0.32
Nodes (8): Key presence recorded, key values never, API-key gate as a FastAPI dependency (design), pyquant doctor command, CLI exit-code taxonomy (0/1/2), Test-first for bug fixes, X-API-Key authentication (shipped), HTTP endpoint surface (v1), HTTP status-code taxonomy (401/404/409/422/500)

### Community 71 - "Conf Cluster"
Cohesion: 0.36
Nodes (7): _colon_led_listings(), _indent(), Sphinx configuration for the PyQuant documentation site (PYQ-232). Build: uv…, Retry an unresolved Python reference under its documented public name., Treat ``text:`` followed by an indented block as an RST literal block. Several…, _resolve_canonical(), setup()

### Community 72 - "Test Interpret Cluster"
Cohesion: 0.25
Nodes (8): explain_forecast(), Settings, Compute feature importance + temporal attention for ``symbol``., PYQ-119: explain shares the forecast's schema problem, so it needs the fix too., PYQ-314: skill_vs_baseline is a @property on EvaluationMetrics, not a stored…, test_explain_forecast_rebuilds_the_panel_with_the_bundles_recorded_config(), test_explain_forecast_records_the_bundles_skill_vs_baseline(), test_explain_forecast_reuses_the_panel_it_built()

### Community 73 - "Test Sentiment Cluster"
Cohesion: 0.25
Nodes (8): align_to_sessions(), DataFrame, DatetimeIndex, Roll a daily sentiment series onto real trading sessions. :func:`session_date`…, News dated on a weekend must land on Monday, not be silently dropped.…, News with no session left to land on is dropped, never rolled backwards., test_align_to_sessions_drops_news_after_the_last_session(), test_align_to_sessions_rolls_non_trading_dates_onto_the_next_session()

### Community 74 - "Manifest Cluster"
Cohesion: 0.25
Nodes (8): yfinance_options_aapl.pkl, expiry, library_version, n_calls, n_puts, recorded_at, symbol, vendor

### Community 75 - "Investigations Cluster"
Cohesion: 0.33
Nodes (7): PYQ-116: pooled groups aligned by position rather than by date, PYQ-204: Pooled / cross-sectional multi-ticker training, scripts/compare_pooling.py, Pre-registering the success threshold before running, PYQ-315: Is pooling actually helping now that PYQ-116 aligned the calendar?, Smoke-scale evidence is directional, not a verdict, Multi-symbol repeat before changing any default

### Community 76 - "Forecast Cluster"
Cohesion: 0.33
Nodes (7): _get_forecast(), get, post, Settings, p10/p50/p90 quantile forecast for symbol, from its trained bundle., Forecast every requested symbol; one flaky symbol must not fail the rest., scan()

### Community 77 - "Test Dataset Cluster"
Cohesion: 0.29
Nodes (7): _cache_fingerprint(), What a cached panel's validity depends on -- change any of these, different…, PYQ-121 redefined RSI_14 and PYQ-123 changed which rows survive, neither of…, Key *presence* is fingerprinted; key values never are., test_cache_fingerprint_changes_with_the_package_version(), test_cache_fingerprint_is_stable_for_identical_inputs(), test_cache_fingerprint_records_no_secret_values()

### Community 78 - "Ablate Features Cluster"
Cohesion: 0.48
Nodes (6): indicator_correlations(), main(), DataFrame, Which of the 25+ features earn their place? (PYQ-316) One-off investigation…, run_ablation(), _settings()

### Community 79 - "Conftest Cluster"
Cohesion: 0.33
Nodes (6): fixture, Pytest configuration and shared fixtures for PyQuant., A realistic, date-indexed OHLCV DataFrame (no network)., Default settings with all enrichments off (pure-OHLCV baseline)., sample_ohlcv_df(), settings()

### Community 80 - "Manifest Cluster"
Cohesion: 0.29
Nodes (7): finnhub_news_aapl.json, library_version, note, recorded_at, rows, symbol, vendor

### Community 81 - "Manifest Cluster"
Cohesion: 0.29
Nodes (7): fred_dff.json, library_version, note, recorded_at, rows, series_id, vendor

### Community 82 - "Investigations Cluster"
Cohesion: 0.40
Nodes (6): PYQ-258: Pluggable price-provider interface with a licensed fallback, Use a licensed provider before serving anything publicly, Pickle trust boundary for bundles and panel cache, PYQ-306: Is weights_only=False required for dataset_params.pt?, PYQ-309: No LICENSE file despite pyproject declaring MIT, PYQ-320: Data-source licensing and ToS review before anything public-facing

### Community 83 - "Providers Cluster"
Cohesion: 0.33
Nodes (5): Protocol, PriceProvider, DataFrame, Anything that can supply adjusted daily OHLCV for one symbol., Return adjusted daily OHLCV. Must satisfy ``assert_ohlcv_contract``.

### Community 84 - "Manifest Cluster"
Cohesion: 0.33
Nodes (6): yfinance_prices_aapl.pkl, library_version, recorded_at, rows, symbol, vendor

### Community 85 - "Manifest Cluster"
Cohesion: 0.33
Nodes (6): yfinance_vix.pkl, library_version, recorded_at, rows, symbol, vendor

### Community 86 - "Test Cli Cluster"
Cohesion: 0.33
Nodes (6): _empty_snapshot(), _settings_in(), _simple_forecast(), test_explain_on_untrained_symbol_exits_cleanly_without_a_traceback(), test_forecast_on_untrained_symbol_exits_cleanly_without_a_traceback(), test_forecast_skips_the_options_fetch_when_use_options_is_false()

### Community 87 - "Investigations Cluster"
Cohesion: 0.50
Nodes (5): PYQ-114: transient FinBERT failure poisoned the pipeline cache, Declining a dependency or CI job on measured evidence, PYQ-304: Re-run full test suite with the complete ML stack installed, PYQ-308: FinBERT/Finnhub scoring path has zero CI coverage, PYQ-310: Would type-checking (mypy/pyright) catch anything real, cheaply?

### Community 88 - "Development Cluster"
Cohesion: 0.40
Nodes (5): Private helpers omitted from the reference, backlog/ as source of truth, scripts/backlog.py check|list, Comments explain why and cite ticket IDs, Resolution note standard

### Community 89 - "Test Cli Cluster"
Cohesion: 0.40
Nodes (5): An Optuna hyperparameter search (PYQ-253), plus its winner's honest score.…, TuneResult, PYQ-253: the in-search value and the held-out evaluation are different numbers…, test_tune_command_json_output(), test_tune_command_reports_the_held_out_score_not_the_in_search_value()

### Community 90 - "Compare Pooling Cluster"
Cohesion: 0.60
Nodes (4): main(), per_symbol_skill(), Is pooling actually helping, now that PYQ-116 aligned the calendar? (PYQ-315)…, _settings()

### Community 91 - "Test Cli Cluster"
Cohesion: 0.40
Nodes (5): _fake_interpretation(), PYQ-314: an interpretation of a model that does not beat persistence describes…, test_explain_json_carries_bundle_skill(), test_explain_stays_quiet_when_the_bundle_beats_the_baseline(), test_explain_warns_when_the_bundle_does_not_beat_the_baseline()

### Community 92 - "Test Cli Cluster"
Cohesion: 0.40
Nodes (5): With quantiles [0.05, 0.5, 0.95] the band is p5-p95, not the hardcoded p10-p90., test_train_json_output_includes_the_metric_sample_size(), test_train_table_labels_the_calibration_band_from_configured_quantiles(), test_train_table_reports_the_metric_sample_size(), _train_result_with()

### Community 93 - "Cli Cluster"
Cohesion: 0.50
Nodes (4): Enrichment flag is a request, not an assertion, Long-running server is where schema drift bites (PYQ-302), Options snapshot is display-only, never a model input, pyquant snapshot command (options history accrual)

### Community 94 - "Cache Cluster"
Cohesion: 0.50
Nodes (4): fingerprint_key(), Stable short key for a cache fingerprint dict (order-independent)., test_fingerprint_key_differs_for_different_inputs(), test_fingerprint_key_is_order_independent()

## Ambiguous Edges - Review These
- `Pickle trust boundary for bundles and panel cache` → `Use a licensed provider before serving anything publicly`  [AMBIGUOUS]
  backlog/investigations.md · relation: semantically_similar_to
- `p10-p90 Prediction Interval Band` → `Level Discontinuity Between Last Close and Forecast Median`  [AMBIGUOUS]
  nvo.png · relation: conceptually_related_to
- `p50 Median Forecast Path` → `Level Discontinuity Between Last Close and Forecast Median`  [AMBIGUOUS]
  nvo.png · relation: rationale_for

## Knowledge Gaps
- **130 isolated node(s):** `pyquant`, `recorded_at`, `vendor`, `library_version`, `symbol` (+125 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **83 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Pickle trust boundary for bundles and panel cache` and `Use a licensed provider before serving anything publicly`?**
  _Edge tagged AMBIGUOUS (relation: semantically_similar_to) - confidence is low._
- **What is the exact relationship between `p10-p90 Prediction Interval Band` and `Level Discontinuity Between Last Close and Forecast Median`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `p50 Median Forecast Path` and `Level Discontinuity Between Last Close and Forecast Median`?**
  _Edge tagged AMBIGUOUS (relation: rationale_for) - confidence is low._
- **Why does `Settings` connect `Test Config Cluster` to `CLI Output Formatting & Options Snapshot`, `TimeSeriesDataSet & Window Geometry`, `API Job Registry`, `API Dependency Injection & Bundle Cache`, `Feature Schema Mismatch & API Tests`, `TFT Bundle, Interpret & Predict`, `Backtest Aggregation`, `Typer CLI Application`, `Forecast Entry Point & Invariant Tests`, `Forecast Dataclass & Serialization`, `Long-Format Reshape & Latency Profiling`, `CLI Smoke Tests`, `Pydantic Settings & Config`, `Doctor Health Checks`, `Log-Return Reconstruction`, `Test Config Cluster`, `Test Sectors Cluster`, `Interpret Cluster`, `Record Fixtures Cluster`, `Test Cli Cluster`, `Runs Cluster`, `Test Forecast Cluster`, `Test Interpret Cluster`, `Ablate Features Cluster`, `Conftest Cluster`, `Test Cli Cluster`, `Test Cli Cluster`, `Compare Pooling Cluster`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Why does `build_panel()` connect `Panel Assembly & Leak Tests` to `CLI Output Formatting & Options Snapshot`, `FRED Macro & VIX Ingestion`, `News Sentiment & FinBERT`, `TFT Bundle, Interpret & Predict`, `Panel Cache & Pins`, `Model Construction & Purged Cutoff`, `Forecast Entry Point & Invariant Tests`, `Long-Format Reshape & Latency Profiling`, `Test Sectors Cluster`, `Test Prices Cluster`, `Tft Cluster`, `Interpret Cluster`, `Cache Cluster`, `Test Forecast Cluster`, `Test Interpret Cluster`, `Test Sentiment Cluster`, `Test Dataset Cluster`, `Ablate Features Cluster`, `Cache Cluster`?**
  _High betweenness centrality (0.059) - this node is a cross-community bridge._
- **Why does `add_technical_indicators()` connect `TimeSeriesDataSet & Window Geometry` to `Test Prices Cluster`, `Panel Assembly & Leak Tests`, `Test Tft Cluster`, `Tft Cluster`, `Prices Cluster`, `Test Interpret Cluster`, `Test Cli Cluster`, `Interpret Cluster`, `TFT Bundle, Interpret & Predict`, `Model Construction & Purged Cutoff`, `Technical Indicator Tests`, `Forecast Entry Point & Invariant Tests`, `Test Learnability Cluster`, `CLI Smoke Tests`, `Long-Format Reshape & Latency Profiling`, `Log-Return Reconstruction`, `Test Forecast Cluster`?**
  _High betweenness centrality (0.045) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `Settings` (e.g. with `BundleHealth` and `DoctorReport`) actually correct?**
  _`Settings` has 12 INFERRED edges - model-reasoned connections that need verification._