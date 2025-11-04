## Copilot / AI assistant instructions for Pytorch-Quant-Model

This file gives focused, actionable guidance for an AI coding assistant working on this repo.

- Quick orientation
  - Purpose: a production-ready PyTorch time-series forecasting project (LSTM) for stock prices.
  - Entry points: `pystock.py` (main CLI), `src/realtime_predict.py` (legacy real-time usage), `src/train.py` (training/StockTrainer).
  - Data: `src/data/stock_data.py` implements Yahoo Finance fetching and preprocessing. Configuration lives in `config.yaml`.

- Important locations (frequently edited)
  - Models and factory: `src/model.py` (StockLSTM - enhanced with attention & residual connections)
  - Training loop and orchestration: `src/train.py` (StockTrainer), `pystock.py` for CLI wiring
  - Evaluation & realtime API: `src/evaluate.py`, `src/realtime_predict.py` (legacy)
  - Utilities: `src/utils.py` (device selection, plotting, metrics)
  - Tests: `tests/test_model.py` (unit tests that validate model interfaces)
  - Checkpoints: `checkpoints/` contains production checkpoint naming patterns (e.g. `AAPL_lstm_128h_2l_60s_best.pth`).

- Developer workflows & concrete commands
  - Install deps: `pip install -r requirements.txt && pip install -e .`
  - Train model: `pystock train --symbol AAPL [--start yyyy-mm-dd --end yyyy-mm-dd]`
  - Evaluate model: `pystock evaluate --symbol AAPL`
  - Get prediction: `pystock predict --symbol AAPL`
  - Run tests: `pytest tests/test_model.py -v` or `pytest -q` for the whole suite.

- Conventions and patterns to follow
  - CLI-first: operations are wired through `pystock.py`. The CLI has exactly three commands: `train`, `evaluate`, and `predict`.
  - Model API: new models must subclass `torch.nn.Module` and implement a `forward(self, x)` that returns predictions for the last timestep; keep device-agnostic code (use utils.device_or_cpu pattern).
  - Checkpoint naming: use the existing convention `<SYMBOL>_lstm_<hidden>h_<layers>l_<seq>s_epoch_X.pth` and include a `*_best.pth` when saving best weights.
  - Config-driven: prefer `config.yaml` for defaults; CLI flags are minimal (only --symbol, --start, --end for train; only --symbol for evaluate/predict).
  - Tests: keep unit tests small and deterministic; reuse provided synthetic inputs in `tests/test_model.py` where useful.

- Integration & external dependencies
  - Uses `yfinance` for live/historical data (see `src/data/stock_data.py`) — network access is expected for realtime runs.
  - Checkpoints are file-based; models are loaded with `torch.load` and created via the model factory in `src/model.py`.

- Typical changes an assistant will be asked to make
  - Add a new model: update `src/model.py` and add a small unit test in `tests/` that verifies forward shape and device behavior.
  - Add CLI flags: extend `pystock.py` (limited to the three commands: train, evaluate, predict), update README/QUICK_REFERENCE.md.
  - Fix data bugs: edit `src/data/stock_data.py` and add regression tests using saved sample CSVs or mocked `yfinance` calls.

- Examples (copyable snippets)
  - Create model instance (pattern used across codebase):
    `model = create_model("lstm", input_size=15, hidden_size=128, num_layers=2)`
  - Use trainer pattern:
    `trainer = StockTrainer(model, device, run_name, "checkpoints"); trainer.train(train_loader, val_loader, epochs=100)`
  - CLI usage:
    `pystock train --symbol AAPL --start 2020-01-01 --end 2023-12-31`
    `pystock evaluate --symbol AAPL`
    `pystock predict --symbol AAPL`

- Edge-cases & gotchas
  - Tests expect deterministic shapes and dtype; ensure any random seeds or device-dependent ops are controlled in tests.
  - `yfinance` rate-limits can make realtime runs flaky — use cached data or mock network calls for CI tests.
  - Model forward implementations must return the last timestep prediction shape (batch, 1) — many evaluation routines assume this.

If anything in this file is unclear or you want additional examples (unit-test templates, code snippets for adding a model, or a checklist for PRs), tell me which section to expand. 
