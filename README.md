# PyStock

LSTM pytorch learning model with real time Yahoo Finance data.

## Install

```bash
pip install -r requirements.txt
```

## Run Pystock

Three simple commands:

```bash
# Train a model
python pystock.py train --symbol AAPL
python pystock.py train --symbol TSLA --start 2020-01-01 --end 2023-12-31

# Evaluate model performance
python pystock.py evaluate --symbol AAPL

# Predict next 5 days
python pystock.py predict --symbol AAPL
```

## Configuration

Edit `config.yaml` to adjust model and training parameters:

```yaml
model:
  hidden_size: 128
  num_layers: 2
  dropout: 0.2

training:
  sequence_length: 60
  batch_size: 32
  epochs: 50
  learning_rate: 0.001
  train_split: 0.8

predict:
  forecast_days: 5
```
