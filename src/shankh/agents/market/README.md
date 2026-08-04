# Market Regime Analysis (Market Module)

This module implements an unsupervised Hidden Markov Model (HMM) to discover latent market regimes based on macro indicators.

## Pipeline
- **train_regime.py**: Entrypoint for end-to-end training.
- **inference.py**: Exposes the trained HMM to predict current regime.
- **tune_regime.py**: Hyperparameter optimization.

## Features
- Market Return and Volatility
- Market Breadth (percentage above 20-DMA)
- Correlation Density
- Advance/Decline Ratio

## Expected Inputs
- OHLCV market data CSVs for the universe of stocks.

## Outputs
- `regime_scaler.joblib`: Scaler used to normalize macro features.
- `regime_model.joblib`: Gaussian HMM model.
- `historical_regimes.csv`: Time-series of predicted regimes.
- `regime_metadata.json`: Latest state, transition statistics, and metrics.
