# Stock Clustering & Anomaly Detection (Macro Module)

This module implements unsupervised machine learning pipelines to discover latent relationships in the equity universe.

## Pipeline
- **train_cluster.py**: Entrypoint for end-to-end training.
- **inference.py**: Exposes the trained models to make new predictions.
- **tune_cluster.py**: Hyperparameter optimization.

## Features
- Return and momentum metrics
- Risk/volatility profiles (Sharpe, Drawdown, ATR)
- Market correlations (Beta)
- Technical ratios (RSI, Dist 20DMA)

## Expected Inputs
- OHLCV market data CSVs for the universe of stocks.

## Outputs
- `cluster_scaler.joblib`: Robust scaler used to normalize factors.
- `kmeans_cluster_model.joblib`: Multi-factor K-Means model.
- `isolation_forest_model.joblib`: Anomaly detection model.
- `cluster_results.json`: Training metrics and assignments.
