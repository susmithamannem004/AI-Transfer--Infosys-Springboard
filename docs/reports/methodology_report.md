# TransferIQ Methodology Report

## Objective

TransferIQ predicts football player transfer values by combining performance statistics, market history, social sentiment, injury risk, and contract status.

## Data Sources

This self-contained implementation generates realistic sample datasets that mirror the requested sources:

- StatsBomb-style player performance by quarter
- Transfermarkt-style market value and contract data
- Twitter/X-style fan sentiment samples
- Historical injury records

The pipeline writes raw source tables to `data/raw` and feature tables to `data/processed`.

## Feature Engineering

The model uses goals per 90, assists per 90, action volume, rolling injury risk, contract duration, age flags, sentiment score, lagged market value, and one-hot categorical features for position, club, and country.

## Models

- Linear multi-source baseline: dependency-free regularized regression.
- LSTM-lite sequence model: a gated memory encoder over four-quarter player windows with a ridge output layer.
- Ensemble boosted stumps: gradient boosting-style residual learner over baseline predictions and current market value.

## Evaluation

Holdout periods are the 2025 quarters. Metrics are RMSE, MAE, and R-squared.

| model | rmse | mae | r2 |
| --- | --- | --- | --- |
| linear_multisource_baseline | 1.1957 | 0.8873 | 0.9334 |
| lstm_lite_sequence | 3.5153 | 2.6444 | 0.4242 |
| ensemble_boosted_stumps | 1.5915 | 1.1447 | 0.882 |

Best model by RMSE: **linear_multisource_baseline** with RMSE **1.196**.

## Findings

The strongest model in this run is selected by holdout RMSE. The ensemble remains useful as a robustness check because it combines current market value anchors with the multi-source regression signal. Sentiment and injury features help explain short-term value movement, while lagged value and performance trend provide the strongest continuity signals.

## Deployment

The interactive dashboard is exported to `reports/dashboard.html`. Model artifacts are saved as JSON in `models`, making the pipeline transparent and easy to inspect.

## Next Steps

Replace the sample generators in `src/data_sources.py` with real StatsBomb, Transfermarkt, Twitter/X, and injury feeds. After that, keep the same feature, modeling, and reporting steps for a production dataset.
