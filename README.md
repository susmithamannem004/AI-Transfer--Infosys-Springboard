# TransferIQ

Dynamic player transfer value prediction using AI and multi-source data.

This repository implements the full project brief from scratch in a self-contained way. It creates sample multi-source football player data, engineers performance, injury, contract, sentiment, and market-trend features, trains time-series and ensemble-style models, evaluates them, and exports an interactive HTML dashboard.

The implementation uses only `numpy` and `pandas` so it can run in a lean internship environment without TensorFlow, XGBoost, API keys, or browser scraping. The code is structured so real StatsBomb, Transfermarkt, Twitter/X, and injury feeds can replace the sample data later.

## Quick Start

```powershell
python -m pip install pandas numpy
python run_pipeline.py
```

Outputs are written to:

- `data/raw/`: generated source-like datasets
- `data/processed/`: cleaned feature datasets
- `models/`: trained lightweight model artifacts
- `reports/`: metrics, predictions, methodology, and dashboard
- `docs/architecture.mmd`: Mermaid architecture diagram

Open `reports/dashboard.html` in a browser to explore predicted transfer values.

## Project Scope

- Multi-source data generation and ingestion
- Data cleaning and preprocessing
- Sentiment analysis from social text
- Injury risk and contract features
- Univariate and multivariate sequence forecasting
- Gradient boosting-style ensemble modeling
- Evaluation with RMSE, MAE, and R-squared
- Interactive HTML dashboard
- Comprehensive project report

## Real Data Extension Points

The files in `src/data_sources.py` isolate data loading. Replace the sample generators with:

- StatsBomb Open Data event/player aggregates
- Transfermarkt market value scraping/export
- Twitter/X API search results
- Historical injury records

Keep the returned columns compatible with `src/features.py`, then the same modeling and reporting pipeline will continue to work.
