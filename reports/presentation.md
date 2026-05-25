# TransferIQ Presentation

## Slide 1: Project Title

TransferIQ: Dynamic Player Transfer Value Prediction using AI and Multi-source Data

## Slide 2: Problem

Transfer value depends on performance, age, contract status, public sentiment, injuries, and market momentum. Manual valuation is slow and inconsistent, so TransferIQ builds a repeatable data-driven workflow.

## Slide 3: Data Sources

- Player performance data inspired by StatsBomb aggregates
- Market value and contract data inspired by Transfermarkt
- Social sentiment text inspired by Twitter/X mentions
- Player injury history

## Slide 4: Feature Engineering

- Goals, assists, and actions per 90
- Rolling injury risk
- Contract duration and short-contract indicator
- Sentiment score from social text
- Lagged market values and performance trend
- Position, club, and country encodings

## Slide 5: Modeling

- Regularized multi-source regression baseline
- LSTM-lite gated sequence model using four-quarter history
- Boosted-stump ensemble over regression outputs and current market value

## Slide 6: Evaluation

Models are evaluated on 2025 holdout periods with RMSE, MAE, and R-squared. Results are saved to `reports/metrics.csv`.

## Slide 7: Dashboard

`reports/dashboard.html` shows model metrics, latest player forecasts, sentiment leaders, and current-versus-predicted market value bars.

## Slide 8: Findings

Lagged market value and performance trend are strong valuation signals. Sentiment and injury risk provide short-term adjustment signals. The pipeline selects the best model by holdout RMSE.

## Slide 9: Deployment and Next Steps

The current project is fully reproducible with sample data. Production deployment would replace sample generators with real StatsBomb, Transfermarkt, Twitter/X, and injury feeds while keeping the same modeling pipeline.

