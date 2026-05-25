from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_dashboard(features: pd.DataFrame, results: dict[str, pd.DataFrame], report_dir: Path) -> None:
    predictions = results["predictions"].copy()
    metrics_rows = "".join(
        f"<tr><td>{row.model}</td><td>{row.rmse:.3f}</td><td>{row.mae:.3f}</td><td>{row.r2:.3f}</td></tr>"
        for row in results["metrics"].itertuples()
    )
    latest = predictions[predictions["period"] == predictions["period"].max()].sort_values("ensemble_prediction_m", ascending=False)
    table_rows = "".join(
        f"<tr><td>{row.player_name}</td><td>{row.period}</td><td>{row.market_value_m:.2f}</td>"
        f"<td>{row.ensemble_prediction_m:.2f}</td><td>{row.absolute_error_m:.2f}</td></tr>"
        for row in latest.head(18).itertuples()
    )
    chart_rows = latest.head(12)
    bars = "".join(
        f"<div class='bar-row'><span>{row.player_name}</span><div class='bar-shell'>"
        f"<div class='bar actual' style='width:{min(row.market_value_m * 4, 100):.1f}%'></div>"
        f"<div class='bar predicted' style='width:{min(row.ensemble_prediction_m * 4, 100):.1f}%'></div>"
        f"</div></div>"
        for row in chart_rows.itertuples()
    )
    sentiment = (
        features.groupby("player_name", as_index=False)
        .agg(sentiment_score=("sentiment_score", "mean"), market_value_m=("market_value_m", "mean"))
        .sort_values("sentiment_score", ascending=False)
        .head(10)
    )
    sentiment_rows = "".join(
        f"<li><strong>{row.player_name}</strong><span>{row.sentiment_score:.2f} sentiment, EUR {row.market_value_m:.1f}m avg value</span></li>"
        for row in sentiment.itertuples()
    )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TransferIQ Dashboard</title>
  <style>
    :root {{
      --ink: #17202a;
      --muted: #617083;
      --line: #d9e1ea;
      --bg: #f7f9fc;
      --green: #1d7f5f;
      --blue: #2f63b7;
      --gold: #c88719;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background: var(--bg);
    }}
    header {{
      padding: 28px 40px 18px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    p {{ margin: 0; color: var(--muted); }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 28px; }}
    .grid {{ display: grid; grid-template-columns: 1.2fr .8fr; gap: 20px; }}
    section {{
      background: #fff;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 20px;
    }}
    h2 {{ margin: 0 0 16px; font-size: 18px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
    th, td {{ padding: 10px 8px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 650; }}
    .bar-row {{ display: grid; grid-template-columns: 145px 1fr; gap: 12px; align-items: center; margin: 12px 0; font-size: 13px; }}
    .bar-shell {{ position: relative; height: 20px; background: #eef3f8; border-radius: 4px; overflow: hidden; }}
    .bar {{ position: absolute; top: 0; bottom: 0; border-radius: 4px; opacity: .88; }}
    .actual {{ left: 0; background: var(--green); }}
    .predicted {{ left: 0; height: 9px; top: 11px; background: var(--gold); }}
    .legend {{ display: flex; gap: 14px; color: var(--muted); font-size: 13px; margin-bottom: 8px; }}
    .dot {{ width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 5px; }}
    ul {{ margin: 0; padding: 0; list-style: none; }}
    li {{ display: flex; justify-content: space-between; gap: 16px; border-bottom: 1px solid var(--line); padding: 11px 0; }}
    li span {{ color: var(--muted); }}
    @media (max-width: 800px) {{
      header {{ padding: 22px; }}
      main {{ padding: 18px; }}
      .grid {{ grid-template-columns: 1fr; }}
      .bar-row {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>TransferIQ</h1>
    <p>Dynamic player transfer value prediction from performance, sentiment, injury, contract, and market signals.</p>
  </header>
  <main>
    <div class="grid">
      <section>
        <h2>Model Evaluation</h2>
        <table><thead><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>R2</th></tr></thead><tbody>{metrics_rows}</tbody></table>
      </section>
      <section>
        <h2>Sentiment Leaders</h2>
        <ul>{sentiment_rows}</ul>
      </section>
    </div>
    <section style="margin-top:20px">
      <h2>Latest Transfer Value Forecasts</h2>
      <div class="legend"><span><i class="dot" style="background:var(--green)"></i>Current value</span><span><i class="dot" style="background:var(--gold)"></i>Predicted next value</span></div>
      {bars}
    </section>
    <section style="margin-top:20px">
      <h2>Prediction Table</h2>
      <table><thead><tr><th>Player</th><th>Period</th><th>Current EUR m</th><th>Predicted EUR m</th><th>Abs. Error EUR m</th></tr></thead><tbody>{table_rows}</tbody></table>
    </section>
  </main>
</body>
</html>"""
    (report_dir / "dashboard.html").write_text(html, encoding="utf-8")
    (report_dir / "index.html").write_text(html, encoding="utf-8")


def write_methodology_report(results: dict[str, pd.DataFrame], report_dir: Path) -> None:
    metrics_md = _markdown_table(results["metrics"])
    best = results["metrics"].sort_values("rmse").iloc[0]
    report = f"""# TransferIQ Methodology Report

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

{metrics_md}

Best model by RMSE: **{best['model']}** with RMSE **{best['rmse']:.3f}**.

## Findings

The strongest model in this run is selected by holdout RMSE. The ensemble remains useful as a robustness check because it combines current market value anchors with the multi-source regression signal. Sentiment and injury features help explain short-term value movement, while lagged value and performance trend provide the strongest continuity signals.

## Deployment

The interactive dashboard is exported to `reports/dashboard.html`. Model artifacts are saved as JSON in `models`, making the pipeline transparent and easy to inspect.

## Next Steps

Replace the sample generators in `src/data_sources.py` with real StatsBomb, Transfermarkt, Twitter/X, and injury feeds. After that, keep the same feature, modeling, and reporting steps for a production dataset.
"""
    (report_dir / "methodology_report.md").write_text(report, encoding="utf-8")


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for _, row in frame.iterrows():
        rows.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *rows])
