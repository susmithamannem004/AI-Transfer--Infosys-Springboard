from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data_sources import RawDatasets


POSITIVE_WORDS = {"brilliant", "clinical", "elite", "underrated", "creative", "strong", "excellent"}
NEGATIVE_WORDS = {"inconsistent", "overpriced", "injury-prone", "quiet", "risky", "poor", "weak"}


def build_feature_table(raw: RawDatasets, output_dir: Path) -> pd.DataFrame:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = (
        raw.performance.merge(raw.market, on=["player_id", "period"])
        .merge(raw.sentiment, on=["player_id", "period"])
        .merge(raw.injuries, on=["player_id", "period"])
        .merge(raw.players[["player_id", "player_name", "position", "country"]], on="player_id")
    )
    df = df.sort_values(["player_id", "period"]).reset_index(drop=True)

    social_text = (df["positive_examples"].fillna("") + " " + df["negative_examples"].fillna("")).str.lower()
    df["lexicon_sentiment"] = social_text.map(_lexicon_sentiment)
    df["sentiment_score"] = (df["source_sentiment"] * 0.7 + df["lexicon_sentiment"] * 0.3).clip(-1, 1)

    grouped = df.groupby("player_id", group_keys=False)
    df["goals_per_90"] = _safe_rate(df["goals"], df["minutes"])
    df["assists_per_90"] = _safe_rate(df["assists"], df["minutes"])
    df["actions_per_90"] = _safe_rate(df["progressive_actions"] + df["defensive_actions"], df["minutes"])
    df["injury_risk"] = grouped["injury_days"].transform(lambda s: s.rolling(4, min_periods=1).mean()) / 90
    df["performance_index"] = (
        df["goals_per_90"] * 4
        + df["assists_per_90"] * 3
        + df["actions_per_90"] * 0.35
        - df["injury_risk"] * 1.5
    )
    df["performance_trend"] = grouped["performance_index"].diff().fillna(0)
    df["value_lag_1"] = grouped["market_value_m"].shift(1)
    df["value_lag_2"] = grouped["market_value_m"].shift(2)
    df["next_market_value_m"] = grouped["market_value_m"].shift(-1)

    df["short_contract"] = (df["contract_months_remaining"] <= 18).astype(int)
    df["prime_age"] = df["age"].between(23, 28).astype(int)
    df = pd.get_dummies(df, columns=["position", "club", "country"], drop_first=False, dtype=int)

    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns if col != "next_market_value_m"]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))

    df.to_csv(output_dir / "features.csv", index=False)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "player_id",
        "period",
        "player_name",
        "positive_examples",
        "negative_examples",
        "market_value_m",
        "next_market_value_m",
        "source_sentiment",
        "lexicon_sentiment",
    }
    return [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]


def _safe_rate(value: pd.Series, minutes: pd.Series) -> pd.Series:
    return (value / minutes.replace(0, np.nan) * 90).fillna(0)


def _lexicon_sentiment(text: str) -> float:
    tokens = [token.strip(".,;:!?|") for token in text.split()]
    pos = sum(token in POSITIVE_WORDS for token in tokens)
    neg = sum(token in NEGATIVE_WORDS for token in tokens)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total
