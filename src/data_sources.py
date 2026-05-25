from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RawDatasets:
    performance: pd.DataFrame
    market: pd.DataFrame
    sentiment: pd.DataFrame
    injuries: pd.DataFrame
    players: pd.DataFrame


POSITIONS = ["Forward", "Midfielder", "Defender", "Goalkeeper"]
CLUBS = ["Northbridge FC", "Athletic Union", "Real Portside", "East City", "Sporting Vale"]
COUNTRIES = ["England", "Spain", "France", "Germany", "Portugal", "Brazil", "Argentina"]


def generate_raw_datasets(output_dir: Path, seed: int = 42) -> RawDatasets:
    rng = np.random.default_rng(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    players = _generate_players(rng)
    periods = pd.period_range("2021Q1", "2025Q4", freq="Q").astype(str)

    performance_rows = []
    market_rows = []
    sentiment_rows = []
    injury_rows = []

    for _, player in players.iterrows():
        age_start = player.age
        base_skill = player.base_skill
        value = player.base_value_m
        durability = player.durability
        popularity = player.popularity

        for idx, period in enumerate(periods):
            age = age_start + idx / 4
            minutes = max(0, rng.normal(780, 180) * durability)
            form_wave = np.sin(idx / 2 + player.player_id * 0.37)
            goals = _position_goals(player.position, rng, base_skill, minutes, form_wave)
            assists = _position_assists(player.position, rng, base_skill, minutes, form_wave)
            progressive_actions = max(0, rng.normal(18 + base_skill * 8, 6))
            defensive_actions = max(0, rng.normal(12 + (1 if player.position == "Defender" else 0) * 15, 5))

            injury_days = int(max(0, rng.normal((1 - durability) * 20, 9)))
            if rng.random() < (1 - durability) * 0.25:
                injury_days += int(rng.integers(20, 80))

            sentiment_score = float(np.clip(rng.normal(0.15 + popularity * 0.45 + form_wave * 0.08, 0.28), -1, 1))
            mention_count = int(max(5, rng.normal(300 + popularity * 2200 + goals * 35, 130)))
            positive_posts, negative_posts = _make_social_posts(player.player_name, sentiment_score, mention_count)

            contract_months = max(3, player.contract_months_start - idx * 3)
            value_growth = (
                0.035 * goals
                + 0.025 * assists
                + 0.012 * progressive_actions
                - 0.018 * injury_days
                + 0.11 * sentiment_score
                - 0.015 * max(age - 27, 0)
                + rng.normal(0, 0.8)
            )
            value = max(1.0, value + value_growth)

            performance_rows.append(
                {
                    "player_id": player.player_id,
                    "period": period,
                    "minutes": round(minutes, 1),
                    "goals": int(goals),
                    "assists": int(assists),
                    "progressive_actions": round(progressive_actions, 2),
                    "defensive_actions": round(defensive_actions, 2),
                }
            )
            market_rows.append(
                {
                    "player_id": player.player_id,
                    "period": period,
                    "market_value_m": round(value, 2),
                    "contract_months_remaining": int(contract_months),
                    "age": round(age, 2),
                    "club": player.club,
                }
            )
            sentiment_rows.append(
                {
                    "player_id": player.player_id,
                    "period": period,
                    "mention_count": mention_count,
                    "positive_examples": " | ".join(positive_posts),
                    "negative_examples": " | ".join(negative_posts),
                    "source_sentiment": round(sentiment_score, 4),
                }
            )
            injury_rows.append(
                {
                    "player_id": player.player_id,
                    "period": period,
                    "injury_days": injury_days,
                    "injury_count": int(injury_days > 0) + int(injury_days > 35),
                }
            )

    performance = pd.DataFrame(performance_rows)
    market = pd.DataFrame(market_rows)
    sentiment = pd.DataFrame(sentiment_rows)
    injuries = pd.DataFrame(injury_rows)

    players.to_csv(output_dir / "players.csv", index=False)
    performance.to_csv(output_dir / "performance.csv", index=False)
    market.to_csv(output_dir / "market_values.csv", index=False)
    sentiment.to_csv(output_dir / "sentiment.csv", index=False)
    injuries.to_csv(output_dir / "injuries.csv", index=False)

    return RawDatasets(performance, market, sentiment, injuries, players)


def _generate_players(rng: np.random.Generator, count: int = 36) -> pd.DataFrame:
    first = ["Alex", "Mateo", "Lucas", "Noah", "Leo", "Milan", "Rafael", "Theo", "Andre", "Bruno", "Nico", "Victor"]
    last = ["Silva", "Martin", "Kane", "Costa", "Rossi", "Diallo", "Santos", "Meyer", "Garcia", "Moreau", "Reed", "Lopez"]
    rows = []
    for player_id in range(1, count + 1):
        position = rng.choice(POSITIONS, p=[0.28, 0.34, 0.28, 0.10])
        age = int(rng.integers(18, 31))
        base_skill = float(np.clip(rng.normal(0.62, 0.16), 0.25, 0.94))
        position_multiplier = {"Forward": 1.35, "Midfielder": 1.15, "Defender": 0.95, "Goalkeeper": 0.8}[position]
        rows.append(
            {
                "player_id": player_id,
                "player_name": f"{rng.choice(first)} {rng.choice(last)}",
                "position": position,
                "country": rng.choice(COUNTRIES),
                "club": rng.choice(CLUBS),
                "age": age,
                "base_skill": round(base_skill, 4),
                "base_value_m": round(max(2, rng.normal(12, 4) * base_skill * position_multiplier), 2),
                "durability": round(float(np.clip(rng.normal(0.82, 0.09), 0.55, 0.98)), 4),
                "popularity": round(float(np.clip(rng.normal(0.45, 0.22), 0.05, 0.98)), 4),
                "contract_months_start": int(rng.integers(24, 61)),
            }
        )
    return pd.DataFrame(rows)


def _position_goals(position: str, rng: np.random.Generator, skill: float, minutes: float, wave: float) -> int:
    rate = {"Forward": 0.55, "Midfielder": 0.22, "Defender": 0.06, "Goalkeeper": 0.0}[position]
    expected = max(0, rate * skill * minutes / 450 + wave * 0.3)
    return int(rng.poisson(expected))


def _position_assists(position: str, rng: np.random.Generator, skill: float, minutes: float, wave: float) -> int:
    rate = {"Forward": 0.24, "Midfielder": 0.34, "Defender": 0.11, "Goalkeeper": 0.01}[position]
    expected = max(0, rate * skill * minutes / 450 + wave * 0.25)
    return int(rng.poisson(expected))


def _make_social_posts(player_name: str, score: float, mentions: int) -> tuple[list[str], list[str]]:
    positive_words = ["brilliant", "clinical", "elite", "underrated", "creative"]
    negative_words = ["inconsistent", "overpriced", "injury-prone", "quiet", "risky"]
    pos_count = 2 if score >= 0 else 1
    neg_count = 2 if score < 0 else 1
    positives = [f"{player_name} looks {word} after {mentions} fan mentions" for word in positive_words[:pos_count]]
    negatives = [f"{player_name} feels {word} to some supporters" for word in negative_words[:neg_count]]
    return positives, negatives

