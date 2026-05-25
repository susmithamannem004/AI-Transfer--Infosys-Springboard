from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import feature_columns


def train_and_evaluate(features: pd.DataFrame, model_dir: Path, report_dir: Path) -> dict[str, pd.DataFrame]:
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    model_frame = features.dropna(subset=["next_market_value_m"]).copy()
    train = model_frame[model_frame["period"] < "2025Q1"].copy()
    test = model_frame[model_frame["period"] >= "2025Q1"].copy()
    cols = feature_columns(features)

    scaler = StandardScaler().fit(train[cols].to_numpy(float))
    x_train = scaler.transform(train[cols].to_numpy(float))
    x_test = scaler.transform(test[cols].to_numpy(float))
    y_train = train["next_market_value_m"].to_numpy(float)
    y_test = test["next_market_value_m"].to_numpy(float)

    baseline = LinearRegressor(learning_rate=0.03, epochs=1400, l2=0.001).fit(x_train, y_train)
    baseline_pred = baseline.predict(x_test)

    sequence_train_x, sequence_train_y = _make_sequences(features, cols, scaler, lookback=4, target_index=train.index)
    sequence_test_x, sequence_test_y, sequence_index = _make_sequences(
        features, cols, scaler, lookback=4, target_index=test.index, include_index=True
    )
    lstm_like = LSTMLiteRegressor(hidden_size=18, learning_rate=0.01, epochs=360).fit(sequence_train_x, sequence_train_y)
    lstm_pred_seq = lstm_like.predict(sequence_test_x)

    ensemble_train_base = np.column_stack([baseline.predict(x_train), train["market_value_m"].to_numpy(float)])
    ensemble = GradientBoostedStumps(rounds=80, learning_rate=0.08).fit(ensemble_train_base, y_train)
    ensemble_test_base = np.column_stack([baseline_pred, test["market_value_m"].to_numpy(float)])
    ensemble_pred = ensemble.predict(ensemble_test_base)

    metrics = pd.DataFrame(
        [
            _metrics_row("linear_multisource_baseline", y_test, baseline_pred),
            _metrics_row("lstm_lite_sequence", sequence_test_y, lstm_pred_seq),
            _metrics_row("ensemble_boosted_stumps", y_test, ensemble_pred),
        ]
    )

    predictions = test[["player_id", "player_name", "period", "market_value_m", "next_market_value_m"]].copy()
    predictions["baseline_prediction_m"] = baseline_pred
    predictions["ensemble_prediction_m"] = ensemble_pred
    predictions["absolute_error_m"] = np.abs(predictions["next_market_value_m"] - predictions["ensemble_prediction_m"])
    predictions = predictions.sort_values(["period", "absolute_error_m"])

    sequence_predictions = test.loc[sequence_index][["player_id", "player_name", "period", "market_value_m", "next_market_value_m"]].copy()
    sequence_predictions["lstm_lite_prediction_m"] = lstm_pred_seq

    metrics.to_csv(report_dir / "metrics.csv", index=False)
    predictions.to_csv(report_dir / "predictions.csv", index=False)
    sequence_predictions.to_csv(report_dir / "sequence_predictions.csv", index=False)

    _save_json(model_dir / "scaler.json", scaler.to_dict(cols))
    _save_json(model_dir / "linear_model.json", baseline.to_dict(cols))
    _save_json(model_dir / "lstm_lite_model.json", lstm_like.to_dict())
    _save_json(model_dir / "ensemble_model.json", ensemble.to_dict())

    return {"metrics": metrics, "predictions": predictions, "sequence_predictions": sequence_predictions}


class StandardScaler:
    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = x.mean(axis=0)
        self.std_ = x.std(axis=0)
        self.std_[self.std_ == 0] = 1
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def to_dict(self, columns: list[str]) -> dict[str, object]:
        return {"columns": columns, "mean": self.mean_.tolist(), "std": self.std_.tolist()}


class LinearRegressor:
    def __init__(self, learning_rate: float, epochs: int, l2: float) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LinearRegressor":
        x_aug = np.column_stack([np.ones(len(x)), x])
        self.weights = np.zeros(x_aug.shape[1])
        for _ in range(self.epochs):
            pred = x_aug @ self.weights
            grad = x_aug.T @ (pred - y) / len(y)
            grad[1:] += self.l2 * self.weights[1:]
            self.weights -= self.learning_rate * grad
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.column_stack([np.ones(len(x)), x]) @ self.weights

    def to_dict(self, columns: list[str]) -> dict[str, object]:
        return {"intercept": float(self.weights[0]), "features": columns, "weights": self.weights[1:].tolist()}


class LSTMLiteRegressor:
    """Small gated recurrent forecaster trained with random features plus ridge output.

    It preserves the LSTM idea from the brief: a gated memory summarizes lookback windows
    before regression. The recurrent gates are deterministic from the seed; the output
    layer is trained by ridge regression, making this lightweight and dependency-free.
    """

    def __init__(self, hidden_size: int, learning_rate: float, epochs: int, seed: int = 7) -> None:
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.seed = seed

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LSTMLiteRegressor":
        rng = np.random.default_rng(self.seed)
        feature_count = x.shape[-1]
        self.w_i = rng.normal(0, 0.25, (feature_count, self.hidden_size))
        self.w_f = rng.normal(0, 0.25, (feature_count, self.hidden_size))
        self.w_o = rng.normal(0, 0.25, (feature_count, self.hidden_size))
        self.w_c = rng.normal(0, 0.25, (feature_count, self.hidden_size))
        encoded = self._encode(x)
        design = np.column_stack([np.ones(len(encoded)), encoded])
        ridge = 0.8 * np.eye(design.shape[1])
        ridge[0, 0] = 0
        self.output_weights = np.linalg.pinv(design.T @ design + ridge) @ design.T @ y
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        encoded = self._encode(x)
        return np.column_stack([np.ones(len(encoded)), encoded]) @ self.output_weights

    def _encode(self, x: np.ndarray) -> np.ndarray:
        h = np.zeros((x.shape[0], self.hidden_size))
        c = np.zeros_like(h)
        for step in range(x.shape[1]):
            current = x[:, step, :]
            i = _sigmoid(current @ self.w_i)
            f = _sigmoid(current @ self.w_f)
            o = _sigmoid(current @ self.w_o)
            candidate = np.tanh(current @ self.w_c)
            c = f * c + i * candidate
            h = o * np.tanh(c)
        return h

    def to_dict(self) -> dict[str, object]:
        return {
            "hidden_size": self.hidden_size,
            "seed": self.seed,
            "output_weights": self.output_weights.tolist(),
            "note": "Dependency-free gated sequence encoder with ridge output layer.",
        }


class GradientBoostedStumps:
    def __init__(self, rounds: int, learning_rate: float) -> None:
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.stumps: list[dict[str, float]] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "GradientBoostedStumps":
        self.base = float(np.mean(y))
        pred = np.full(len(y), self.base)
        for _ in range(self.rounds):
            residual = y - pred
            stump = self._best_stump(x, residual)
            update = self._predict_stump(x, stump)
            pred += self.learning_rate * update
            self.stumps.append(stump)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = np.full(len(x), self.base)
        for stump in self.stumps:
            pred += self.learning_rate * self._predict_stump(x, stump)
        return pred

    def _best_stump(self, x: np.ndarray, residual: np.ndarray) -> dict[str, float]:
        best = {"feature": 0.0, "threshold": 0.0, "left": 0.0, "right": 0.0, "loss": float("inf")}
        for feature in range(x.shape[1]):
            for threshold in np.quantile(x[:, feature], [0.2, 0.4, 0.6, 0.8]):
                left_mask = x[:, feature] <= threshold
                if left_mask.all() or (~left_mask).all():
                    continue
                left_value = float(residual[left_mask].mean())
                right_value = float(residual[~left_mask].mean())
                update = np.where(left_mask, left_value, right_value)
                loss = float(np.mean((residual - update) ** 2))
                if loss < best["loss"]:
                    best = {
                        "feature": float(feature),
                        "threshold": float(threshold),
                        "left": left_value,
                        "right": right_value,
                        "loss": loss,
                    }
        best.pop("loss")
        return best

    def _predict_stump(self, x: np.ndarray, stump: dict[str, float]) -> np.ndarray:
        feature = int(stump["feature"])
        return np.where(x[:, feature] <= stump["threshold"], stump["left"], stump["right"])

    def to_dict(self) -> dict[str, object]:
        return {"base": self.base, "learning_rate": self.learning_rate, "stumps": self.stumps}


def _make_sequences(
    frame: pd.DataFrame,
    columns: list[str],
    scaler: StandardScaler,
    lookback: int,
    target_index: pd.Index,
    include_index: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[int]]:
    x_seq, y_seq, indices = [], [], []
    target_set = set(target_index)
    for _, group in frame.groupby("player_id"):
        group = group.sort_values("period")
        x = scaler.transform(group[columns].to_numpy(float))
        y = group["next_market_value_m"].to_numpy(float)
        for idx in range(lookback - 1, len(group)):
            row_index = group.index[idx]
            if row_index not in target_set or np.isnan(y[idx]):
                continue
            x_seq.append(x[idx - lookback + 1 : idx + 1])
            y_seq.append(y[idx])
            indices.append(row_index)
    arrays = (np.asarray(x_seq), np.asarray(y_seq))
    if include_index:
        return arrays[0], arrays[1], indices
    return arrays


def _metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | str]:
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.sum((y_true - y_true.mean()) ** 2)
    r2 = float(1 - np.sum((y_true - y_pred) ** 2) / denom) if denom else 0.0
    return {"model": name, "rmse": round(rmse, 4), "mae": round(mae, 4), "r2": round(r2, 4)}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -35, 35)))


def _save_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
