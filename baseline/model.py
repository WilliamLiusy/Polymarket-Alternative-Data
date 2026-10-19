"""Qlib MLP with training-only normalization and manually configured epochs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn
from qlib.data.dataset.handler import DataHandlerLP
from qlib.model.base import Model


_NORMALIZATION_CHUNK_ROWS = 8192


def build_mlp(input_size: int, hidden_size: int) -> nn.Sequential:
    """Prediction head shared by the baseline and Polymarket extension."""
    return nn.Sequential(nn.Linear(input_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 1))


def _normalization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two-pass finite-value statistics with bounded temporary memory."""
    count = np.zeros(values.shape[1], dtype="int64")
    total = np.zeros(values.shape[1], dtype="float64")
    for start in range(0, len(values), _NORMALIZATION_CHUNK_ROWS):
        block = values[start:start + _NORMALIZATION_CHUNK_ROWS]
        valid = np.isfinite(block)
        count += valid.sum(axis=0)
        total += np.sum(block, axis=0, dtype="float64", where=valid)
    denominator = np.maximum(count, 1)
    mean = total / denominator
    squared_deviations = np.zeros(values.shape[1], dtype="float64")
    for start in range(0, len(values), _NORMALIZATION_CHUNK_ROWS):
        block = np.array(values[start:start + _NORMALIZATION_CHUNK_ROWS], dtype="float64", copy=True)
        valid = np.isfinite(block)
        np.subtract(block, mean, out=block, where=valid)
        np.square(block, out=block, where=valid)
        squared_deviations += np.sum(block, axis=0, where=valid)
    scale = np.sqrt(squared_deviations / denominator)
    scale[scale < 1e-6] = 1
    return mean.astype("float32"), scale.astype("float32")


class MLPModel(Model):
    """Fit Alpha158 features through Qlib's model and dataset interfaces.

    Normalization uses only finite-label training rows. Every fit starts from
    scratch and keeps the final configured epoch's weights. Other dataset
    segments never update statistics, weights, or the number of training epochs.
    """

    def __init__(self, epochs: int = 10, batch_size: int = 512, hidden_size: int = 32,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 seed: int = 42, num_threads: int = 2):
        if type(epochs) is not int or epochs <= 0:
            raise ValueError("epochs must be a positive integer")
        self.epochs, self.batch_size, self.hidden_size = epochs, batch_size, hidden_size
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.seed, self.num_threads = seed, num_threads
        self.network = None
        self.trained_epochs = 0

    def _initialize_network(self, train: pd.DataFrame) -> None:
        self.network = build_mlp(len(self.feature_names), self.hidden_size)

    def _arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
        if list(frame.columns) == self.feature_names:
            alpha = frame.to_numpy(dtype="float32", copy=True)
        else:
            # Align inference columns without constructing another full frame.
            alpha = np.full((len(frame), len(self.feature_names)), np.nan, dtype="float32")
            for column, name in enumerate(self.feature_names):
                if name in frame:
                    alpha[:, column] = frame[name].to_numpy()
        np.subtract(alpha, self.alpha_mean, out=alpha)
        np.divide(alpha, self.alpha_scale, out=alpha)
        for start in range(0, len(alpha), _NORMALIZATION_CHUNK_ROWS):
            np.nan_to_num(alpha[start:start + _NORMALIZATION_CHUNK_ROWS], copy=False,
                          nan=0.0, posinf=0.0, neginf=0.0)
        return (alpha,)

    def _batch(self, arrays, positions):
        return (torch.from_numpy(arrays[0][positions]),)

    def _optimizer(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate,
                                weight_decay=self.weight_decay)

    def _regularization(self):
        return 0.0

    def _predict_arrays(self, arrays) -> np.ndarray:
        predictions = []
        self.network.eval()
        with torch.no_grad():
            for start in range(0, len(arrays[0]), self.batch_size):
                positions = np.arange(start, min(start + self.batch_size, len(arrays[0])))
                predictions.append(self.network(*self._batch(arrays, positions)).reshape(-1).numpy())
        return np.concatenate(predictions) if predictions else np.array([], dtype="float32")

    def fit(self, dataset, **kwargs) -> None:
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.num_threads)
        rng = np.random.default_rng(self.seed)

        def prepare_segment(name):
            selector = dataset.segments[name]
            if not isinstance(selector, slice):
                selector = slice(*selector)
            # The sorted index is (datetime, instrument). A plain date slice
            # (level=None) keeps a view; Qlib's default multilevel selection copies.
            return dataset.prepare(selector, col_set=DataHandlerLP.CS_RAW,
                                   data_key=DataHandlerLP.DK_L, level=None)

        train = prepare_segment("train")
        finite_label = np.isfinite(train["label"].iloc[:, 0])
        if not finite_label.all():
            train = train.loc[finite_label]
        if train.empty:
            raise ValueError("No finite training labels")
        features = train["feature"]
        self.feature_names = list(features.columns)
        self.alpha_mean, self.alpha_scale = _normalization(features.to_numpy(dtype="float32"))
        self._initialize_network(train)
        arrays = self._arrays(features)
        # A label view can otherwise keep the entire frame's backing array alive.
        targets = train["label"].iloc[:, 0].to_numpy(dtype="float32", copy=True)
        del train, features, finite_label
        optimizer = self._optimizer()
        self.history = []
        self.trained_epochs = 0
        for epoch in range(self.epochs):
            self.network.train()
            permutation = rng.permutation(len(targets))
            total = 0.0
            for start in range(0, len(targets), self.batch_size):
                positions = permutation[start:start + self.batch_size]
                optimizer.zero_grad()
                prediction = self.network(*self._batch(arrays, positions)).reshape(-1)
                mse = (prediction - torch.from_numpy(targets[positions])).square().mean()
                loss = mse + self._regularization()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss; inspect feature values and learning rate")
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10)
                optimizer.step()
                total += float(mse.detach()) * len(positions)
            record = {"epoch": epoch + 1, "train_mse": total / len(targets)}
            self.history.append(record)
            self.trained_epochs = epoch + 1
            print(f"{type(self).__name__} epoch {epoch + 1}: {record}", flush=True)
        self.network.eval()

    def predict(self, dataset, segment="test") -> pd.Series:
        if self.network is None:
            raise ValueError("Model has not been fitted")
        frame = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        return pd.Series(self._predict_arrays(self._arrays(frame)), index=frame.index, name="score")
