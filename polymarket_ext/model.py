"""Qlib model with five trainable Polymarket sensitivities per stock.

Every available market contributes a factor vector. A shared encoder for each
category consumes the stock-scaled vectors before averaging their embeddings.
This accommodates new markets without a growing matrix of market-ID columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from baseline.model import MLPModel, _normalization, build_mlp

from .factors import CATEGORY_FACTORS


CATEGORIES = tuple(CATEGORY_FACTORS)
INPUT_SCHEMA_VERSION = 2
REBUILD_INSTRUCTIONS = (
    "Rebuild inputs and retrain with `python -m polymarket_ext classify --refresh`, "
    "`python -m polymarket_ext download` (reuses downloaded history), "
    "`python -m polymarket_ext prepare --refresh`, then `python -m polymarket_ext train --refresh`."
)


class GatedSetNetwork(nn.Module):
    def __init__(self, stock_count: int, alpha_size: int, hidden_size: int):
        super().__init__()
        # Row zero is the neutral sensitivity of stocks absent from training.
        self.sensitivities = nn.Embedding(stock_count + 1, len(CATEGORIES), padding_idx=0)
        nn.init.ones_(self.sensitivities.weight)
        self.encoders = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * len(CATEGORY_FACTORS[category]), hidden_size),
                          nn.Tanh(), nn.Linear(hidden_size, hidden_size), nn.Tanh())
            for category in CATEGORIES
        ])
        self.head = build_mlp(alpha_size + len(CATEGORIES) * (hidden_size + 2), hidden_size)

    def forward(self, alpha, stocks, markets):
        gates = self.sensitivities(stocks)
        inputs = [alpha]
        for category, (values, missing, present) in enumerate(markets):
            # Missing values have been filled with zero; masks bypass the gates.
            scaled = values * gates[:, category, None, None]
            encoded = self.encoders[category](torch.cat([scaled, missing], dim=-1))
            count = present.sum(dim=1, keepdim=True)
            pooled = (encoded * present[..., None]).sum(dim=1) / count.clamp(min=1)
            available = ((missing == 0) & present[..., None].bool()).any(dim=2).any(dim=1, keepdim=True)
            category_missing = (~available).to(values.dtype)
            inputs.extend([pooled, count.log1p(), category_missing])
        return self.head(torch.cat(inputs, dim=1)).squeeze(-1)


class PolymarketModel(MLPModel):
    """Jointly fit an Alpha158/set network and unrestricted stock sensitivities.

    Normalization and stock IDs are learned exclusively from the train segment.
    The configured epoch count is fixed; test observations never update parameters.
    Missing factors and categories have explicit masks. All active market rows
    contribute, including those with missing quotes. Dates without markets have
    zero pooled features and missing category flags. Sensitivities are relative
    learned model coefficients, not identified causal exposures (encoder weights
    can absorb their scale).
    """

    def __init__(self, factor_panel: pd.DataFrame | str | Path, epochs: int = 10,
                 batch_size: int = 512, hidden_size: int = 32,
                 learning_rate: float = 1e-3, weight_decay: float = 1e-4,
                 sensitivity_l2: float = 1e-3,
                 seed: int = 42, num_threads: int = 2):
        super().__init__(epochs=epochs, batch_size=batch_size, hidden_size=hidden_size,
                         learning_rate=learning_rate, weight_decay=weight_decay,
                         seed=seed, num_threads=num_threads)
        self.sensitivity_l2 = sensitivity_l2
        self.set_factor_panel(factor_panel)

    def set_factor_panel(self, panel: pd.DataFrame | str | Path) -> None:
        """Supply newly observed markets after loading a fitted model."""
        if self.network is not None:
            self._check_input_schema()
        frame = pd.read_parquet(panel) if isinstance(panel, (str, Path)) else panel.copy()
        if "datetime" not in frame:
            frame = frame.reset_index()
        if frame["category"].eq("direction").any():
            missing = {"WINDOW_LENGTH", "TIME_TO_END"} - set(frame.columns)
            if missing:
                raise ValueError(
                    f"Incompatible direction factor panel: missing {', '.join(sorted(missing))}. "
                    + REBUILD_INSTRUCTIONS
                )
        frame["datetime"] = pd.to_datetime(frame["datetime"]).dt.normalize()
        self.factor_panel = frame.sort_values(["datetime", "category", "market_id"]).reset_index(drop=True)
        if self.network is not None:
            self._index_markets()

    def _index_markets(self) -> None:
        self.market_values = {}
        for category in CATEGORIES:
            subset = self.factor_panel.loc[self.factor_panel["category"] == category]
            by_date = {}
            for date, frame in subset.groupby("datetime", sort=False):
                values = frame.reindex(columns=CATEGORY_FACTORS[category]).to_numpy(dtype="float32")
                missing = ~np.isfinite(values)
                normalized = (values - self.factor_mean[category]) / self.factor_scale[category]
                normalized = np.where(missing, 0, normalized).astype("float32")
                by_date[pd.Timestamp(date).value] = (normalized, missing.astype("float32"))
            self.market_values[category] = by_date

    def _arrays(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        alpha = super()._arrays(frame)[0]
        stocks = np.array([self.stock_ids.get(str(stock), 0)
                           for stock in frame.index.get_level_values("instrument")], dtype="int64")
        dates = pd.DatetimeIndex(frame.index.get_level_values("datetime")).normalize().as_unit("ns").asi8
        return alpha, stocks, dates

    def _batch(self, arrays, positions):
        alpha, stocks, dates = arrays
        selected_dates = dates[positions]
        markets = []
        for category in CATEGORIES:
            by_date = self.market_values[category]
            rows = [by_date.get(date) for date in selected_dates]
            width = max(1, max((len(row[0]) if row is not None else 0 for row in rows), default=0))
            values = np.zeros((len(positions), width, len(CATEGORY_FACTORS[category])), dtype="float32")
            missing = np.ones_like(values)
            present = np.zeros((len(positions), width), dtype="float32")
            for index, row in enumerate(rows):
                if row is not None:
                    length = len(row[0])
                    values[index, :length], missing[index, :length] = row
                    present[index, :length] = 1
            markets.append(tuple(torch.from_numpy(value) for value in (values, missing, present)))
        return torch.from_numpy(alpha[positions]), torch.from_numpy(stocks[positions]), markets

    def _initialize_network(self, train: pd.DataFrame) -> None:
        self.stock_ids = {stock: index + 1 for index, stock in enumerate(
            sorted(set(train.index.get_level_values("instrument").astype(str))))}
        train_dates = pd.DatetimeIndex(train.index.get_level_values("datetime")).normalize().unique()
        historical = self.factor_panel.loc[self.factor_panel["datetime"].isin(train_dates)]
        quoted = historical.loc[historical.get(
            "quote_valid", pd.Series(False, index=historical.index)).fillna(False).astype(bool)]
        self.training_coverage = {
            "quoted_market_rows": len(quoted),
            "quoted_days": int(quoted["datetime"].nunique()),
            "category_rows": {category: int((quoted["category"] == category).sum()) for category in CATEGORIES},
            "first_quoted_date": quoted["datetime"].min().strftime("%Y-%m-%d") if len(quoted) else None,
            "last_quoted_date": quoted["datetime"].max().strftime("%Y-%m-%d") if len(quoted) else None,
            "active_market_rows": len(historical),
            "active_days": int(historical["datetime"].nunique()),
            "category_active_rows": {},
            "category_missing_days": {},
        }
        self.factor_mean, self.factor_scale = {}, {}
        for category in CATEGORIES:
            category_rows = historical.loc[historical["category"] == category]
            values = category_rows.reindex(columns=CATEGORY_FACTORS[category]).to_numpy(dtype="float32")
            self.factor_mean[category], self.factor_scale[category] = _normalization(values)
            available_dates = category_rows.loc[np.isfinite(values).any(axis=1), "datetime"]
            self.training_coverage["category_active_rows"][category] = len(category_rows)
            self.training_coverage["category_missing_days"][category] = len(train_dates) - int(available_dates.nunique())
        self.network = GatedSetNetwork(len(self.stock_ids), len(self.feature_names), self.hidden_size)
        # Public attributes survive Qlib Serializable's default pickle filtering.
        self.input_schema_version = INPUT_SCHEMA_VERSION
        self._index_markets()

    def _check_input_schema(self) -> None:
        if getattr(self, "input_schema_version", None) != INPUT_SCHEMA_VERSION:
            raise ValueError(
                "Incompatible Polymarket model input schema for interval direction factors. "
                + REBUILD_INSTRUCTIONS
            )

    def predict(self, dataset, segment="test") -> pd.Series:
        if self.network is not None:
            self._check_input_schema()
        return super().predict(dataset, segment)

    def _optimizer(self):
        # Weight decay on the gate table would move even never-observed stocks.
        ordinary = [value for name, value in self.network.named_parameters()
                    if name != "sensitivities.weight"]
        return torch.optim.Adam([
            {"params": ordinary, "weight_decay": self.weight_decay},
            {"params": self.network.sensitivities.parameters(), "weight_decay": 0},
        ], lr=self.learning_rate)

    def _regularization(self):
        return self.sensitivity_l2 * (self.network.sensitivities.weight[1:] - 1).square().mean()

    def sensitivity_frame(self) -> pd.DataFrame:
        """One row per training stock, columns in the five-category order."""
        if self.network is None:
            raise ValueError("Model has not been fitted")
        self._check_input_schema()
        values = self.network.sensitivities.weight.detach().numpy()[1:]
        return pd.DataFrame(values, index=pd.Index(self.stock_ids, name="instrument"), columns=CATEGORIES)
