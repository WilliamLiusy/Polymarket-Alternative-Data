import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from baseline.model import MLPModel, _normalization


def sample_data():
    dates = pd.bdate_range("2026-01-05", periods=12)
    index = pd.MultiIndex.from_product([dates, ["SH600000", "SZ000001"]], names=["datetime", "instrument"])
    columns = pd.MultiIndex.from_tuples([
        ("feature", "X"), ("feature", "MISSING"), ("label", "LABEL0"),
    ])
    values = np.tile([-1.0, 1.0], len(dates))
    frame = pd.DataFrame(np.column_stack([values, np.full(len(values), np.nan), values]),
                         index=index, columns=columns)
    segments = {"train": (dates[0], dates[7]), "valid": (dates[8], dates[9]),
                "test": (dates[10], dates[11])}
    return frame, segments


def test_mlp_learns_serializes_and_ignores_extra_validation_and_test_values(tmp_path):
    frame, segments = sample_data()
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    settings = dict(epochs=10, batch_size=8, hidden_size=8, learning_rate=0.03, num_threads=1)
    model = MLPModel(**settings)
    model.fit(dataset)
    np.testing.assert_array_equal(model.alpha_mean, [0, 0])
    np.testing.assert_array_equal(model.alpha_scale, [1, 1])
    assert model.history[-1]["train_mse"] < model.history[0]["train_mse"]
    assert model.trained_epochs == len(model.history) == settings["epochs"]
    assert not hasattr(model, "best_epoch")
    assert all("valid_mse" not in epoch for epoch in model.history)
    prediction = model.predict(dataset)
    assert prediction.index.equals(frame.loc[segments["test"][0]:].index)
    assert np.isfinite(prediction).all()
    assert prediction.corr(frame.loc[segments["test"][0]:, ("label", "LABEL0")]) > 0.99

    model.to_pickle(tmp_path / "model.pkl")
    with (tmp_path / "model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    pd.testing.assert_series_equal(restored.predict(dataset), prediction)

    changed_frame = frame.copy()
    changed_frame.loc[segments["valid"][0]:, ("feature", "X")] = 1e6
    changed_frame.loc[segments["valid"][0]:, ("label", "LABEL0")] = -1e6
    changed = MLPModel(**settings)
    changed.fit(DatasetH(handler=DataHandlerLP.from_df(changed_frame), segments=segments))
    assert changed.history == model.history
    np.testing.assert_array_equal(changed.alpha_mean, model.alpha_mean)
    np.testing.assert_array_equal(changed.alpha_scale, model.alpha_scale)
    for name, value in model.network.state_dict().items():
        torch.testing.assert_close(changed.network.state_dict()[name], value, rtol=0, atol=0)


def test_mlp_keeps_last_epoch_weights_even_if_earlier_validation_loss_was_better(monkeypatch):
    frame, segments = sample_data()
    # Observe an unused, reversed-label segment to distinguish last weights from
    # the weights an old validation-based selector would restore.
    frame.loc[segments["valid"][0]:segments["valid"][1], ("label", "LABEL0")] *= -1
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = MLPModel(epochs=10, batch_size=16, hidden_size=8, learning_rate=0.03, num_threads=1)
    optimizer_factory = model._optimizer
    weights, heldout_losses = [], []
    labels = frame.loc[segments["valid"][0]:segments["valid"][1], ("label", "LABEL0")]

    def recording_optimizer():
        optimizer = optimizer_factory()
        step = optimizer.step

        def record_step(*args, **kwargs):
            result = step(*args, **kwargs)
            weights.append({name: value.detach().clone() for name, value in model.network.state_dict().items()})
            heldout_losses.append(np.mean((model.predict(dataset, "valid") - labels) ** 2))
            return result

        monkeypatch.setattr(optimizer, "step", record_step)
        return optimizer

    monkeypatch.setattr(model, "_optimizer", recording_optimizer)
    model.fit(dataset)
    assert len(weights) == len(model.history) == model.trained_epochs == 10
    assert heldout_losses[-1] > min(heldout_losses)
    for name, value in model.network.state_dict().items():
        torch.testing.assert_close(value, weights[-1][name], rtol=0, atol=0)


def test_normalization_across_chunks_matches_finite_population_statistics():
    rng = np.random.default_rng(19)
    values = rng.normal(size=(25001, 6)).astype("float32")
    values[:, 1] = np.nan
    values[:, 2] = 7.5
    values[::3, 0] = np.nan
    values[::5, 3] = np.inf
    values[::7, 3] = -np.inf
    values[:, 4] = (1e8 + rng.normal(0, 20, len(values))).astype("float32")
    values[:, 5] = np.nan
    values[[0, 9000, 18000, -1], 5] = [-4, -2, 2, 4]
    original = values.copy()
    values.setflags(write=False)
    mean, scale = _normalization(values)
    expected_mean, expected_scale = [], []
    for column in original.T:
        finite = column[np.isfinite(column)].astype("float64")
        expected_mean.append(finite.mean() if len(finite) else 0)
        standard_deviation = finite.std() if len(finite) else 0
        expected_scale.append(standard_deviation if standard_deviation >= 1e-6 else 1)
    np.testing.assert_allclose(mean, np.array(expected_mean, dtype="float32"), rtol=1e-6)
    np.testing.assert_allclose(scale, np.array(expected_scale, dtype="float32"), rtol=1e-6)
    np.testing.assert_array_equal(values, original)
    assert mean.dtype == scale.dtype == np.dtype("float32")
    empty_mean, empty_scale = _normalization(np.empty((0, 3), dtype="float32"))
    np.testing.assert_array_equal(empty_mean, [0, 0, 0])
    np.testing.assert_array_equal(empty_scale, [1, 1, 1])


@pytest.mark.parametrize("columns", [["X", "Y"], ["Y", "EXTRA", "X"], ["X"]])
def test_arrays_normalize_independent_storage_and_align_missing_features(columns):
    frame = pd.DataFrame({"X": [1, np.nan, np.inf, -np.inf], "Y": [2, 4, 6, 8],
                          "EXTRA": [10, 20, 30, 40]}, dtype="float32")[columns]
    original = frame.copy(deep=True)
    model = MLPModel()
    model.feature_names = ["X", "Y"]
    model.alpha_mean = np.array([1, 2], dtype="float32")
    model.alpha_scale = np.array([2, 2], dtype="float32")
    (alpha,) = model._arrays(frame)
    expected_y = [0, 1, 2, 3] if "Y" in columns else [0, 0, 0, 0]
    np.testing.assert_array_equal(alpha, np.column_stack([np.zeros(4), expected_y]))
    assert alpha.dtype == np.dtype("float32")
    assert not np.shares_memory(alpha, frame.to_numpy())
    alpha[:] = 100
    pd.testing.assert_frame_equal(frame, original)


def test_fit_excludes_nonfinite_labels_from_normalization():
    frame, segments = sample_data()
    frame.loc[frame.index[:2], ("feature", "X")] = 1e6
    frame.loc[frame.index[0], ("label", "LABEL0")] = np.nan
    frame.loc[frame.index[1], ("label", "LABEL0")] = np.inf
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = MLPModel(epochs=1, hidden_size=4, num_threads=1)
    model.fit(dataset)
    np.testing.assert_array_equal(model.alpha_mean, [0, 0])
    np.testing.assert_array_equal(model.alpha_scale, [1, 1])
    assert np.isfinite(model.predict(dataset)).all()


@pytest.mark.parametrize("use_slices", [False, True])
def test_fit_uses_date_views_without_changing_shared_dataset(monkeypatch, use_slices):
    frame, segments = sample_data()
    frame = frame.astype("float32")
    original = frame.copy(deep=True)
    if use_slices:
        segments = {name: slice(*bounds) for name, bounds in segments.items()}
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = MLPModel(epochs=1, hidden_size=4, num_threads=1)
    initialize = model._initialize_network
    arrays = model._arrays
    observed_feature_lengths = []

    def check_training_view(train):
        expected = original.loc["2026-01-05":"2026-01-14"]
        pd.testing.assert_frame_equal(train, expected)
        assert np.shares_memory(train.to_numpy(), frame.to_numpy())
        initialize(train)

    def check_feature_view(features):
        assert np.shares_memory(features.to_numpy(), frame.to_numpy())
        observed_feature_lengths.append(len(features))
        return arrays(features)

    monkeypatch.setattr(model, "_initialize_network", check_training_view)
    monkeypatch.setattr(model, "_arrays", check_feature_view)
    model.fit(dataset)
    assert observed_feature_lengths == [16]
    pd.testing.assert_frame_equal(frame, original)
