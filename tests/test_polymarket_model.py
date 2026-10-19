import json
import pickle

import numpy as np
import pandas as pd
import pytest
import torch
import yaml
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

from polymarket_ext.__main__ import load_config, validate_training_config
from polymarket_ext.factors import CATEGORY_FACTORS, VERSION as FACTOR_VERSION
from polymarket_ext.model import CATEGORIES, INPUT_SCHEMA_VERSION, GatedSetNetwork, PolymarketModel
from polymarket_ext.train import train


@pytest.mark.parametrize("epochs", [0, -1, 1.5, True, "2", None])
def test_extension_rejects_invalid_epoch_overrides(epochs):
    with pytest.raises(ValueError, match="epochs"):
        validate_training_config({"model": {"epochs": epochs, "sensitivity_l2": 0.01}})


@pytest.mark.parametrize("removed", [
    {"validation_months": 0}, {"validation_months": 3},
    {"model": {"early_stopping_rounds": 0}}, {"model": {"early_stopping_rounds": 3}},
])
def test_extension_rejects_removed_validation_settings(removed):
    with pytest.raises(ValueError):
        validate_training_config(removed)


def test_extension_config_preserves_epoch_inheritance_and_explicit_override(tmp_path):
    with open("configs/polymarket.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    path = tmp_path / "polymarket.yaml"
    path.write_text(yaml.safe_dump(config))
    inherited = load_config(path)
    assert inherited["research_end"] == "2026-09-30"
    assert inherited["model"] == {"sensitivity_l2": 0.001}
    config["model"]["epochs"] = 4
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["model"]["epochs"] == 4


def test_all_market_factor_values_are_gated_before_pooling():
    torch.manual_seed(4)
    network = GatedSetNetwork(stock_count=1, alpha_size=2, hidden_size=4)
    gates = torch.tensor([-2.0, 3.0, 4.0, 5.0, 6.0])
    with torch.no_grad():
        network.sensitivities.weight[1] = gates
    observed_inputs = []
    hooks = [encoder.register_forward_pre_hook(lambda module, args: observed_inputs.append(args[0]))
             for encoder in network.encoders]
    markets = []
    for category in CATEGORIES:
        width = len(CATEGORY_FACTORS[category])
        values = torch.arange(1, 1 + 2 * width, dtype=torch.float32).reshape(1, 2, width) / 10
        markets.append((values, torch.zeros_like(values), torch.ones((1, 2))))
    output = network(torch.ones((1, 2)), torch.tensor([1]), markets)
    for index, inputs in enumerate(observed_inputs):
        width = len(CATEGORY_FACTORS[CATEGORIES[index]])
        torch.testing.assert_close(inputs[..., :width], markets[index][0] * gates[index])
        torch.testing.assert_close(inputs[..., width:], markets[index][1])
    output.sum().backward()
    assert torch.all(network.sensitivities.weight.grad[1] != 0)
    assert torch.all(network.sensitivities.weight.grad[0] == 0)
    for hook in hooks:
        hook.remove()


def test_direction_time_factors_keep_five_encoders_and_default_head_width():
    network = GatedSetNetwork(stock_count=2, alpha_size=158, hidden_size=32)
    assert len(CATEGORIES) == len(network.encoders) == 5
    assert network.sensitivities.weight.shape == (3, 5)
    assert CATEGORY_FACTORS["direction"] == (
        "PROB", "ENTROPY", "DIRECTION_SCORE", "DIRECTION_CERTAINTY", "WINDOW_LENGTH", "TIME_TO_END",
    )
    assert network.encoders[CATEGORIES.index("direction")][0].in_features == 12
    assert network.head[0].in_features == 328


def sample_data():
    dates = pd.bdate_range("2026-01-05", periods=9)
    index = pd.MultiIndex.from_product([dates, ["SH600000", "SZ000001"]], names=["datetime", "instrument"])
    columns = pd.MultiIndex.from_tuples([("feature", "X"), ("label", "LABEL0")])
    values = np.column_stack([np.repeat(np.arange(len(dates)), 2), np.tile([-1.0, 1.0], len(dates))])
    frame = pd.DataFrame(values, index=index, columns=columns)
    panel = []
    for day, date in enumerate(dates):
        for category, factors in CATEGORY_FACTORS.items():
            row = {"datetime": date, "category": category, "market_id": f"{category}-{day}",
                   "token_id": f"token-{category}-{day}", "quote_valid": True}
            row.update({factor: (day + 1) * (position + 1) / 10 for position, factor in enumerate(factors)})
            panel.append(row)
    segments = {"train": (dates[0], dates[5]), "test": (dates[8], dates[8])}
    return frame, pd.DataFrame(panel), segments


def test_category_missing_separates_empty_zero_and_all_missing_markets():
    network = GatedSetNetwork(stock_count=1, alpha_size=2, hidden_size=4)
    head_inputs = []
    hook = network.head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))
    markets = []
    for category in CATEGORIES:
        size = len(CATEGORY_FACTORS[category])
        values = torch.zeros((4, 2, size))
        missing = torch.ones_like(values)
        present = torch.tensor([[0., 0.], [1., 0.], [1., 0.], [1., 1.]])
        # Padding has deliberately misleading values and masks; it must be ignored.
        values[0] = 1e3
        values[1:3, 1] = 1e3
        missing[0] = 0
        missing[1:3, 1] = 0
        missing[1, 0] = 0  # Real, observed zeros.
        missing[3, 1, 0] = 0  # One finite factor makes this category available.
        markets.append((values, missing, present))
    output = network(torch.zeros((4, 2)), torch.ones(4, dtype=torch.long), markets)
    hook.remove()
    assert torch.isfinite(output).all()
    head = head_inputs[0]
    assert head.shape == (4, 2 + len(CATEGORIES) * (4 + 2))
    for index in range(len(CATEGORIES)):
        offset = 2 + index * 6
        values, missing, present = markets[index]
        encoded = network.encoders[index](torch.cat([values, missing], dim=-1))
        for row in range(len(present)):
            real_rows = present[row].bool()
            expected = encoded[row, real_rows].mean(dim=0) if real_rows.any() else torch.zeros(4)
            torch.testing.assert_close(head[row, offset:offset + 4], expected)
        torch.testing.assert_close(head[:, offset + 4], torch.tensor([0., 1., 1., 2.]).log1p())
        torch.testing.assert_close(head[:, offset + 5], torch.tensor([1., 0., 1., 0.]))


def test_missing_markers_train_the_head_without_task_loss_sensitivity_gradients():
    network = GatedSetNetwork(stock_count=2, alpha_size=1, hidden_size=4)
    # Isolate the missing indicators' direct contribution to the prediction head.
    with torch.no_grad():
        network.head[0].weight.zero_()
        network.head[0].bias.zero_()
        network.head[2].weight.fill_(1)
        network.head[2].bias.zero_()
    markets = []
    for category in CATEGORIES:
        values = torch.zeros((2, 1, len(CATEGORY_FACTORS[category])))
        markets.append((values, torch.ones_like(values), torch.tensor([[0.], [1.]])))
    prediction = network(torch.zeros((2, 1)), torch.tensor([1, 2]), markets)
    (prediction - 1).square().mean().backward()
    torch.testing.assert_close(network.sensitivities.weight.grad, torch.zeros((3, len(CATEGORIES))))
    for index in range(len(CATEGORIES)):
        missing_column = 1 + index * 6 + 5
        assert torch.all(network.head[0].weight.grad[:, missing_column] != 0)


def test_partial_missing_masks_bypass_stock_sensitivities():
    torch.manual_seed(4)
    network = GatedSetNetwork(stock_count=1, alpha_size=1, hidden_size=4)
    with torch.no_grad():
        network.sensitivities.weight[1].fill_(-2)
    encoder_inputs = []
    hook = network.encoders[0].register_forward_pre_hook(lambda module, args: encoder_inputs.append(args[0]))
    markets = []
    for index, category in enumerate(CATEGORIES):
        values = torch.zeros((1, 1, len(CATEGORY_FACTORS[category])))
        missing = torch.ones_like(values)
        if index == 0:
            values[0, 0, 0] = 0.5
            missing[0, 0, 0] = 0
        markets.append((values, missing, torch.ones((1, 1))))
    network(torch.zeros((1, 1)), torch.tensor([1]), markets).sum().backward()
    hook.remove()
    width = len(CATEGORY_FACTORS[CATEGORIES[0]])
    torch.testing.assert_close(encoder_inputs[0][..., :width], markets[0][0] * -2)
    torch.testing.assert_close(encoder_inputs[0][..., width:], markets[0][1])
    assert network.sensitivities.weight.grad[1, 0] != 0
    torch.testing.assert_close(network.sensitivities.weight.grad[1, 1:], torch.zeros(len(CATEGORIES) - 1))


def test_batches_keep_unquoted_rows_metadata_factors_and_nonfinite_masks():
    frame, _, segments = sample_data()
    dates = frame.index.get_level_values("datetime").unique()
    direction = CATEGORY_FACTORS["direction"]
    panel = pd.DataFrame([
        {"datetime": dates[1], "category": "direction", "market_id": "zero", "quote_valid": True,
         **dict.fromkeys(direction, 0.0)},
        {"datetime": dates[2], "category": "direction", "market_id": "no-quote", "quote_valid": False},
        {"datetime": dates[3], "category": "threshold", "market_id": "metadata", "quote_valid": False,
         "THRESHOLD_REL_POSITION": 0.0},
        {"datetime": dates[4], "category": "direction", "market_id": "a-partial", "quote_valid": True,
         "PROB": 0.0, "DIRECTION_SCORE": np.inf, "DIRECTION_CERTAINTY": -np.inf},
        {"datetime": dates[4], "category": "direction", "market_id": "b-finite", "quote_valid": True,
         **dict.fromkeys(direction, 2.0)},
        {"datetime": dates[8], "category": "threshold", "market_id": "future", "quote_valid": False,
         "THRESHOLD_REL_POSITION": 1e6},
    ])
    model = PolymarketModel(panel, hidden_size=4)
    model.feature_names, model.alpha_mean, model.alpha_scale = ["X"], np.array([0.]), np.array([1.])
    model._initialize_network(frame.loc[:segments["train"][1]])
    assert len(model.factor_panel) == len(panel)
    np.testing.assert_allclose(model.factor_mean["direction"], [2 / 3, 1, 1, 1, 1, 1])
    np.testing.assert_array_equal(model.factor_mean["threshold"], np.zeros(5))
    features = frame["feature"].iloc[::2]
    _, _, markets = model._batch(model._arrays(features), np.arange(6))
    values, missing, present = markets[CATEGORIES.index("direction")]
    assert values.shape == (6, 2, len(direction))
    torch.testing.assert_close(present, torch.tensor([[0., 0.], [1., 0.], [1., 0.],
                                                     [0., 0.], [1., 1.], [0., 0.]]))
    torch.testing.assert_close(missing[0], torch.ones((2, len(direction))))
    torch.testing.assert_close(missing[1, 0], torch.zeros(len(direction)))
    torch.testing.assert_close(missing[1, 1], torch.ones(len(direction)))
    torch.testing.assert_close(missing[2], torch.ones((2, len(direction))))
    torch.testing.assert_close(missing[4, 0], torch.tensor([0., 1., 1., 1., 1., 1.]))
    assert torch.isfinite(values).all()
    assert torch.all(values[missing.bool()] == 0)
    _, threshold_missing, threshold_present = markets[CATEGORIES.index("threshold")]
    position = CATEGORY_FACTORS["threshold"].index("THRESHOLD_REL_POSITION")
    expected = torch.ones(len(CATEGORY_FACTORS["threshold"]))
    expected[position] = 0
    torch.testing.assert_close(threshold_missing[3, 0], expected)
    assert threshold_present[3, 0] == 1

    coverage = model.training_coverage
    assert coverage["quoted_market_rows"] == 3
    assert coverage["quoted_days"] == 2
    assert coverage["active_market_rows"] == 5
    assert coverage["active_days"] == 4
    assert coverage["category_rows"]["threshold"] == 0
    assert coverage["category_active_rows"]["threshold"] == 1
    assert coverage["category_missing_days"] == {
        "direction": 4, "threshold": 5, "touch": 6, "range": 6, "record": 6,
    }


def test_early_missing_training_and_mixed_inference_survive_pickle(tmp_path):
    frame, panel, _ = sample_data()
    old_dates = frame.index.get_level_values("datetime").unique()
    dates = pd.bdate_range("2008-01-02", periods=6).append(old_dates[-3:])
    frame.index = pd.MultiIndex.from_product([dates, ["SH600000", "SZ000001"]],
                                           names=["datetime", "instrument"])
    panel = panel.loc[(panel["datetime"] >= dates[7]) & panel["category"].eq("direction")].copy()
    panel.loc[:, list(CATEGORY_FACTORS["direction"])] = 0.0
    panel.loc[panel["datetime"] == dates[8], "quote_valid"] = False
    panel.loc[panel["datetime"] == dates[8], list(CATEGORY_FACTORS["direction"])] = np.nan
    segments = {"train": (dates[0], dates[5]),
                "test": (dates[8], dates[8]), "all": (dates[0], dates[8])}
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = PolymarketModel(panel, epochs=2, batch_size=4, hidden_size=4, num_threads=1)
    model.fit(dataset)
    assert model.input_schema_version == INPUT_SCHEMA_VERSION == 2
    assert model.training_coverage["active_market_rows"] == 0
    assert all(value == 6 for value in model.training_coverage["category_missing_days"].values())
    np.testing.assert_array_equal(model.sensitivity_frame(), np.ones((2, 5)))
    original = model.predict(dataset, "all")
    assert len(original) == len(frame)
    assert np.isfinite(original).all()
    model.to_pickle(tmp_path / "missing-model.pkl")
    with (tmp_path / "missing-model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    assert restored.input_schema_version == INPUT_SCHEMA_VERSION
    pd.testing.assert_series_equal(restored.predict(dataset, "all"), original)
    restored.set_factor_panel(panel)
    pd.testing.assert_series_equal(restored.predict(dataset, "all"), original)


def test_unquoted_direction_markets_learn_from_different_horizons_and_survive_pickle(tmp_path):
    frame, _, segments = sample_data()
    dates = frame.index.get_level_values("datetime").unique()
    panel = pd.DataFrame([
        {"datetime": date, "category": "direction", "market_id": f"{horizon}-{day}",
         "quote_valid": False,
         **dict.fromkeys(CATEGORY_FACTORS["direction"][:4], np.nan),
         "WINDOW_LENGTH": np.log1p(horizon),
         "TIME_TO_END": np.log1p(remaining)}
        for day, date in enumerate(dates)
        for horizon, remaining in ((1, 0.5), (30, 20 - day))
    ])
    # A non-finite remaining time is missing even though the window length exists.
    panel.loc[panel.index[-1], "TIME_TO_END"] = np.inf
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = PolymarketModel(panel, epochs=3, batch_size=6, hidden_size=8,
                            learning_rate=0.02, num_threads=1)
    model.fit(dataset)
    assert model.training_coverage["quoted_market_rows"] == 0
    assert model.training_coverage["category_missing_days"]["direction"] == 0
    np.testing.assert_array_equal(model.factor_mean["direction"][:4], np.zeros(4))
    training_times = panel.loc[panel["datetime"] <= segments["train"][1], ["WINDOW_LENGTH", "TIME_TO_END"]]
    np.testing.assert_allclose(model.factor_mean["direction"][4:], training_times.mean(), rtol=1e-6)
    np.testing.assert_allclose(model.factor_scale["direction"][4:], training_times.std(ddof=0), rtol=1e-6)
    assert np.any(np.abs(model.sensitivity_frame()["direction"].to_numpy() - 1) > 1e-5)

    features = frame.loc[segments["test"][0]:, "feature"]
    batch = model._batch(model._arrays(features), np.arange(len(features)))
    values, missing, present = batch[2][CATEGORIES.index("direction")]
    assert values.shape == (2, 2, 6)
    torch.testing.assert_close(missing[..., :4], torch.ones((2, 2, 4)))
    torch.testing.assert_close(missing[..., 4], torch.zeros((2, 2)))
    torch.testing.assert_close(missing[..., 5], torch.tensor([[0., 1.], [0., 1.]]))
    assert torch.all(values[missing.bool()] == 0)
    assert torch.all(values[:, 0, 4] != values[:, 1, 4])
    torch.testing.assert_close(present, torch.ones((2, 2)))
    head_inputs = []
    hook = model.network.head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))
    prediction = model.predict(dataset)
    hook.remove()
    offset = len(model.feature_names) + CATEGORIES.index("direction") * (model.hidden_size + 2)
    torch.testing.assert_close(head_inputs[0][:, offset + model.hidden_size], torch.full((2,), np.log(3)))
    torch.testing.assert_close(head_inputs[0][:, offset + model.hidden_size + 1], torch.zeros(2))
    assert np.isfinite(prediction).all()
    model.to_pickle(tmp_path / "horizon-model.pkl")
    with (tmp_path / "horizon-model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    assert restored.input_schema_version == 2
    pd.testing.assert_series_equal(restored.predict(dataset), prediction)


def test_range_with_only_missing_d5_remains_an_available_category():
    frame, _, segments = sample_data()
    date = frame.index.get_level_values("datetime").unique()[1]
    panel = pd.DataFrame([{
        "datetime": date, "category": "range", "market_id": "new-range", "quote_valid": True,
        "PROB": 0.0, "D1": 0.0, "D5": np.nan,
        "RANGE_REL_WIDTH": 0.5, "RANGE_REL_DENSITY": 0.0,
    }])
    model = PolymarketModel(panel, hidden_size=4)
    model.feature_names, model.alpha_mean, model.alpha_scale = ["X"], np.array([0.]), np.array([1.])
    model._initialize_network(frame.loc[:segments["train"][1]])
    features = frame["feature"].iloc[::2]
    batch = model._batch(model._arrays(features), np.array([1]))
    category_index = CATEGORIES.index("range")
    values, missing, present = batch[2][category_index]
    expected_missing = torch.zeros_like(missing)
    expected_missing[..., CATEGORY_FACTORS["range"].index("D5")] = 1
    torch.testing.assert_close(missing, expected_missing)
    torch.testing.assert_close(present, torch.ones((1, 1)))
    assert torch.isfinite(values).all()
    head_inputs = []
    hook = model.network.head.register_forward_pre_hook(lambda module, args: head_inputs.append(args[0]))
    prediction = model.network(*batch)
    hook.remove()
    assert torch.isfinite(prediction).all()
    offset = len(model.feature_names) + category_index * (model.hidden_size + 2)
    assert head_inputs[0][0, offset + model.hidden_size + 1] == 0
    torch.testing.assert_close(head_inputs[0][0, offset + model.hidden_size], torch.tensor(1.).log1p())


@pytest.mark.parametrize("schema", [None, 1, INPUT_SCHEMA_VERSION + 1])
@pytest.mark.parametrize("operation", ["predict", "set_factor_panel", "sensitivity_frame"])
def test_incompatible_fitted_models_require_explicit_retraining(schema, operation):
    frame, panel, segments = sample_data()
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = PolymarketModel(panel, epochs=1, hidden_size=4, num_threads=1)
    model.fit(dataset)
    if schema is None:
        del model.input_schema_version
    else:
        model.input_schema_version = schema
    with pytest.raises(ValueError) as error:
        if operation == "predict":
            model.predict(dataset)
        elif operation == "set_factor_panel":
            model.set_factor_panel(panel)
        else:
            model.sensitivity_frame()
    assert "retrain" in str(error.value).lower()
    for command in ("classify --refresh", "download", "prepare --refresh", "train --refresh"):
        assert command in str(error.value)


@pytest.mark.parametrize("missing_columns", [
    ["WINDOW_LENGTH"], ["TIME_TO_END"], ["WINDOW_LENGTH", "TIME_TO_END"],
])
def test_legacy_direction_panel_is_rejected_before_missing_values_are_filled(missing_columns):
    _, panel, _ = sample_data()
    with pytest.raises(ValueError, match="Incompatible direction factor panel") as error:
        PolymarketModel(panel.drop(columns=missing_columns))
    for command in ("classify --refresh", "download", "prepare --refresh", "train --refresh"):
        assert command in str(error.value)


def test_qlib_fit_learns_sensitivities_without_test_statistics_and_serializes(tmp_path):
    frame, panel, segments = sample_data()
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = PolymarketModel(panel, epochs=3, batch_size=6, hidden_size=8,
                            learning_rate=0.02, num_threads=1)
    model.fit(dataset)
    assert model.trained_epochs == len(model.history) == 3
    assert all(set(record) == {"epoch", "train_mse"} for record in model.history)
    assert model.alpha_mean[0] == pytest.approx(2.5)
    assert model.factor_mean["direction"][0] == pytest.approx(0.35)
    assert model.sensitivity_frame().shape == (2, 5)
    assert np.any(np.abs(model.sensitivity_frame().to_numpy() - 1) > 1e-5)
    torch.testing.assert_close(model.network.sensitivities.weight[0], torch.ones(5))
    original_prediction = model.predict(dataset)
    model.to_pickle(tmp_path / "model.pkl")
    with (tmp_path / "model.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    pd.testing.assert_series_equal(restored.predict(dataset), original_prediction)

    # Future market identities/values and future Alpha158 values do not alter fitting.
    changed_frame = frame.copy()
    changed_frame.loc[segments["test"][0], ("feature", "X")] = 1e6
    changed_panel = panel.copy()
    future = changed_panel["datetime"] == segments["test"][0]
    changed_panel.loc[future, ["PROB", "WINDOW_LENGTH", "TIME_TO_END"]] = 1e6
    changed_panel.loc[future, "market_id"] = "brand-new-market"
    changed = PolymarketModel(changed_panel, epochs=3, batch_size=6, hidden_size=8,
                             learning_rate=0.02, num_threads=1)
    changed.fit(DatasetH(handler=DataHandlerLP.from_df(changed_frame), segments=segments))
    for name, value in model.network.state_dict().items():
        torch.testing.assert_close(changed.network.state_dict()[name], value, rtol=0, atol=0)


def test_new_stock_and_new_event_require_no_market_specific_training_columns():
    frame, panel, segments = sample_data()
    dataset = DatasetH(handler=DataHandlerLP.from_df(frame), segments=segments)
    model = PolymarketModel(panel, epochs=1, batch_size=6, hidden_size=4, num_threads=1)
    model.fit(dataset)
    new_frame = frame.loc[segments["test"][0]:].copy()
    new_frame.index = pd.MultiIndex.from_product(
        [[segments["test"][0]], ["SH600999", "SZ000999"]], names=["datetime", "instrument"])
    new_panel = panel.loc[panel["datetime"] == segments["test"][0]].copy()
    new_panel["market_id"] = "unseen"
    model.set_factor_panel(new_panel)
    prediction = model.predict(DatasetH(handler=DataHandlerLP.from_df(new_frame), segments=segments))
    assert len(prediction) == 2
    assert np.isfinite(prediction).all()
    assert model._arrays(new_frame["feature"])[1].tolist() == [0, 0]


@pytest.mark.parametrize("quoted", [True, False])
def test_monthly_training_uses_all_known_history_fixed_epochs_and_resumes(tmp_path, monkeypatch, quoted):
    cache, provider, panels, output = [tmp_path / name for name in ("cache", "provider", "panels", "output")]
    cache.mkdir()
    panels.mkdir()
    (provider / "calendars").mkdir(parents=True)
    calendar = pd.bdate_range("2025-01-01", "2026-04-20")
    (provider / "calendars" / "day.txt").write_text("\n".join(calendar.strftime("%Y-%m-%d")))
    baseline = {
        "data_dir": str(tmp_path / "data"), "provider_dir": str(provider), "cache_dir": str(cache),
        "output_dir": str(tmp_path / "baseline_output"), "research_start": "2025-01-01",
        "research_end": "2026-04-10", "train_start": "2025-01-01", "first_test_month": "2026-04-01",
        "download_start": "2025-01-01", "download_end": "2026-04-20",
        "model": {"epochs": 2}, "qlib": {"kernels": 1},
    }
    baseline_path = tmp_path / "baseline.yaml"
    baseline_path.write_text(yaml.safe_dump(baseline))
    manifest = {"fingerprint": "cache-id", "provider_fingerprint": "provider-id", "years": [2025, 2026],
                "research_start": baseline["research_start"], "research_end": baseline["research_end"],
                "year_files": {}}
    for year in manifest["years"]:
        dates = calendar[calendar.year == year]
        index = pd.MultiIndex.from_product([dates, ["SH600000", "SZ000001"]], names=["datetime", "instrument"])
        columns = pd.MultiIndex.from_tuples([("feature", "X"), ("label", "LABEL0"), ("raw_label", "LABEL0")])
        labels = np.tile([-1.0, 1.0], len(dates))
        annual = pd.DataFrame(np.column_stack([labels, labels, labels / 100]), index=index, columns=columns)
        annual.to_parquet(cache / f"{year}.parquet")
        manifest["year_files"][str(year)] = {"bytes": (cache / f"{year}.parquet").stat().st_size}
    (cache / "manifest.json").write_text(json.dumps(manifest))
    (provider / "manifest.json").write_text(json.dumps({"fingerprint": "provider-id"}))
    _, panel, _ = sample_data()
    if not quoted:
        panel = panel.loc[panel["category"].eq("direction")].copy()
        panel["quote_valid"] = False
        panel.loc[:, list(CATEGORY_FACTORS["direction"][:4])] = np.nan
    panel.to_parquet(panels / "factors.parquet", index=False)
    (panels / "manifest.json").write_text(json.dumps({
        "version": FACTOR_VERSION,
        "research_start": baseline["research_start"], "research_end": baseline["research_end"],
    }))
    config = {"baseline_config": str(baseline_path), "panel_dir": str(panels), "output_dir": str(output),
              "research_start": baseline["research_start"], "research_end": baseline["research_end"],
              "model": {"batch_size": 128, "hidden_size": 4, "num_threads": 1}}
    fits = []
    original_fit = PolymarketModel.fit

    def record_fit(model, dataset, **kwargs):
        training = dataset.prepare("train", col_set="label")
        fits.append({"epochs": model.epochs, "segments": dict(dataset.segments), "index": training.index})
        return original_fit(model, dataset, **kwargs)

    monkeypatch.setattr(PolymarketModel, "fit", record_fit)
    summary = train(config)
    assert summary["months"] == 1
    assert summary["evaluated_days"] == 6
    assert summary["training_scheme"] == "expanding_monthly_fixed_epochs"
    assert len(fits) == 1
    assert fits[0]["epochs"] == 2
    assert set(fits[0]["segments"]) == {"train", "test"}
    assert fits[0]["segments"]["train"] == ("2025-01-01", "2026-03-27")
    expected_index = pd.MultiIndex.from_product(
        [calendar[calendar <= "2026-03-27"], ["SH600000", "SZ000001"]],
        names=["datetime", "instrument"],
    )
    pd.testing.assert_index_equal(fits[0]["index"], expected_index)
    assert (output / "2026-04" / "model.pkl").is_file()
    gates = pd.read_csv(output / "2026-04" / "sensitivities.csv", index_col="instrument")
    assert gates.shape == (2, 5)
    # Recent market rows now participate in training and update stock sensitivities.
    assert np.any(np.abs(gates.to_numpy() - 1) > 1e-5)
    run = json.loads((output / "2026-04" / "run.json").read_text())
    assert run["training_polymarket_coverage"]["quoted_market_rows"] == (len(panel) if quoted else 0)
    assert run["training_polymarket_coverage"]["active_market_rows"] == len(panel)
    assert run["trained_epochs"] == 2
    assert "best_epoch" not in run
    assert "validation_months" not in run
    assert run["training_scheme"] == "expanding_monthly_fixed_epochs"
    history = pd.read_csv(output / "2026-04" / "training_history.csv")
    assert history["epoch"].tolist() == [1, 2]
    assert "valid_mse" not in history
    assert run["input_schema_version"] == INPUT_SCHEMA_VERSION
    assert run["missing_semantics"] == {
        "factor": "1 for non-finite values; 0 for finite values, including zero",
        "category": "1 if no real market row has any finite factor; 0 otherwise",
        "padding": "excluded from pooling, counts and category availability",
        "values": "missing values filled with 0 after normalization; only values scaled by stock sensitivities",
    }
    assert "all active market rows" in run["pooling"]
    assert not list((output / "2026-04").glob("validation*"))
    predictions = pd.read_parquet(output / "predictions.parquet")
    assert predictions.loc["2026-04-09":, "label"].isna().all()
    original_mtime = (output / "2026-04" / "model.pkl").stat().st_mtime_ns
    assert train(config) == summary
    assert len(fits) == 1
    assert (output / "2026-04" / "model.pkl").stat().st_mtime_ns == original_mtime

    config["model"]["epochs"] = 1
    with pytest.raises(ValueError, match="--refresh"):
        train(config)
    assert len(fits) == 1
    train(config, refresh=True)
    assert len(fits) == 2
    assert fits[1]["epochs"] == 1
    rerun = json.loads((output / "2026-04" / "run.json").read_text())
    assert rerun["trained_epochs"] == 1

    old_manifest = json.loads((panels / "manifest.json").read_text())
    old_manifest["version"] = FACTOR_VERSION - 1
    (panels / "manifest.json").write_text(json.dumps(old_manifest))
    with pytest.raises(ValueError, match="Incompatible Polymarket factor panel version") as error:
        train(config, refresh=True)
    for command in ("classify --refresh", "download", "prepare --refresh", "train --refresh"):
        assert command in str(error.value)
    assert len(fits) == 2
