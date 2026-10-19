from pathlib import Path

import pytest
import yaml

from baseline.__main__ import load_config


def test_default_config_uses_requested_protocol():
    config = load_config("configs/baseline.yaml")
    assert config["train_start"] == "2008-01-01"
    assert config["first_test_month"] == "2026-04-01"
    assert config["research_end"] == "2026-09-30"
    assert not {"common_start", "common_end", "lookback_months"}.intersection(config)
    assert "validation_months" not in config
    assert "early_stopping_rounds" not in config["model"]
    assert config["model"]["epochs"] == 10
    assert config["model"]["hidden_size"] == 32
    assert "num_boost_round" not in config["model"]
    assert Path(config["data_dir"]).is_absolute()


@pytest.mark.parametrize("key,value", [
    ("download_end", "2020-01-01"),
    ("first_test_month", "2026-04-15"),
    ("train_start", "2007-12-31"),
    ("train_start", "2026-04-01"),
    ("train_start", "2026-05-01"),
])
def test_invalid_dates_are_rejected(tmp_path, key, value):
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    config[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_config(path)


@pytest.mark.parametrize("months", [0, 3, -1, 1.5, None])
def test_removed_validation_months_are_rejected(tmp_path, months):
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    config["validation_months"] = months
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="validation_months"):
        load_config(path)


@pytest.mark.parametrize("rounds", [0, 3, None])
def test_removed_early_stopping_is_rejected(tmp_path, rounds):
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    config["model"]["early_stopping_rounds"] = rounds
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="early_stopping_rounds"):
        load_config(path)


@pytest.mark.parametrize("epochs", [0, -1, 1.5, True, False, "10", None])
def test_epochs_must_be_a_positive_integer(tmp_path, epochs):
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    config["model"]["epochs"] = epochs
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="epochs"):
        load_config(path)


def test_omitted_epochs_defaults_to_ten(tmp_path):
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    del config["model"]["epochs"]
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    assert load_config(path)["model"]["epochs"] == 10


@pytest.mark.parametrize("extra,parameter", [
    ({"validation_months": 0}, "validation_months"),
    ({"model": {"early_stopping_rounds": 0}}, "early_stopping_rounds"),
    ({"model": {"epochs": False}}, "epochs"),
])
def test_direct_train_rejects_stale_or_invalid_config_before_reading_data(extra, parameter):
    from baseline.train import train

    # Missing cache paths deliberately ensure config validation runs first.
    with pytest.raises(ValueError, match=parameter):
        train(extra)


def test_old_lightgbm_parameters_are_rejected(tmp_path):
    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    config["model"]["num_boost_round"] = 5
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="num_boost_round"):
        load_config(path)
