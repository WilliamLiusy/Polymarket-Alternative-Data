from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

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
    assert config["model"]["epochs"] == 20
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


def test_timestamped_output_dir_uses_beijing_time(tmp_path):
    from baseline.__main__ import timestamped_output_dir

    now = datetime(2026, 10, 22, 18, 2, 3, 456789, tzinfo=timezone.utc)
    assert timestamped_output_dir(tmp_path, now=now) == tmp_path / "20261023T020203456789"


@pytest.mark.parametrize("command", ["train", "all"])
def test_training_cli_creates_timestamped_run_and_saves_effective_config(tmp_path, monkeypatch, command):
    import baseline.__main__ as cli
    import baseline.download as download_module
    import baseline.prepare as prepare_module
    import baseline.train as train_module

    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    output_root = tmp_path / "outputs"
    config["output_dir"] = str(output_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    expected = load_config(config_path)
    run_dir = output_root / "20261022T180203456789"
    monkeypatch.setattr(cli, "timestamped_output_dir", lambda root: Path(root) / run_dir.name)
    calls = []

    def observe(stage):
        def run(effective_config, refresh=False):
            calls.append((stage, deepcopy(effective_config), refresh))
            if stage == "train":
                assert json.loads((run_dir / "training_config.json").read_text()) == effective_config
                assert yaml.safe_load((run_dir / "training_config.yaml").read_text()) == effective_config
            return {"stage": stage}

        return run

    monkeypatch.setattr(download_module, "download", observe("download"))
    monkeypatch.setattr(prepare_module, "prepare", observe("prepare"))
    monkeypatch.setattr(train_module, "train", observe("train"))
    monkeypatch.setattr(sys, "argv", ["baseline", command, "--config", str(config_path)])
    cli.main()

    assert [stage for stage, _, _ in calls] == (["train"] if command == "train" else ["download", "prepare", "train"])
    expected["output_dir"] = str(run_dir)
    assert all(effective_config == expected and refresh is False for _, effective_config, refresh in calls)
    assert list(output_root.iterdir()) == [run_dir]
    assert json.loads((run_dir / "training_config.json").read_text()) == expected
    assert yaml.safe_load((run_dir / "training_config.yaml").read_text()) == expected
    assert load_config(config_path)["output_dir"] == str(output_root)


@pytest.mark.parametrize("command", ["download", "prepare"])
def test_nontraining_cli_does_not_create_run_directory(tmp_path, monkeypatch, command):
    import baseline.__main__ as cli
    import baseline.download as download_module
    import baseline.prepare as prepare_module

    config = yaml.safe_load(Path("configs/baseline.yaml").read_text())
    output_root = tmp_path / "outputs"
    config["output_dir"] = str(output_root)
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(yaml.safe_dump(config))
    calls = []

    def run(effective_config, refresh=False):
        calls.append(deepcopy(effective_config))

    module = download_module if command == "download" else prepare_module
    monkeypatch.setattr(module, command, run)
    monkeypatch.setattr(sys, "argv", ["baseline", command, "--config", str(config_path)])
    cli.main()

    assert calls == [load_config(config_path)]
    assert not output_root.exists()


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
