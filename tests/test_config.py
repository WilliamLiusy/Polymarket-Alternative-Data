from pathlib import Path

import pytest
import yaml

from baseline.__main__ import load_config


def test_default_config_uses_requested_protocol():
    config = load_config("configs/baseline.yaml")
    assert config["train_start"] == "2008-01-01"
    assert config["first_test_month"] == "2026-04-01"
    assert not {"common_start", "common_end", "lookback_months"}.intersection(config)
    assert config["model"]["early_stopping_rounds"] == 0
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
