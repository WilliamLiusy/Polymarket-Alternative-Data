"""Run ``python -m polymarket_ext <stage> --config configs/polymarket.yaml``."""

import argparse
from datetime import date, datetime
import json
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


def validate_training_config(config):
    """Validate common MLP overrides without changing baseline inheritance."""
    from baseline.__main__ import validate_training_config as validate_common_training

    common = {**config, "model": dict(config.get("model", {}))}
    common["model"].pop("sensitivity_l2", None)
    validate_common_training(common)


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for key in ("data_dir", "panel_dir", "calendar_path", "baseline_config", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    for section, key in (("collection", "events_file"), ("classification", "overrides_file")):
        if config.get(section, {}).get(key):
            config[section][key] = str(Path(config[section][key]).expanduser().resolve())
    for key in ("research_start", "research_end"):
        config[key] = date.fromisoformat(str(config[key])).isoformat()
    if config["research_start"] > config["research_end"]:
        raise ValueError("research_start must not exceed research_end")
    validate_training_config(config)
    return config


def timestamped_output_dir(output_dir, now=None):
    """Return a unique per-run output directory using Beijing local time."""
    now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = now.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    return Path(output_dir) / run_id


def main():
    parser = argparse.ArgumentParser(description="Polymarket 原油数据、收盘因子与 Qlib 敏感度模型")
    parser.add_argument("command", choices=("discover", "classify", "download", "prepare", "train", "all"))
    parser.add_argument("--config", default="configs/polymarket.yaml")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    stages = ("discover", "classify", "download", "prepare", "train") if args.command == "all" else (args.command,)
    if "train" in stages:
        config["output_dir"] = str(timestamped_output_dir(config["output_dir"]))
        logging.info("本次训练输出目录：%s", config["output_dir"])
    for stage in stages:
        if stage == "discover":
            from .collect import discover as run
        elif stage == "classify":
            from .classify import classify as run
        elif stage == "download":
            from .collect import download as run
        elif stage == "prepare":
            from .factors import prepare as run
        else:
            from .train import train as run
        logging.info("开始 %s", stage)
        result = run(config, refresh=args.refresh)
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
