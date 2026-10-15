"""Command-line entrypoint; run from the repository root."""

import argparse
import json
import logging
from datetime import date
from pathlib import Path

import yaml


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    for key in ("data_dir", "provider_dir", "cache_dir", "output_dir"):
        config[key] = str(Path(config[key]).expanduser().resolve())
    date_keys = (
        "research_start", "research_end", "train_start",
        "first_test_month", "download_start",
    )
    for key in date_keys:
        config[key] = date.fromisoformat(str(config[key])).isoformat()
    config["download_end"] = date.fromisoformat(
        str(config.get("download_end") or date.today().isoformat())
    ).isoformat()
    if not (
        config["download_start"] <= config["research_start"]
        <= config["train_start"] < config["first_test_month"] <= config["research_end"]
        <= config["download_end"]
    ):
        raise ValueError("日期应满足 download_start ≤ research_start ≤ train_start < "
                         "first_test_month ≤ research_end ≤ download_end")
    if date.fromisoformat(config["first_test_month"]).day != 1:
        raise ValueError("first_test_month 必须是测试月第一天")
    if config["model"].get("early_stopping_rounds") != 0:
        raise ValueError("本实验没有验证集，early_stopping_rounds 必须为 0")
    return config


def main():
    parser = argparse.ArgumentParser(description="历史沪深300 Qlib 月度扩展窗口 baseline")
    parser.add_argument("command", choices=("download", "prepare", "train", "all"))
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--refresh", action="store_true", help="重新生成指定阶段的缓存和产物")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    commands = ("download", "prepare", "train") if args.command == "all" else (args.command,)
    for command in commands:
        if command == "download":
            from .download import download as run
        elif command == "prepare":
            from .prepare import prepare as run
        else:
            from .train import train as run
        logging.info("开始 %s", command)
        result = run(config, refresh=args.refresh)
        if result is not None:
            print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
