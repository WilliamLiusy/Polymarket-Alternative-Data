"""Command-line entrypoint; run from the repository root."""

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml


def validate_training_config(config):
    """Require an explicit fixed-epoch protocol without legacy holdout options."""
    if "validation_months" in config:
        raise ValueError("已取消验证集；请删除 validation_months，直接设置 model.epochs")
    model = config.setdefault("model", {})
    allowed = {"epochs", "batch_size", "hidden_size", "learning_rate", "weight_decay",
               "seed", "num_threads"}
    unknown = set(model) - allowed
    if unknown:
        raise ValueError(f"MLP 不支持这些模型参数：{', '.join(sorted(unknown))}；"
                         "请删除旧参数，直接设置 model.epochs")
    epochs = model.setdefault("epochs", 10)
    if type(epochs) is not int or epochs <= 0:
        raise ValueError("model.epochs 必须是正整数")


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
    validate_training_config(config)
    return config


def timestamped_output_dir(output_dir, now=None):
    """Return a per-run output directory using Beijing local time."""
    now = now or datetime.now(ZoneInfo("Asia/Shanghai"))
    run_id = now.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S%f")
    return Path(output_dir) / run_id


def main():
    parser = argparse.ArgumentParser(description="历史沪深300 Qlib MLP 月度扩展窗口 baseline")
    parser.add_argument("command", choices=("download", "prepare", "train", "all"))
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--refresh", action="store_true", help="重新生成指定阶段的缓存和产物")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(args.config)
    commands = ("download", "prepare", "train") if args.command == "all" else (args.command,)
    if "train" in commands:
        output_dir = timestamped_output_dir(config["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=False)
        config["output_dir"] = str(output_dir)
        (output_dir / "training_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )
        (output_dir / "training_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
        logging.info("本次训练输出目录：%s", output_dir)
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
