"""Verify that prepare/train reuse the completed extension without rewriting models."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/polymarket"
MONTHS = [f"2026-{month:02d}" for month in range(4, 10)]


def snapshot():
    paths = list((ROOT / "data/polymarket_factors").glob("*"))
    paths += [path for month in MONTHS for path in (OUTPUT / month).iterdir()]
    result = {}
    for path in paths:
        if path.is_file():
            stat = path.stat()
            result[str(path.relative_to(ROOT))] = {
                "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    return result


def main():
    for month in MONTHS:
        assert (OUTPUT / month / "completed.json").is_file(), month
    assert (OUTPUT / "summary.json").is_file()
    before = snapshot()
    codes = {}
    log_path = OUTPUT / "cache_rerun.log"
    with log_path.open("w") as log:
        for stage in ("prepare", "train"):
            result = subprocess.run(
                [sys.executable, "-m", "polymarket_ext", stage,
                 "--config", "configs/polymarket.yaml"],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False,
            )
            codes[stage] = result.returncode
            if result.returncode:
                break
    log_text = log_path.read_text()
    after = snapshot()
    reused = [month for month in MONTHS
              if f"Reusing verified Polymarket results for {month}" in log_text]
    changed = [path for path in before if before[path] != after.get(path)]
    checks = {
        "commands_succeeded": codes == {"prepare": 0, "train": 0},
        "factor_panel_reused": "Reuse Polymarket factor panel" in log_text,
        "all_six_months_reused": reused == MONTHS,
        "no_training_epochs_executed": "PolymarketModel epoch" not in log_text,
        "panel_and_monthly_artifacts_unchanged": before == after,
    }
    report = {
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks, "return_codes": codes,
        "reused_months": reused, "changed_files": changed,
        "snapshot": before, "log": str(log_path.relative_to(ROOT)),
    }
    (OUTPUT / "cache_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "checks", "changed_files")}, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
