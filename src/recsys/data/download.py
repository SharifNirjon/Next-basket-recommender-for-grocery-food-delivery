"""Download the Instacart dataset with the Kaggle API.

Tries the competition first, then falls back to the public dataset mirror.
Credentials are read from ``~/.kaggle/kaggle.json`` (or KAGGLE_USERNAME/KAGGLE_KEY).
"""

from __future__ import annotations

import os
import subprocess
import sys
import zipfile
from pathlib import Path

from recsys.config import cli_config
from recsys.utils import configure_logging, get_logger

REQUIRED = [
    "orders.csv",
    "order_products__prior.csv",
    "order_products__train.csv",
    "products.csv",
    "aisles.csv",
    "departments.csv",
]
log = get_logger(__name__)


def have_credentials() -> bool:
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


def _extract_all(raw_dir: Path) -> None:
    # Archives may be nested (competition zips contain per-file zips).
    for _ in range(2):
        for zpath in raw_dir.glob("*.zip"):
            with zipfile.ZipFile(zpath) as zf:
                zf.extractall(raw_dir)
            zpath.unlink()


def download(raw_dir: Path, competition: str, dataset: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if all((raw_dir / f).exists() for f in REQUIRED):
        log.info("raw data already present", raw_dir=str(raw_dir))
        return
    if not have_credentials():
        sys.exit(
            "Kaggle credentials missing: put kaggle.json in ~/.kaggle/ (chmod 600) "
            "or set KAGGLE_USERNAME/KAGGLE_KEY."
        )
    cmds = [
        ["kaggle", "competitions", "download", "-c", competition, "-p", str(raw_dir)],
        ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir)],
    ]
    for cmd in cmds:
        log.info("downloading", cmd=" ".join(cmd))
        if subprocess.run(cmd, check=False).returncode == 0:
            break
        log.warning("download failed, trying fallback", cmd=" ".join(cmd))
    else:
        sys.exit("Kaggle download failed for both competition and dataset mirror.")
    _extract_all(raw_dir)
    missing = [f for f in REQUIRED if not (raw_dir / f).exists()]
    if missing:
        sys.exit(f"Download incomplete, missing: {missing}")
    log.info("download complete", files=REQUIRED)


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Download Instacart data from Kaggle")
    download(cfg.paths.raw_dir, cfg["data"]["kaggle_competition"], cfg["data"]["kaggle_dataset"])


if __name__ == "__main__":
    main()
