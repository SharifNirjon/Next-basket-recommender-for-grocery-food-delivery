"""Configuration loading and path resolution."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw_dir: Path
    processed_dir: Path
    artifacts_dir: Path
    reports_dir: Path
    mlruns_dir: Path

    @property
    def features_dir(self) -> Path:
        return self.processed_dir / "features"

    @property
    def models_dir(self) -> Path:
        return self.processed_dir / "models"


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    paths: Paths
    sample: bool

    @property
    def seed(self) -> int:
        return int(self.raw["seed"])

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]


def _resolve(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else base / path


def load_config(path: str | Path | None = None, sample: bool = False) -> Config:
    """Load YAML config. With ``sample=True`` outputs go to a ``sample/`` subfolder."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path) as f:
        raw = yaml.safe_load(f)
    base = REPO_ROOT if cfg_path.resolve().is_relative_to(REPO_ROOT) else cfg_path.parent
    p = {k: _resolve(base, v) for k, v in raw["paths"].items()}
    if sample:
        for k in ("processed_dir", "artifacts_dir", "reports_dir"):
            p[k] = p[k] / "sample"
    paths = Paths(**p)
    for d in (paths.processed_dir, paths.artifacts_dir, paths.reports_dir):
        d.mkdir(parents=True, exist_ok=True)
    return Config(raw=raw, paths=paths, sample=sample)


def cli_config(description: str) -> tuple[Config, argparse.Namespace]:
    """Shared CLI: ``--config`` and ``--sample`` for every pipeline step."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--sample", action="store_true", help="run on ~10%% of users")
    args = parser.parse_args()
    return load_config(args.config, sample=args.sample), args
