"""Session fixtures: synthetic raw data -> full offline pipeline -> tiny artifacts."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from recsys.config import DEFAULT_CONFIG, Config, load_config
from recsys.data.synthetic import make_synthetic
from recsys.pipeline import run_pipeline
from recsys.utils import configure_logging


def write_test_config(root: Path, raw_dir: Path) -> Path:
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["paths"] = {
        "raw_dir": str(raw_dir),
        "processed_dir": str(root / "processed"),
        "artifacts_dir": str(root / "artifacts"),
        "reports_dir": str(root / "reports"),
        "mlruns_dir": str(root / "mlruns"),
    }
    raw["candidates"].update(n_item2vec=10, n_top_aisles=2, n_per_aisle=5, n_global=10)
    raw["candidates"]["max_candidates"] = 40
    raw["candidates"]["item2vec"].update(vector_size=8, min_count=1, epochs=3, window=5, workers=1)
    raw["features"]["n_user_chunks"] = 2
    raw["ranker"]["params"].update(min_child_samples=5, num_leaves=7, n_jobs=1)
    raw["ranker"].update(n_estimators=40, early_stopping_rounds=10)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def build(root: Path, seed: int = 0, mutate_targets: bool = False) -> Config:
    raw_dir = root / "raw"
    make_synthetic(raw_dir, n_users=200, seed=seed)
    if mutate_targets:
        import pandas as pd

        t = pd.read_csv(raw_dir / "order_products__train.csv")
        t["product_id"] = (t["product_id"] * 7) % 200 + 1  # completely different targets
        t = t.drop_duplicates(["order_id", "product_id"])
        t.to_csv(raw_dir / "order_products__train.csv", index=False)
    cfg = load_config(write_test_config(root, raw_dir))
    run_pipeline(cfg)
    return cfg


@pytest.fixture(scope="session")
def pipeline_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    configure_logging()
    return build(tmp_path_factory.mktemp("pipeline"))


@pytest.fixture(scope="session")
def mutated_cfg(tmp_path_factory: pytest.TempPathFactory) -> Config:
    return build(tmp_path_factory.mktemp("mutated"), mutate_targets=True)
