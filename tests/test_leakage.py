"""Leakage tests: nothing from the target ("train") order may reach the features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.features.schema import STATIC_FEATURES
from recsys.ranker.dataset import load_split


def _features(cfg: Config) -> pd.DataFrame:
    df = pd.read_parquet(cfg.paths.features_dir)
    return df.sort_values(["user_id", "product_id"]).reset_index(drop=True)


def test_prior_table_excludes_target_orders(pipeline_cfg: Config) -> None:
    d = pipeline_cfg.paths.processed_dir
    prior = pd.read_parquet(d / "prior.parquet")
    targets = pd.read_parquet(d / "target_orders.parquet")
    assert not prior["order_id"].isin(targets["order_id"]).any()
    # every history order precedes the user's target order
    last_prior = prior.groupby("user_id")["order_number"].max()
    tgt = targets.set_index("user_id")["order_number"]
    common = last_prior.index.intersection(tgt.index)
    assert (last_prior.loc[common] < tgt.loc[common]).all()


def test_features_invariant_to_target_order_contents(
    pipeline_cfg: Config, mutated_cfg: Config
) -> None:
    """Replacing every target-order product must not change a single feature value."""
    a, b = _features(pipeline_cfg), _features(mutated_cfg)
    pd.testing.assert_frame_equal(a[["user_id", "product_id"]], b[["user_id", "product_id"]])
    # Bit-for-bit equal, including Item2Vec similarities (single-threaded, seeded in tests).
    np.testing.assert_array_equal(a[STATIC_FEATURES].to_numpy(), b[STATIC_FEATURES].to_numpy())

    xa, xb = load_split(pipeline_cfg, "train"), load_split(mutated_cfg, "train")
    ctx = [c for c in xa.X.columns if c.startswith("ctx_")]
    np.testing.assert_array_equal(xa.X[ctx].to_numpy(), xb.X[ctx].to_numpy())
    assert not np.array_equal(xa.y, xb.y), "labels should change when targets change"


def test_no_target_or_label_columns_in_features(pipeline_cfg: Config) -> None:
    data = load_split(pipeline_cfg, "valid")
    forbidden = {"label", "reordered", "eval_set", "order_id", "add_to_cart_order"}
    assert not forbidden & set(data.X.columns)
