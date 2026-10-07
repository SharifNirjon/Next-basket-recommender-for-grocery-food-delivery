"""Assemble ranking datasets: static features + target-order context + labels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.features.schema import FEATURES, STATIC_FEATURES, add_context


@dataclass
class RankingData:
    keys: pd.DataFrame  # user_id, product_id (sorted by user)
    X: pd.DataFrame  # FEATURES
    y: np.ndarray  # 1 if product is in the user's target order
    group: np.ndarray  # candidates per user, in key order


def load_split(cfg: Config, split: str) -> RankingData:
    d = cfg.paths.processed_dir
    splits = pd.read_parquet(d / "splits.parquet")
    users = splits.loc[splits["split"] == split, "user_id"].to_numpy()
    feats = pd.read_parquet(cfg.paths.features_dir, filters=[("user_id", "in", users.tolist())])
    feats = feats.sort_values(["user_id", "product_id"], kind="stable").reset_index(drop=True)

    # Context comes from the target order's metadata only (when/how long since last order),
    # never from its contents.
    ctx = pd.read_parquet(
        d / "target_orders.parquet",
        columns=["user_id", "order_hour_of_day", "order_dow", "days_since_prior_order"],
    ).set_index("user_id")
    c = ctx.loc[feats["user_id"]]
    X = add_context(
        feats[STATIC_FEATURES].to_numpy(dtype="float32"),
        c["order_hour_of_day"].to_numpy(),
        c["order_dow"].to_numpy(),
        c["days_since_prior_order"].to_numpy(),
    )

    targets = pd.read_parquet(d / "targets.parquet", columns=["user_id", "product_id"])
    labeled = feats[["user_id", "product_id"]].merge(
        targets.assign(label=1), on=["user_id", "product_id"], how="left"
    )
    y = labeled["label"].fillna(0).to_numpy(dtype="int8")
    group = feats.groupby("user_id", sort=True).size().to_numpy()
    return RankingData(
        keys=feats[["user_id", "product_id"]],
        X=pd.DataFrame(X, columns=FEATURES),
        y=y,
        group=group,
    )
