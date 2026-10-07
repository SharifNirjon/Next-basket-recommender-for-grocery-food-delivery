"""Non-learned baselines. All are computed from prior orders only."""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_popularity(prior: pd.DataFrame, n: int) -> pd.DataFrame:
    """Top-n products by number of prior order lines: product_id, pop_count, pop_rank."""
    counts = prior["product_id"].value_counts()
    counts = counts.sort_index().sort_values(ascending=False, kind="stable").head(n)
    return pd.DataFrame(
        {
            "product_id": counts.index.astype("int32"),
            "pop_count": counts.to_numpy(),
            "pop_rank": np.arange(1, len(counts) + 1),
        }
    )


def popularity_recs(prior: pd.DataFrame, users: np.ndarray, k: int) -> pd.DataFrame:
    """(a) Same global top-k list for everyone."""
    pop = global_popularity(prior, k)
    u = pd.DataFrame({"user_id": users})
    return u.merge(
        pop[["product_id", "pop_rank"]].rename(columns={"pop_rank": "rank"}), how="cross"
    )


def user_frequency_recs(prior: pd.DataFrame, users: np.ndarray, k: int) -> pd.DataFrame:
    """(b) User's most frequently bought items (ties -> most recently bought).

    Users with fewer than k distinct items are padded with global popularity,
    which makes the baseline strictly stronger.
    """
    p = prior[prior["user_id"].isin(users)]
    freq = (
        p.groupby(["user_id", "product_id"], observed=True)
        .agg(times=("order_id", "size"), last=("order_number", "max"))
        .reset_index()
        .sort_values(
            ["user_id", "times", "last", "product_id"], ascending=[True, False, False, True]
        )
    )
    freq["rank"] = freq.groupby("user_id").cumcount() + 1
    own = freq.loc[freq["rank"] <= k, ["user_id", "product_id", "rank"]]

    pad = popularity_recs(prior, users, 2 * k)
    pad = pad.merge(own[["user_id", "product_id"]], how="left", indicator=True)
    pad = pad[pad["_merge"] == "left_only"].drop(columns="_merge")
    n_own = own.groupby("user_id").size().reindex(users, fill_value=0)
    pad["rank"] = pad.groupby("user_id").cumcount() + 1 + pad["user_id"].map(n_own).to_numpy()
    out = pd.concat([own, pad[pad["rank"] <= k]], ignore_index=True)
    return out.sort_values(["user_id", "rank"]).reset_index(drop=True)
