"""Vectorized aggregates over PRIOR orders only.

These are shared by candidate generation (stage 1) and feature building (stage 2),
so retrieval scores and ranking features are computed by exactly the same code.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def user_order_stats(orders: pd.DataFrame) -> pd.DataFrame:
    """Per user, from prior orders: last order number and cumulative days at that order."""
    prior_orders = orders[orders["eval_set"] == "prior"]
    return prior_orders.groupby("user_id").agg(
        max_on=("order_number", "max"), max_cum=("cumdays", "max")
    )


def user_item_history(prior: pd.DataFrame, orders: pd.DataFrame, decay: float) -> pd.DataFrame:
    """One row per (user, product) ever bought in prior orders.

    hist_score = sum over purchases of decay ** orders_back (orders_back=0 is the latest
    prior order): a recency-weighted frequency used as the history retrieval score.
    """
    uo = user_order_stats(orders)
    p = prior[["user_id", "product_id", "order_number", "add_to_cart_order", "cumdays"]]
    max_on = p["user_id"].map(uo["max_on"]).to_numpy()
    orders_back = (max_on - p["order_number"].to_numpy()).astype("int32")
    p = p.assign(orders_back=orders_back, w=np.power(decay, orders_back).astype("float32"))

    g = p.groupby(["user_id", "product_id"], sort=True)
    h = g.agg(
        times_bought=("order_number", "size"),
        first_on=("order_number", "min"),
        last_on=("order_number", "max"),
        last_cum=("cumdays", "max"),
        avg_cart_pos=("add_to_cart_order", "mean"),
        hist_score=("w", "sum"),
    ).reset_index()

    # Streak: number of consecutive most-recent prior orders containing the item.
    # Sorted by orders_back ascending, a purchase is part of the streak iff its
    # orders_back equals its position in the sequence.
    s = p[["user_id", "product_id", "orders_back"]].sort_values(
        ["user_id", "product_id", "orders_back"]
    )
    pos = s.groupby(["user_id", "product_id"], sort=False).cumcount().to_numpy()
    s = s.assign(in_streak=(s["orders_back"].to_numpy() == pos).astype("int16"))
    streak = s.groupby(["user_id", "product_id"], sort=True)["in_streak"].sum()
    h["streak"] = streak.to_numpy()

    h = h.merge(uo, left_on="user_id", right_index=True, how="left")
    h["orders_since_last"] = (h["max_on"] - h["last_on"]).astype("int16")
    h["orders_since_first"] = (h["max_on"] - h["first_on"]).astype("int16")
    h["days_since_last"] = (h["max_cum"] - h["last_cum"]).astype("float32")
    h["share_since_first"] = (h["times_bought"] / (h["orders_since_first"] + 1)).astype("float32")
    h["order_share"] = (h["times_bought"] / h["max_on"]).astype("float32")
    return h.drop(columns=["first_on", "last_on", "last_cum", "max_cum", "max_on"])


def item_stats(prior: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Global item features from prior orders, one row per product in the catalogue."""
    g = prior.groupby("product_id")
    per_user = prior.groupby(["product_id", "user_id"]).size()
    repeat = (per_user >= 2).groupby(level=0).mean()
    st = pd.DataFrame(
        {
            "i_n_purchases": g.size(),
            "i_n_users": per_user.groupby(level=0).size(),
            "i_reorder_rate": g["reordered"].mean(),
            "i_repeat_user_ratio": repeat,
            "i_avg_cart_pos": g["add_to_cart_order"].mean(),
        }
    )
    out = products[["product_id", "aisle_id", "department_id"]].merge(
        st, left_on="product_id", right_index=True, how="left"
    )
    out[["i_n_purchases", "i_n_users"]] = out[["i_n_purchases", "i_n_users"]].fillna(0)
    out = out.rename(columns={"aisle_id": "i_aisle_id", "department_id": "i_department_id"})
    return out.astype({c: "float32" for c in out.columns if c != "product_id"})


def user_stats(prior: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Per-user behavioural features from prior orders."""
    po = orders[orders["eval_set"] == "prior"]
    og = po.groupby("user_id")
    lines = prior.groupby("user_id")
    reorder_ratio = prior[prior["order_number"] > 1].groupby("user_id")["reordered"].mean()

    def mode(col: str) -> pd.Series:
        c = po.groupby(["user_id", col]).size().reset_index(name="n")
        c = c.sort_values(["user_id", "n", col], ascending=[True, False, True])
        return c.drop_duplicates("user_id").set_index("user_id")[col]

    out = pd.DataFrame(
        {
            "u_n_orders": og.size(),
            "u_avg_days_between": og["days_since_prior_order"].mean(),
            "u_days_span": og["cumdays"].max(),
            "u_total_items": lines.size(),
            "u_n_distinct_items": lines["product_id"].nunique(),
            "u_reorder_ratio": reorder_ratio,
            "u_pref_hour": mode("order_hour_of_day"),
            "u_pref_dow": mode("order_dow"),
        }
    )
    out["u_avg_basket"] = out["u_total_items"] / out["u_n_orders"]
    return out.astype("float32")


def user_aisle_share(hist: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Share of each user's purchases falling into each aisle: user_id, aisle_id, ua_share."""
    h = hist[["user_id", "product_id", "times_bought"]].merge(
        products[["product_id", "aisle_id"]], on="product_id"
    )
    ua = h.groupby(["user_id", "aisle_id"])["times_bought"].sum().reset_index()
    ua["ua_share"] = (
        ua["times_bought"] / ua.groupby("user_id")["times_bought"].transform("sum")
    ).astype("float32")
    return ua.drop(columns="times_bought")
