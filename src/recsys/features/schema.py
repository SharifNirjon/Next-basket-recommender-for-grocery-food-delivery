"""Feature schema shared by offline training and online serving.

STATIC features depend only on prior orders and are precomputed per (user, candidate).
CONTEXT features depend on the order being predicted (hour, day of week, days since
the previous order) and are computed by ``add_context`` at train *and* serve time,
so both paths run identical code (no train/serve skew).
"""

from __future__ import annotations

import numpy as np

USER_FEATURES = [
    "u_n_orders",
    "u_avg_basket",
    "u_avg_days_between",
    "u_reorder_ratio",
    "u_pref_hour",
    "u_pref_dow",
    "u_n_distinct_items",
    "u_total_items",
    "u_days_span",
]
ITEM_FEATURES = [
    "i_n_purchases",
    "i_n_users",
    "i_reorder_rate",
    "i_repeat_user_ratio",
    "i_avg_cart_pos",
    "i_aisle_id",
    "i_department_id",
]
USER_ITEM_FEATURES = [
    "ui_times_bought",
    "ui_order_share",
    "ui_orders_since_last",
    "ui_orders_since_first",
    "ui_share_since_first",
    "ui_days_since_last",
    "ui_avg_cart_pos",
    "ui_streak",
    "ua_share",
]
RETRIEVAL_FEATURES = [
    "hist_score",
    "i2v_sim",
    "retrieval_score",
    "src_history",
    "src_i2v",
    "src_aisle",
    "src_global",
]
STATIC_FEATURES = USER_FEATURES + ITEM_FEATURES + USER_ITEM_FEATURES + RETRIEVAL_FEATURES
CONTEXT_FEATURES = [
    "ctx_hour",
    "ctx_dow",
    "ctx_days_since_prior",
    "ctx_days_since_last_at_target",
    "ctx_hour_diff",
    "ctx_dow_match",
    "ctx_gap_ratio",
]
FEATURES = STATIC_FEATURES + CONTEXT_FEATURES
CATEGORICAL_FEATURES = ["i_aisle_id", "i_department_id"]

_S = {name: i for i, name in enumerate(STATIC_FEATURES)}


def add_context(
    static: np.ndarray,
    hour: np.ndarray | float,
    dow: np.ndarray | float,
    days_since_prior: np.ndarray | float,
) -> np.ndarray:
    """Append CONTEXT_FEATURES to a (n, len(STATIC_FEATURES)) float32 matrix.

    ``hour``/``dow``/``days_since_prior`` are scalars (serving) or per-row arrays (training).
    """
    n = static.shape[0]
    hour = np.broadcast_to(np.asarray(hour, dtype="float32"), (n,))
    dow = np.broadcast_to(np.asarray(dow, dtype="float32"), (n,))
    dsp = np.broadcast_to(np.asarray(days_since_prior, dtype="float32"), (n,))
    pref_hour = static[:, _S["u_pref_hour"]]
    diff = np.abs(hour - pref_hour)
    ctx = np.column_stack(
        [
            hour,
            dow,
            dsp,
            static[:, _S["ui_days_since_last"]] + dsp,  # NaN for never-bought items
            np.minimum(diff, 24 - diff),
            (dow == static[:, _S["u_pref_dow"]]).astype("float32"),
            dsp / np.maximum(static[:, _S["u_avg_days_between"]], 1.0),
        ]
    ).astype("float32")
    return np.hstack([static.astype("float32", copy=False), ctx])
