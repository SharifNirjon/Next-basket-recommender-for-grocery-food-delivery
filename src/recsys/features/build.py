"""Stage 2 feature building: static features for every (user, candidate) pair.

Reads candidate parts written by stage 1 and joins user, item and user x item
features computed from PRIOR orders only. Output: data/processed/features/part-*.parquet
with columns user_id, product_id + STATIC_FEATURES (float32).
"""

from __future__ import annotations

import pandas as pd

from recsys.config import Config, cli_config
from recsys.features.aggregates import (
    item_stats,
    user_aisle_share,
    user_item_history,
    user_stats,
)
from recsys.features.schema import STATIC_FEATURES
from recsys.utils import configure_logging, get_logger, timer

log = get_logger(__name__)

UI_RENAME = {
    "times_bought": "ui_times_bought",
    "order_share": "ui_order_share",
    "orders_since_last": "ui_orders_since_last",
    "orders_since_first": "ui_orders_since_first",
    "share_since_first": "ui_share_since_first",
    "days_since_last": "ui_days_since_last",
    "avg_cart_pos": "ui_avg_cart_pos",
    "streak": "ui_streak",
}


def build_static_features(
    cands: pd.DataFrame,
    prior: pd.DataFrame,
    orders: pd.DataFrame,
    items: pd.DataFrame,
    products: pd.DataFrame,
    decay: float,
) -> pd.DataFrame:
    """Join all static features onto candidate rows. ``prior``/``orders`` cover the same users."""
    hist = user_item_history(prior, orders, decay)
    ui = hist.rename(columns=UI_RENAME)[["user_id", "product_id", *UI_RENAME.values()]]
    users = user_stats(prior, orders)
    ua = user_aisle_share(hist, products).rename(columns={"aisle_id": "i_aisle_id"})
    ua["i_aisle_id"] = ua["i_aisle_id"].astype("float32")

    df = cands.drop(columns=[c for c in cands.columns if c.startswith("ui_")], errors="ignore")
    df = df.merge(ui, on=["user_id", "product_id"], how="left")
    df = df.merge(items, on="product_id", how="left")
    df = df.merge(users, left_on="user_id", right_index=True, how="left")
    df = df.merge(ua, on=["user_id", "i_aisle_id"], how="left")
    # Never-bought items: counts are truly 0; recency features stay NaN (undefined).
    zero_if_unseen = ["ui_times_bought", "ui_order_share", "ui_share_since_first", "ui_streak"]
    df[zero_if_unseen + ["ua_share"]] = df[zero_if_unseen + ["ua_share"]].fillna(0)
    out = df[["user_id", "product_id"]].copy()
    out[STATIC_FEATURES] = df[STATIC_FEATURES].astype("float32")
    return out


def build_all(cfg: Config) -> None:
    d = cfg.paths.processed_dir
    prior = pd.read_parquet(d / "prior.parquet")
    orders = pd.read_parquet(d / "orders.parquet")
    products = pd.read_parquet(d / "products.parquet")
    items = item_stats(prior, products)
    out_dir = cfg.paths.features_dir
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    n_rows = 0
    for part in sorted((d / "candidates").glob("part-*.parquet")):
        with timer(log, f"features {part.name}"):
            cands = pd.read_parquet(part)
            users = cands["user_id"].unique()
            p = prior[prior["user_id"].isin(users)]
            o = orders[orders["user_id"].isin(users)]
            feats = build_static_features(
                cands, p, o, items, products, cfg["candidates"]["history_decay"]
            )
            feats.to_parquet(out_dir / part.name, index=False)
            n_rows += len(feats)
    log.info("features built", rows=n_rows, n_features=len(STATIC_FEATURES))


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Build static ranking features")
    build_all(cfg)


if __name__ == "__main__":
    main()
