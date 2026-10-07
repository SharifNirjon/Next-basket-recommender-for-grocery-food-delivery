"""Raw CSV -> typed parquet tables + leakage-safe user split.

Outputs (in ``processed_dir``):
  products.parquet       product_id, product_name, aisle_id, department_id, aisle, department
  orders.parquet         all orders of the selected users (+ cumulative days per user)
  prior.parquet          prior order lines joined with order metadata (features come ONLY from here)
  target_orders.parquet  the single "train" order per eval user (context only: dow/hour/days gap)
  targets.parquet        products of that order = ground truth (never used for features)
  splits.parquet         user_id -> train/valid/test (70/15/15, seeded, by user)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import Config, cli_config
from recsys.utils import configure_logging, get_logger, timer

log = get_logger(__name__)

ORDER_DTYPES = {
    "order_id": "int32",
    "user_id": "int32",
    "eval_set": "category",
    "order_number": "int16",
    "order_dow": "int8",
    "order_hour_of_day": "int8",
    "days_since_prior_order": "float32",
}
LINE_DTYPES = {
    "order_id": "int32",
    "product_id": "int32",
    "add_to_cart_order": "int16",
    "reordered": "int8",
}


def load_products(raw: Path) -> pd.DataFrame:
    products = pd.read_csv(
        raw / "products.csv",
        dtype={"product_id": "int32", "aisle_id": "int16", "department_id": "int8"},
    )
    aisles = pd.read_csv(raw / "aisles.csv", dtype={"aisle_id": "int16"})
    depts = pd.read_csv(raw / "departments.csv", dtype={"department_id": "int8"})
    return products.merge(aisles, on="aisle_id", how="left").merge(
        depts, on="department_id", how="left"
    )


def sample_users(user_ids: np.ndarray, frac: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = max(1, int(round(len(user_ids) * frac)))
    return np.sort(rng.choice(np.sort(user_ids), size=n, replace=False))


def split_users(user_ids: np.ndarray, fractions: dict[str, float], seed: int) -> pd.DataFrame:
    """Random user-level split; every order of a user lands in exactly one split."""
    rng = np.random.default_rng(seed)
    users = rng.permutation(np.sort(user_ids))
    n = len(users)
    n_train = int(n * fractions["train"])
    n_valid = int(n * fractions["valid"])
    split = np.array(["test"] * n, dtype=object)
    split[:n_train] = "train"
    split[n_train : n_train + n_valid] = "valid"
    return (
        pd.DataFrame({"user_id": users.astype("int32"), "split": pd.Categorical(split)})
        .sort_values("user_id")
        .reset_index(drop=True)
    )


def prepare(cfg: Config) -> None:
    raw, out = cfg.paths.raw_dir, cfg.paths.processed_dir
    with timer(log, "read orders"):
        orders = pd.read_csv(raw / "orders.csv", dtype=ORDER_DTYPES)

    if cfg.sample:
        keep = sample_users(orders["user_id"].unique(), cfg["data"]["sample_frac"], cfg.seed)
        orders = orders[orders["user_id"].isin(keep)]
        log.info("sampled users", n_users=len(keep))

    orders = orders.sort_values(["user_id", "order_number"]).reset_index(drop=True)
    # Cumulative days since the user's first order -> lets us compute "days since X".
    orders["cumdays"] = (
        orders["days_since_prior_order"].fillna(0).groupby(orders["user_id"]).cumsum()
    ).astype("float32")
    orders.to_parquet(out / "orders.parquet", index=False)

    with timer(log, "read prior lines"):
        prior_lines = pd.read_csv(raw / "order_products__prior.csv", dtype=LINE_DTYPES)
    prior_orders = orders.loc[
        orders["eval_set"] == "prior", ["order_id", "user_id", "order_number"]
    ]
    prior = prior_lines.merge(prior_orders, on="order_id", how="inner")
    del prior_lines
    prior = prior.sort_values(["user_id", "order_number", "add_to_cart_order"]).reset_index(
        drop=True
    )
    prior.to_parquet(out / "prior.parquet", index=False)

    train_lines = pd.read_csv(raw / "order_products__train.csv", dtype=LINE_DTYPES)
    target_orders = orders.loc[orders["eval_set"] == "train"].drop(columns=["eval_set"])
    targets = train_lines.merge(target_orders[["order_id", "user_id"]], on="order_id")
    target_orders.reset_index(drop=True).to_parquet(out / "target_orders.parquet", index=False)
    targets[["user_id", "order_id", "product_id"]].to_parquet(out / "targets.parquet", index=False)

    splits = split_users(target_orders["user_id"].unique(), cfg["data"]["split"], cfg.seed)
    splits.to_parquet(out / "splits.parquet", index=False)

    load_products(raw).to_parquet(out / "products.parquet", index=False)

    log.info(
        "prepared",
        users=int(orders["user_id"].nunique()),
        orders=len(orders),
        prior_lines=len(prior),
        eval_users=len(target_orders),
        target_lines=len(targets),
        split_sizes=splits["split"].value_counts().to_dict(),
    )


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Prepare processed parquet tables")
    prepare(cfg)


if __name__ == "__main__":
    main()
