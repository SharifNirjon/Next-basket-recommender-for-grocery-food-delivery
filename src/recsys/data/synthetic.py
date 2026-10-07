"""Tiny synthetic dataset in the exact Instacart CSV format (for tests and CI).

Users have latent aisle preferences and a personal "staples" list, so reorder
signal exists and every pipeline stage has something to learn. No Kaggle data.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_synthetic(out_dir: Path, n_users: int = 300, n_products: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_aisles, n_depts = 12, 4
    pd.DataFrame(
        {"aisle_id": np.arange(1, n_aisles + 1), "aisle": [f"aisle {i}" for i in range(n_aisles)]}
    ).to_csv(out_dir / "aisles.csv", index=False)
    pd.DataFrame(
        {
            "department_id": np.arange(1, n_depts + 1),
            "department": [f"dept {i}" for i in range(n_depts)],
        }
    ).to_csv(out_dir / "departments.csv", index=False)
    aisle_of = rng.integers(1, n_aisles + 1, n_products)
    pd.DataFrame(
        {
            "product_id": np.arange(1, n_products + 1),
            "product_name": [f"Product {i}" for i in range(1, n_products + 1)],
            "aisle_id": aisle_of,
            "department_id": (aisle_of - 1) % n_depts + 1,
        }
    ).to_csv(out_dir / "products.csv", index=False)

    item_pop = rng.dirichlet(np.full(n_products, 0.3))
    orders, prior_lines, train_lines = [], [], []
    order_id = 1
    for user in range(1, n_users + 1):
        staples = rng.choice(n_products, size=rng.integers(3, 12), replace=False, p=item_pop) + 1
        n_orders = int(rng.integers(4, 15))
        last_set = "train" if rng.random() < 0.8 else "test"
        for num in range(1, n_orders + 1):
            eval_set = last_set if num == n_orders else "prior"
            orders.append(
                (
                    order_id,
                    user,
                    eval_set,
                    num,
                    int(rng.integers(0, 7)),
                    int(rng.integers(7, 22)),
                    np.nan if num == 1 else float(rng.integers(1, 31)),
                )
            )
            if eval_set != "test":
                keep = staples[rng.random(len(staples)) < 0.6]
                extra = rng.choice(n_products, size=rng.integers(1, 5), p=item_pop) + 1
                basket = list(dict.fromkeys([*keep.tolist(), *extra.tolist()]))
                rows = [(order_id, p, pos + 1, 0) for pos, p in enumerate(basket)]
                (prior_lines if eval_set == "prior" else train_lines).extend(rows)
            order_id += 1

    orders_df = pd.DataFrame(
        orders,
        columns=[
            "order_id",
            "user_id",
            "eval_set",
            "order_number",
            "order_dow",
            "order_hour_of_day",
            "days_since_prior_order",
        ],
    )
    orders_df.to_csv(out_dir / "orders.csv", index=False)

    cols = ["order_id", "product_id", "add_to_cart_order"]
    lines = pd.concat(
        [
            pd.DataFrame(prior_lines, columns=[*cols, "_r"]).assign(_file="prior"),
            pd.DataFrame(train_lines, columns=[*cols, "_r"]).assign(_file="train"),
        ]
    ).merge(orders_df[["order_id", "user_id", "order_number"]], on="order_id")
    lines = lines.sort_values(["user_id", "order_number", "add_to_cart_order"])
    # reordered = the user bought this product in an earlier order
    lines["reordered"] = (lines.groupby(["user_id", "product_id"]).cumcount() > 0).astype(int)
    for name in ("prior", "train"):
        part = lines.loc[lines["_file"] == name, [*cols, "reordered"]]
        part.to_csv(out_dir / f"order_products__{name}.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Write a tiny synthetic Instacart-format dataset")
    ap.add_argument("--out", default="data/synthetic/raw")
    ap.add_argument("--users", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    make_synthetic(Path(args.out), n_users=args.users, seed=args.seed)


if __name__ == "__main__":
    main()
