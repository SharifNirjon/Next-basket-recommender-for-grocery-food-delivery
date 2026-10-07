"""Offline batch step: package everything the API needs into ``artifacts_dir``.

  static_features.npy   float32 (n_rows, n_static) - memory-mapped by the API
  candidate_ids.npy     int32  (n_rows,)           - product id of each row
  user_index.npz        user_ids (sorted), start, end -> row slice per user
  ranker.txt            LightGBM model (best iteration)
  item2vec.faiss        FAISS IndexFlatIP over normalized Item2Vec vectors
  item2vec_ids.npy      product id of each FAISS row
  products.json         product_id -> name / aisle / department
  popular.json          global popularity fallback list
  metadata.json         model version, trained-at, feature lists, offline metrics
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from recsys.config import Config, cli_config
from recsys.evaluate.baselines import global_popularity
from recsys.features.schema import CONTEXT_FEATURES, STATIC_FEATURES
from recsys.utils import configure_logging, get_logger, timer

log = get_logger(__name__)


def export(cfg: Config) -> None:
    d, out = cfg.paths.processed_dir, cfg.paths.artifacts_dir
    out.mkdir(parents=True, exist_ok=True)
    parts = sorted(cfg.paths.features_dir.glob("part-*.parquet"))
    n_rows = sum(pd.read_parquet(p, columns=["user_id"]).shape[0] for p in parts)

    feats_mm = np.lib.format.open_memmap(
        out / "static_features.npy",
        mode="w+",
        dtype="float32",
        shape=(n_rows, len(STATIC_FEATURES)),
    )
    cand_ids = np.empty(n_rows, dtype="int32")
    users, starts, ends = [], [], []
    pos = 0
    with timer(log, "write feature matrix"):
        for p in parts:
            df = pd.read_parquet(p).sort_values(["user_id", "product_id"], kind="stable")
            n = len(df)
            feats_mm[pos : pos + n] = df[STATIC_FEATURES].to_numpy(dtype="float32")
            cand_ids[pos : pos + n] = df["product_id"].to_numpy()
            uid = df["user_id"].to_numpy()
            first = np.flatnonzero(np.r_[True, uid[1:] != uid[:-1]])
            users.append(uid[first])
            starts.append(first + pos)
            ends.append(np.r_[first[1:], n] + pos)
            pos += n
    feats_mm.flush()
    del feats_mm
    np.save(out / "candidate_ids.npy", cand_ids)
    u, s, e = np.concatenate(users), np.concatenate(starts), np.concatenate(ends)
    order = np.argsort(u)
    np.savez(out / "user_index.npz", user_ids=u[order], start=s[order], end=e[order])

    products = pd.read_parquet(d / "products.parquet")
    catalogue = {
        int(r.product_id): [r.product_name, r.aisle, r.department]
        for r in products.itertuples(index=False)
    }
    (out / "products.json").write_text(json.dumps(catalogue))

    prior = pd.read_parquet(d / "prior.parquet", columns=["product_id"])
    pop = global_popularity(prior, cfg["serve"]["max_k"])
    popular = {
        "product_ids": pop["product_id"].tolist(),
        "scores": (pop["pop_count"] / pop["pop_count"].max()).round(6).tolist(),
    }
    (out / "popular.json").write_text(json.dumps(popular))

    models = cfg.paths.models_dir
    for name in ("ranker.txt", "item2vec.faiss", "item2vec_ids.npy"):
        shutil.copy(models / name, out / name)

    meta = json.loads((models / "ranker_meta.json").read_text())
    metrics_path = cfg.paths.reports_dir / "metrics.json"
    test_metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else None
    metadata = {
        "model_version": meta["model_version"],
        "trained_at": meta["trained_at"],
        "best_iteration": meta["best_iteration"],
        "sample": cfg.sample,
        "static_features": STATIC_FEATURES,
        "context_features": CONTEXT_FEATURES,
        "n_users": len(u),
        "n_candidate_rows": n_rows,
        "metrics": {
            "valid": meta["valid_metrics"],
            "test": test_metrics["systems"]["two_stage"] if test_metrics else None,
            "candidate_recall_test": (
                test_metrics["retrieval"]["candidate_recall"] if test_metrics else None
            ),
        },
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    log.info("artifacts exported", dir=str(out), users=len(u), rows=n_rows)


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Export serving artifacts")
    export(cfg)


if __name__ == "__main__":
    main()
