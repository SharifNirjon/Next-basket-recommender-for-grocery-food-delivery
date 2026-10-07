"""Stage 1: candidate generation.

Sources (merged, deduplicated, source flags kept as ranking features):
  history  every item the user bought in prior orders, scored by recency-weighted frequency
  i2v      FAISS nearest items to the user's recency-weighted Item2Vec embedding
  aisle    most popular items within the user's top aisles
  global   globally popular items (fallback for short histories / exploration)

The merged pool is capped at ``max_candidates`` per user by ``retrieval_score``:
history items first (by hist_score), then unseen items by Item2Vec similarity.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from recsys.candidates.item2vec import ItemIndex
from recsys.config import Config, cli_config
from recsys.evaluate.metrics import candidate_recall, evaluate_rankings, top_k_from_scores
from recsys.features.aggregates import item_stats, user_aisle_share, user_item_history
from recsys.utils import configure_logging, get_logger, timer

log = get_logger(__name__)

SOURCES = ["src_history", "src_i2v", "src_aisle", "src_global"]


@dataclass
class GlobalTables:
    """Tables computed once over all prior data and reused for every user chunk."""

    products: pd.DataFrame
    items: pd.DataFrame  # item_stats()
    index: ItemIndex


def user_chunks(user_ids: np.ndarray, n_chunks: int) -> Iterator[np.ndarray]:
    users = np.sort(np.unique(user_ids))
    for i in range(n_chunks):
        chunk = users[users % n_chunks == i]
        if len(chunk):
            yield chunk


def item2vec_sim(index: ItemIndex, user_vecs: pd.DataFrame, cands: pd.DataFrame) -> np.ndarray:
    """Cosine between user embedding and item embedding for each candidate row (NaN if unknown)."""
    urow = user_vecs.index.get_indexer(cands["user_id"])
    irow = index.rows(cands["product_id"].to_numpy())
    uv = np.stack(user_vecs["vec"].to_numpy()) if len(user_vecs) else np.zeros((0, 1))
    sim = np.full(len(cands), np.nan, dtype="float32")
    ok = np.flatnonzero((urow >= 0) & (irow >= 0))
    for start in range(0, len(ok), 500_000):  # bounded memory for the row gathers
        sl = ok[start : start + 500_000]
        sim[sl] = np.einsum("ij,ij->i", uv[urow[sl]], index.vectors[irow[sl]])
    return sim


def generate_for_chunk(
    prior: pd.DataFrame,
    orders: pd.DataFrame,
    g: GlobalTables,
    cfg: Config,
    hist: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Candidates for all users present in ``prior`` (one row per user x product)."""
    c = cfg["candidates"]
    if hist is None:
        hist = user_item_history(prior, orders, c["history_decay"])
    users = hist["user_id"].unique()

    # --- history -------------------------------------------------------------
    parts = [hist[["user_id", "product_id"]].assign(src_history=1)]

    # --- item2vec + FAISS ----------------------------------------------------
    u_ids, u_vecs = g.index.user_vectors(
        hist["user_id"].to_numpy(),
        hist["product_id"].to_numpy(),
        hist["hist_score"].to_numpy(),
    )
    user_vecs = pd.DataFrame({"vec": list(u_vecs)}, index=pd.Index(u_ids, name="user_id"))
    has_vec = np.linalg.norm(u_vecs, axis=1) > 0
    n_i2v = c["n_item2vec"]
    if has_vec.any():
        k_search = min(2 * n_i2v, len(g.index.product_ids))
        sims, rows = g.index.index.search(u_vecs[has_vec], k_search)
        nn = pd.DataFrame(
            {
                "user_id": np.repeat(u_ids[has_vec], k_search),
                "product_id": g.index.product_ids[rows.ravel()],
            }
        )
        nn = nn.merge(parts[0], on=["user_id", "product_id"], how="left")
        # keep up to n_i2v *unseen* items; seen ones just get the flag
        unseen = nn["src_history"].isna()
        nn["_r"] = unseen.groupby(nn["user_id"]).cumsum()
        nn = nn[~unseen | (nn["_r"] <= n_i2v)]
        parts.append(nn[["user_id", "product_id"]].assign(src_i2v=1))

    # --- popular items in the user's top aisles ------------------------------
    ua = user_aisle_share(hist, g.products)
    top_aisles = ua.sort_values(["user_id", "ua_share"], ascending=[True, False])
    top_aisles = top_aisles.groupby("user_id").head(c["n_top_aisles"])
    aisle_pop = (
        g.items.sort_values("i_n_purchases", ascending=False)
        .groupby("i_aisle_id")
        .head(c["n_per_aisle"])[["product_id", "i_aisle_id"]]
        .astype({"i_aisle_id": "int16"})
    )
    ap = top_aisles.merge(aisle_pop, left_on="aisle_id", right_on="i_aisle_id")
    parts.append(ap[["user_id", "product_id"]].assign(src_aisle=1))

    # --- global popularity fallback -----------------------------------------
    top_global = g.items.nlargest(c["n_global"], "i_n_purchases")["product_id"].to_numpy()
    gp = pd.DataFrame(
        {
            "user_id": np.repeat(users, len(top_global)),
            "product_id": np.tile(top_global, len(users)),
        }
    )
    parts.append(gp.assign(src_global=1))

    # --- merge & dedupe ------------------------------------------------------
    cands = pd.concat(parts, ignore_index=True)
    for s in SOURCES:
        if s not in cands:
            cands[s] = 0
    cands[SOURCES] = cands[SOURCES].fillna(0).astype("int8")
    cands = cands.groupby(["user_id", "product_id"], as_index=False)[SOURCES].max()

    cands = cands.merge(
        hist[["user_id", "product_id", "hist_score"]], on=["user_id", "product_id"], how="left"
    )
    cands["hist_score"] = cands["hist_score"].fillna(0).astype("float32")
    cands["i2v_sim"] = item2vec_sim(g.index, user_vecs, cands)
    # History items always rank above unseen ones; unseen ordered by i2v similarity.
    unseen_score = np.clip(np.nan_to_num(cands["i2v_sim"].to_numpy(), nan=-1.0), -1, 1)
    cands["retrieval_score"] = np.where(
        cands["src_history"] == 1, 2.0 + cands["hist_score"], unseen_score
    ).astype("float32")

    cands = cands.sort_values(
        ["user_id", "retrieval_score", "product_id"], ascending=[True, False, True]
    )
    cands = cands.groupby("user_id").head(c["max_candidates"]).reset_index(drop=True)
    return cands


def load_global_tables(cfg: Config, prior: pd.DataFrame) -> GlobalTables:
    products = pd.read_parquet(cfg.paths.processed_dir / "products.parquet")
    return GlobalTables(
        products=products,
        items=item_stats(prior, products),
        index=ItemIndex.load(cfg.paths.models_dir),
    )


def generate_all(cfg: Config) -> None:
    d = cfg.paths.processed_dir
    prior = pd.read_parquet(d / "prior.parquet")
    orders = pd.read_parquet(d / "orders.parquet")
    g = load_global_tables(cfg, prior)
    out_dir = d / "candidates"
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("part-*.parquet"):
        old.unlink()
    n_chunks = cfg["features"]["n_user_chunks"]
    for i, users in enumerate(user_chunks(prior["user_id"].to_numpy(), n_chunks)):
        with timer(log, f"candidates chunk {i}"):
            p = prior[prior["user_id"].isin(users)]
            o = orders[orders["user_id"].isin(users)]
            generate_for_chunk(p, o, g, cfg).to_parquet(out_dir / f"part-{i:03d}.parquet")
    report_candidates(cfg)


def report_candidates(cfg: Config, split: str = "test") -> dict[str, float]:
    d = cfg.paths.processed_dir
    splits = pd.read_parquet(d / "splits.parquet")
    users = splits.loc[splits["split"] == split, "user_id"]
    cands = pd.read_parquet(d / "candidates")
    cands = cands[cands["user_id"].isin(users)]
    truth = pd.read_parquet(d / "targets.parquet")
    truth = truth[truth["user_id"].isin(users)]
    stats: dict[str, float] = {
        "candidate_recall": candidate_recall(cands, truth),
        "avg_candidates_per_user": float(cands.groupby("user_id").size().mean()),
    }
    for s in SOURCES:
        stats[f"candidate_recall_{s}"] = candidate_recall(cands[cands[s] == 1], truth)
        stats[f"share_{s}"] = float(cands[s].mean())
    retrieval = evaluate_rankings(top_k_from_scores(cands, "retrieval_score", 20), truth, [10, 20])
    stats.update({f"retrieval_{k}": v for k, v in retrieval.items()})
    log.info("candidate report", split=split, **{k: round(v, 4) for k, v in stats.items()})
    return stats


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Generate candidates for all users")
    generate_all(cfg)


if __name__ == "__main__":
    main()
