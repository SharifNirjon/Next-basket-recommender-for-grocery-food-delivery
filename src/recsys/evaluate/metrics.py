"""Ranking metrics for next-basket recommendation.

Two implementations:
* reference per-user functions (plain Python, easy to verify by hand; used in tests)
* ``evaluate_rankings``: vectorized pandas version used on hundreds of thousands of users.

Conventions (binary relevance, ground truth = all products of the target order):
  Recall@K    = |top-K ∩ truth| / |truth|
  Precision@K = |top-K ∩ truth| / K
  NDCG@K      = DCG@K / IDCG@K, gain 1/log2(rank+1), IDCG over min(|truth|, K) hits
  MAP@K       = sum_{i<=K} P@i * rel_i / min(|truth|, K)
  HitRate@K   = 1 if any of top-K is relevant
Users with no recommendations still count (all metrics 0).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def recall_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    return len(set(recs[:k]) & truth) / len(truth) if truth else 0.0


def precision_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    return len(set(recs[:k]) & truth) / k


def hit_rate_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    return float(any(r in truth for r in recs[:k]))


def ndcg_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(recs[:k]) if r in truth)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(truth), k)))
    return dcg / idcg if idcg > 0 else 0.0


def average_precision_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    hits, score = 0, 0.0
    for i, r in enumerate(recs[:k]):
        if r in truth:
            hits += 1
            score += hits / (i + 1)
    denom = min(len(truth), k)
    return score / denom if denom else 0.0


def per_user_metrics(recs: pd.DataFrame, truth: pd.DataFrame, ks: Sequence[int]) -> pd.DataFrame:
    """Per-user metrics (index = every user in ``truth``).

    recs:  user_id, product_id, rank (1-based, unique per user)
    truth: user_id, product_id
    """
    truth = truth[["user_id", "product_id"]].drop_duplicates()
    n_truth = truth.groupby("user_id").size().rename("n_truth")
    users = n_truth.index
    recs = recs.loc[recs["user_id"].isin(users), ["user_id", "product_id", "rank"]]
    scored = recs.merge(truth.assign(_rel=1), on=["user_id", "product_id"], how="left")
    scored["_rel"] = scored["_rel"].fillna(0).astype("float64")
    scored = scored.sort_values(["user_id", "rank"])

    max_k = max(ks)
    idcg_table = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(np.arange(2, max_k + 2)))])
    out = pd.DataFrame(index=users)
    n = n_truth.to_numpy()
    for k in ks:
        top = scored[scored["rank"] <= k].copy()
        top["_cumhits"] = top.groupby("user_id")["_rel"].cumsum()
        top["_gain"] = top["_rel"] / np.log2(top["rank"] + 1)
        top["_prec_rel"] = top["_rel"] * top["_cumhits"] / top["rank"]
        agg = top.groupby("user_id")[["_rel", "_gain", "_prec_rel"]].sum()
        agg = agg.reindex(users, fill_value=0.0)
        hits = agg["_rel"].to_numpy()
        denom = np.minimum(n, k)
        out[f"recall@{k}"] = hits / n
        out[f"precision@{k}"] = hits / k
        out[f"ndcg@{k}"] = agg["_gain"].to_numpy() / idcg_table[denom]
        out[f"map@{k}"] = agg["_prec_rel"].to_numpy() / denom
        out[f"hit_rate@{k}"] = (hits > 0).astype("float64")
    return out


def evaluate_rankings(
    recs: pd.DataFrame, truth: pd.DataFrame, ks: Sequence[int]
) -> dict[str, float]:
    """Mean of ``per_user_metrics`` over all users present in ``truth``."""
    pu = per_user_metrics(recs, truth, ks)
    out = {c: float(v) for c, v in pu.mean().items()}
    out["n_users"] = float(len(pu))
    return out


def paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of (a - b) over users with a 95% paired-bootstrap confidence interval."""
    diff = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n_boot, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def candidate_recall(candidates: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Mean per-user share of ground-truth items present in the candidate set."""
    truth = truth[["user_id", "product_id"]].drop_duplicates()
    found = truth.merge(
        candidates[["user_id", "product_id"]].drop_duplicates(),
        on=["user_id", "product_id"],
        how="left",
        indicator=True,
    )
    per_user = (found["_merge"] == "both").groupby(found["user_id"]).mean()
    return float(per_user.mean())


def top_k_from_scores(df: pd.DataFrame, score_col: str, k: int) -> pd.DataFrame:
    """Rank candidates per user by score (desc, ties by product_id) and keep top-k."""
    ranked = df.sort_values(["user_id", score_col, "product_id"], ascending=[True, False, True])
    ranked = ranked.assign(rank=ranked.groupby("user_id").cumcount() + 1)
    return ranked.loc[ranked["rank"] <= k, ["user_id", "product_id", "rank"]]
