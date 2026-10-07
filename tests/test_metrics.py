"""Metric functions vs hand-computed examples."""

import math

import pandas as pd
import pytest

from recsys.evaluate.metrics import (
    average_precision_at_k,
    candidate_recall,
    evaluate_rankings,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

# recs = [1, 2, 3, 4, 5], truth = {2, 5, 9}; hits at ranks 2 and 5.
RECS = [1, 2, 3, 4, 5]
TRUTH = {2, 5, 9}


def test_recall_precision_hit():
    assert recall_at_k(RECS, TRUTH, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RECS, TRUTH, 1) == 0.0
    assert precision_at_k(RECS, TRUTH, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RECS, TRUTH, 2) == pytest.approx(1 / 2)
    assert hit_rate_at_k(RECS, TRUTH, 1) == 0.0
    assert hit_rate_at_k(RECS, TRUTH, 2) == 1.0


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(RECS, TRUTH, 5) == pytest.approx(dcg / idcg)
    assert ndcg_at_k([2, 5, 9], TRUTH, 3) == pytest.approx(1.0)


def test_map_hand_computed():
    # P@2 = 1/2, P@5 = 2/5, normalised by min(|truth|, K) = 3
    assert average_precision_at_k(RECS, TRUTH, 5) == pytest.approx((0.5 + 0.4) / 3)
    # K=2 -> only the first hit, denominator min(3, 2) = 2
    assert average_precision_at_k(RECS, TRUTH, 2) == pytest.approx(0.5 / 2)


def test_vectorized_matches_reference():
    recs = pd.DataFrame(
        {
            "user_id": [1] * 5 + [2] * 3,
            "product_id": RECS + [7, 8, 9],
            "rank": [1, 2, 3, 4, 5, 1, 2, 3],
        }
    )
    truth = pd.DataFrame(
        {"user_id": [1, 1, 1, 2, 3], "product_id": [2, 5, 9, 8, 4]}
    )  # user 3 has no recs -> counts as zero
    m = evaluate_rankings(recs, truth, ks=[2, 5])
    per_user = {1: (RECS, TRUTH), 2: ([7, 8, 9], {8}), 3: ([], {4})}
    for k in (2, 5):
        for name, fn in [
            ("recall", recall_at_k),
            ("precision", precision_at_k),
            ("ndcg", ndcg_at_k),
            ("map", average_precision_at_k),
            ("hit_rate", hit_rate_at_k),
        ]:
            expected = sum(fn(r, t, k) for r, t in per_user.values()) / 3
            assert m[f"{name}@{k}"] == pytest.approx(expected), (name, k)
    assert m["n_users"] == 3


def test_candidate_recall():
    cands = pd.DataFrame({"user_id": [1, 1, 2], "product_id": [2, 3, 8]})
    truth = pd.DataFrame({"user_id": [1, 1, 2, 2], "product_id": [2, 5, 8, 9]})
    assert candidate_recall(cands, truth) == pytest.approx((0.5 + 0.5) / 2)
