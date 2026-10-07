"""Candidate generator output shape and invariants."""

from __future__ import annotations

import pandas as pd

from recsys.candidates.generate import SOURCES, generate_for_chunk, load_global_tables
from recsys.config import Config


def test_candidate_output_shape(pipeline_cfg: Config) -> None:
    d = pipeline_cfg.paths.processed_dir
    prior = pd.read_parquet(d / "prior.parquet")
    orders = pd.read_parquet(d / "orders.parquet")
    g = load_global_tables(pipeline_cfg, prior)
    cands = generate_for_chunk(prior, orders, g, pipeline_cfg)
    cap = pipeline_cfg["candidates"]["max_candidates"]

    expected_cols = {"user_id", "product_id", "hist_score", "i2v_sim", "retrieval_score", *SOURCES}
    assert expected_cols <= set(cands.columns)
    assert not cands.duplicated(["user_id", "product_id"]).any()
    per_user = cands.groupby("user_id").size()
    assert set(per_user.index) == set(prior["user_id"].unique())  # every user gets candidates
    assert per_user.max() <= cap
    assert per_user.min() >= min(cap, pipeline_cfg["candidates"]["n_global"])
    for s in SOURCES:
        assert set(cands[s].unique()) <= {0, 1}
    assert (cands[SOURCES].sum(axis=1) >= 1).all()  # each row comes from some source
    # history items are present (unless pushed out by the cap) and rank above unseen items
    hist = prior[["user_id", "product_id"]].drop_duplicates()
    n_hist = hist.groupby("user_id").size()
    small = n_hist[n_hist <= cap].index
    got = cands[(cands["src_history"] == 1) & cands["user_id"].isin(small)]
    assert len(got) == len(hist[hist["user_id"].isin(small)])
    seen_min = cands[cands["src_history"] == 1]["retrieval_score"].min()
    unseen_max = cands[cands["src_history"] == 0]["retrieval_score"].max()
    assert seen_min > unseen_max
