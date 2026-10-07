"""API tests with TestClient against tiny fixture artifacts built by the pipeline."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from recsys.config import Config
from recsys.ranker.dataset import load_split
from recsys.ranker.train import score


@pytest.fixture(scope="module")
def client(pipeline_cfg: Config, monkeypatch_module: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch_module.setenv("ARTIFACTS_DIR", str(pipeline_cfg.paths.artifacts_dir))
    from recsys.serve.app import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module() -> Iterator[pytest.MonkeyPatch]:
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def known_user(pipeline_cfg: Config) -> int:
    idx = np.load(pipeline_cfg.paths.artifacts_dir / "user_index.npz")
    return int(idx["user_ids"][0])


def test_health_and_metadata(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"
    m = client.get("/metadata").json()
    assert m["model_version"].startswith("lgbm-lambdarank-")
    assert "trained_at" in m and "test" in m["metrics"]


def test_recommend_known_user(client: TestClient, known_user: int) -> None:
    r = client.get(f"/recommend/{known_user}", params={"k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["fallback"] is False and len(body["items"]) == 5
    assert {"product_id", "product_name", "aisle", "score"} <= set(body["items"][0])
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert len({i["product_id"] for i in body["items"]}) == 5


def test_unknown_user_falls_back_to_popularity(client: TestClient) -> None:
    body = client.get("/recommend/987654321", params={"k": 3}).json()
    assert body["fallback"] is True and len(body["items"]) == 3
    assert body["items"][0]["score"] == 1.0


@pytest.mark.parametrize(
    "url",
    [
        "/recommend/abc",
        "/recommend/0",
        "/recommend/-5",
        "/recommend/1?k=0",
        "/recommend/1?k=101",
        "/recommend/1?hour=24",
    ],
)
def test_validation_errors(client: TestClient, url: str) -> None:
    assert client.get(url).status_code == 422


def test_404s(client: TestClient) -> None:
    assert client.get("/users/987654321/history").status_code == 404
    assert client.get("/similar/987654321").status_code == 404
    assert client.get("/no/such/route").status_code == 404


def test_history_and_similar(client: TestClient, known_user: int) -> None:
    h = client.get(f"/users/{known_user}/history").json()
    assert h["items"] and all(i["times_bought"] >= 1 for i in h["items"])
    pid = h["items"][0]["product_id"]
    s = client.get(f"/similar/{pid}", params={"k": 3}).json()
    assert len(s["items"]) == 3 and pid not in {i["product_id"] for i in s["items"]}


def test_serving_matches_offline_scores(client: TestClient, pipeline_cfg: Config) -> None:
    """Train/serve skew guard: API ranking == offline ranking under the same context."""
    import lightgbm as lgb

    data = load_split(pipeline_cfg, "test")
    booster = lgb.Booster(model_file=str(pipeline_cfg.paths.models_dir / "ranker.txt"))
    offline = score(booster, data)
    ctx = pd.read_parquet(pipeline_cfg.paths.processed_dir / "target_orders.parquet")
    for user in data.keys["user_id"].unique()[:5]:
        t = ctx[ctx["user_id"] == user].iloc[0]
        params = {
            "k": 10,
            "hour": int(t["order_hour_of_day"]),
            "dow": int(t["order_dow"]),
            "days_since_prior": float(t["days_since_prior_order"]),
        }
        body = client.get(f"/recommend/{user}", params=params).json()
        exp = offline[offline["user_id"] == user].nlargest(10, "score")
        np.testing.assert_allclose(
            [i["score"] for i in body["items"]], exp["score"].to_numpy(), rtol=1e-5
        )
