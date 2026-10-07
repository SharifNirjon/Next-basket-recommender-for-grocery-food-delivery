"""In-memory artifact store and request-time ranking."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import lightgbm as lgb
import numpy as np

from recsys.features.schema import STATIC_FEATURES, add_context

_COL = {name: i for i, name in enumerate(STATIC_FEATURES)}


@dataclass
class Ranked:
    product_ids: np.ndarray
    scores: np.ndarray


class ArtifactStore:
    """Loads artifacts once; the feature matrix is memory-mapped (OS page cache)."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        self.metadata: dict = json.loads((self.dir / "metadata.json").read_text())
        if self.metadata["static_features"] != STATIC_FEATURES:
            raise RuntimeError("artifact feature schema does not match code")
        self.features = np.load(self.dir / "static_features.npy", mmap_mode="r")
        self.candidate_ids = np.load(self.dir / "candidate_ids.npy")
        idx = np.load(self.dir / "user_index.npz")
        self.user_ids, self.starts, self.ends = idx["user_ids"], idx["start"], idx["end"]
        self.booster = lgb.Booster(model_file=str(self.dir / "ranker.txt"))
        raw = json.loads((self.dir / "products.json").read_text())
        self.products: dict[int, tuple[str, str, str]] = {int(k): tuple(v) for k, v in raw.items()}
        pop = json.loads((self.dir / "popular.json").read_text())
        self.popular_ids = np.asarray(pop["product_ids"], dtype="int64")
        self.popular_scores = np.asarray(pop["scores"], dtype="float32")
        self.faiss_index = faiss.read_index(str(self.dir / "item2vec.faiss"))
        self.faiss_ids = np.load(self.dir / "item2vec_ids.npy")
        self.faiss_row = {int(p): i for i, p in enumerate(self.faiss_ids)}

    @property
    def model_version(self) -> str:
        return str(self.metadata["model_version"])

    def _slice(self, user_id: int) -> slice | None:
        i = int(np.searchsorted(self.user_ids, user_id))
        if i < len(self.user_ids) and self.user_ids[i] == user_id:
            return slice(int(self.starts[i]), int(self.ends[i]))
        return None

    def has_user(self, user_id: int) -> bool:
        return self._slice(user_id) is not None

    def user_defaults(self, user_id: int) -> tuple[int, int, float]:
        """Default context = the user's habitual hour, weekday and order gap."""
        sl = self._slice(user_id)
        assert sl is not None
        row = self.features[sl.start]
        gap = row[_COL["u_avg_days_between"]]
        return (
            int(row[_COL["u_pref_hour"]]),
            int(row[_COL["u_pref_dow"]]),
            float(np.clip(np.nan_to_num(gap, nan=7.0), 0, 30)),
        )

    def rank(self, user_id: int, k: int, hour: int, dow: int, days_since_prior: float) -> Ranked:
        sl = self._slice(user_id)
        assert sl is not None
        static = np.asarray(self.features[sl])
        X = add_context(static, hour, dow, days_since_prior)
        scores = self.booster.predict(X, num_threads=1)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return Ranked(self.candidate_ids[sl][top], scores[top].astype("float32"))

    def popular(self, k: int) -> Ranked:
        return Ranked(self.popular_ids[:k], self.popular_scores[:k])

    def history(self, user_id: int, n: int) -> list[tuple[int, int]]:
        sl = self._slice(user_id)
        assert sl is not None
        times = np.nan_to_num(self.features[sl, _COL["ui_times_bought"]])
        order = np.argsort(-times, kind="stable")
        ids = self.candidate_ids[sl]
        return [(int(ids[i]), int(times[i])) for i in order[:n] if times[i] > 0]

    def similar(self, product_id: int, k: int) -> Ranked | None:
        row = self.faiss_row.get(product_id)
        if row is None:
            return None
        vec = self.faiss_index.reconstruct(row).reshape(1, -1)
        sims, rows = self.faiss_index.search(vec, k + 1)
        keep = rows[0] != row
        return Ranked(self.faiss_ids[rows[0][keep]][:k], sims[0][keep][:k])
