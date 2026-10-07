"""Item2Vec: Word2Vec (skip-gram, negative sampling) over prior-order baskets + FAISS index.

Each prior order is a "sentence" of product ids in add-to-cart order. Only prior
orders are used, so the target ("train") orders never influence the embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import Word2Vec

from recsys.config import Config, cli_config
from recsys.utils import configure_logging, get_logger, timer

log = get_logger(__name__)
logging.getLogger("gensim").setLevel(logging.WARNING)


@dataclass
class ItemIndex:
    """Normalized item vectors, their product ids and an inner-product FAISS index."""

    product_ids: np.ndarray  # (n_items,) int32
    vectors: np.ndarray  # (n_items, d) float32, L2-normalized
    index: faiss.Index
    lookup: np.ndarray  # product_id -> row in `vectors`, -1 if not embedded

    @classmethod
    def build(cls, product_ids: np.ndarray, vectors: np.ndarray) -> ItemIndex:
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        lookup = np.full(int(product_ids.max()) + 1, -1, dtype="int32")
        lookup[product_ids] = np.arange(len(product_ids), dtype="int32")
        return cls(product_ids.astype("int32"), vectors, index, lookup)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "item2vec_ids.npy", self.product_ids)
        np.save(directory / "item2vec_vectors.npy", self.vectors)
        faiss.write_index(self.index, str(directory / "item2vec.faiss"))

    @classmethod
    def load(cls, directory: Path) -> ItemIndex:
        ids = np.load(directory / "item2vec_ids.npy")
        vecs = np.load(directory / "item2vec_vectors.npy")
        index = faiss.read_index(str(directory / "item2vec.faiss"))
        lookup = np.full(int(ids.max()) + 1, -1, dtype="int32")
        lookup[ids] = np.arange(len(ids), dtype="int32")
        return cls(ids, vecs, index, lookup)

    def rows(self, product_ids: np.ndarray) -> np.ndarray:
        """Vector row for each product id (-1 if unknown / out of range)."""
        pid = np.asarray(product_ids)
        out = np.full(len(pid), -1, dtype="int32")
        ok = pid < len(self.lookup)
        out[ok] = self.lookup[pid[ok]]
        return out

    def user_vectors(
        self, user_ids: np.ndarray, product_ids: np.ndarray, weights: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Weighted mean of item vectors per user (rows given as COO triples).

        Returns (unique_user_ids, normalized user vectors). Users without any
        embedded item get a zero vector.
        """
        users, u_idx = np.unique(user_ids, return_inverse=True)
        rows = self.rows(product_ids)
        ok = rows >= 0
        m = sp.csr_matrix(
            (weights[ok].astype("float32"), (u_idx[ok], rows[ok])),
            shape=(len(users), len(self.product_ids)),
        )
        uv = np.asarray(m @ self.vectors, dtype="float32")
        norms = np.linalg.norm(uv, axis=1, keepdims=True)
        uv = np.divide(uv, norms, out=np.zeros_like(uv), where=norms > 0)
        return users, uv


def write_corpus(prior: pd.DataFrame, path: Path) -> int:
    """One line per prior order: space-separated product ids in add-to-cart order."""
    p = prior.sort_values(["order_id", "add_to_cart_order"])
    tokens = p["product_id"].astype(str).to_numpy()
    bounds = np.flatnonzero(np.diff(p["order_id"].to_numpy())) + 1
    with open(path, "w") as f:
        for basket in np.split(tokens, bounds):
            f.write(" ".join(basket))
            f.write("\n")
    return len(bounds) + 1


def train_item2vec(cfg: Config) -> ItemIndex:
    params = cfg["candidates"]["item2vec"]
    models_dir = cfg.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    prior = pd.read_parquet(
        cfg.paths.processed_dir / "prior.parquet",
        columns=["order_id", "product_id", "add_to_cart_order"],
    )
    corpus = models_dir / "baskets.txt"
    with timer(log, "write corpus"):
        n_sent = write_corpus(prior, corpus)
    with timer(log, "train word2vec"):
        model = Word2Vec(
            corpus_file=str(corpus),
            vector_size=params["vector_size"],
            window=params["window"],
            min_count=params["min_count"],
            sg=params["sg"],
            negative=params["negative"],
            sample=params["sample"],
            epochs=params["epochs"],
            workers=4,
            seed=cfg.seed,
        )
    corpus.unlink()
    ids = np.array([int(k) for k in model.wv.index_to_key], dtype="int32")
    index = ItemIndex.build(ids, model.wv.vectors)
    index.save(models_dir)
    log.info("item2vec trained", sentences=n_sent, vocab=len(ids), dim=index.vectors.shape[1])
    return index


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Train Item2Vec and build the FAISS index")
    train_item2vec(cfg)


if __name__ == "__main__":
    main()
