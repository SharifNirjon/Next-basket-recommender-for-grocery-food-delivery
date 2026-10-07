"""Run every offline step in order (used by tests and ``make all``-style runs)."""

from __future__ import annotations

from recsys.candidates.generate import generate_all
from recsys.candidates.item2vec import train_item2vec
from recsys.config import Config, cli_config
from recsys.data.prepare import prepare
from recsys.evaluate.run import run_evaluation
from recsys.features.build import build_all
from recsys.ranker.train import train
from recsys.serve.export import export
from recsys.utils import configure_logging


def run_pipeline(cfg: Config) -> None:
    prepare(cfg)
    train_item2vec(cfg)
    generate_all(cfg)
    build_all(cfg)
    train(cfg)
    run_evaluation(cfg)
    export(cfg)


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Run the full offline pipeline (raw data must exist)")
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
