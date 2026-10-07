"""Final evaluation on held-out TEST users: baselines vs two-stage model.

(a) global popularity   (b) user's most frequent items
(c) retrieval only      (d) retrieval + LightGBM lambdarank
"""

from __future__ import annotations

import json
from typing import Any

import lightgbm as lgb
import mlflow
import pandas as pd

from recsys.candidates.generate import SOURCES
from recsys.config import Config, cli_config
from recsys.evaluate.baselines import popularity_recs, user_frequency_recs
from recsys.evaluate.metrics import (
    candidate_recall,
    paired_bootstrap_ci,
    per_user_metrics,
    top_k_from_scores,
)
from recsys.ranker.dataset import load_split
from recsys.ranker.train import score
from recsys.utils import configure_logging, get_logger, timer

log = get_logger(__name__)

SYSTEMS = {
    "popularity": "(a) Global popularity",
    "user_frequency": "(b) User's most frequent items",
    "retrieval_only": "(c) Retrieval only (retrieval score)",
    "two_stage": "(d) Two-stage: retrieval + LGBM lambdarank",
}


def evaluate(cfg: Config, split: str = "test") -> dict[str, Any]:
    d = cfg.paths.processed_dir
    ks = cfg["evaluation"]["ks"]
    max_k = max(ks)
    data = load_split(cfg, split)
    users = data.keys["user_id"].unique()
    truth = pd.read_parquet(d / "targets.parquet")
    truth = truth[truth["user_id"].isin(users)]
    prior = pd.read_parquet(d / "prior.parquet")

    booster = lgb.Booster(model_file=str(cfg.paths.models_dir / "ranker.txt"))
    with timer(log, f"evaluate {split}"):
        cands = data.keys.assign(retrieval_score=data.X["retrieval_score"].to_numpy())
        recs = {
            "popularity": popularity_recs(prior, users, max_k),
            "user_frequency": user_frequency_recs(prior, users, max_k),
            "retrieval_only": top_k_from_scores(cands, "retrieval_score", max_k),
            "two_stage": top_k_from_scores(score(booster, data), "score", max_k),
        }
        per_user = {name: per_user_metrics(r, truth, ks) for name, r in recs.items()}
        results = {
            name: {**{c: float(v) for c, v in pu.mean().items()}, "n_users": float(len(pu))}
            for name, pu in per_user.items()
        }
        # Is the two-stage gain real? Paired bootstrap over test users.
        significance = {
            f"two_stage_minus_{base}_{metric}": dict(
                zip(
                    ("mean", "ci95_low", "ci95_high"),
                    paired_bootstrap_ci(
                        per_user["two_stage"][metric].to_numpy(),
                        per_user[base][metric].to_numpy(),
                        seed=cfg.seed,
                    ),
                    strict=True,
                )
            )
            for base in ("user_frequency", "retrieval_only")
            for metric in ("recall@10", "ndcg@10")
        }

    cand_flags = data.keys.assign(**{s: data.X[s].to_numpy() for s in SOURCES})
    retrieval_stats = {
        "candidate_recall": candidate_recall(cand_flags, truth),
        "avg_candidates_per_user": float(len(data.keys) / len(users)),
        **{
            f"candidate_recall_{s}": candidate_recall(cand_flags[cand_flags[s] == 1], truth)
            for s in SOURCES
        },
        "share_truth_items_never_bought_before": float(
            truth.merge(
                prior[["user_id", "product_id"]].drop_duplicates(),
                how="left",
                indicator=True,
            )["_merge"]
            .eq("left_only")
            .mean()
        ),
    }
    meta = json.loads((cfg.paths.models_dir / "ranker_meta.json").read_text())
    return {
        "split": split,
        "n_users": int(len(users)),
        "sample": cfg.sample,
        "model_version": meta["model_version"],
        "systems": results,
        "significance": significance,
        "retrieval": retrieval_stats,
    }


def _diff_label(key: str) -> str:
    base, metric = key.removeprefix("two_stage_minus_").rsplit("_", 1)
    return f"(d) minus {SYSTEMS[base][:3]} - {metric}"


def to_markdown(m: dict[str, Any], ks: list[int]) -> str:
    cols = [
        f"{name}@{k}" for k in ks for name in ("recall", "precision", "ndcg", "map", "hit_rate")
    ]
    header = "| System | " + " | ".join(c.replace("_", " ") for c in cols) + " |"
    sep = "|---|" + "---:|" * len(cols)
    best = {c: max(r[c] for r in m["systems"].values()) for c in cols}
    rows = []
    for key, label in SYSTEMS.items():
        r = m["systems"][key]
        cells = [f"**{r[c]:.4f}**" if r[c] == best[c] else f"{r[c]:.4f}" for c in cols]
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    rt = m["retrieval"]
    return "\n".join(
        [
            f"# Results ({m['split']} users, n={m['n_users']:,}"
            f"{', SAMPLE' if m['sample'] else ''})",
            "",
            f"Model: `{m['model_version']}`. Ground truth: every product in the user's "
            "held-out last order (including first-time purchases).",
            "",
            header,
            sep,
            *rows,
            "",
            "## Candidate generation",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Candidate recall (all sources, upper bound for the ranker) "
            f"| {rt['candidate_recall']:.4f} |",
            f"| Avg candidates per user | {rt['avg_candidates_per_user']:.1f} |",
            *[
                f"| Candidate recall, `{s}` only | {rt[f'candidate_recall_{s}']:.4f} |"
                for s in SOURCES
            ],
            f"| Share of truth items never bought before by the user "
            f"| {rt['share_truth_items_never_bought_before']:.4f} |",
            "",
            "## Is the gain real? (paired bootstrap over test users, 1,000 resamples)",
            "",
            "| Difference | Mean | 95% CI |",
            "|---|---:|---:|",
            *[
                f"| {_diff_label(k)} | {v['mean']:+.4f} "
                f"| [{v['ci95_low']:+.4f}, {v['ci95_high']:+.4f}] |"
                for k, v in m["significance"].items()
            ],
            "",
        ]
    )


def run_evaluation(cfg: Config) -> dict[str, Any]:
    """Evaluate on test users, write reports/metrics.json + results.md, log to MLflow."""
    m = evaluate(cfg, "test")
    reports = cfg.paths.reports_dir
    (reports / "metrics.json").write_text(json.dumps(m, indent=2))
    (reports / "results.md").write_text(to_markdown(m, cfg["evaluation"]["ks"]))

    meta = json.loads((cfg.paths.models_dir / "ranker_meta.json").read_text())
    mlflow.set_tracking_uri(cfg.paths.mlruns_dir.resolve().as_uri())
    with mlflow.start_run(run_id=meta["mlflow_run_id"]):
        for system, res in m["systems"].items():
            mlflow.log_metrics(
                {f"test_{system}_{k.replace('@', '_at_')}": v for k, v in res.items()}
            )
        mlflow.log_metrics({f"test_{k}": v for k, v in m["retrieval"].items()})
        mlflow.log_artifact(str(reports / "metrics.json"))
        mlflow.log_artifact(str(reports / "results.md"))
    for system, res in m["systems"].items():
        log.info(
            "test metrics", system=system, **{k: round(v, 4) for k, v in res.items() if "@" in k}
        )
    log.info("retrieval", **{k: round(v, 4) for k, v in m["retrieval"].items()})
    return m


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Evaluate baselines and the two-stage model on test users")
    run_evaluation(cfg)


if __name__ == "__main__":
    main()
