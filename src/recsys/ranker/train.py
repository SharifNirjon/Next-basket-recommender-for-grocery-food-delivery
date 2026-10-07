"""Train the LightGBM lambdarank ranker (group = user) with early stopping on valid NDCG@10."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from recsys.config import Config, cli_config  # noqa: E402
from recsys.evaluate.metrics import evaluate_rankings, top_k_from_scores  # noqa: E402
from recsys.features.schema import CATEGORICAL_FEATURES, FEATURES  # noqa: E402
from recsys.ranker.dataset import RankingData, load_split  # noqa: E402
from recsys.utils import configure_logging, get_logger, timer  # noqa: E402

log = get_logger(__name__)
EXPERIMENT = "next-basket-ranker"


def groups_with_positives(data: RankingData) -> RankingData:
    """Drop users whose candidates contain no positive: they carry no lambdarank gradient
    and LightGBM scores them NDCG=1, which would inflate early-stopping metrics.
    (Final evaluation always uses *all* users.)"""
    user_pos = pd.Series(data.y).groupby(data.keys["user_id"].to_numpy()).transform("sum")
    keep = user_pos.to_numpy() > 0
    keys = data.keys[keep].reset_index(drop=True)
    return RankingData(
        keys=keys,
        X=data.X[keep].reset_index(drop=True),
        y=data.y[keep],
        group=keys.groupby("user_id", sort=True).size().to_numpy(),
    )


def score(booster: lgb.Booster, data: RankingData) -> pd.DataFrame:
    s = booster.predict(data.X[FEATURES], num_threads=0)
    return data.keys.assign(score=s.astype("float32"))


def plot_importance(booster: lgb.Booster, path: Path) -> pd.DataFrame:
    imp = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance("gain"),
            "split": booster.feature_importance("split"),
        }
    ).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    top = imp.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["gain_share"], color="#3b6ea5")
    ax.set_xlabel("share of total gain")
    ax.set_title("LightGBM lambdarank - feature importance (gain)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return imp


def train(cfg: Config) -> dict[str, float]:
    rc = cfg["ranker"]
    models_dir = cfg.paths.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)
    with timer(log, "load datasets"):
        train_full = load_split(cfg, "train")
        valid_full = load_split(cfg, "valid")
        tr, va = groups_with_positives(train_full), groups_with_positives(valid_full)
    log.info(
        "datasets",
        train_rows=len(tr.y),
        train_users=len(tr.group),
        valid_rows=len(va.y),
        valid_users=len(va.group),
        train_pos_rate=round(float(tr.y.mean()), 4),
    )

    mlflow.set_tracking_uri(cfg.paths.mlruns_dir.resolve().as_uri())
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="sample" if cfg.sample else "full") as run:
        params = {**rc["params"], "n_estimators": rc["n_estimators"], "random_state": cfg.seed}
        mlflow.log_params(params)
        mlflow.log_params(
            {
                "sample": cfg.sample,
                "n_features": len(FEATURES),
                "max_candidates": cfg["candidates"]["max_candidates"],
                "train_users": len(tr.group),
            }
        )
        model = lgb.LGBMRanker(**params)
        with timer(log, "fit ranker"):
            model.fit(
                tr.X,
                tr.y,
                group=tr.group,
                eval_set=[(va.X, va.y)],
                eval_group=[va.group],
                eval_at=[rc["eval_at"]],
                categorical_feature=CATEGORICAL_FEATURES,
                callbacks=[
                    lgb.early_stopping(rc["early_stopping_rounds"], verbose=False),
                    lgb.log_evaluation(50),
                ],
            )
        booster = model.booster_
        best_iter = int(model.best_iteration_ or booster.current_iteration())
        booster.save_model(str(models_dir / "ranker.txt"), num_iteration=best_iter)
        booster = lgb.Booster(model_file=str(models_dir / "ranker.txt"))

        # Honest validation metrics: all valid users (incl. those with no positive candidate).
        valid_truth = pd.read_parquet(cfg.paths.processed_dir / "targets.parquet")
        valid_truth = valid_truth[valid_truth["user_id"].isin(valid_full.keys["user_id"])]
        recs = top_k_from_scores(score(booster, valid_full), "score", 20)
        valid_metrics = evaluate_rankings(recs, valid_truth, cfg["evaluation"]["ks"])
        lgb_ndcg = float(model.best_score_["valid_0"][f"ndcg@{rc['eval_at']}"])

        imp = plot_importance(booster, cfg.paths.reports_dir / "feature_importance.png")
        imp.to_csv(cfg.paths.reports_dir / "feature_importance.csv", index=False)

        mlflow.log_metric("best_iteration", best_iter)
        mlflow.log_metric("lgb_valid_ndcg_at_10_pos_groups", lgb_ndcg)
        mlflow.log_metrics({f"valid_{k.replace('@', '_at_')}": v for k, v in valid_metrics.items()})
        mlflow.log_artifact(str(models_dir / "ranker.txt"), "model")
        mlflow.log_artifact(str(cfg.paths.reports_dir / "feature_importance.png"))
        mlflow.log_artifact(str(cfg.paths.reports_dir / "feature_importance.csv"))

        meta = {
            "model_version": f"lgbm-lambdarank-{run.info.run_id[:8]}",
            "mlflow_run_id": run.info.run_id,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "best_iteration": best_iter,
            "features": FEATURES,
            "sample": cfg.sample,
            "valid_metrics": valid_metrics,
        }
        (models_dir / "ranker_meta.json").write_text(json.dumps(meta, indent=2))
    log.info(
        "ranker trained",
        best_iteration=best_iter,
        lgb_valid_ndcg10_pos_groups=round(lgb_ndcg, 4),
        **{f"valid_{k}": round(v, 4) for k, v in valid_metrics.items()},
        top_features=imp["feature"].head(8).tolist(),
    )
    return valid_metrics


def main() -> None:
    configure_logging()
    cfg, _ = cli_config("Train the LightGBM lambdarank ranker")
    np.random.seed(cfg.seed)
    train(cfg)


if __name__ == "__main__":
    main()
