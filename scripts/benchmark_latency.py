"""Latency benchmark: N sequential GET /recommend requests against a running API.

Usage: python scripts/benchmark_latency.py --url http://localhost:8000 --n 1000 --k 10
       [--artifacts artifacts]  (user ids are sampled from the exported user index)
Reports client-side p50/p95/p99 (ms) and writes reports/latency.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import httpx
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--unknown-share", type=float, default=0.0, help="share of unknown users")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--artifacts", default="artifacts", help="to sample real user ids")
    ap.add_argument("--out", default="reports/latency.json")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    with httpx.Client(base_url=args.url, timeout=10.0) as client:
        meta = client.get("/metadata").json()
        health = client.get("/health").json()
        known = np.load(Path(args.artifacts) / "user_index.npz")["user_ids"]
        users = rng.choice(known, size=args.n + args.warmup).astype("int64")
        unknown = rng.random(len(users)) < args.unknown_share
        users[unknown] += 10_000_000

        latencies, fallbacks, errors = [], 0, 0
        for i, uid in enumerate(users):
            t0 = time.perf_counter()
            r = client.get(f"/recommend/{int(uid)}", params={"k": args.k})
            dt = (time.perf_counter() - t0) * 1000
            if r.status_code != 200:
                errors += 1
                continue
            if i >= args.warmup:
                latencies.append(dt)
                fallbacks += r.json()["fallback"]

    lat = np.asarray(latencies)
    result = {
        "requests": int(len(lat)),
        "errors": errors,
        "fallback_responses": int(fallbacks),
        "k": args.k,
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "p99_ms": round(float(np.percentile(lat, 99)), 2),
        "mean_ms": round(float(lat.mean()), 2),
        "max_ms": round(float(lat.max()), 2),
        "model_version": meta["model_version"],
        "n_users_in_artifacts": health["n_users"],
        "target": args.url,
        "client_host": f"{platform.system()} {platform.machine()} py{platform.python_version()}",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
