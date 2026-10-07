"""FastAPI recommendation service.

GET /recommend/{user_id}?k=10[&hour=&dow=&days_since_prior=]
GET /users/{user_id}/history      (404 if unknown)
GET /similar/{product_id}?k=10    (404 if not embedded)
GET /health, GET /metadata
"""

from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi import Path as PathParam
from fastapi.responses import JSONResponse

from recsys.serve.schemas import (
    Context,
    HealthResponse,
    HistoryItem,
    HistoryResponse,
    MetadataResponse,
    RecommendedItem,
    RecommendResponse,
    SimilarResponse,
)
from recsys.serve.store import ArtifactStore, Ranked
from recsys.utils import configure_logging, get_logger

MAX_K = 100
log = get_logger("recsys.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(json=True)
    directory = Path(os.environ.get("ARTIFACTS_DIR", "artifacts"))
    t0 = time.perf_counter()
    app.state.store = ArtifactStore(directory)
    log.info(
        "artifacts loaded",
        dir=str(directory),
        users=len(app.state.store.user_ids),
        model_version=app.state.store.model_version,
        seconds=round(time.perf_counter() - t0, 2),
    )
    yield


app = FastAPI(title="Next-basket recommender", version="1.0.0", lifespan=lifespan)


def store_of(request: Request) -> ArtifactStore:
    return request.app.state.store


@app.middleware("http")
async def access_log(request: Request, call_next):  # type: ignore[no-untyped-def]
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("unhandled error", path=request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})
    log.info(
        "request",
        method=request.method,
        path=request.url.path,
        query=str(request.url.query),
        status=response.status_code,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )
    return response


def _items(store: ArtifactStore, ranked: Ranked) -> list[RecommendedItem]:
    out = []
    for pid, score in zip(ranked.product_ids.tolist(), ranked.scores.tolist(), strict=True):
        name, aisle, dept = store.products.get(pid, ("unknown", "unknown", "unknown"))
        out.append(
            RecommendedItem(
                product_id=pid, product_name=name, aisle=aisle, department=dept, score=score
            )
        )
    return out


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    store = store_of(request)
    return HealthResponse(status="ok", artifacts_loaded=True, n_users=len(store.user_ids))


@app.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request) -> MetadataResponse:
    m = store_of(request).metadata
    return MetadataResponse(**{k: m[k] for k in MetadataResponse.model_fields})


@app.get("/recommend/{user_id}", response_model=RecommendResponse)
def recommend(
    request: Request,
    user_id: Annotated[int, PathParam(gt=0, description="Instacart user id")],
    k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
    hour: Annotated[int | None, Query(ge=0, le=23)] = None,
    dow: Annotated[int | None, Query(ge=0, le=6)] = None,
    days_since_prior: Annotated[float | None, Query(ge=0, le=30)] = None,
) -> RecommendResponse:
    store = store_of(request)
    if not store.has_user(user_id):
        return RecommendResponse(
            user_id=user_id,
            k=k,
            fallback=True,
            model_version=store.model_version,
            context=None,
            items=_items(store, store.popular(k)),
        )
    d_hour, d_dow, d_gap = store.user_defaults(user_id)
    given = (hour, dow, days_since_prior)
    ctx = Context(
        hour=d_hour if hour is None else hour,
        dow=d_dow if dow is None else dow,
        days_since_prior=d_gap if days_since_prior is None else days_since_prior,
        source="request" if all(v is not None for v in given) else "user_default",
    )
    ranked = store.rank(user_id, k, ctx.hour, ctx.dow, ctx.days_since_prior)
    return RecommendResponse(
        user_id=user_id,
        k=k,
        fallback=False,
        model_version=store.model_version,
        context=ctx,
        items=_items(store, ranked),
    )


@app.get("/users/{user_id}/history", response_model=HistoryResponse)
def history(
    request: Request,
    user_id: Annotated[int, PathParam(gt=0)],
    n: Annotated[int, Query(ge=1, le=MAX_K)] = 20,
) -> HistoryResponse:
    store = store_of(request)
    if not store.has_user(user_id):
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    items = []
    for pid, times in store.history(user_id, n):
        name, aisle, _ = store.products.get(pid, ("unknown", "unknown", "unknown"))
        items.append(
            HistoryItem(product_id=pid, product_name=name, aisle=aisle, times_bought=times)
        )
    return HistoryResponse(user_id=user_id, items=items)


@app.get("/similar/{product_id}", response_model=SimilarResponse)
def similar(
    request: Request,
    product_id: Annotated[int, PathParam(gt=0)],
    k: Annotated[int, Query(ge=1, le=MAX_K)] = 10,
) -> SimilarResponse:
    store = store_of(request)
    ranked = store.similar(product_id, k)
    if ranked is None:
        raise HTTPException(status_code=404, detail=f"product {product_id} has no embedding")
    name = store.products.get(product_id, ("unknown",))[0]
    return SimilarResponse(product_id=product_id, product_name=name, items=_items(store, ranked))
