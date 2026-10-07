"""Pydantic request/response models for the recommendation API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RecommendedItem(BaseModel):
    product_id: int
    product_name: str
    aisle: str
    department: str
    score: float


class Context(BaseModel):
    hour: int = Field(ge=0, le=23)
    dow: int = Field(ge=0, le=6)
    days_since_prior: float = Field(ge=0, le=30)
    source: Literal["request", "user_default", "none"]


class RecommendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    user_id: int
    k: int
    fallback: bool = Field(description="True when the user is unknown -> popularity list")
    model_version: str
    context: Context | None
    items: list[RecommendedItem]


class HistoryItem(BaseModel):
    product_id: int
    product_name: str
    aisle: str
    times_bought: int


class HistoryResponse(BaseModel):
    user_id: int
    items: list[HistoryItem]


class SimilarResponse(BaseModel):
    product_id: int
    product_name: str
    items: list[RecommendedItem]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    artifacts_loaded: bool
    n_users: int


class MetadataResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    trained_at: str
    best_iteration: int
    sample: bool
    n_users: int
    n_candidate_rows: int
    static_features: list[str]
    context_features: list[str]
    metrics: dict
