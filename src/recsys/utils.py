"""Logging and small helpers shared by the pipeline and the API."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

import structlog


def configure_logging(level: int = logging.INFO, json: bool = False) -> None:
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level)
    renderer = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer(colors=False)
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


@contextmanager
def timer(log: structlog.stdlib.BoundLogger, step: str) -> Iterator[None]:
    start = time.perf_counter()
    log.info("start", step=step)
    yield
    log.info("done", step=step, seconds=round(time.perf_counter() - start, 2))
