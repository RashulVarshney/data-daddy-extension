"""Structured JSON logging with a per-request trace id."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid

trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        d = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": trace_id_var.get(),
        }
        d.update(getattr(record, "fields", {}))
        return json.dumps(d, default=str)


def get_logger(name: str = "datadaddy", stream=None) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler(stream or sys.stdout)
        h.setFormatter(JsonFormatter())
        log.addHandler(h)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


def log_event(log: logging.Logger, msg: str, **fields):
    log.info(msg, extra={"fields": fields})
