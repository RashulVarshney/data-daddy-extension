"""FastAPI service: POST /scan, GET /search, POST /ask. JSON logs with a trace id per request."""

from __future__ import annotations

import io
import json
import os
import pathlib
import time

import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from ..agent.catalog import Catalog
from ..agent.loop import run_agent
from ..agent.naive_mock import NaiveMockAgent
from ..agent.providers import ProviderConfigError, get_provider
from ..agent.tools import Guardrails, Session, catalog_to_search
from ..classifier.model import ColumnClassifier
from ..classifier.scan import DEFAULT_MODEL, scan_dataframe
from ..common.logging import get_logger, log_event, new_trace_id, trace_id_var
from ..retrieval.corpus import load_cards
from ..retrieval.search import DatasetSearch

MAX_UPLOAD = 5 * 1024 * 1024
log = get_logger()


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


def _sessions() -> dict[str, int]:
    return json.loads(os.environ.get("SESSIONS_JSON", '{"demo-tier1": 1, "demo-tier2": 2, "demo-tier3": 3}'))


def create_app(
    model=None, search: DatasetSearch | None = None, catalog: Catalog | None = None, provider=None
) -> FastAPI:
    app = FastAPI(title="datadaddy-ai", version="0.1.0")
    state: dict = {"model": model, "search": search, "catalog": catalog, "provider": provider}

    def get_model():
        if state["model"] is None:
            if not pathlib.Path(DEFAULT_MODEL).exists():
                raise HTTPException(503, "classifier model missing; run `make eval-classifier`")
            state["model"] = ColumnClassifier.load(DEFAULT_MODEL)
        return state["model"]

    def get_search() -> DatasetSearch:
        if state["search"] is None:
            p = pathlib.Path("data/hf_cards/cards.jsonl")
            if p.exists():
                state["search"] = DatasetSearch(load_cards(p))
            else:
                state["search"] = catalog_to_search(get_catalog())
        return state["search"]

    def get_catalog() -> Catalog:
        if state["catalog"] is None:
            state["catalog"] = Catalog.load()
        return state["catalog"]

    @app.middleware("http")
    async def trace(request: Request, call_next):
        tid = request.headers.get("x-trace-id") or new_trace_id()
        trace_id_var.set(tid)
        t0 = time.perf_counter()
        try:
            resp = await call_next(request)
        except Exception as e:  # noqa: BLE001
            log_event(log, "request_error", path=request.url.path, error=repr(e))
            raise
        resp.headers["x-trace-id"] = tid
        log_event(
            log,
            "request",
            method=request.method,
            path=request.url.path,
            status=resp.status_code,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )
        return resp

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/scan")
    async def scan(file: UploadFile = File(...), headerless: bool = False):
        raw = await file.read(MAX_UPLOAD + 1)
        if len(raw) > MAX_UPLOAD:
            raise HTTPException(413, "file too large (max 5MB)")
        try:
            df = pd.read_csv(io.BytesIO(raw), nrows=5000, dtype=str, keep_default_na=False, na_values=[""])
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"cannot parse CSV: {e}") from e
        rep = scan_dataframe(df, get_model(), headerless)
        log_event(
            log,
            "scan",
            filename=file.filename,
            rows=rep["n_rows"],
            cols=rep["n_columns"],
            risk=rep["risk"]["level"],
        )
        return rep

    @app.get("/search")
    def search_ep(
        q: str, k: int = 5, task: str | None = None, language: str | None = None, license: str | None = None
    ):  # noqa: A002
        if not 1 <= k <= 20:
            raise HTTPException(422, "k must be in 1..20")
        t0 = time.perf_counter()
        res = get_search().search(q, k, {"task": task, "language": language, "license": license})
        log_event(
            log, "search", query=q, k=k, n=len(res), latency_ms=round((time.perf_counter() - t0) * 1000, 2)
        )
        return {"query": q, "results": res}

    @app.post("/ask")
    def ask(body: AskBody, x_session_token: str | None = Header(default=None)):
        tier = _sessions().get(x_session_token or "")
        if tier is None:  # buyer tier comes only from trusted session state, never from the request body
            raise HTTPException(401, "missing or unknown X-Session-Token")
        prov = state["provider"]
        if prov is None:
            name = (os.environ.get("LLM_PROVIDER") or "").lower()
            try:
                prov = NaiveMockAgent() if name == "mock" else get_provider(name)
            except ProviderConfigError as e:
                raise HTTPException(503, str(e)) from e
            state["provider"] = prov
        cat = get_catalog()
        rec = run_agent(
            prov,
            cat,
            Guardrails.all_on(),
            Session(buyer_tier=tier),
            body.question,
            "api",
            classifier=get_model(),
            search=catalog_to_search(cat),
        )
        for e in rec.tool_events:
            log_event(
                log,
                "tool_call",
                tool=e["name"],
                status=e["status"],
                reason=e["reason"],
                latency_ms=e["latency_ms"],
            )
        log_event(
            log,
            "ask",
            provider=rec.provider,
            model=rec.model,
            steps=rec.steps,
            blocked=rec.blocked,
            input_tokens=rec.usage["input_tokens"],
            output_tokens=rec.usage["output_tokens"],
            llm_latency_s=round(rec.llm_latency_s, 3),
            citations_verified=rec.citations.get("verified"),
        )
        return {
            "answer": rec.delivered,
            "blocked": rec.blocked,
            "block_reasons": rec.block_reasons,
            "citations": rec.citations,
            "buyer_tier": tier,
            "provider": f"{rec.provider}/{rec.model}",
            "usage": rec.usage,
            "tool_calls": [{"tool": e["name"], "status": e["status"]} for e in rec.tool_events],
        }

    return app


app = create_app()
