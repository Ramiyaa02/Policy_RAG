"""Phase 7 — Application layer (SRS §22 Phase 7, §21 deployment concept).

FastAPI service exposing the verified Phase 5 pipeline over HTTP:

- ``GET  /``                the bundled chat UI (single static page, no build step)
- ``GET  /api/health``      liveness + LLM availability
- ``POST /api/query``       JSON request/response over the verified pipeline
- ``GET  /api/stream?q=..`` Server-Sent-Events variant of the same pipeline
                            (emits ``working`` stage events, then the final result)
- ``GET  /api/reports``     index of the phase 1-6 evaluation/report artifacts
- ``GET  /api/reports/{name}`` one report artifact (text)

Access control (SRS: "authentication/access control where required"): if the
``POLICY_RAG_API_KEY`` environment variable (or ``--api-key`` on run_phase7.py)
is set, every ``/api/*`` route requires a matching ``X-API-Key`` header. When it
is unset the demo/API is open — the deployment is local-first, so the key is
optional by design.

Logging: every API request is appended as one JSON line to
``reports/phase7_api_log.jsonl`` (path, status, duration, query length, error) —
append-only, machine-local, git-ignored like the other phase logs.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from typing import Annotated
from pydantic import BaseModel, Field, field_validator

REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
REPORTS_DIR = REPO_ROOT / "reports"

logger = logging.getLogger("phase7.api")


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int | None = Field(default=None, ge=1, le=30)

    @field_validator("query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped


class AppState:
    """Lazy pipeline holder — the agent build touches the vector store and BM25."""

    def __init__(self, loader: Callable[[], tuple[Any, str]]) -> None:
        self._loader = loader
        self._lock = threading.Lock()
        self._agent: Any | None = None
        self._llm_mode = "unknown"
        self._load_error: str | None = None

    def agent(self) -> Any:
        if self._agent is None:
            with self._lock:
                if self._agent is None:
                    self._agent, self._llm_mode = self._loader()
        return self._agent

    @property
    def llm_mode(self) -> str:
        return self._llm_mode

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def fail(self, message: str) -> None:
        self._load_error = message


def _default_loader(use_extractive: bool, config_path: str | None = None):
    """Build the verified agent exactly like ``run_phase5.py`` does."""
    from run_phase5 import build_agent, load_config

    config = load_config(config_path or str(REPO_ROOT / "configs" / "phase5.json"))
    agent = build_agent(config, use_extractive=use_extractive)
    return agent, ("extractive (no LLM)" if use_extractive else "verified (LLM)")


def create_app(
    agent: Any | None = None,
    llm_mode: str = "verified (LLM)",
    loader: Callable[[], tuple[Any, str]] | None = None,
    api_key_env: str = "POLICY_RAG_API_KEY",
) -> FastAPI:
    """Create the FastAPI app.

    Pass ``agent`` (tests, embedders) to inject a ready pipeline; otherwise
    ``loader`` (default: the run_phase5 builder) is used on first request.
    """

    if loader is None:
        loader = lambda: _default_loader(False)  # noqa: E731
    state = AppState(loader)
    if agent is not None:
        state._agent, state._llm_mode = agent, llm_mode

    app = FastAPI(
        title="Policy Intelligence — Verified RAG API",
        description=(
            "Phase 7 application layer: verified, citation-backed question "
            "answering over Indian insurance policy wordings.\n\n"
            "- `/api/query` — JSON question → verified answer + citations + grounding\n"
            "- `/api/stream` — same pipeline as Server-Sent-Events\n"
            "- `/api/health` — liveness + model info\n"
            "- `/api/reports` — phase 1-6 evaluation artifacts"
        ),
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.app_state = state
    app.state.api_log_path = REPORTS_DIR / "phase7_api_log.jsonl"

    # ------------------------------------------------------------------ auth

    def check_api_key(api_key: Annotated[str | None, Header(alias="X-API-Key")] = None) -> None:
        expected = os.environ.get(api_key_env, "")
        if not expected:
            return
        if api_key != expected:
            raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    key_dep = Depends(check_api_key)

    # ----------------------------------------------------------------- log

    @app.middleware("http")
    async def api_log(request, call_next):
        t0 = time.time()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception as exc:  # pragma: no cover - logged and re-raised
            status = 500
            _append_jsonl(
                app.state.api_log_path,
                {
                    "ts": _now(),
                    "path": request.url.path,
                    "status": 500,
                    "ms": round((time.time() - t0) * 1000, 1),
                    "error": repr(exc),
                },
            )
            raise
        _append_jsonl(
            app.state.api_log_path,
            {
                "ts": _now(),
                "path": request.url.path,
                "status": status,
                "ms": round((time.time() - t0) * 1000, 1),
            },
        )
        return response

    # -------------------------------------------------------------- routes

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", dependencies=[key_dep])
    def health() -> dict[str, Any]:
        agent_ready = True
        llm_available = None
        model = None
        try:
            agent = state.agent()
            llm = getattr(getattr(agent, "generator", None), "llm", None)
            llm_available = None if llm is None else llm.is_available()
            model = getattr(llm, "model", None)
        except Exception as exc:  # noqa: BLE001 — health reports why
            agent_ready = False
            state.fail(str(exc))
            logger.exception("pipeline build failed")
        return {
            "status": "ok" if agent_ready else "degraded",
            "pipeline": "phase5-verified",
            "llm_available": llm_available if llm_available is None else bool(llm_available),
            "model": model,
            "llm_mode": state.llm_mode,
        }

    @app.post("/api/query", dependencies=[key_dep])
    def api_query(body: QueryRequest) -> dict[str, Any]:
        agent = state.agent()
        response = agent.run(body.query, k=body.k)
        return _serialize(response, state.llm_mode)

    @app.get("/api/stream", dependencies=[key_dep])
    def api_stream(q: str = Query(min_length=1, max_length=2000), k: int | None = Query(default=None, ge=1, le=30)) -> StreamingResponse:
        def events() -> Iterator[str]:
            yield f"data: {json.dumps({'status': 'working', 'stage': 'planning retrieval…'})}\n\n"
            try:
                agent = state.agent()
                response = agent.run(q, k=k)
                yield "data: " + json.dumps({"status": "done", "result": _serialize(response, state.llm_mode)}) + "\n\n"
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI
                logger.exception("query failed")
                yield "data: " + json.dumps({"status": "error", "detail": str(exc)}) + "\n\n"

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/reports", dependencies=[key_dep])
    def reports_index() -> dict[str, Any]:
        items = []
        for path in sorted(REPORTS_DIR.iterdir()):
            if path.is_file() and path.suffix in {".json", ".md"}:
                items.append(
                    {
                        "name": path.name,
                        "bytes": path.stat().st_size,
                        "modified": path.stat().st_mtime,
                    }
                )
        return {"reports": items}

    @app.get("/api/reports/{name}", dependencies=[key_dep])
    def report_file(name: str):
        path = (REPORTS_DIR / name).resolve()
        if path.parent != REPORTS_DIR.resolve() or path.suffix not in {".json", ".md"}:
            raise HTTPException(status_code=404, detail="not found")
        if not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        media = "application/json" if path.suffix == ".json" else "text/markdown"
        return PlainTextResponse(path.read_text(encoding="utf-8"), media_type=media)

    return app


# ----------------------------------------------------------------- helpers


def _serialize(response: Any, llm_mode: str) -> dict[str, Any]:
    """VerifiedResponse → the JSON contract the UI consumes."""
    citations = []
    for i, citation in enumerate(getattr(response, "citations", []) or [], start=1):
        entry = {"n": i}
        if isinstance(citation, dict):
            entry.update(citation)
        else:  # Citation dataclass
            entry.update(getattr(citation, "to_dict", lambda: citation)())
        citations.append(entry)
    grounding = {
        "supported": getattr(response, "supported_count", 0)
        or sum(1 for v in getattr(response, "verifications", []) or [] if getattr(v.verdict, "value", "") == "supported"),
        "contradicted": getattr(response, "contradicted_count", None),
        "insufficient": getattr(response, "insufficient_count", None),
        "total": len(getattr(response, "verifications", []) or []),
        "unsupported_claims": getattr(response, "unsupported_claims", []) or [],
        "note": getattr(response, "notes", []) or [],
    }
    return {
        "query": getattr(response, "query", ""),
        "answer": getattr(response, "final_answer", "") or getattr(response, "answer", ""),
        "query_type": getattr(response, "query_type", ""),
        "method": getattr(response, "method", ""),
        "model": getattr(response, "model", ""),
        "llm_mode": llm_mode,
        "products": getattr(response, "products", []) or [],
        "criteria": getattr(response, "criteria", []) or [],
        "citations": citations,
        "abstained": bool(getattr(response, "abstained", False)),
        "clarification": getattr(response, "clarification", None),
        "grounding": grounding,
        "latency_ms": getattr(response, "latency_ms", {}) or {},
        "steps": getattr(response, "steps", []) or [],
        "evidence_count": getattr(response, "evidence_count", 0),
    }


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except OSError:  # logging must never take the API down
        logger.warning("could not write API log to %s", path)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z"
