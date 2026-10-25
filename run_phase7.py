"""Phase 7 — Policy Intelligence API + UI.

Serves the verified Phase 5 pipeline over HTTP (FastAPI) with the bundled
single-page chat UI, optional API-key access control and per-request JSONL
logging (SRS §22 Phase 7: FastAPI, user interface, access control, logging).

    python run_phase7.py serve                       # http://127.0.0.1:8000
    python run_phase7.py serve --port 8600 --host 0.0.0.0
    python run_phase7.py serve --no-llm              # extractive mode, no Ollama
    python run_phase7.py serve --api-key s3cret      # require X-API-Key on /api/*

Endpoints are documented at http://<host>:<port>/docs (auto-generated OpenAPI).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 7: API + UI server")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="run the FastAPI server")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--config", default="configs/phase5.json")
    p_serve.add_argument("--no-llm", action="store_true", help="extractive pipeline (no Ollama needed)")
    p_serve.add_argument(
        "--api-key",
        help="require X-API-Key == this value on /api/* (also settable via POLICY_RAG_API_KEY)",
    )

    args = parser.parse_args()
    if args.command == "serve":
        return _serve(args)
    return 2


def _serve(args) -> int:  # noqa: ANN001
    if args.api_key:
        os.environ["POLICY_RAG_API_KEY"] = args.api_key

    from src.api.service import _default_loader, create_app

    app = create_app(loader=lambda: _default_loader(args.no_llm, args.config))
    try:
        import uvicorn
    except ImportError:  # pragma: no cover
        print("uvicorn is missing — pip install -r requirements.txt", file=sys.stderr)
        return 2

    print(f"Policy Intelligence API + UI  →  http://{args.host}:{args.port}  (docs at /docs)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
