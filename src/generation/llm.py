"""Phase 2 — LLM access over the local Ollama server.

Thin HTTP client (stdlib only) around Ollama's /api/generate. The model and
host are configurable; the baseline uses the already-pulled ``llama3``.

Generation is grounded and citation-based; see generator.py.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    """Raised when the Ollama server is unreachable or returns an error."""


class OllamaLLM:
    """Minimal blocking client for a local Ollama server."""

    def __init__(
        self,
        model: str = "llama3",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        num_predict: int = 512,
        num_gpu: int | None = None,
        timeout_seconds: int = 180,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_gpu = num_gpu
        self.timeout_seconds = timeout_seconds

    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """True when the server answers /api/tags (cheap health check)."""
        try:
            with urllib.request.urlopen(
                f"{self.host}/api/tags", timeout=5
            ) as response:
                return response.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate a completion (non-streaming). Raises OllamaError on failure."""
        payload: dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        # num_gpu=0 forces CPU inference; needed when the local GPU cannot run
        # the model (e.g. CUDA PTX toolchain errors on older GPUs).
        if self.num_gpu is not None:
            payload["options"]["num_gpu"] = self.num_gpu
        if system:
            payload["system"] = system

        request = urllib.request.Request(
            f"{self.host}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            raise OllamaError(f"Ollama HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise OllamaError(
                f"Cannot reach Ollama at {self.host}: {exc.reason}. "
                "Start it with: ollama serve"
            ) from exc
        except TimeoutError as exc:
            raise OllamaError(
                f"Ollama timed out after {self.timeout_seconds}s"
            ) from exc

        if body.get("error"):
            raise OllamaError(f"Ollama error: {body['error']}")
        return (body.get("response") or "").strip()
