"""Phase 5 — Verified end-to-end agent (SRS §5.2, §8, §11, FR-015→FR-018).

Composes the Phase 4 agent workflow (Agent 1-4, 6) with the grounding layer
(Agent 5 + response handler):

    analyze -> plan -> retrieve -> generate/compare
        -> claim extraction -> evidence verification -> response shaping

Short-circuits inherited from Phase 4 keep working: out-of-domain queries and
plan-level clarification needs skip retrieval and generation entirely.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import src.agent.agent as agent_module

from .handler import GroundingHandler
from .verifier import GroundingValidator
from .schema import VerifiedResponse

logger = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return round((time.time() - start) * 1000, 1)


class VerifiedAgent:
    """End-to-end agentic query processing with grounding verification."""

    def __init__(
        self,
        analyzer: Any,
        planner: Any,
        executor: Any,
        generator: Any,
        validator: GroundingValidator | None = None,
        handler: GroundingHandler | None = None,
        comparison: Any | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.planner = planner
        self.executor = executor
        self.generator = generator
        self.comparison = comparison
        self.validator = validator or GroundingValidator()
        self.handler = handler or GroundingHandler()

    # ------------------------------------------------------------------

    def run(self, query: str, k: int | None = None) -> VerifiedResponse:
        t0 = time.time()
        raw: dict[str, Any] = {"query": query}

        t = time.time()
        analysis = self.analyzer.analyze(query)
        plan = self.planner.plan(analysis, k=k)
        analysis_ms = _ms(t)

        base_latency = {"analysis": analysis_ms, "planning": 0.0}
        raw.update(
            {
                "query_type": analysis.query_type.value,
                "products": plan.products,
                "criteria": plan.criteria,
                "steps": [s.to_dict() for s in plan.steps],
            }
        )

        # FR-019 — out-of-domain short-circuit (no retrieval, no generation).
        if agent_module.QueryType.OUT_OF_DOMAIN.value == raw["query_type"]:
            base_latency["total"] = _ms(t0)
            return _short_circuit(
                raw,
                agent_module.OUT_OF_DOMAIN_ANSWER,
                base_latency,
                abstained=True,
            )

        # FR-018 — plan-level clarification short-circuit.
        if plan.needs_clarification:
            base_latency["total"] = _ms(t0)
            return _short_circuit(
                raw,
                plan.clarification_question or "",
                base_latency,
                abstained=True,
                clarification=plan.clarification_question,
            )

        # FR-014 — multi-step retrieval.
        t = time.time()
        bundles = self.executor.execute(plan)
        retrieval_ms = _ms(t)
        chunks = self.executor.flatten(bundles)
        chunks_by_id = {getattr(c, "chunk_id", ""): c for c in chunks}
        raw["evidence_count"] = len(chunks)

        # Agent 4 comparison / Agent 6 generated answer.
        t = time.time()
        if plan.needs_comparison and self.comparison is not None:
            result = self.comparison.answer(query, plan.products, plan.criteria, chunks)
        else:
            result = self.generator.answer(query, chunks)
        generation_ms = _ms(t)

        raw.update(
            {
                "answer": result.answer,
                "citations": [c.to_dict() for c in result.citations],
                "model": getattr(result, "model", ""),
                "abstained": bool(result.abstained),
            }
        )

        # Abstained generators produce no claims to verify.
        if result.abstained:
            t = time.time()
            response = self.handler.build(raw, [], [], query_type=raw["query_type"])
            response.latency_ms = {
                "analysis": analysis_ms,
                "planning": 0.0,
                "retrieval": retrieval_ms,
                "generation": generation_ms,
                "verification": _ms(t),
                "total": _ms(t0),
            }
            return response

        # Agent 5 — claim extraction + evidence verification.
        t = time.time()
        evidence = _aligned_evidence(result, chunks_by_id)
        verifications = self.validator.verify(result.answer, evidence)
        verification_ms = _ms(t)

        t = time.time()
        claims = [v.claim for v in verifications]
        response = self.handler.build(
            raw, claims, verifications, query_type=raw["query_type"]
        )
        response.latency_ms = {
            "analysis": analysis_ms,
            "planning": 0.0,
            "retrieval": retrieval_ms,
            "generation": generation_ms,
            "verification": verification_ms,
            "total": _ms(t0),
        }
        return response


def _short_circuit(
    raw: dict[str, Any],
    answer: str,
    latency: dict[str, float],
    abstained: bool,
    clarification: str | None = None,
) -> VerifiedResponse:
    return VerifiedResponse(
        query=raw.get("query", ""),
        query_type=raw.get("query_type", ""),
        method="verified",
        answer=answer,
        final_answer=answer,
        abstained=abstained,
        clarification=clarification,
        products=raw.get("products", []),
        criteria=raw.get("criteria", []),
        steps=raw.get("steps", []),
        latency_ms=latency,
    )


def _aligned_evidence(result: Any, chunks_by_id: dict[str, Any]) -> list[Any]:
    """Evidence list aligned with citation markers: index i <-> marker [i+1]."""
    texts_by_marker: dict[int, Any] = {}
    for citation in getattr(result, "citations", []) or []:
        if hasattr(citation, "marker"):
            marker = int(citation.marker)
            chunk_id = str(getattr(citation, "chunk_id", ""))
            excerpt = str(getattr(citation, "excerpt", ""))
        else:
            marker = int(citation.get("marker", 0))
            chunk_id = citation.get("chunk_id", "")
            excerpt = citation.get("excerpt", "")
        if marker < 1:
            continue
        chunk = chunks_by_id.get(chunk_id)
        texts_by_marker[marker] = chunk if chunk is not None else {
            "chunk_id": chunk_id,
            "text": excerpt,
        }
    if not texts_by_marker:
        return []
    width = max(texts_by_marker)
    return [texts_by_marker.get(i + 1, {"chunk_id": "", "text": ""}) for i in range(width)]
