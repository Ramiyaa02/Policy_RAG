"""Phase 4 — Agent orchestrator (SRS §5.2, §11, FR-014/FR-018/FR-019).

Wires the agents into one workflow:

    analyze (Agent 1) -> plan (Agent 2) -> retrieve (Agent 3)
        -> compare (Agent 4) | answer (Agent 6) -> AgentResponse

It handles the two short-circuits the SRS calls out explicitly:

- **out-of-domain** (FR-019): a query with no insurance footing is answered with a
  scope statement, without spending retrieval or generation.
- **clarification** (FR-018): a plan that lacks enough information returns a
  targeted question instead of guessing.

Grounding verification (Agent 5) is Phase 5, so claims are cited but not yet
individually verified.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .schema import AgentResponse, QueryType, RetrievalPlan

logger = logging.getLogger(__name__)

OUT_OF_DOMAIN_ANSWER = (
    "This question is outside the insurance policy knowledge base I support. "
    "I can only answer from the ingested insurance policy documents — for example "
    "coverage, exclusions, waiting periods, eligibility and product comparisons."
)


class PolicyRAGAgent:
    """End-to-end agentic query processing over the policy corpus."""

    def __init__(
        self,
        analyzer: Any,
        planner: Any,
        executor: Any,
        generator: Any,
        comparison: Any | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.planner = planner
        self.executor = executor
        self.generator = generator
        self.comparison = comparison

    # ------------------------------------------------------------------

    def plan(self, query: str, k: int | None = None) -> tuple[Any, RetrievalPlan]:
        """Analyze + plan without retrieving (used by the ``analyze`` CLI)."""
        analysis = self.analyzer.analyze(query)
        return analysis, self.planner.plan(analysis, k=k)

    def run(self, query: str, k: int | None = None) -> AgentResponse:
        t0 = time.time()

        t = time.time()
        analysis = self.analyzer.analyze(query)
        analysis_ms = _ms(t)

        t = time.time()
        plan = self.planner.plan(analysis, k=k)
        plan_ms = _ms(t)

        base = AgentResponse(
            query=query,
            query_type=analysis.query_type.value,
            method=analysis.method,
            requirements=analysis.requirements.to_dict(),
            criteria=plan.criteria,
            products=plan.products,
            steps=[s.to_dict() for s in plan.steps],
        )
        base.latency_ms = {
            "analysis": analysis_ms,
            "planning": plan_ms,
            "retrieval": 0.0,
            "generation": 0.0,
            "total": _ms(t0),
        }

        # FR-019 — out-of-domain short-circuit.
        if analysis.query_type is QueryType.OUT_OF_DOMAIN:
            base.answer = OUT_OF_DOMAIN_ANSWER
            base.abstained = True
            base.latency_ms["total"] = _ms(t0)
            return base

        # FR-018 — clarification short-circuit.
        if plan.needs_clarification:
            base.answer = plan.clarification_question or ""
            base.clarification = plan.clarification_question
            base.abstained = True
            base.latency_ms["total"] = _ms(t0)
            return base

        # FR-014 — multi-step retrieval.
        t = time.time()
        bundles = self.executor.execute(plan)
        retrieval_ms = _ms(t)
        chunks = self.executor.flatten(bundles)
        base.evidence_count = len(chunks)
        base.latency_ms["retrieval"] = retrieval_ms

        # Agent 4 — comparison / recommendation; Agent 6 — grounded answer.
        t = time.time()
        if plan.needs_comparison and self.comparison is not None:
            result = self.comparison.answer(query, plan.products, plan.criteria, chunks)
        else:
            result = self.generator.answer(query, chunks)
        base.latency_ms["generation"] = _ms(t)

        base.answer = result.answer
        base.citations = [c.to_dict() for c in result.citations]
        base.used_markers = list(result.used_markers)
        base.abstained = bool(result.abstained)
        base.model = getattr(result, "model", "")
        base.latency_ms["total"] = _ms(t0)
        return base


def _ms(start: float) -> float:
    return round((time.time() - start) * 1000, 1)
