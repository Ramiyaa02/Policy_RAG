"""Phase 4 — Agent 2: Retrieval Planner (SRS FR-014, FR-018, §11).

Turns a :class:`QueryAnalysis` into an explicit :class:`RetrievalPlan` — the
list of retrieval operations (sub-queries plus metadata filters) the executor
will run. This is where "agentic" behaviour lives:

- **simple queries** -> one semantic/keyword step;
- **comparisons** -> one step per (product x criterion), each with a product
  metadata filter, so evidence is gathered per policy (FR-012);
- **recommendations** -> candidate products x criteria, seeded with the user's
  requirements (FR-013);
- **multi-step queries** -> the question is decomposed into sub-questions or
  facets, one step each (FR-014);
- **clarification** (FR-018) -> when a comparison names fewer than two policies,
  or a recommendation carries no requirements, the plan asks a targeted question
  instead of guessing.

Steps are capped (``max_steps``, ``max_products``) so a broad comparison cannot
balloon into hundreds of retrievals (risk R5).
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from .schema import (
    DEFAULT_COMPARISON_CRITERIA,
    PlanStep,
    QueryAnalysis,
    QueryType,
    RetrievalPlan,
)

logger = logging.getLogger(__name__)


class RetrievalPlanner:
    """Agent 2 — plans retrieval operations for an analysed query."""

    def __init__(
        self,
        products: Sequence[str] = (),
        default_k: int = 5,
        max_steps: int = 24,
        max_products: int = 4,
    ) -> None:
        self.products = [p for p in products if p]
        self.default_k = default_k
        self.max_steps = max_steps
        self.max_products = max_products

    # ------------------------------------------------------------------

    def plan(self, analysis: QueryAnalysis, k: int | None = None) -> RetrievalPlan:
        k = k or self.default_k
        if analysis.query_type is QueryType.OUT_OF_DOMAIN:
            return RetrievalPlan(
                query=analysis.query,
                query_type=analysis.query_type,
                products=analysis.products,
            )
        if analysis.query_type is QueryType.PRODUCT_COMPARISON:
            return self._comparison_plan(analysis, k)
        if analysis.query_type is QueryType.RECOMMENDATION:
            return self._recommendation_plan(analysis, k)
        if analysis.query_type is QueryType.MULTI_STEP:
            return self._multi_step_plan(analysis, k)
        return self._simple_plan(analysis, k)

    # ------------------------------------------------------------------
    # Plan builders
    # ------------------------------------------------------------------

    def _simple_plan(self, analysis: QueryAnalysis, k: int) -> RetrievalPlan:
        step = PlanStep(
            step_id=1,
            description=f"{analysis.query_type.value} query",
            query=analysis.query,
            top_k=k,
            criteria=analysis.criteria,
        )
        return RetrievalPlan(
            query=analysis.query,
            query_type=analysis.query_type,
            steps=[step],
            criteria=analysis.criteria,
            products=analysis.products,
        )

    def _comparison_plan(self, analysis: QueryAnalysis, k: int) -> RetrievalPlan:
        products = list(analysis.products)
        text = analysis.query.lower()
        if len(products) < 2:
            # Only fall back to "all policies" when the query is explicitly
            # generic. A query that names products we cannot resolve (e.g.
            # "Policy A and Policy B") gets a clarification instead of a
            # silently wrong comparison (FR-018).
            wants_all = any(
                cue in text
                for cue in ("all", "these", "every", "various", "different", "the policies", "the plans", "policies")
            )
            if len(self.products) >= 2 and wants_all:
                products = self.products[: self.max_products]

        if len(products) < 2:
            return RetrievalPlan(
                query=analysis.query,
                query_type=analysis.query_type,
                products=products,
                criteria=analysis.criteria,
                needs_clarification=True,
                clarification_question=(
                    "Which policies would you like me to compare? Please name at least two "
                    "(for example, the Digit Health Insurance Policy and Elevate)."
                ),
            )

        criteria = analysis.criteria or list(DEFAULT_COMPARISON_CRITERIA)
        steps = self._product_criteria_steps(products, criteria, k)
        return RetrievalPlan(
            query=analysis.query,
            query_type=analysis.query_type,
            steps=steps,
            criteria=criteria,
            products=products,
            needs_comparison=True,
        )

    def _recommendation_plan(self, analysis: QueryAnalysis, k: int) -> RetrievalPlan:
        requirements = analysis.requirements
        if requirements.is_empty:
            return RetrievalPlan(
                query=analysis.query,
                query_type=analysis.query_type,
                criteria=analysis.criteria,
                products=analysis.products,
                needs_clarification=True,
                clarification_question=(
                    "To recommend a policy I need the applicant's age and the cover you care "
                    "about (for example hospitalization or maternity). Any priority such as a "
                    "low waiting period or a lower premium also helps."
                ),
            )

        criteria = analysis.criteria or list(DEFAULT_COMPARISON_CRITERIA)
        candidates = analysis.products or self.products[: self.max_products]
        steps = self._product_criteria_steps(
            candidates, criteria, k, requirements=requirements
        )
        return RetrievalPlan(
            query=analysis.query,
            query_type=analysis.query_type,
            steps=steps,
            criteria=criteria,
            products=candidates,
            needs_comparison=True,
        )

    def _multi_step_plan(self, analysis: QueryAnalysis, k: int) -> RetrievalPlan:
        sub_queries = self._decompose(analysis)
        steps = [
            PlanStep(
                step_id=i,
                description=f"sub-question {i}",
                query=sub_query,
                top_k=k,
                criteria=analysis.criteria,
            )
            for i, sub_query in enumerate(sub_queries, start=1)
        ]
        return RetrievalPlan(
            query=analysis.query,
            query_type=analysis.query_type,
            steps=steps,
            criteria=analysis.criteria,
            products=analysis.products,
            needs_comparison=False,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _product_criteria_steps(
        self,
        products: Sequence[str],
        criteria: Sequence[str],
        k: int,
        requirements=None,
    ) -> list[PlanStep]:
        products = list(products)[: self.max_products]
        criteria = list(criteria)
        steps: list[PlanStep] = []
        step_id = 1
        for product in products:
            for criterion in criteria:
                if len(steps) >= self.max_steps:
                    logger.info("Plan truncated at max_steps=%d", self.max_steps)
                    return steps
                query = self._criterion_query(criterion, product, requirements)
                steps.append(
                    PlanStep(
                        step_id=step_id,
                        description=f"{criterion} — {product}",
                        query=query,
                        product=product,
                        top_k=k,
                        criteria=[criterion],
                    )
                )
                step_id += 1
        return steps

    @staticmethod
    def _criterion_query(criterion: str, product: str, requirements) -> str:
        """Build a retrieval sub-query for one (criterion, product) pair."""
        query = f"{criterion} — {product}"
        if requirements is not None:
            extras = []
            if requirements.age and criterion in ("eligibility", "coverage"):
                extras.append(f"age {requirements.age}")
            if requirements.coverage_objective and criterion == "coverage":
                extras.append(requirements.coverage_objective)
            if extras:
                query = f"{query} ({', '.join(extras)})"
        return query

    def _decompose(self, analysis: QueryAnalysis) -> list[str]:
        """Split a complex query into sub-queries (or per-facet queries)."""
        segments = [
            seg.strip(" ?.")
            for seg in re.split(r"\s*(?:;|\band\b|\?)\s+", analysis.query)
            if seg.strip(" ?.")
        ]
        segments = [s for s in segments if len(s.split()) >= 2]
        if len(segments) >= 2:
            return segments[: self.max_steps]

        facets = analysis.detected_types or []
        if len(facets) >= 2:
            return [
                f"{facet.replace('_', ' ')} for: {analysis.query}" for facet in facets
            ][: self.max_steps]
        return [analysis.query]
