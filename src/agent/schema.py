"""Phase 4 — Agentic layer data schema (SRS §5.2, §10, §11).

Shared dataclasses and enums for the offline agent workflow:

    User Query
       │
       ▼  QueryAnalyzer (Agent 1)      -> QueryAnalysis
    Query type · user requirements · criteria · products
       │
       ▼  RetrievalPlanner (Agent 2)   -> RetrievalPlan
    Sub-queries with filters (multi-step / comparison / clarification)
       │
       ▼  PlanExecutor (Agent 3)       -> list[EvidenceBundle]
    Hybrid retrieval per step
       │
       ▼  ComparisonGenerator / AnswerGenerator (Agent 4/6) -> AgentResponse

Grounding validation (Agent 5) is Phase 5, so claims here are cited but not yet
verified.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """SRS FR-007 query categories."""

    POLICY_INFORMATION = "policy_information"
    COVERAGE = "coverage"
    ELIGIBILITY = "eligibility"
    EXCLUSION = "exclusion"
    WAITING_PERIOD = "waiting_period"
    PRODUCT_COMPARISON = "product_comparison"
    RECOMMENDATION = "recommendation"
    MULTI_STEP = "multi_step"
    OUT_OF_DOMAIN = "out_of_domain"


# SRS FR-012 comparison criteria.
COMPARISON_CRITERIA = (
    "coverage",
    "eligibility",
    "exclusions",
    "waiting period",
    "benefits",
    "limitations",
)

# Default criteria a comparison/recommendation plan will retrieve when the query
# does not name any.
DEFAULT_COMPARISON_CRITERIA = ("coverage", "waiting period", "exclusions", "eligibility")


def _clean(value: Any) -> Any:
    """Recursively make a value JSON-serialisable (enums -> str, dataclass -> dict)."""
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


@dataclass
class UserRequirements:
    """Structured user requirements for recommendation queries (SRS FR-013)."""

    age: int | None = None
    sum_insured: str | None = None
    coverage_objective: str | None = None
    priorities: list[str] = field(default_factory=list)
    tenure: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not (
            self.age
            or self.sum_insured
            or self.coverage_objective
            or self.priorities
            or self.tenure
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QueryAnalysis:
    """Output of Agent 1 — Query Analyzer."""

    query: str
    query_type: QueryType = QueryType.POLICY_INFORMATION
    confidence: float = 0.0
    method: str = "rules"  # "rules" | "llm"
    is_complex: bool = False
    requirements: UserRequirements = field(default_factory=UserRequirements)
    criteria: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    matched_cues: list[str] = field(default_factory=list)
    detected_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class PlanStep:
    """One retrieval operation in a :class:`RetrievalPlan` (SRS FR-014)."""

    step_id: int
    description: str
    query: str
    product: str | None = None
    document_id: str | None = None
    chunk_type: str | None = None
    top_k: int = 5
    criteria: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))


@dataclass
class RetrievalPlan:
    """Output of Agent 2 — Retrieval Planner."""

    query: str
    query_type: QueryType = QueryType.POLICY_INFORMATION
    steps: list[PlanStep] = field(default_factory=list)
    criteria: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    needs_comparison: bool = False
    needs_clarification: bool = False
    clarification_question: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "criteria": self.criteria,
            "products": self.products,
            "needs_comparison": self.needs_comparison,
            "needs_clarification": self.needs_clarification,
            "clarification_question": self.clarification_question,
            "steps": [s.to_dict() for s in self.steps],
        }


@dataclass
class EvidenceBundle:
    """Retrieved evidence for a single plan step."""

    step: PlanStep
    chunks: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step.to_dict(),
            "chunk_ids": [getattr(c, "chunk_id", "") for c in self.chunks],
        }


@dataclass
class AgentResponse:
    """Final user-facing response from the agent workflow."""

    query: str
    answer: str = ""
    query_type: str = QueryType.POLICY_INFORMATION.value
    method: str = "rules"
    requirements: dict[str, Any] = field(default_factory=dict)
    criteria: list[str] = field(default_factory=list)
    products: list[str] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    used_markers: list[int] = field(default_factory=list)
    abstained: bool = False
    clarification: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    model: str = ""
    latency_ms: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _clean(asdict(self))
